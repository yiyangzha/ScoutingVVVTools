import os
import glob
import json
import shutil
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import xgboost as xgb
import gc

from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier, plot_importance
from typing import List

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)

_EPS = 1e-12


def log_message(message):
    print(message, flush=True)


def log_warning(message):
    log_message(f"Warning: {message}")


def log_info(message):
    log_message(f"Info: {message}")

# -------------------- Config loading --------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

_cfg_path = os.environ.get("BDT_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
if not os.path.isabs(_cfg_path):
    _cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _cfg_path))

cfg     = _load_json(_cfg_path)
br_cfg  = _load_json(os.path.join(_SCRIPT_DIR, "branch.json"))
sel_cfg = _load_json(os.path.join(_SCRIPT_DIR, "selection.json"))

_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

# -------------------- Constants --------------------
RANDOM_STATE       = cfg.get("random_state", 42)
ENTRIES_PER_SAMPLE = cfg.get("entries_per_sample", 1_000_000)
TRAIN_FRACTION     = float(cfg.get("train_fraction", 0.7))
DECOR_LAMBDA       = cfg.get("decor_lambda", 30)
SUBMIT_TREES       = cfg.get("submit_trees", ["fat2"])
INPUT_ROOT         = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN      = cfg["input_pattern"]
OUTPUT_ROOT_PATTERN = cfg.get("output_root", ".")
MODEL_PATTERN      = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
CLASS_TARGET_WEIGHT = float(cfg.get("class_target_weight", 1e10))

if not 0.0 < TRAIN_FRACTION < 1.0:
    raise ValueError(f"train_fraction must be in (0, 1), got {TRAIN_FRACTION}")

# -------------------- Sample registry --------------------
SAMPLE_INFO = {}
for _rule in sample_cfg["sample"]:
    SAMPLE_INFO[_rule["name"]] = {
        "xsection":    _rule["xsection"],
        "raw_entries": _rule.get("raw_entries", -1),
        "is_MC":       _rule["is_MC"],
        "is_signal":   _rule["is_signal"],
        "sample_ID":   _rule["sample_ID"],
    }

# -------------------- Class groups --------------------
CLASS_GROUPS  = cfg["class_groups"]            # {"VVV": [...], "VH": [...], ...}
CLASS_NAMES   = list(CLASS_GROUPS.keys())      # Ordered class names.
NUM_CLASSES   = len(CLASS_NAMES)

CLASS_TYPES = {}
SIGNAL_CLASS_INDICES = []
BACKGROUND_CLASS_INDICES = []
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    _flags = [SAMPLE_INFO[_s]["is_signal"] for _s in _members]
    _kind = "single" if _flags and all(_flags) else "background"
    CLASS_TYPES[_cls] = _kind
    if _kind == "single":
        SIGNAL_CLASS_INDICES.append(_idx)
    else:
        BACKGROUND_CLASS_INDICES.append(_idx)

SAMPLE_TO_CLASS = {}
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    for _s in _members:
        SAMPLE_TO_CLASS[_s] = _idx

# Resolve the training sample list.
TRAINING_SAMPLES = [r["name"] for r in sample_cfg["sample"] if r["name"] in SAMPLE_TO_CLASS]


# -------------------- File discovery --------------------
def _sample_group(sample_name):
    return "signal" if SAMPLE_INFO[sample_name]["is_signal"] else "bkg"

def _input_files(sample_name):
    sg   = _sample_group(sample_name)
    base = INPUT_PATTERN.format(input_root=INPUT_ROOT, sample_group=sg, sample=sample_name)
    stem = base[:-5]  # Drop the ".root" suffix.
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


def _resolve_output_root(tree_name):
    output_root = OUTPUT_ROOT_PATTERN.format(tree_name=tree_name)
    if not os.path.isabs(output_root):
        output_root = os.path.normpath(os.path.join(_SCRIPT_DIR, output_root))
    return output_root


def _slugify(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _figure_path(output_root, stem):
    return os.path.join(output_root, f"{stem}.pdf")


# -------------------- Data loading --------------------
def _report_sample_weights(df_all, stage_label):
    log_message(f"{stage_label}:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask_cls = df_all["class_idx"] == cls_idx
        if not np.any(mask_cls):
            log_message(f"  {cls_name}: no entries")
            continue
        total_w = float(df_all.loc[mask_cls, "weight"].sum())
        log_message(f"  {cls_name}: total_w={total_w:.6g}")
        for sample_name in CLASS_GROUPS[cls_name]:
            mask_sample = mask_cls & (df_all["sample_name"] == sample_name)
            if not np.any(mask_sample):
                continue
            sample_w = float(df_all.loc[mask_sample, "weight"].sum())
            log_message(
                f"    {sample_name}: sum_w={sample_w:.6g}, xsec={SAMPLE_INFO[sample_name]['xsection']:.6g}"
            )


def _validate_sample_weight_totals(df_all, sample_target_totals):
    for sample_name, target_total in sample_target_totals.items():
        info = SAMPLE_INFO[sample_name]
        mask = df_all["sample_name"] == sample_name
        if not np.any(mask):
            continue
        total_w = float(df_all.loc[mask, "weight"].sum())
        if target_total <= 0.0:
            if abs(total_w) > 1e-8:
                raise RuntimeError(
                    f"Sample '{sample_name}' has non-positive target weight {target_total:.6g} "
                    f"but total weight {total_w:.6g}"
                )
            continue
        rel = abs(total_w - target_total) / max(abs(target_total), _EPS)
        if rel > 1e-6:
            raise RuntimeError(
                f"Sample '{sample_name}' weight sum {total_w:.6g} does not match target {target_total:.6g}"
            )


def _validate_class_weight_totals(df_all):
    positive_totals = []
    for cls_idx in range(NUM_CLASSES):
        total_w = float(df_all.loc[df_all["class_idx"] == cls_idx, "weight"].sum())
        if total_w > 0.0:
            positive_totals.append(total_w)
    if not positive_totals:
        raise RuntimeError("No positive class weights after normalisation.")
    ref = positive_totals[0]
    for total_w in positive_totals[1:]:
        rel = abs(total_w - ref) / max(abs(ref), _EPS)
        if rel > 1e-6:
            raise RuntimeError("Class totals are not equal after class normalisation.")


def _rebalance_class_weights(df_all):
    df_all = df_all.copy()
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = df_all["class_idx"] == cls_idx
        w_sum = float(df_all.loc[mask, "weight"].sum())
        if w_sum > 0.0:
            scale = CLASS_TARGET_WEIGHT / w_sum
            df_all.loc[mask, "weight"] *= scale
            log_message(f"  {cls_name}: total_w={w_sum:.4g}, scale={scale:.4g}")
    _validate_class_weight_totals(df_all)
    return df_all


def _segment_length(segment):
    return int(segment["entry_stop"]) - int(segment["entry_start"])


def _sum_segment_lengths(segments):
    return sum(_segment_length(segment) for segment in segments)


def _build_segments(file_infos, global_start, global_stop, max_entries=None):
    segments = []
    cursor = 0
    used_entries = 0

    for info in file_infos:
        next_cursor = cursor + int(info["entries"])
        overlap_start = max(global_start, cursor)
        overlap_stop = min(global_stop, next_cursor)
        if overlap_stop > overlap_start:
            local_start = overlap_start - cursor
            local_stop = overlap_stop - cursor
            if max_entries is not None:
                remain = max_entries - used_entries
                if remain <= 0:
                    break
                local_stop = min(local_stop, local_start + remain)
            if local_stop > local_start:
                segment = {
                    "path": info["path"],
                    "entry_start": int(local_start),
                    "entry_stop": int(local_stop),
                    "global_start": int(overlap_start),
                    "global_stop": int(overlap_start + (local_stop - local_start)),
                }
                segments.append(segment)
                used_entries += _segment_length(segment)
                if max_entries is not None and used_entries >= max_entries:
                    break
        cursor = next_cursor

    return segments


def _inspect_sample_tree(sample_name, tree_name):
    files = _input_files(sample_name)
    if not files:
        log_warning(f"no files for '{sample_name}', skipping")
        return None

    file_infos = []
    total_entries = 0
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree = uf[tree_name]
            n_entries = int(tree.num_entries)
            file_infos.append({
                "path": fpath,
                "entries": n_entries,
            })
            total_entries += n_entries

    if total_entries <= 0:
        log_warning(f"zero entries found for '{sample_name}' in tree '{tree_name}', skipping")
        return None

    train_stop = int(total_entries * TRAIN_FRACTION)
    return {
        "sample_name": sample_name,
        "file_infos": file_infos,
        "total_entries": total_entries,
        "train_stop": train_stop,
        "test_start": train_stop,
        "train_segments_full": _build_segments(file_infos, 0, train_stop),
        "train_segments_read": _build_segments(file_infos, 0, train_stop, max_entries=ENTRIES_PER_SAMPLE),
        "test_segments": _build_segments(file_infos, train_stop, total_entries),
    }


def _load_segments(tree_name, branches, segments):
    parts = []
    n_read = 0
    for segment in segments:
        with uproot.open(segment["path"]) as uf:
            tree = uf[tree_name]
            available = set(tree.keys())
            missing = [branch for branch in branches if branch not in available]
            if missing:
                raise KeyError(
                    f"Missing branches in {segment['path']}:{tree_name}: {', '.join(missing[:10])}"
                    + (" ..." if len(missing) > 10 else "")
                )
            df_part = tree.arrays(
                branches,
                library="pd",
                entry_start=int(segment["entry_start"]),
                entry_stop=int(segment["entry_stop"]),
            )
            parts.append(df_part)
            n_read += len(df_part)

    if not parts:
        return None, 0

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df, n_read


def build_split_plans(tree_name):
    split_plans = {}
    for sample_name in TRAINING_SAMPLES:
        plan = _inspect_sample_tree(sample_name, tree_name)
        if plan is not None:
            split_plans[sample_name] = plan
    if not split_plans:
        raise RuntimeError(f"No data available for tree '{tree_name}'")
    return split_plans


def prepare_split_data(tree_name, branches, split_name, split_plans, shuffle):
    dfs = []
    sample_target_totals = {}

    reweight_branches = list(cfg.get(tree_name, {}).get("event_reweight_branches", []))
    load_branches = list(branches)
    for rb in reweight_branches:
        if rb not in load_branches:
            load_branches.append(rb)

    for sample_name in TRAINING_SAMPLES:
        if sample_name not in split_plans:
            continue

        plan = split_plans[sample_name]
        info = SAMPLE_INFO[sample_name]
        raw_entries = int(info["raw_entries"])
        xsec = float(info["xsection"])
        if xsec > 0.0 and raw_entries <= 0:
            raise RuntimeError(
                f"Sample '{sample_name}' has raw_entries={raw_entries}; "
                "fill src/sample.json before training."
            )

        if split_name == "train":
            full_segments = plan["train_segments_full"]
            read_segments = plan["train_segments_read"]
        elif split_name == "test":
            full_segments = plan["test_segments"]
            read_segments = plan["test_segments"]
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        split_total_entries = _sum_segment_lengths(full_segments)
        df, n_read = _load_segments(tree_name, load_branches, read_segments)
        if split_total_entries == 0 or n_read == 0 or df is None:
            log_warning(f"zero entries read for '{sample_name}' in split '{split_name}' of tree '{tree_name}', skipping")
            continue

        # Raw per-event weight: product of the configured reweight branches.
        # Computed on raw values before any clip/log/threshold so ratios between
        # events within the sample follow raw_w. The sample is then renormalised
        # so sum(weight) equals target_total, independent of raw_w's magnitude.
        if reweight_branches:
            raw_w = np.ones(n_read, dtype=float)
            for rb in reweight_branches:
                raw_w *= df[rb].to_numpy(dtype=float, copy=False)
            df = df.drop(columns=reweight_branches)
        else:
            raw_w = np.ones(n_read, dtype=float)

        total_tree_entries = int(plan["total_entries"])
        if xsec <= 0.0 or raw_entries <= 0:
            target_total = 0.0
        else:
            target_total = xsec * (float(total_tree_entries) / float(raw_entries))

        if target_total <= 0.0:
            df["weight"] = 0.0
            sample_target_totals[sample_name] = 0.0
            if xsec <= 0.0:
                log_warning(
                    f"sample '{sample_name}' has non-positive xsection={xsec:.6g}; assigning zero weight"
                )
        else:
            raw_w_sum = float(raw_w.sum())
            if raw_w_sum <= 0.0:
                raise RuntimeError(
                    f"Sample '{sample_name}' has non-positive raw weight sum "
                    f"{raw_w_sum:.6g} in split '{split_name}' of tree '{tree_name}'"
                )
            df["weight"] = raw_w * (target_total / raw_w_sum)
            sample_target_totals[sample_name] = target_total

        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        dfs.append(df)

        log_message(
            f"  {sample_name}: split={split_name}, tree_entries={plan['total_entries']}, "
            f"split_entries={split_total_entries}, used_entries={n_read}, raw_entries={raw_entries}, "
            f"target_total={target_total:.6g}, class={CLASS_NAMES[SAMPLE_TO_CLASS[sample_name]]}"
        )

    if not dfs:
        raise RuntimeError(f"No data loaded for split '{split_name}' in tree '{tree_name}'")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    _validate_sample_weight_totals(df_all, sample_target_totals)
    _report_sample_weights(df_all, f"Sample totals before class balancing ({split_name})")
    missing_classes = [
        cls_name for cls_idx, cls_name in enumerate(CLASS_NAMES)
        if float(df_all.loc[df_all["class_idx"] == cls_idx, "weight"].sum()) <= 0.0
    ]
    if missing_classes:
        raise RuntimeError(
            f"Missing positive-weight content for split '{split_name}' in classes: "
            + ", ".join(missing_classes)
        )

    df_all = _rebalance_class_weights(df_all)
    _report_sample_weights(df_all, f"Sample totals after class balancing ({split_name})")
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df_all[branches].copy()
    y = df_all["class_idx"].values.astype(int)
    w = df_all["weight"].values
    sample_labels = df_all["sample_name"].astype(str).values

    del df_all
    gc.collect()
    return X, y, w, sample_labels


def write_split_metadata(output_root, tree_name, split_plans):
    metadata = {
        "tree_name": tree_name,
        "train_fraction": TRAIN_FRACTION,
        "test_fraction": 1.0 - TRAIN_FRACTION,
        "samples": {},
    }

    for sample_name, plan in split_plans.items():
        metadata["samples"][sample_name] = {
            "total_entries": int(plan["total_entries"]),
            "train_entries_total": _sum_segment_lengths(plan["train_segments_full"]),
            "train_entries_used": _sum_segment_lengths(plan["train_segments_read"]),
            "test_entries_total": _sum_segment_lengths(plan["test_segments"]),
            "test_global_range": [
                int(plan["test_start"]),
                int(plan["total_entries"]),
            ],
            "test_segments": [
                {
                    "file": segment["path"],
                    "entry_start": int(segment["entry_start"]),
                    "entry_stop": int(segment["entry_stop"]),
                }
                for segment in plan["test_segments"]
            ],
        }

    metadata_path = os.path.join(output_root, "test_ranges.json")
    with open(metadata_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=False)
    log_message(f"Wrote split file: {metadata_path}")


def write_config_copy(output_root):
    config_copy_path = os.path.join(output_root, "config.json")
    shutil.copy2(_cfg_path, config_copy_path)
    log_message(f"Wrote config file: {config_copy_path}")


def write_branch_copy(output_root):
    branch_copy_path = os.path.join(output_root, "branch.json")
    shutil.copy2(os.path.join(_SCRIPT_DIR, "branch.json"), branch_copy_path)
    log_message(f"Wrote branch file: {branch_copy_path}")


def write_selection_copy(output_root):
    selection_copy_path = os.path.join(output_root, "selection.json")
    shutil.copy2(os.path.join(_SCRIPT_DIR, "selection.json"), selection_copy_path)
    log_message(f"Wrote selection file: {selection_copy_path}")


# -------------------- Event filtering --------------------
def filter_X(X: pd.DataFrame, y, w, branch: list,
             thresholds: dict = None, apply_to_sentinel: bool = True,
             sample_labels=None):
    """Apply per-branch threshold cuts.

    Only branches that appear as keys in ``thresholds`` are inspected: for each
    such branch, events with sentinel values (< -990) are dropped (when
    ``apply_to_sentinel`` is True) and the threshold condition is enforced.
    Branches not listed in ``thresholds`` are left untouched, so an event with
    a sentinel value in (for example) a lepton branch is still kept as long as
    no threshold targets that branch. The ``branch`` argument is retained for
    backward compatibility and is not used to drive filtering.
    """
    if not thresholds:
        if sample_labels is None:
            return X.copy(), y.copy(), w.copy()
        return X.copy(), y.copy(), w.copy(), np.asarray(sample_labels).copy()

    mask = pd.Series(True, index=X.index)

    def _combine(masks, op, idx):
        if not masks:
            return pd.Series(op == "&", index=idx)
        out = masks[0]
        for m in masks[1:]:
            out = (out & m) if op == "&" else (out | m)
        return out

    def _mask_from_cond(col, cond):
        idx = col.index
        if cond is None:
            return pd.Series(True, index=idx)
        if isinstance(cond, (int, float, np.integer, np.floating)):
            return col > float(cond)
        if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
            mn, mx = cond
            m = pd.Series(True, index=idx)
            if mn is not None:
                m &= col > mn
            if mx is not None:
                m &= col < mx
            return m
        if isinstance(cond, (list, tuple)):
            return _combine([_mask_from_cond(col, c) for c in cond], "|", idx)
        if isinstance(cond, dict):
            for op_key, op_sym in (("&", "&"), ("and", "&"), ("|", "|"), ("or", "|")):
                if op_key in cond:
                    return _combine([_mask_from_cond(col, c) for c in cond[op_key]], op_sym, idx)
            raise ValueError(f"Unsupported dict condition keys: {cond}")
        raise TypeError(f"Unsupported condition type: {type(cond)}")

    for b, cond in thresholds.items():
        if b not in X.columns:
            raise KeyError(f"Column {b!r} not found in X")
        col      = X[b]
        sentinel = col < -990

        if apply_to_sentinel:
            mask &= ~sentinel
            if cond is not None:
                mask &= _mask_from_cond(col, cond)
        else:
            if cond is not None:
                mask &= (_mask_from_cond(col, cond) | sentinel)

    X_out = X.loc[mask].copy()
    y_out = y[mask.values].copy()
    w_out = w[mask.values].copy()
    if sample_labels is None:
        return X_out, y_out, w_out
    return X_out, y_out, w_out, np.asarray(sample_labels)[mask.values].copy()


# -------------------- Feature standardization --------------------
def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    """Clip values and apply log transform in-place; sentinel values (< -990) are untouched."""
    log_set = set(log_transform)
    for col in X.columns:
        arr  = X[col].values.copy()
        mask = arr < -990   # Sentinel placeholder values.
        valid = ~mask
        if not valid.any():
            continue

        lo, hi = clip_ranges.get(col, (None, None))
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi

        if col in log_set:
            pos = valid & (arr > 0)
            if pos.any():
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                arr[pos] = np.log(arr[pos])

        X[col] = arr
    return X


# -------------------- CvM helpers --------------------
def _weighted_ecdf_positions(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0:
        return np.zeros_like(y)
    order    = np.argsort(y)
    w_sorted = w[order].astype(float)
    W        = float(np.sum(w_sorted)) + _EPS
    w_sorted /= W
    cum      = np.cumsum(w_sorted) - 0.5 * w_sorted
    pos      = np.empty_like(cum)
    pos[order] = cum
    return pos


def _cvm_flatness_value(y, Z, w, n_bins=10, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0 or Z.size == 0:
        return 0.0
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    _, n_decor = Z.shape
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return 0.0
    w_norm      = w / W_abs
    global_pos  = _weighted_ecdf_positions(y, w_norm)
    flat_penalty = 0.0
    for j in range(n_decor):
        zj    = Z[:, j]
        z_min = float(np.min(zj))
        z_max = float(np.max(zj))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            continue
        edges   = np.linspace(z_min, z_max, n_bins + 1)
        bin_idx = np.clip(np.searchsorted(edges, zj, side="right") - 1, 0, n_bins - 1)
        for b in range(n_bins):
            idx_b = np.nonzero(bin_idx == b)[0]
            if idx_b.size < 3:
                continue
            local_pos = _weighted_ecdf_positions(y[idx_b], w_norm[idx_b])
            diff      = local_pos - global_pos[idx_b]
            flat_penalty += float(np.sum(w_norm[idx_b] * (np.abs(diff) ** power)))
    return float(flat_penalty)


def _cvm_flatness_neg_grad_wrt_y(y, groups, w, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0 or not groups:
        return np.zeros_like(y)
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return np.zeros_like(y)
    w_norm     = w / W_abs
    global_pos = _weighted_ecdf_positions(y, w_norm)
    neg_grad   = np.zeros_like(y)
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        if idx.size < 2:
            continue
        local_pos = _weighted_ecdf_positions(y[idx], w_norm[idx])
        diff      = local_pos - global_pos[idx]
        bin_grad  = power * np.sign(diff) * (np.abs(diff) ** (power - 1.0))
        neg_grad[idx] += bin_grad
    neg_grad *= w_norm
    return neg_grad


# -------------------- Diagnostics --------------------
def check_weights(w, name="w"):
    w = np.asarray(w, dtype=float).ravel()
    finite = np.isfinite(w)
    if not np.all(finite):
        bad = np.where(~finite)[0]
        log_warning(f"{name} non-finite count: {bad.size}. e.g. indices: {bad[:10].tolist()}")
    else:
        log_message(f"{name}: all finite")
    n     = w.size
    n_pos = int(np.sum(w > 0))
    n_neg = int(np.sum(w < 0))
    log_message(
        f"{name}: N={n}, >0:{n_pos}, <0:{n_neg}, sum={np.nansum(w):.4g}, "
        f"min={np.nanmin(w):.4g}, max={np.nanmax(w):.4g}"
    )


# -------------------- Decorrelation helpers --------------------
def _resolve_decor_indices(X, decorrelate_feature_names):
    if not decorrelate_feature_names:
        return []
    if isinstance(X, pd.DataFrame):
        name_to_idx = {c: i for i, c in enumerate(X.columns)}
        idx = []
        for key in decorrelate_feature_names:
            if isinstance(key, int):
                idx.append(key)
            else:
                if key not in name_to_idx:
                    raise ValueError(f"Decorrelation feature '{key}' not in DataFrame columns.")
                idx.append(name_to_idx[key])
        return sorted(set(idx))
    idx = []
    for key in decorrelate_feature_names:
        if isinstance(key, int):
            idx.append(key)
        else:
            raise ValueError("X is not a DataFrame; pass integer column indices for decorrelation.")
    return sorted(set(idx))


# -------------------- Custom objective --------------------
def _make_multiclass_objective_with_decor(num_class, Z_train, w_train, lam):
    Z_train = np.asarray(Z_train, dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    n_train = w_train.shape[0]
    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
    n_samples, n_decor = Z_train.shape

    N_DECOR_BINS = 5
    decor_groups_list: List = []
    if lam > 0.0 and n_decor > 0:
        for j in range(n_decor):
            zj    = Z_train[:, j]
            z_min, z_max = float(np.min(zj)), float(np.max(zj))
            if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
                decor_groups_list.append(None)
                continue
            edges   = np.linspace(z_min, z_max, N_DECOR_BINS + 1)
            bin_idx = np.clip(np.searchsorted(edges, zj, side="right") - 1, 0, N_DECOR_BINS - 1)
            groups_j = [np.nonzero(bin_idx == b)[0] for b in range(N_DECOR_BINS)
                        if np.nonzero(bin_idx == b)[0].size >= 3]
            decor_groups_list.append(groups_j if groups_j else None)
    else:
        decor_groups_list = [None] * n_decor

    _scale       = float(n_train) / float(num_class) if n_train > 0 else 1.0
    _base_lr     = 0.05
    _iter_state  = {"t": 0}

    def obj(y_true, y_pred_in):
        t = int(_iter_state["t"])
        _iter_state["t"] = t + 1
        if t < 1000:
            lr_t = 0.01
        elif t < 1200:
            lr_t = 0.01 - (0.01 - 0.005) * (t - 1000) / 200
        else:
            lr_t = 0.005
        lr_mult = float(lr_t) / float(_base_lr)

        y_true    = np.asarray(y_true, dtype=int)
        n         = y_true.shape[0]
        y_pred_in = np.asarray(y_pred_in, dtype=float)
        if y_pred_in.ndim == 1:
            y_pred = y_pred_in.reshape(n, num_class)
        elif y_pred_in.shape == (num_class, n):
            y_pred = y_pred_in.T
        else:
            y_pred = y_pred_in.reshape(n, num_class)

        y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_y   = np.exp(y_shift)
        P       = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        w_used      = w_train
        class_w_sum = np.bincount(y_true, weights=w_used, minlength=num_class).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS
        w_eff       = (w_used / class_w_sum[y_true]) * _scale

        grad_cls = P.copy()
        grad_cls[np.arange(n), y_true] -= 1.0
        grad_cls *= w_eff[:, None]
        hess_cls  = (P * (1.0 - P)) * w_eff[:, None]

        grad_dec = np.zeros_like(grad_cls)
        if lam > 0.0 and n_decor > 0:
            for k in range(num_class):
                mask_k  = y_true == k
                if not np.any(mask_k):
                    continue
                yk      = y_pred[:, k]
                w_cls_k = np.zeros_like(w_used)
                w_cls_k[mask_k] = w_used[mask_k]
                gk = np.zeros(n, dtype=float)
                for j in range(n_decor):
                    groups_j = decor_groups_list[j]
                    if groups_j is None:
                        continue
                    gk += -_cvm_flatness_neg_grad_wrt_y(yk, groups_j, w_cls_k, power=2.0)
                grad_dec[:, k] = (lam * _scale) * gk

        grad = (grad_cls + grad_dec) * lr_mult
        hess = hess_cls + 1e-6
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


def _make_multiclass_total_loss_metric(num_class, Z_train, Z_test, w_train, w_test, lam):
    Z_train = np.asarray(Z_train, dtype=float)
    Z_test  = np.asarray(Z_test,  dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    w_test  = np.asarray(w_test,  dtype=float).ravel()
    N_DECOR_BINS = 5
    datasets = [
        {"Z": Z_train, "w": w_train},
        {"Z": Z_test, "w": w_test},
    ]
    state = {"idx": 0}

    def feval(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int)
        n      = y_true.shape[0]
        ds = datasets[state["idx"] % len(datasets)]
        state["idx"] += 1
        if ds["w"].shape[0] != n:
            matches = [cand for cand in datasets if cand["w"].shape[0] == n]
            ds = matches[0] if matches else {"Z": Z_train[:n], "w": w_train[:n]}
        Z = ds["Z"][:n]
        w = ds["w"][:n]

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(n, num_class)
        elif y_pred.shape == (num_class, n):
            y_pred = y_pred.T
        else:
            y_pred = y_pred.reshape(n, num_class)

        row_sum = np.sum(y_pred, axis=1)
        is_prob = (np.all(np.isfinite(y_pred)) and np.all(y_pred >= -1e-6)
                   and np.all(y_pred <= 1.0 + 1e-6)
                   and np.mean(np.abs(row_sum - 1.0)) < 1e-3)
        if is_prob:
            P = np.clip(y_pred, _EPS, 1.0)
            P = P / (np.sum(P, axis=1, keepdims=True) + _EPS)
        else:
            y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_y   = np.exp(y_shift)
            P       = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        class_w_sum = np.bincount(y_true, weights=w, minlength=num_class).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS
        w_eff  = w / class_w_sum[y_true]
        p_true = P[np.arange(n), y_true]
        ell    = -np.log(p_true + _EPS)
        logloss = float(np.sum(w_eff * ell))

        flat_penalty = 0.0
        if lam > 0.0 and Z.size > 0:
            for k in range(num_class):
                mask_k  = y_true == k
                if not np.any(mask_k):
                    continue
                w_cls_k = np.zeros_like(w)
                w_cls_k[mask_k] = w[mask_k]
                yk = y_pred[:, k] if not is_prob else np.log(P[:, k] + _EPS)
                flat_penalty += _cvm_flatness_value(yk, Z, w_cls_k,
                                                    n_bins=N_DECOR_BINS, power=2.0)
            flat_penalty *= lam

        return float(logloss + flat_penalty)

    return feval


# -------------------- Training --------------------
def train_multi_model(X_train_all, y_train, w_train, X_test_all, y_test, w_test,
                      model_name, tree_name, decorrelate_feature_names=None):
    """Train a multiclass BDT with optional CvM decorrelation.

    Hyperparameters are read from config.json under the key matching tree_name.
    Returns clf and the feature splits used for plotting.
    """
    X_train_all = np.asarray(X_train_all) if not isinstance(X_train_all, pd.DataFrame) else X_train_all
    X_test_all = np.asarray(X_test_all) if not isinstance(X_test_all, pd.DataFrame) else X_test_all
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    w_train = np.asarray(w_train, dtype=float)
    w_test = np.asarray(w_test, dtype=float)

    decor_idx = _resolve_decor_indices(X_train_all, decorrelate_feature_names)
    if decor_idx:
        all_idx  = np.arange(X_train_all.shape[1] if isinstance(X_train_all, np.ndarray)
                             else len(X_train_all.columns))
        keep_idx = np.setdiff1d(all_idx, decor_idx)
        if keep_idx.size == 0:
            raise ValueError("Decorrelation columns cover all features; nothing left to train on.")

        def _slice(Xlike, idx):
            return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

        X_train  = _slice(X_train_all, keep_idx)
        X_test   = _slice(X_test_all,  keep_idx)
        Z_train  = _slice(X_train_all, decor_idx)
        Z_test   = _slice(X_test_all,  decor_idx)
        Z_train  = np.asarray(Z_train, dtype=float)
        Z_test   = np.asarray(Z_test,  dtype=float)
    else:
        X_train, X_test = X_train_all, X_test_all
        Z_train = np.zeros((X_train_all.shape[0], 0), dtype=float)
        Z_test  = np.zeros((X_test_all.shape[0],  0), dtype=float)

    log_message(f"Training arrays: X_train={X_train.shape}, Z_train={Z_train.shape}")

    # Read hyperparameters from the config.
    hp = cfg.get(tree_name, {})
    n_threads = max(1, min(16, os.cpu_count() or 1))
    common_kwargs = dict(
        num_class        = NUM_CLASSES,
        n_estimators     = hp.get("n_estimators",    200),
        max_depth        = hp.get("max_depth",         6),
        learning_rate    = hp.get("learning_rate",   0.1),
        gamma            = hp.get("gamma",             0),
        reg_lambda       = hp.get("reg_lambda",        1),
        reg_alpha        = hp.get("reg_alpha",         0),
        min_child_weight = hp.get("min_child_weight",  1),
        n_jobs           = n_threads,
        random_state     = RANDOM_STATE,
    )
    log_message(f"Thread mode: XGBoost, threads = {n_threads}")
    gpu_kwargs = dict(tree_method="hist", device="cuda")
    cpu_kwargs = dict(tree_method="hist")

    use_decor = Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0

    if use_decor:
        custom_obj    = _make_multiclass_objective_with_decor(
            NUM_CLASSES, Z_train, w_train, DECOR_LAMBDA)
        loss_metric   = _make_multiclass_total_loss_metric(
            NUM_CLASSES, Z_train, Z_test, w_train, w_test, DECOR_LAMBDA)
        extra = dict(objective=custom_obj, eval_metric=loss_metric,
                     early_stopping_rounds=10)
    else:
        extra = dict(objective="multi:softprob", early_stopping_rounds=10)

    try:
        clf = XGBClassifier(**common_kwargs, **gpu_kwargs, **extra)
    except xgb.core.XGBoostError:
        clf = XGBClassifier(**common_kwargs, **cpu_kwargs, **extra)

    fit_kwargs = dict(eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    if not use_decor:
        fit_kwargs["sample_weight"]          = w_train
        fit_kwargs["sample_weight_eval_set"] = [w_train, w_test]
    clf.fit(X_train, y_train, **fit_kwargs)

    if model_name.endswith(".json") or use_decor:
        save_path = model_name if model_name.endswith(".json") else model_name + ".json"
        clf.save_model(save_path)
        log_message(f"Wrote model file: {save_path}")
    else:
        save_path = model_name + ".pkl"
        with open(save_path, "wb") as fout:
            pickle.dump(clf, fout)
        log_message(f"Wrote model file: {save_path}")

    return clf, (X_train_all, X_test_all, y_train, y_test, w_train, w_test)


# -------------------- Plotting --------------------
def plot_results(clf, splits, tree_name, output_root, decorrelate_feature_names=None):
    """ROC curves, feature importance, score distributions, loss curve, decorrelation checks."""
    X_train_full, X_test_full, y_train, y_test, w_train, w_test = splits

    full_feature_names = list(X_train_full.columns) if hasattr(X_train_full, "columns") \
        else [f"f{i}" for i in range(X_train_full.shape[1])]

    def _resolve(names_or_idx):
        if not names_or_idx:
            return []
        name_to_idx = {c: i for i, c in enumerate(full_feature_names)}
        out = []
        for key in names_or_idx:
            if isinstance(key, int):
                if 0 <= key < len(full_feature_names):
                    out.append(key)
            else:
                if key in name_to_idx:
                    out.append(name_to_idx[key])
                else:
                    log_info(f"decor var '{key}' not in feature list, skipping")
        seen, res = set(), []
        for i in out:
            if i not in seen:
                seen.add(i)
                res.append(i)
        return res

    decor_idx_full = _resolve(decorrelate_feature_names)
    all_idx        = np.arange(len(full_feature_names))
    keep_idx       = np.setdiff1d(all_idx, decor_idx_full)

    def _slice(Xlike, idx):
        return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

    X_train_used = _slice(X_train_full, keep_idx)
    X_test_used  = _slice(X_test_full,  keep_idx)
    feat_names_used = [full_feature_names[i] for i in keep_idx]

    if hasattr(clf, "n_features_in_") and clf.n_features_in_ == X_train_full.shape[1]:
        X_train_used, X_test_used = X_train_full, X_test_full
        feat_names_used    = full_feature_names
        decor_idx_full     = []

    n_classes   = NUM_CLASSES
    class_names = CLASS_NAMES
    palette     = plt.cm.get_cmap("tab10", max(n_classes, 3))(np.arange(max(n_classes, 3)))

    def _savefig(stem):
        path = _figure_path(output_root, stem)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        log_message(f"Wrote plot file: {path}")

    def _as_array(Xlike):
        return Xlike.to_numpy() if hasattr(Xlike, "to_numpy") else np.asarray(Xlike)

    def _safe_w(wv):
        return np.abs(np.asarray(wv, float).ravel())

    def _weighted_pearson(x, y_arr, wv, eps=1e-12):
        x  = np.asarray(x, float).ravel()
        y_arr = np.asarray(y_arr, float).ravel()
        wv = _safe_w(wv)
        m  = np.isfinite(x) & np.isfinite(y_arr) & np.isfinite(wv)
        if not np.any(m):
            return 0.0
        x, y_arr, wv = x[m], y_arr[m], wv[m]
        sw = wv.sum()
        if sw <= eps:
            return 0.0
        mx = (wv * x).sum() / (sw + eps)
        my = (wv * y_arr).sum() / (sw + eps)
        x0, y0 = x - mx, y_arr - my
        cov = (wv * x0 * y0).sum() / (sw + eps)
        vx  = (wv * x0 * x0).sum() / (sw + eps)
        vy  = (wv * y0 * y0).sum() / (sw + eps)
        return float(cov / (np.sqrt(vx * vy) + eps))

    probs_train = clf.predict_proba(X_train_used)
    probs_test  = clf.predict_proba(X_test_used)

    def _roc_binary(mask, scores, ys, ws, positive_idx):
        if not np.any(mask):
            return None
        y_bin = (ys[mask] == positive_idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return None
        auc = roc_auc_score(y_bin, scores[mask], sample_weight=ws[mask])
        fpr, tpr, _ = roc_curve(y_bin, scores[mask], sample_weight=ws[mask])
        return fpr, tpr, auc

    def _plot_roc_pair(sig_idx, bkg_idx):
        sig_name = class_names[sig_idx]
        bkg_name = class_names[bkg_idx]
        score_train = probs_train[:, sig_idx] / np.clip(probs_train[:, sig_idx] + probs_train[:, bkg_idx], _EPS, None)
        score_test = probs_test[:, sig_idx] / np.clip(probs_test[:, sig_idx] + probs_test[:, bkg_idx], _EPS, None)
        mask_train = (y_train == sig_idx) | (y_train == bkg_idx)
        mask_test = (y_test == sig_idx) | (y_test == bkg_idx)
        plt.figure(figsize=(10, 10))
        r_tst = _roc_binary(mask_test, score_test, y_test, w_test, sig_idx)
        r_trn = _roc_binary(mask_train, score_train, y_train, w_train, sig_idx)
        if r_tst:
            fpr, tpr, auc = r_tst
            plt.plot(tpr, fpr, color=palette[sig_idx], linestyle="-", label=f"Test AUC={auc:.3f}")
            log_message(f"{tree_name} test AUC ({sig_name} vs {bkg_name}) = {auc:.4f}")
        if r_trn:
            fpr, tpr, auc = r_trn
            plt.plot(tpr, fpr, color=palette[sig_idx], linestyle="--", label=f"Train AUC={auc:.3f}")
            log_message(f"{tree_name} train AUC ({sig_name} vs {bkg_name}) = {auc:.4f}")
        plt.xlabel(rf"$\epsilon_{{\rm {sig_name}}}$", fontsize=20)
        plt.ylabel(rf"$\epsilon_{{\rm {bkg_name}}}$", fontsize=20)
        plt.yscale("log")
        plt.ylim(1e-6, 1)
        plt.xlim(0, 1)
        plt.legend(loc="lower right", fontsize=14)
        _savefig(f"roc_{_slugify(sig_name)}_vs_{_slugify(bkg_name)}")

    if SIGNAL_CLASS_INDICES and BACKGROUND_CLASS_INDICES:
        roc_pairs = [(sig_idx, bkg_idx) for sig_idx in SIGNAL_CLASS_INDICES for bkg_idx in BACKGROUND_CLASS_INDICES]
    else:
        roc_pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
    for sig_idx, bkg_idx in roc_pairs:
        _plot_roc_pair(sig_idx, bkg_idx)

    fig, ax = plt.subplots(figsize=(10, 20))
    plot_importance(clf, ax=ax, max_num_features=200)
    ax.set_title(f"{tree_name} Feature Importance", fontsize=16)
    ax.set_xscale("log")
    widths = [p.get_width() for p in ax.patches]
    if widths:
        nz = [w for w in widths if w > 0]
        if nz:
            ax.set_xlim(min(nz) / 2, max(widths) * 2)
    try:
        raw_labels = [t.get_text() for t in ax.get_yticklabels()]
        mapped = []
        for s in raw_labels:
            if isinstance(s, str) and s.startswith("f") and s[1:].isdigit():
                i = int(s[1:])
                mapped.append(feat_names_used[i] if i < len(feat_names_used) else s)
            else:
                mapped.append(s)
        if mapped:
            ax.set_yticklabels(mapped)
    except Exception:
        pass
    _savefig("importance")

    def _plot_score_dist(sig_idx, bkg_idx):
        sig_name = class_names[sig_idx]
        bkg_name = class_names[bkg_idx]
        score_train = probs_train[:, sig_idx] / np.clip(probs_train[:, sig_idx] + probs_train[:, bkg_idx], _EPS, None)
        score_test = probs_test[:, sig_idx] / np.clip(probs_test[:, sig_idx] + probs_test[:, bkg_idx], _EPS, None)
        mask_train = (y_train == sig_idx) | (y_train == bkg_idx)
        mask_test = (y_test == sig_idx) | (y_test == bkg_idx)
        bins = np.linspace(0, 1, 31)
        plt.figure()
        plt.xlim(0, 1)
        plt.hist(score_train[mask_train & (y_train == bkg_idx)], bins=bins,
                 weights=w_train[mask_train & (y_train == bkg_idx)],
                 density=True, histtype="bar", alpha=0.5, label=f"Train {bkg_name}")
        plt.hist(score_train[mask_train & (y_train == sig_idx)], bins=bins,
                 weights=w_train[mask_train & (y_train == sig_idx)],
                 density=True, histtype="bar", alpha=0.5, label=f"Train {sig_name}")
        plt.hist(score_test[mask_test & (y_test == bkg_idx)], bins=bins,
                 weights=w_test[mask_test & (y_test == bkg_idx)],
                 density=True, histtype="step", linewidth=2, color="lime",
                 label=f"Test {bkg_name}")
        plt.hist(score_test[mask_test & (y_test == sig_idx)], bins=bins,
                 weights=w_test[mask_test & (y_test == sig_idx)],
                 density=True, histtype="step", linewidth=2, color="red",
                 label=f"Test {sig_name}")
        plt.xlabel("BDT Score")
        plt.yscale("log")
        plt.ylim(1e-2,)
        plt.ylabel("Density")
        plt.legend()
        _savefig(f"score_{_slugify(sig_name)}_vs_{_slugify(bkg_name)}")

    for sig_idx, bkg_idx in roc_pairs:
        _plot_score_dist(sig_idx, bkg_idx)

    try:
        evals = clf.evals_result()
        v0    = evals.get("validation_0", {})
        loss_key = next((k for k in v0 if "total_loss" in k or "feval" in k or "logloss" in k), None)
        if loss_key and "validation_1" in evals:
            tr_loss = evals["validation_0"][loss_key]
            te_loss = evals["validation_1"][loss_key]
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(tr_loss) + 1), tr_loss, label="Train")
            plt.plot(range(1, len(te_loss) + 1), te_loss, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("total_loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            _savefig("loss")
    except Exception:
        pass

    X_tr_df = pd.DataFrame(_as_array(X_train_used), columns=feat_names_used)
    corr    = X_tr_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    plt.figure(figsize=(20, 20))
    plt.imshow(corr.values, aspect="equal", interpolation="none", vmin=-1, vmax=1, cmap="bwr")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.index)),   corr.index,   fontsize=10)
    _savefig("feature_corr")

    if not decor_idx_full:
        return

    decor_var_names = [full_feature_names[i] for i in decor_idx_full]

    def _build_corr_matrix(scores, Xfull, y_true, wv):
        Xarr = _as_array(Xfull)
        wv   = _safe_w(wv)
        R    = np.zeros((n_classes, len(decor_idx_full)))
        for r, ci in enumerate(range(n_classes)):
            class_mask = np.asarray(y_true) == ci
            if not np.any(class_mask):
                continue
            s = scores[class_mask, ci]
            for c, j in enumerate(decor_idx_full):
                R[r, c] = _weighted_pearson(Xarr[class_mask, j], s, wv[class_mask])
        return R

    try:
        margins_train = clf.predict(X_train_used, output_margin=True)
        margins_test  = clf.predict(X_test_used, output_margin=True)
    except TypeError:
        margins_train = np.log(np.clip(probs_train, _EPS, 1.0))
        margins_test  = np.log(np.clip(probs_test, _EPS, 1.0))

    for tag, scores, Xfull, y_true, wv in [
        ("train", margins_train, X_train_full, y_train, w_train),
        ("test",  margins_test,  X_test_full,  y_test,  w_test),
    ]:
        R = _build_corr_matrix(scores, Xfull, y_true, wv)
        plt.figure(figsize=(max(6, 1.2 * len(decor_idx_full) + 4), n_classes + 4))
        plt.imshow(R, aspect="auto", interpolation="none", vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(fraction=0.046, pad=0.04, label="weighted Pearson r")
        plt.xticks(range(len(decor_var_names)), decor_var_names, rotation=45, ha="right")
        plt.yticks(range(n_classes), class_names)
        for i in range(n_classes):
            for j in range(len(decor_var_names)):
                v = R[i, j]
                plt.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.5 else "black", fontsize=10)
        _savefig(f"decor_corr_{tag}")
        for i, cls_name in enumerate(class_names):
            stats = ", ".join(
                f"{decor_var_names[j]}={R[i, j]:+.3f}" for j in range(len(decor_var_names))
            )
            log_message(f"{tree_name} {tag} decor corr [{cls_name}] {stats}")


def _validate_filtered_split(tree_name, split_name, y, w, sample_labels):
    filtered_df = pd.DataFrame({
        "weight": w,
        "class_idx": y,
        "sample_name": sample_labels,
    })
    _report_sample_weights(filtered_df, f"Sample totals after thresholding ({split_name})")
    missing_classes = [
        cls_name for cls_idx, cls_name in enumerate(CLASS_NAMES)
        if float(filtered_df.loc[filtered_df["class_idx"] == cls_idx, "weight"].sum()) <= 0.0
    ]
    if missing_classes:
        raise RuntimeError(
            f"Missing positive-weight content after thresholding for split '{split_name}' in tree '{tree_name}': "
            + ", ".join(missing_classes)
        )


def main():
    for tree_name in SUBMIT_TREES:
        output_root = _resolve_output_root(tree_name)
        os.makedirs(output_root, exist_ok=True)
        branches = [b["name"] for b in br_cfg[tree_name]]
        sel = sel_cfg[tree_name]
        clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
        log_tf = sel.get("log_transform", [])
        thresholds = {k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in sel.get("thresholds", {}).items()}
        decorrelate = cfg.get(tree_name, {}).get("decorrelate", [])
        model_path = MODEL_PATTERN.format(output_root=output_root, tree_name=tree_name)

        # Threshold and decorrelate branches that are NOT declared in branch.json
        # still need to be read from the ROOT files so filter_X can cut on them
        # and the decorrelation machinery can reference them. They are removed
        # from X before training so the BDT input feature set stays strictly
        # defined by branch.json.
        extra_cols = []
        for c in list(thresholds.keys()) + list(decorrelate):
            if c not in branches and c not in extra_cols:
                extra_cols.append(c)
        load_cols = branches + extra_cols
        drop_after_filter = [c for c in extra_cols if c not in decorrelate]

        log_message(
            f"Running train.py for tree = {tree_name}, output = {output_root}, classes = {NUM_CLASSES}"
        )
        split_plans = build_split_plans(tree_name)
        write_config_copy(output_root)
        write_branch_copy(output_root)
        write_selection_copy(output_root)
        write_split_metadata(output_root, tree_name, split_plans)

        log_message(f"Loading training split for tree = {tree_name}")
        X_train, y_train, w_train, sample_labels_train = prepare_split_data(
            tree_name, load_cols, "train", split_plans, shuffle=True
        )
        check_weights(w_train, f"{tree_name}_train_weight_before_filter")

        log_message(f"Loading test split for tree = {tree_name}")
        X_test, y_test, w_test, sample_labels_test = prepare_split_data(
            tree_name, load_cols, "test", split_plans, shuffle=False
        )
        check_weights(w_test, f"{tree_name}_test_weight_before_filter")

        log_message(f"Applying thresholds for training split of tree = {tree_name}")
        X_train, y_train, w_train, sample_labels_train = filter_X(
            X_train, y_train, w_train, load_cols, thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_train
        )
        _validate_filtered_split(tree_name, "train", y_train, w_train, sample_labels_train)
        check_weights(w_train, f"{tree_name}_train_weight_after_filter")

        log_message(f"Applying thresholds for test split of tree = {tree_name}")
        X_test, y_test, w_test, sample_labels_test = filter_X(
            X_test, y_test, w_test, load_cols, thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_test
        )
        _validate_filtered_split(tree_name, "test", y_test, w_test, sample_labels_test)
        check_weights(w_test, f"{tree_name}_test_weight_after_filter")

        if drop_after_filter:
            X_train = X_train.drop(columns=drop_after_filter, errors="ignore")
            X_test = X_test.drop(columns=drop_after_filter, errors="ignore")

        log_message(f"Standardising training split for tree = {tree_name}")
        X_train_std = standardize_X(X_train.copy(), clip_ranges, log_tf)
        log_message(f"Standardising test split for tree = {tree_name}")
        X_test_std = standardize_X(X_test.copy(), clip_ranges, log_tf)

        log_message(f"Training model for tree = {tree_name}")
        clf, splits = train_multi_model(
            X_train_std, y_train, w_train,
            X_test_std, y_test, w_test,
            model_path, tree_name,
            decorrelate_feature_names=decorrelate
        )

        log_message(f"Plotting results for tree = {tree_name}")
        plot_results(clf, splits, tree_name, output_root, decorrelate_feature_names=decorrelate)
        log_message(f"Finished train.py for tree = {tree_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
