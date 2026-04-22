import os
import glob
import json
import shutil
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import xgboost as xgb
import gc

from sklearn.metrics import roc_auc_score, roc_curve
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
DECOR_LOSS_MODE    = str(cfg.get("decor_loss_mode", "smooth_cvm")).strip().lower()
DECOR_N_BINS       = int(cfg.get("decor_n_bins", 5))
DECOR_N_THRESHOLDS = int(cfg.get("decor_n_thresholds", 31))
DECOR_SCORE_TAU    = float(cfg.get("decor_score_tau", 0.20))
DECOR_BIN_TAU_SCALE = float(cfg.get("decor_bin_tau_scale", 0.35))
SUBMIT_TREES       = cfg.get("submit_trees", ["fat2"])
INPUT_ROOT         = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN      = cfg["input_pattern"]
OUTPUT_ROOT_PATTERN = cfg.get("output_root", ".")
MODEL_PATTERN      = cfg.get("model_pattern", "{output_root}/{tree_name}_model")

if not 0.0 < TRAIN_FRACTION < 1.0:
    raise ValueError(f"train_fraction must be in (0, 1), got {TRAIN_FRACTION}")

if DECOR_LOSS_MODE == "soft_cvm":
    DECOR_LOSS_MODE = "smooth_cvm"
if DECOR_LOSS_MODE not in {"smooth_cvm", "cvm"}:
    raise ValueError(
        f"decor_loss_mode must be one of ['smooth_cvm', 'cvm'], got {DECOR_LOSS_MODE!r}"
    )

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


def _reference_path(output_root, stem):
    return os.path.join(output_root, f"{stem}.npz")


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
    target_total_per_class = float(len(df_all)) / float(NUM_CLASSES) if len(df_all) > 0 else 0.0
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = df_all["class_idx"] == cls_idx
        w_sum = float(df_all.loc[mask, "weight"].sum())
        if w_sum > 0.0:
            scale = target_total_per_class / w_sum
            df_all.loc[mask, "weight"] *= scale
            log_message(
                f"  {cls_name}: total_w={w_sum:.4g}, target_total={target_total_per_class:.4g}, scale={scale:.4g}"
            )
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
            df["weight_physics"] = 0.0
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
            df["weight_physics"] = raw_w * (target_total / raw_w_sum)
            sample_target_totals[sample_name] = target_total
        if "weight_physics" not in df.columns:
            df["weight_physics"] = 0.0

        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        df["weight"] = df["weight_physics"]
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
    w_physics = df_all["weight_physics"].values
    sample_labels = df_all["sample_name"].astype(str).values

    del df_all
    gc.collect()
    return X, y, w, sample_labels, w_physics


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


# -------------------- Score helpers --------------------
def _reshape_multiclass_margin(predt, num_class, n_rows=None):
    predt = np.asarray(predt, dtype=float)
    if predt.ndim == 2:
        if predt.shape[1] == num_class:
            return predt
        if predt.shape[0] == num_class:
            return predt.T
    if n_rows is None:
        n_rows = predt.size // num_class
    return predt.reshape(int(n_rows), int(num_class))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / (np.sum(exp_v, axis=1, keepdims=True) + _EPS)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# -------------------- CvM helpers --------------------
def _weighted_ecdf_positions(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0:
        return np.zeros_like(y)
    order = np.argsort(y)
    w_sorted = w[order].astype(float)
    W = float(np.sum(w_sorted)) + _EPS
    w_sorted /= W
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    pos = np.empty_like(cum)
    pos[order] = cum
    return pos


def _build_cvm_groups(Z: np.ndarray, n_bins: int = DECOR_N_BINS):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    groups_by_feature = []
    for j in range(Z.shape[1]):
        zj = Z[:, j]
        z_min = float(np.min(zj))
        z_max = float(np.max(zj))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            groups_by_feature.append(None)
            continue
        edges = np.linspace(z_min, z_max, max(2, n_bins) + 1)
        bin_idx = np.clip(np.searchsorted(edges, zj, side="right") - 1, 0, len(edges) - 2)
        groups_j = [np.nonzero(bin_idx == b)[0] for b in range(len(edges) - 1)]
        groups_j = [idx for idx in groups_j if idx.size >= 3]
        groups_by_feature.append(groups_j if groups_j else None)
    return groups_by_feature


def _cvm_flatness_value_from_groups(y, groups_by_feature, w, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if y.size == 0 or not groups_by_feature:
        return 0.0
    W_abs = float(np.sum(w))
    if W_abs <= _EPS:
        return 0.0
    w_norm = w / W_abs
    global_pos = _weighted_ecdf_positions(y, w_norm)
    flat_penalty = 0.0
    for groups in groups_by_feature:
        if not groups:
            continue
        for idx in groups:
            idx = np.asarray(idx, dtype=int)
            if idx.size < 3:
                continue
            local_pos = _weighted_ecdf_positions(y[idx], w_norm[idx])
            diff = local_pos - global_pos[idx]
            flat_penalty += float(np.sum(w_norm[idx] * (np.abs(diff) ** power)))
    return float(flat_penalty)


def _cvm_flatness_neg_grad_wrt_y(y, groups, w, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0 or not groups:
        return np.zeros_like(y)
    W_abs = float(np.sum(w))
    if W_abs <= _EPS:
        return np.zeros_like(y)
    w_norm = w / W_abs
    global_pos = _weighted_ecdf_positions(y, w_norm)
    neg_grad = np.zeros_like(y)
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        if idx.size < 2:
            continue
        local_pos = _weighted_ecdf_positions(y[idx], w_norm[idx])
        diff = local_pos - global_pos[idx]
        bin_grad = power * np.sign(diff) * (np.abs(diff) ** (power - 1.0))
        neg_grad[idx] += bin_grad
    neg_grad *= w_norm
    return neg_grad


# -------------------- Smooth CvM helpers --------------------
def _build_smooth_cvm_memberships(Z: np.ndarray, n_bins: int = DECOR_N_BINS):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    memberships = []
    for j in range(Z.shape[1]):
        zj = Z[:, j]
        z_min = float(np.min(zj))
        z_max = float(np.max(zj))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            memberships.append(None)
            continue
        edges = np.linspace(z_min, z_max, max(2, n_bins) + 1)
        width = max(float(edges[1] - edges[0]), _EPS)
        tau_z = max(width * DECOR_BIN_TAU_SCALE, _EPS)
        left = _sigmoid((zj[:, None] - edges[:-1][None, :]) / tau_z)
        right = _sigmoid((edges[1:][None, :] - zj[:, None]) / tau_z)
        memb = left * right
        row_sum = np.sum(memb, axis=1, keepdims=True)
        valid = row_sum[:, 0] > _EPS
        if np.any(valid):
            memb[valid] /= row_sum[valid]
        if np.any(~valid):
            hard = np.clip(np.searchsorted(edges, zj[~valid], side="right") - 1, 0, len(edges) - 2)
            memb[~valid] = 0.0
            memb[np.where(~valid)[0], hard] = 1.0
        memberships.append(memb.astype(float))
    return memberships


def _build_decor_state(Z: np.ndarray, mode: str):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.size == 0 or Z.shape[1] == 0:
        return {"mode": "none"}
    if mode == "cvm":
        return {"mode": "cvm", "groups": _build_cvm_groups(Z)}
    if mode == "smooth_cvm":
        prob_grid = np.linspace(0.02, 0.98, max(3, DECOR_N_THRESHOLDS))
        score_thresholds = np.log(prob_grid / (1.0 - prob_grid))
        return {
            "mode": "smooth_cvm",
            "memberships": _build_smooth_cvm_memberships(Z),
            "score_thresholds": score_thresholds.astype(float),
            "score_tau": max(float(DECOR_SCORE_TAU), _EPS),
        }
    raise ValueError(f"Unsupported decorrelation loss mode: {mode}")


def _smooth_cvm_value_and_grad_1d(score, memberships, weights, thresholds, score_tau):
    score = np.asarray(score, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    n = score.size
    grad = np.zeros(n, dtype=float)
    hess = np.zeros(n, dtype=float)
    if memberships is None or n == 0:
        return 0.0, grad, hess

    total_w = float(np.sum(weights))
    if total_w <= _EPS:
        return 0.0, grad, hess

    thresholds = np.asarray(thresholds, dtype=float).ravel()
    sig = _sigmoid((score[:, None] - thresholds[None, :]) / score_tau)
    dsig = sig * (1.0 - sig) / score_tau

    global_eff = np.sum(weights[:, None] * sig, axis=0) / total_w
    weighted_memberships = weights[:, None] * memberships
    bin_totals = np.sum(weighted_memberships, axis=0)
    valid_bins = bin_totals > _EPS
    if not np.any(valid_bins):
        return 0.0, grad, hess

    memberships_v = memberships[:, valid_bins]
    bin_totals_v = bin_totals[valid_bins]
    rho = bin_totals_v / total_w
    local_eff = (weighted_memberships[:, valid_bins].T @ sig) / bin_totals_v[:, None]
    delta = local_eff - global_eff[None, :]
    n_thr = float(sig.shape[1])

    loss = float(np.sum(rho[:, None] * delta * delta) / n_thr)
    local_term = (memberships_v / bin_totals_v) @ (rho[:, None] * delta)
    global_term = np.sum(rho[:, None] * delta, axis=0) / total_w
    coeff = local_term - global_term[None, :]
    grad = (2.0 * weights[:, None] * dsig * coeff).sum(axis=1) / n_thr
    # Positive Gauss-Newton-style diagonal surrogate for the coupled decorrelation term.
    hess = (2.0 * (weights[:, None] * dsig * coeff) ** 2).sum(axis=1) / n_thr
    return loss, grad, hess


# -------------------- Diagnostics --------------------
def check_weights(w, name="w"):
    w = np.asarray(w, dtype=float).ravel()
    finite = np.isfinite(w)
    if not np.all(finite):
        bad = np.where(~finite)[0]
        log_warning(f"{name} non-finite count: {bad.size}. e.g. indices: {bad[:10].tolist()}")
    else:
        log_message(f"{name}: all finite")
    n = w.size
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


def _multiclass_classification_terms(labels, predt, weights, num_class, prediction_mode="margin"):
    labels = np.asarray(labels, dtype=int).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    predt = _reshape_multiclass_margin(predt, num_class, labels.size)
    if prediction_mode == "margin":
        probs = _softmax_rows(predt)
        grad = probs.copy()
        grad[np.arange(labels.size), labels] -= 1.0
        grad *= weights[:, None]
        hess = np.maximum(2.0 * probs * (1.0 - probs) * weights[:, None], 1e-6)
    elif prediction_mode == "probability":
        probs = np.clip(predt.astype(float, copy=True), _EPS, None)
        row_sum = np.sum(probs, axis=1, keepdims=True)
        probs /= np.where(row_sum > _EPS, row_sum, 1.0)
        grad = None
        hess = None
    else:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode!r}")
    loss = float(np.sum(weights * (-np.log(probs[np.arange(labels.size), labels] + _EPS))))
    return probs, grad, hess, loss


def _weighted_mlogloss(loss_sum, weights):
    weights = np.asarray(weights, dtype=float).ravel()
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(loss_sum / weight_sum)


def _decorrelation_loss_components(logits, labels, weights, decor_state, num_class, decor_scale=1.0):
    labels = np.asarray(labels, dtype=int).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    logits = _reshape_multiclass_margin(logits, num_class, labels.size)
    mode = decor_state.get("mode", "none")
    grad = np.zeros_like(logits, dtype=float)
    hess = np.zeros_like(logits, dtype=float)
    loss = 0.0

    if mode == "none":
        return loss, grad, hess

    for cls_idx in range(num_class):
        mask_cls = labels == cls_idx
        if not np.any(mask_cls):
            continue
        cls_weights = np.zeros_like(weights)
        cls_weights[mask_cls] = weights[mask_cls]
        score = logits[:, cls_idx]

        if mode == "cvm":
            groups_by_feature = decor_state.get("groups", [])
            cls_loss = _cvm_flatness_value_from_groups(score, groups_by_feature, cls_weights, power=2.0)
            cls_grad = np.zeros_like(score)
            for groups in groups_by_feature:
                if not groups:
                    continue
                # Hard-bin CvM is non-smooth; keep the legacy surrogate gradient but
                # make it consistent with the loss recorded below.
                cls_grad += -_cvm_flatness_neg_grad_wrt_y(score, groups, cls_weights, power=2.0)
            cls_hess = np.maximum(np.abs(cls_grad), 1e-6)
        elif mode == "smooth_cvm":
            cls_loss = 0.0
            cls_grad = np.zeros_like(score)
            cls_hess = np.zeros_like(score)
            for memberships in decor_state.get("memberships", []):
                part_loss, part_grad, part_hess = _smooth_cvm_value_and_grad_1d(
                    score,
                    memberships,
                    cls_weights,
                    decor_state["score_thresholds"],
                    decor_state["score_tau"],
                )
                cls_loss += part_loss
                cls_grad += part_grad
                cls_hess += part_hess
            cls_hess = np.maximum(cls_hess, 1e-6)
        else:
            raise ValueError(f"Unsupported decorrelation loss mode: {mode}")

        loss += float(decor_scale * cls_loss)
        grad[:, cls_idx] = decor_scale * cls_grad
        hess[:, cls_idx] = decor_scale * cls_hess

    return float(loss), grad, hess


def _loss_components(predt, labels, weights, decor_state, num_class, lam, decor_scale,
                     prediction_mode="margin"):
    labels = np.asarray(labels, dtype=int).ravel()
    _, _, _, cls_loss = _multiclass_classification_terms(
        labels, predt, weights, num_class, prediction_mode=prediction_mode
    )
    mlogloss = _weighted_mlogloss(cls_loss, weights)
    if decor_state.get("mode", "none") == "none" or lam <= 0.0:
        decor_loss_raw = 0.0
    else:
        if prediction_mode != "margin":
            raise ValueError("Decorrelation loss requires raw margin predictions.")
        decor_loss_raw, _, _ = _decorrelation_loss_components(
            predt, labels, weights, decor_state, num_class, decor_scale
        )
    decor_loss = float(lam * decor_loss_raw)
    return {
        "classification": float(cls_loss),
        "mlogloss": float(mlogloss),
        "decorrelation": decor_loss,
        "regularization": 0.0,
        "total": float(cls_loss + decor_loss),
    }


def _make_multiclass_objective(num_class, decor_state, lam, decor_scale):
    def obj(predt, dtrain):
        labels = dtrain.get_label().astype(int)
        weights = dtrain.get_weight()
        if weights.size == 0:
            weights = np.ones(labels.size, dtype=float)
        logits = _reshape_multiclass_margin(predt, num_class, labels.size)
        _, grad_cls, hess_cls, _ = _multiclass_classification_terms(labels, logits, weights, num_class)
        if lam > 0.0 and decor_state.get("mode", "none") != "none":
            _, grad_dec, hess_dec = _decorrelation_loss_components(
                logits, labels, weights, decor_state, num_class, decor_scale
            )
            grad = grad_cls + lam * grad_dec
            hess = hess_cls + lam * hess_dec
        else:
            grad = grad_cls
            hess = hess_cls
        return grad.reshape(-1, 1).astype(np.float32), np.maximum(hess, 1e-6).reshape(-1, 1).astype(np.float32)

    return obj


def _collect_leaf_weights(node, out):
    if "leaf" in node:
        out.append(float(node["leaf"]))
        return
    for child in node.get("children", []):
        _collect_leaf_weights(child, out)


def _booster_regularization_loss(model, reg_lambda, reg_alpha, gamma, learning_rate):
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    eta = float(learning_rate)
    if eta <= 0.0:
        raise ValueError(f"learning_rate must be positive to reconstruct native regularization, got {eta}")
    total = 0.0
    for tree_json in booster.get_dump(dump_format="json"):
        leaf_weights = []
        _collect_leaf_weights(json.loads(tree_json), leaf_weights)
        if not leaf_weights:
            continue
        leaf_weights = np.asarray(leaf_weights, dtype=float)
        unshrunk_leaf_weights = leaf_weights / eta
        total += float(gamma) * float(leaf_weights.size)
        total += 0.5 * float(reg_lambda) * float(np.sum(unshrunk_leaf_weights * unshrunk_leaf_weights))
        total += float(reg_alpha) * float(np.sum(np.abs(unshrunk_leaf_weights)))
    return float(total)


class _TotalLossMetricRecorder:
    def __init__(self, datasets, num_class, lam, decor_scale, prediction_mode="margin",
                 selection_metric_key="mlogloss", selection_metric_name=None):
        self.datasets = list(datasets)
        self.num_class = int(num_class)
        self.lam = float(lam)
        self.decor_scale = float(decor_scale)
        self.prediction_mode = str(prediction_mode)
        self.selection_metric_key = str(selection_metric_key)
        self.selection_metric_name = (
            str(selection_metric_name)
            if selection_metric_name is not None
            else self.selection_metric_key
        )
        self._call_idx = 0
        self.history = {
            tag: {
                "classification": [],
                "mlogloss": [],
                "decorrelation": [],
                "regularization": [],
                "total": [],
            }
            for tag, _, _, _ in self.datasets
        }

    def __call__(self, predt, dtrain):
        tag, labels, weights, decor_state = self.datasets[self._call_idx % len(self.datasets)]
        self._call_idx += 1
        comp = _loss_components(
            predt,
            labels,
            weights,
            decor_state,
            self.num_class,
            self.lam,
            self.decor_scale,
            prediction_mode=self.prediction_mode,
        )
        for key in ("classification", "mlogloss", "decorrelation", "regularization", "total"):
            self.history[tag][key].append(comp[key])
        return self.selection_metric_name, comp[self.selection_metric_key]

    def finalize_iteration(self, reg_loss):
        reg_loss = float(reg_loss)
        for tag in self.history:
            metrics = self.history[tag]
            if len(metrics["regularization"]) < len(metrics["classification"]):
                metrics["regularization"].append(reg_loss)
            else:
                metrics["regularization"][-1] = reg_loss
            if metrics["total"]:
                metrics["total"][-1] = (
                    metrics["classification"][-1]
                    + metrics["decorrelation"][-1]
                )


def _loss_value_at(loss_history, split_name, metric_key, epoch):
    values = loss_history.get(split_name, {}).get(metric_key, [])
    return values[epoch] if epoch < len(values) else float("nan")


def _format_detailed_loss_line(epoch, loss_history, prefix="", compact=False):
    head = f"{prefix}[{epoch}]" if prefix else f"[{epoch}]"
    if compact:
        return (
            f"{head}"
            f"\ttrain-mlogloss:{_loss_value_at(loss_history, 'train', 'mlogloss', epoch):.5f}"
            f"\ttest-mlogloss:{_loss_value_at(loss_history, 'test', 'mlogloss', epoch):.5f}"
        )
    return (
        f"{head}"
        f"\ttrain-mlogloss:{_loss_value_at(loss_history, 'train', 'mlogloss', epoch):.5f}"
        f"\ttrain-classification_loss:{_loss_value_at(loss_history, 'train', 'classification', epoch):.5f}"
        f"\ttrain-decorrelation_loss:{_loss_value_at(loss_history, 'train', 'decorrelation', epoch):.5f}"
        f"\ttrain-total_loss:{_loss_value_at(loss_history, 'train', 'total', epoch):.5f}"
        f"\ttest-mlogloss:{_loss_value_at(loss_history, 'test', 'mlogloss', epoch):.5f}"
        f"\ttest-classification_loss:{_loss_value_at(loss_history, 'test', 'classification', epoch):.5f}"
        f"\ttest-decorrelation_loss:{_loss_value_at(loss_history, 'test', 'decorrelation', epoch):.5f}"
        f"\ttest-total_loss:{_loss_value_at(loss_history, 'test', 'total', epoch):.5f}"
    )


class _DetailedLossMonitor(xgb.callback.TrainingCallback):
    def __init__(self, recorder, reg_lambda, reg_alpha, gamma, learning_rate, early_stopping_rounds,
                 stage_label="", compact_log=False, initial_reg=0.0, tree_offset=0,
                 monitor_metric_key="mlogloss", monitor_metric_label=None):
        self.recorder = recorder
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.stage_label = str(stage_label)
        self.compact_log = bool(compact_log)
        self.cumulative_regularization = float(initial_reg)
        self.tree_offset = int(tree_offset)
        self.monitor_metric_key = str(monitor_metric_key)
        self.monitor_metric_label = (
            str(monitor_metric_label)
            if monitor_metric_label is not None
            else self.monitor_metric_key
        )
        self.best_iteration = None  # stage-local best iteration
        self.best_score = float("inf")
        self._stale_rounds = 0

    def after_iteration(self, model, epoch, evals_log):
        local_epoch = int(epoch)
        tree_index = self.tree_offset + local_epoch
        self.cumulative_regularization += _booster_regularization_loss(
            model[tree_index:tree_index + 1],
            self.reg_lambda,
            self.reg_alpha,
            self.gamma,
            self.learning_rate,
        )
        self.recorder.finalize_iteration(self.cumulative_regularization)
        prefix = f"[{self.stage_label}]" if self.stage_label else ""
        log_message(_format_detailed_loss_line(
            local_epoch, self.recorder.history, prefix=prefix, compact=self.compact_log
        ))
        current_score = _loss_value_at(
            self.recorder.history, "test", self.monitor_metric_key, local_epoch
        )
        if np.isfinite(current_score) and (
            self.best_iteration is None or current_score < self.best_score - 1e-12
        ):
            self.best_iteration = int(local_epoch)
            self.best_score = float(current_score)
            self._stale_rounds = 0
            model.set_attr(
                best_iteration=str(self.tree_offset + self.best_iteration),
                best_score=str(self.best_score),
            )
        else:
            self._stale_rounds += 1

        if self._stale_rounds >= self.early_stopping_rounds:
            tag = f" ({self.stage_label})" if self.stage_label else ""
            log_message(
                f"Info: early stopping{tag} on {self.monitor_metric_label} "
                f"(best_iteration={self.best_iteration}, "
                f"best_test_{self.monitor_metric_label}={self.best_score:.5f})"
            )
            return True
        return False


def _trim_loss_history(loss_history, n_rounds):
    trimmed = {}
    for split_name, metrics in loss_history.items():
        trimmed[split_name] = {
            key: list(values[:n_rounds]) for key, values in metrics.items()
        }
    return trimmed


def _make_dmatrix(Xlike, y=None, w=None):
    data = Xlike
    feature_names = list(Xlike.columns) if hasattr(Xlike, "columns") else None
    return xgb.DMatrix(data, label=y, weight=w, feature_names=feature_names)


# -------------------- Training --------------------
def train_multi_model(X_train_all, y_train, w_train, X_test_all, y_test, w_test,
                      model_name, tree_name, decorrelate_feature_names=None):
    """Two-stage multiclass BDT training.

    Stage 1 uses native ``multi:softprob`` (cls-only). When decorrelation is
    enabled (non-empty ``decorrelate`` and ``decor_lambda > 0``), stage 2
    continues from the stage-1 best model with a custom objective that adds
    the smooth-CvM (or hard-CvM) decorrelation term to the native softprob
    gradient. Stage 1 early-stops on test ``classification_loss``; stage 2
    early-stops on test ``total_loss = classification_loss +
    decorrelation_loss``. The sum-scale ``classification_loss``, exact
    native-style ``regularization_loss``, and ``total_loss`` remain
    diagnostic outputs shared across both stages.

    Returns ``(stage1_model, stage2_model_or_None, splits, combined_loss_history, stage_boundary)``
    where ``stage_boundary`` is the number of stage-1 iterations kept.
    """
    X_train_all = np.asarray(X_train_all) if not isinstance(X_train_all, pd.DataFrame) else X_train_all
    X_test_all = np.asarray(X_test_all) if not isinstance(X_test_all, pd.DataFrame) else X_test_all
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    w_train = np.asarray(w_train, dtype=float)
    w_test = np.asarray(w_test, dtype=float)

    decor_idx = _resolve_decor_indices(X_train_all, decorrelate_feature_names)
    if decor_idx:
        all_idx = np.arange(X_train_all.shape[1] if isinstance(X_train_all, np.ndarray)
                            else len(X_train_all.columns))
        keep_idx = np.setdiff1d(all_idx, decor_idx)
        if keep_idx.size == 0:
            raise ValueError("Decorrelation columns cover all features; nothing left to train on.")

        def _slice(Xlike, idx):
            return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

        X_train = _slice(X_train_all, keep_idx)
        X_test = _slice(X_test_all, keep_idx)
        Z_train = np.asarray(_slice(X_train_all, decor_idx), dtype=float)
        Z_test = np.asarray(_slice(X_test_all, decor_idx), dtype=float)
    else:
        X_train, X_test = X_train_all, X_test_all
        Z_train = np.zeros((X_train_all.shape[0], 0), dtype=float)
        Z_test = np.zeros((X_test_all.shape[0], 0), dtype=float)

    log_message(
        f"Training arrays: X_train={X_train.shape}, Z_train={Z_train.shape}, decor_mode={DECOR_LOSS_MODE}"
    )

    hp = cfg.get(tree_name, {})
    n_threads = max(1, min(16, os.cpu_count() or 1))
    n_estimators = int(hp.get("n_estimators", 200))
    n_estimators_decorr = int(hp.get("n_estimators_decorr", 1000))
    early_stopping_rounds = int(hp.get("early_stopping_rounds", 10))
    if early_stopping_rounds <= 0:
        raise ValueError(
            f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}"
        )
    learning_rate = float(hp.get("learning_rate", 0.1))
    learning_rate_decorr = float(hp.get("learning_rate_decorr", 0.01))
    log_message(f"Thread mode: XGBoost, threads = {n_threads}")

    use_decor = Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0
    splits = (X_train_all, X_test_all, y_train, y_test, w_train, w_test)

    dtrain = _make_dmatrix(X_train, y_train, w_train)
    dtest = _make_dmatrix(X_test, y_test, w_test)

    base_params = dict(
        num_class=NUM_CLASSES,
        objective="multi:softprob",
        max_depth=hp.get("max_depth", 6),
        gamma=hp.get("gamma", 0),
        reg_lambda=hp.get("reg_lambda", 1),
        reg_alpha=hp.get("reg_alpha", 0),
        min_child_weight=hp.get("min_child_weight", 1),
        nthread=n_threads,
        seed=RANDOM_STATE,
        disable_default_eval_metric=1,
        tree_method="hist",
    )

    # Build decor state on both splits when decor is enabled; stage-1 recorder
    # uses an explicit "none" decor_state so it contributes zero loss/grad/hess.
    train_decor_state = _build_decor_state(Z_train, DECOR_LOSS_MODE) if use_decor else {"mode": "none"}
    test_decor_state = _build_decor_state(Z_test, DECOR_LOSS_MODE) if use_decor else {"mode": "none"}

    # ---------- Stage 1: native cls-only ----------
    stage1_params = dict(base_params)
    stage1_params["eta"] = learning_rate

    def _run_stage1(extra_params):
        recorder = _TotalLossMetricRecorder(
            [
                ("train", y_train, w_train, {"mode": "none"}),
                ("test", y_test, w_test, {"mode": "none"}),
            ],
            NUM_CLASSES, 0.0, 1.0, prediction_mode="probability",
            selection_metric_key="classification",
            selection_metric_name="classification_loss",
        )
        monitor = _DetailedLossMonitor(
            recorder,
            reg_lambda=stage1_params["reg_lambda"],
            reg_alpha=stage1_params["reg_alpha"],
            gamma=stage1_params["gamma"],
            learning_rate=stage1_params["eta"],
            early_stopping_rounds=early_stopping_rounds,
            stage_label="stage1",
            compact_log=True,
            monitor_metric_key="classification",
            monitor_metric_label="classification_loss",
        )
        train_kwargs = dict(
            params={**stage1_params, **extra_params},
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=False,
            callbacks=[monitor],
        )
        try:
            model = xgb.train(custom_metric=recorder, **train_kwargs)
        except TypeError:
            model = xgb.train(feval=recorder, **train_kwargs)
        return model, recorder, monitor

    log_message(
        f"Starting stage 1 (native multi:softprob, n_estimators={n_estimators}, eta={learning_rate})"
    )
    try:
        stage1_model, stage1_recorder, stage1_monitor = _run_stage1({"device": "cuda"})
    except xgb.core.XGBoostError:
        stage1_model, stage1_recorder, stage1_monitor = _run_stage1({})

    stage1_best = stage1_monitor.best_iteration
    if stage1_best is None:
        stage1_best = stage1_model.num_boosted_rounds() - 1
    stage1_rounds = int(stage1_best) + 1
    if stage1_model.num_boosted_rounds() != stage1_rounds:
        stage1_model = stage1_model[:stage1_rounds]
    stage1_history = _trim_loss_history(stage1_recorder.history, stage1_rounds)

    base_path = model_name[:-5] if model_name.endswith(".json") else model_name
    stage1_save_path = f"{base_path}_stage1.json"
    stage1_model.save_model(stage1_save_path)
    log_message(f"Wrote model file: {stage1_save_path}")

    if not use_decor:
        # No stage 2. Copy stage 1 as the main model file so downstream paths stay unchanged.
        main_save_path = f"{base_path}.json"
        stage1_model.save_model(main_save_path)
        log_message(f"Wrote model file: {main_save_path}")
        return stage1_model, None, splits, stage1_history, stage1_rounds

    # ---------- Stage 2: continuation with decorrelation ----------
    # Calibrate fixed decor scale from stage 1 best-model state so that
    # decor_lambda=1 keeps the logged decor_loss at a magnitude comparable to
    # cls_loss + reg_loss at the point where stage 2 actually starts.
    stage1_logits = _reshape_multiclass_margin(
        stage1_model.predict(dtrain, output_margin=True), NUM_CLASSES, len(y_train)
    )
    _, _, _, cls_loss_ref = _multiclass_classification_terms(
        y_train, stage1_logits, w_train, NUM_CLASSES
    )
    reg_loss_ref = _booster_regularization_loss(
        stage1_model,
        stage1_params["reg_lambda"],
        stage1_params["reg_alpha"],
        stage1_params["gamma"],
        stage1_params["eta"],
    )
    decor_loss_ref_raw, _, _ = _decorrelation_loss_components(
        stage1_logits, y_train, w_train, train_decor_state, NUM_CLASSES, decor_scale=1.0
    )
    numerator = max(float(cls_loss_ref) + float(reg_loss_ref), _EPS)
    if not np.isfinite(decor_loss_ref_raw) or decor_loss_ref_raw <= _EPS:
        log_warning(
            f"Stage-1 raw decorrelation loss is non-positive ({decor_loss_ref_raw:.6g}); "
            "falling back to scale=1.0"
        )
        decor_scale = 1.0
    else:
        decor_scale = numerator / float(decor_loss_ref_raw)

    log_message(
        f"Stage-1 end: best_iter={stage1_rounds - 1}, cls_loss={cls_loss_ref:.6g}, "
        f"reg_loss={reg_loss_ref:.6g}, raw_decor_loss={decor_loss_ref_raw:.6g}"
    )
    log_message(f"Decorrelation scale: mode={DECOR_LOSS_MODE}, fixed_scale={decor_scale:.6g}")

    stage2_params = dict(base_params)
    stage2_params["eta"] = learning_rate_decorr

    def _run_stage2(extra_params):
        recorder = _TotalLossMetricRecorder(
            [
                ("train", y_train, w_train, train_decor_state),
                ("test", y_test, w_test, test_decor_state),
            ],
            NUM_CLASSES, DECOR_LAMBDA, decor_scale, prediction_mode="margin",
            selection_metric_key="total",
            selection_metric_name="total_loss",
        )
        monitor = _DetailedLossMonitor(
            recorder,
            reg_lambda=stage2_params["reg_lambda"],
            reg_alpha=stage2_params["reg_alpha"],
            gamma=stage2_params["gamma"],
            learning_rate=stage2_params["eta"],
            early_stopping_rounds=early_stopping_rounds,
            stage_label="stage2",
            compact_log=False,
            initial_reg=reg_loss_ref,
            tree_offset=stage1_rounds,
            monitor_metric_key="total",
            monitor_metric_label="total_loss",
        )
        custom_obj = _make_multiclass_objective(
            NUM_CLASSES, train_decor_state, DECOR_LAMBDA, decor_scale
        )
        # In xgb.train continuation, num_boost_round is the number of additional
        # rounds to add, while the callback epoch counter restarts from 0.
        train_kwargs = dict(
            params={**stage2_params, **extra_params},
            dtrain=dtrain,
            num_boost_round=n_estimators_decorr,
            evals=[(dtrain, "train"), (dtest, "test")],
            obj=custom_obj,
            xgb_model=stage1_save_path,
            verbose_eval=False,
            callbacks=[monitor],
        )
        try:
            model = xgb.train(custom_metric=recorder, **train_kwargs)
        except TypeError:
            model = xgb.train(feval=recorder, **train_kwargs)
        return model, recorder, monitor

    log_message(
        f"Starting stage 2 (cls+decor, n_estimators_decorr={n_estimators_decorr}, eta={learning_rate_decorr})"
    )
    try:
        stage2_model, stage2_recorder, stage2_monitor = _run_stage2({"device": "cuda"})
    except xgb.core.XGBoostError:
        stage2_model, stage2_recorder, stage2_monitor = _run_stage2({})

    stage2_best = stage2_monitor.best_iteration  # stage-local index
    if stage2_best is None:
        stage2_best = stage2_model.num_boosted_rounds() - stage1_rounds - 1
    stage2_rounds = int(stage2_best) + 1
    total_rounds = stage1_rounds + stage2_rounds
    if stage2_model.num_boosted_rounds() != total_rounds:
        stage2_model = stage2_model[:total_rounds]
    stage2_history = _trim_loss_history(stage2_recorder.history, stage2_rounds)

    combined_history = {
        "train": {
            k: list(stage1_history["train"].get(k, [])) + list(stage2_history["train"].get(k, []))
            for k in ("classification", "mlogloss", "decorrelation", "regularization", "total")
        },
        "test": {
            k: list(stage1_history["test"].get(k, [])) + list(stage2_history["test"].get(k, []))
            for k in ("classification", "mlogloss", "decorrelation", "regularization", "total")
        },
    }

    main_save_path = f"{base_path}.json"
    stage2_model.save_model(main_save_path)
    log_message(f"Wrote model file: {main_save_path}")

    return stage1_model, stage2_model, splits, combined_history, stage1_rounds


# -------------------- Plotting --------------------
def _booster_from_model(model):
    return model.get_booster() if hasattr(model, "get_booster") else model


def _predict_margins(model, Xlike):
    booster = _booster_from_model(model)
    dmat = _make_dmatrix(Xlike)
    pred = booster.predict(dmat, output_margin=True)
    return _reshape_multiclass_margin(pred, NUM_CLASSES, len(Xlike))


def _predict_proba(model, Xlike):
    return _softmax_rows(_predict_margins(model, Xlike))


def plot_results(stage1_model, stage2_model, splits, tree_name, output_root,
                 loss_history, stage_boundary, decorrelate_feature_names=None):
    """ROC curves, feature importance, score distributions, loss curves, and decorrelation checks.

    Model-dependent plots (ROC, importance, score distributions, decor_corr)
    are saved twice with ``_cls`` / ``_decorr`` suffixes (for the stage-1
    baseline and stage-2 final model respectively). ``feature_corr.pdf`` and
    the shared ``loss_mlogloss.pdf`` / ``loss_classification.pdf`` /
    ``loss_decorrelation.pdf`` / ``loss_total.pdf`` files are saved once.
    ``stage_boundary`` is the number of stage-1 iterations kept; it is drawn
    as a dotted vertical line on the loss curves.
    """
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
    all_idx = np.arange(len(full_feature_names))
    keep_idx = np.setdiff1d(all_idx, decor_idx_full)

    def _slice(Xlike, idx):
        return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

    X_train_used = _slice(X_train_full, keep_idx)
    X_test_used = _slice(X_test_full, keep_idx)
    feat_names_used = [full_feature_names[i] for i in keep_idx]

    booster_ref = _booster_from_model(stage1_model if stage1_model is not None else stage2_model)
    booster_features = booster_ref.feature_names or []
    if booster_features and len(booster_features) == len(full_feature_names):
        X_train_used, X_test_used = X_train_full, X_test_full
        feat_names_used = full_feature_names
        decor_idx_full = []

    n_classes = NUM_CLASSES
    class_names = CLASS_NAMES
    palette = plt.cm.get_cmap("tab10", max(n_classes, 3))(np.arange(max(n_classes, 3)))

    def _savefig(stem, fig=None, tight=True):
        fig = plt.gcf() if fig is None else fig
        path = _figure_path(output_root, stem)
        if tight:
            fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        log_message(f"Wrote plot file: {path}")

    def _as_array(Xlike):
        return Xlike.to_numpy() if hasattr(Xlike, "to_numpy") else np.asarray(Xlike)

    def _safe_w(wv):
        return np.abs(np.asarray(wv, float).ravel())

    def _weighted_pearson(x, y_arr, wv, eps=1e-12):
        x = np.asarray(x, float).ravel()
        y_arr = np.asarray(y_arr, float).ravel()
        wv = _safe_w(wv)
        m = np.isfinite(x) & np.isfinite(y_arr) & np.isfinite(wv)
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
        vx = (wv * x0 * x0).sum() / (sw + eps)
        vy = (wv * y0 * y0).sum() / (sw + eps)
        return float(cov / (np.sqrt(vx * vy) + eps))

    def _plot_matrix_heatmap(matrix, row_labels, col_labels, stem, *, aspect, annotate, cbar_label=None):
        matrix = np.asarray(matrix, dtype=float)
        fig_w = max(6.5, 0.55 * len(col_labels) + 4.0)
        fig_h = max(5.0, 0.48 * len(row_labels) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        x_edges = np.arange(matrix.shape[1] + 1, dtype=float)
        y_edges = np.arange(matrix.shape[0] + 1, dtype=float)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap="bwr",
            vmin=-1.0,
            vmax=1.0,
            shading="flat",
            edgecolors="white",
            linewidth=0.35,
            antialiased=False,
            rasterized=False,
        )
        ax.set_xlim(0.0, float(matrix.shape[1]))
        ax.set_ylim(0.0, float(matrix.shape[0]))
        ax.invert_yaxis()
        ax.set_aspect(aspect)
        ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
        ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
        ax.set_xticklabels(col_labels, rotation=90 if aspect == "equal" else 45,
                           ha="center" if aspect == "equal" else "right", fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        if cbar_label:
            cbar.set_label(cbar_label)
        if annotate:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v = matrix[i, j]
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{v:+.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(v) > 0.5 else "black",
                        fontsize=10,
                    )
        _savefig(stem, fig=fig)

    def _roc_binary(mask, scores, ys, ws, positive_idx):
        if not np.any(mask):
            return None
        y_bin = (ys[mask] == positive_idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return None
        auc = roc_auc_score(y_bin, scores[mask], sample_weight=ws[mask])
        fpr, tpr, _ = roc_curve(y_bin, scores[mask], sample_weight=ws[mask])
        return fpr, tpr, auc

    if SIGNAL_CLASS_INDICES and BACKGROUND_CLASS_INDICES:
        roc_pairs = [
            (sig_idx, bkg_idx)
            for sig_idx in SIGNAL_CLASS_INDICES for bkg_idx in BACKGROUND_CLASS_INDICES
        ]
        roc_signal_groups = [
            (sig_idx, list(BACKGROUND_CLASS_INDICES)) for sig_idx in SIGNAL_CLASS_INDICES
        ]
    else:
        roc_pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        roc_signal_groups = [
            (i, [j for j in range(n_classes) if j != i]) for i in range(n_classes)
        ]

    decor_var_names = [full_feature_names[i] for i in decor_idx_full]

    def _plot_for_model(model, suffix, stage_tag):
        if model is None:
            return
        probs_train = _predict_proba(model, X_train_used)
        probs_test = _predict_proba(model, X_test_used)
        margins_train = _predict_margins(model, X_train_used)
        margins_test = _predict_margins(model, X_test_used)
        booster = _booster_from_model(model)

        # ROC plots
        def _plot_roc_for_signal(sig_idx, bkg_indices):
            sig_name = class_names[sig_idx]
            fig, ax = plt.subplots(figsize=(10, 10))
            any_curve = False
            for bkg_idx in bkg_indices:
                bkg_name = class_names[bkg_idx]
                score_train = probs_train[:, sig_idx] / np.clip(
                    probs_train[:, sig_idx] + probs_train[:, bkg_idx], _EPS, None
                )
                score_test = probs_test[:, sig_idx] / np.clip(
                    probs_test[:, sig_idx] + probs_test[:, bkg_idx], _EPS, None
                )
                mask_train = (y_train == sig_idx) | (y_train == bkg_idx)
                mask_test = (y_test == sig_idx) | (y_test == bkg_idx)
                color = palette[bkg_idx]
                r_tst = _roc_binary(mask_test, score_test, y_test, w_test, sig_idx)
                r_trn = _roc_binary(mask_train, score_train, y_train, w_train, sig_idx)
                if r_tst:
                    fpr, tpr, auc = r_tst
                    ax.plot(tpr, fpr, color=color, linestyle="-",
                            label=f"Test vs {bkg_name} AUC={auc:.3f}")
                    log_message(
                        f"{tree_name} [{stage_tag}] test AUC ({sig_name} vs {bkg_name}) = {auc:.4f}"
                    )
                    any_curve = True
                if r_trn:
                    fpr, tpr, auc = r_trn
                    ax.plot(tpr, fpr, color=color, linestyle="--",
                            label=f"Train vs {bkg_name} AUC={auc:.3f}")
                    log_message(
                        f"{tree_name} [{stage_tag}] train AUC ({sig_name} vs {bkg_name}) = {auc:.4f}"
                    )
                    any_curve = True
            if not any_curve:
                plt.close(fig)
                return
            ax.set_xlabel(rf"$\epsilon_{{\rm {sig_name}}}$", fontsize=20)
            ax.set_ylabel(r"$\epsilon_{\rm bkg}$", fontsize=20)
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 1)
            ax.set_xlim(0, 1)
            ax.legend(loc="lower right", fontsize=12)
            _savefig(f"roc_{_slugify(sig_name)}{suffix}", fig=fig)

        for sig_idx, bkg_indices in roc_signal_groups:
            _plot_roc_for_signal(sig_idx, bkg_indices)

        # Importance plot
        score_map = booster.get_score(importance_type="gain")
        importances = []
        for i, name in enumerate(feat_names_used):
            importances.append(float(score_map.get(name, score_map.get(f"f{i}", 0.0))))
        importances = np.asarray(importances, dtype=float)
        positive = importances > 0.0
        if not np.any(positive):
            positive = np.ones_like(importances, dtype=bool)
        imp_names = [feat_names_used[i] for i in np.where(positive)[0]]
        imp_vals = importances[positive]
        order = np.argsort(imp_vals)
        imp_names = [imp_names[i] for i in order]
        imp_vals = imp_vals[order]
        max_label_len = max((len(name) for name in imp_names), default=10)
        fig_h = max(4.0, 0.24 * len(imp_names) + 1.4)
        fig_w = max(7.0, min(13.0, 4.8 + 0.055 * max_label_len))
        left_margin = min(0.45, max(0.16, 0.0065 * max_label_len + 0.06))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        y_pos = np.arange(len(imp_names))
        ax.barh(y_pos, np.maximum(imp_vals, 1e-12), color="steelblue", edgecolor="none", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_names, fontsize=10)
        ax.set_title(f"{tree_name} Feature Importance [{stage_tag}]", fontsize=16)
        ax.set_xlabel("Gain", fontsize=12)
        positive_vals = imp_vals[imp_vals > 0.0]
        if positive_vals.size > 0:
            ax.set_xscale("log")
            ax.set_xlim(max(np.min(positive_vals) / 2.0, 1e-12), np.max(positive_vals) * 2.0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.94, bottom=0.08)
        _savefig(f"importance{suffix}", fig=fig, tight=False)

        # Score distributions
        def _plot_score_dist(sig_idx, bkg_idx):
            sig_name = class_names[sig_idx]
            bkg_name = class_names[bkg_idx]
            score_train = probs_train[:, sig_idx] / np.clip(
                probs_train[:, sig_idx] + probs_train[:, bkg_idx], _EPS, None
            )
            score_test = probs_test[:, sig_idx] / np.clip(
                probs_test[:, sig_idx] + probs_test[:, bkg_idx], _EPS, None
            )
            mask_train = (y_train == sig_idx) | (y_train == bkg_idx)
            mask_test = (y_test == sig_idx) | (y_test == bkg_idx)
            bins = np.linspace(0, 1, 31)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 1)
            ax.hist(
                score_train[mask_train & (y_train == bkg_idx)],
                bins=bins,
                weights=w_train[mask_train & (y_train == bkg_idx)],
                density=True,
                histtype="bar",
                alpha=0.5,
                label=f"Train {bkg_name}",
            )
            ax.hist(
                score_train[mask_train & (y_train == sig_idx)],
                bins=bins,
                weights=w_train[mask_train & (y_train == sig_idx)],
                density=True,
                histtype="bar",
                alpha=0.5,
                label=f"Train {sig_name}",
            )
            ax.hist(
                score_test[mask_test & (y_test == bkg_idx)],
                bins=bins,
                weights=w_test[mask_test & (y_test == bkg_idx)],
                density=True,
                histtype="step",
                linewidth=2,
                color="lime",
                label=f"Test {bkg_name}",
            )
            ax.hist(
                score_test[mask_test & (y_test == sig_idx)],
                bins=bins,
                weights=w_test[mask_test & (y_test == sig_idx)],
                density=True,
                histtype="step",
                linewidth=2,
                color="red",
                label=f"Test {sig_name}",
            )
            ax.set_xlabel("BDT Score")
            ax.set_yscale("log")
            ax.set_ylim(1e-2,)
            ax.set_ylabel("Density")
            ax.legend()
            _savefig(f"score_{_slugify(sig_name)}_vs_{_slugify(bkg_name)}{suffix}", fig=fig)

        for sig_idx, bkg_idx in roc_pairs:
            _plot_score_dist(sig_idx, bkg_idx)

        # Decorrelation correlation matrices (one per split), save under suffix.
        if decor_idx_full:
            def _build_corr_matrix(scores, Xfull, y_true, wv):
                Xarr = _as_array(Xfull)
                wv_abs = _safe_w(wv)
                R = np.zeros((n_classes, len(decor_idx_full)))
                for r, ci in enumerate(range(n_classes)):
                    class_mask = np.asarray(y_true) == ci
                    if not np.any(class_mask):
                        continue
                    s = scores[class_mask, ci]
                    for c, j in enumerate(decor_idx_full):
                        R[r, c] = _weighted_pearson(Xarr[class_mask, j], s, wv_abs[class_mask])
                return R

            for tag, scores, Xfull, y_true, wv in [
                ("train", margins_train, X_train_full, y_train, w_train),
                ("test", margins_test, X_test_full, y_test, w_test),
            ]:
                R = _build_corr_matrix(scores, Xfull, y_true, wv)
                _plot_matrix_heatmap(
                    R,
                    class_names,
                    decor_var_names,
                    f"decor_corr_{tag}{suffix}",
                    aspect="auto",
                    annotate=True,
                    cbar_label="weighted Pearson r",
                )
                for i, cls_name in enumerate(class_names):
                    stats = ", ".join(
                        f"{decor_var_names[j]}={R[i, j]:+.3f}" for j in range(len(decor_var_names))
                    )
                    log_message(f"{tree_name} [{stage_tag}] {tag} decor corr [{cls_name}] {stats}")

    _plot_for_model(stage1_model, "_cls", "stage1")
    _plot_for_model(stage2_model, "_decorr", "stage2")

    # ---- Shared plots (saved once) ----
    def _plot_loss_metric(metric_key, ylabel, stem):
        tr_loss = list(loss_history.get("train", {}).get(metric_key, []))
        te_loss = list(loss_history.get("test", {}).get(metric_key, []))
        if not tr_loss and not te_loss:
            return
        n_rounds = max(len(tr_loss), len(te_loss))
        fig, ax = plt.subplots(figsize=(8, 5))
        if tr_loss:
            ax.plot(range(1, len(tr_loss) + 1), tr_loss, label="Train")
        if te_loss:
            ax.plot(range(1, len(te_loss) + 1), te_loss, label="Test")
        if stage_boundary is not None and 0 < int(stage_boundary) < n_rounds:
            ax.axvline(
                float(stage_boundary) + 0.5,
                color="gray", linestyle=":", alpha=0.7,
                label="stage 2 start",
            )
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        ax.set_xlim(1, max(1, n_rounds))
        finite_vals = [v for v in tr_loss + te_loss if np.isfinite(v)]
        if finite_vals:
            top = max(finite_vals)
            if top <= 0.0:
                top = 1.0
            ax.set_ylim(0.0, top * 1.05)
        else:
            ax.set_ylim(bottom=0.0)
        _savefig(stem, fig=fig)

    _plot_loss_metric("classification", "classification_loss", "loss_classification")
    _plot_loss_metric("mlogloss", "mlogloss", "loss_mlogloss")
    _plot_loss_metric("decorrelation", "decorrelation_loss", "loss_decorrelation")
    _plot_loss_metric("total", "total_loss", "loss_total")

    X_tr_df = pd.DataFrame(_as_array(X_train_used), columns=feat_names_used)
    corr = X_tr_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if not corr.empty:
        _plot_matrix_heatmap(
            corr.values,
            list(corr.index),
            list(corr.columns),
            "feature_corr",
            aspect="equal",
            annotate=False,
        )


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


def _split_mass_thresholds(thresholds):
    mass_thresholds = {}
    other_thresholds = {}
    for name, cond in thresholds.items():
        if str(name).startswith("ScoutingFatPFJetRecluster_msoftdrop_"):
            mass_thresholds[name] = cond
        else:
            other_thresholds[name] = cond
    return mass_thresholds, other_thresholds


def _drop_decorrelated_features(X, decorrelate_feature_names):
    if not decorrelate_feature_names:
        return X.copy()
    X_out = X.copy()
    drop_cols = [name for name in decorrelate_feature_names if name in X_out.columns]
    if drop_cols:
        X_out = X_out.drop(columns=drop_cols)
    return X_out


def _write_prediction_reference(
    output_root,
    stem,
    tree_name,
    pipeline_name,
    feature_names,
    sample_labels,
    class_idx,
    weights,
    proba,
    *,
    weight_rtol=1e-10,
    weight_atol=1e-12,
    proba_rtol=1e-6,
    proba_atol=1e-9,
):
    path = _reference_path(output_root, stem)
    np.savez_compressed(
        path,
        tree_name=np.asarray(str(tree_name)),
        pipeline_name=np.asarray(str(pipeline_name)),
        feature_names=np.asarray(list(feature_names), dtype=str),
        sample_name=np.asarray(sample_labels, dtype=str),
        class_idx=np.asarray(class_idx, dtype=np.int32),
        weight=np.asarray(weights, dtype=np.float64),
        proba=np.asarray(proba, dtype=np.float64),
        weight_rtol=np.asarray(float(weight_rtol)),
        weight_atol=np.asarray(float(weight_atol)),
        proba_rtol=np.asarray(float(proba_rtol)),
        proba_atol=np.asarray(float(proba_atol)),
    )
    log_message(f"Wrote reference file: {path}")


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
        X_train, y_train, w_train, sample_labels_train, _ = prepare_split_data(
            tree_name, load_cols, "train", split_plans, shuffle=True
        )
        check_weights(w_train, f"{tree_name}_train_weight_before_filter")

        log_message(f"Loading test split for tree = {tree_name}")
        X_test, y_test, w_test, sample_labels_test, w_test_physics = prepare_split_data(
            tree_name, load_cols, "test", split_plans, shuffle=False
        )
        check_weights(w_test, f"{tree_name}_test_weight_before_filter")
        check_weights(w_test_physics, f"{tree_name}_test_physics_weight_before_filter")
        X_test_unfiltered = X_test.copy()
        y_test_unfiltered = y_test.copy()
        w_test_unfiltered = w_test.copy()
        w_test_physics_unfiltered = w_test_physics.copy()
        sample_labels_test_unfiltered = np.asarray(sample_labels_test).copy()

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
        X_test_ref, y_test_ref, w_test_ref, sample_labels_test_ref = filter_X(
            X_test_unfiltered.copy(),
            y_test_unfiltered.copy(),
            w_test_physics_unfiltered.copy(),
            load_cols,
            thresholds,
            apply_to_sentinel=True,
            sample_labels=sample_labels_test_unfiltered.copy(),
        )
        if not X_test_ref.index.equals(X_test.index):
            raise RuntimeError("Filtered model/reference test splits are misaligned")
        _validate_filtered_split(tree_name, "test", y_test, w_test, sample_labels_test)
        check_weights(w_test, f"{tree_name}_test_weight_after_filter")
        check_weights(w_test_ref, f"{tree_name}_test_physics_weight_after_filter")

        if drop_after_filter:
            X_train = X_train.drop(columns=drop_after_filter, errors="ignore")
            X_test = X_test.drop(columns=drop_after_filter, errors="ignore")
            X_test_ref = X_test_ref.drop(columns=drop_after_filter, errors="ignore")

        log_message(f"Standardising training split for tree = {tree_name}")
        X_train_std = standardize_X(X_train.copy(), clip_ranges, log_tf)
        log_message(f"Standardising test split for tree = {tree_name}")
        X_test_std = standardize_X(X_test.copy(), clip_ranges, log_tf)

        log_message(f"Training model for tree = {tree_name}")
        stage1_model, stage2_model, splits, loss_history, stage_boundary = train_multi_model(
            X_train_std, y_train, w_train,
            X_test_std, y_test, w_test,
            model_path, tree_name,
            decorrelate_feature_names=decorrelate
        )
        final_model = stage2_model if stage2_model is not None else stage1_model

        X_test_signal_model = _drop_decorrelated_features(X_test_std, decorrelate)
        proba_signal_test = _predict_proba(final_model, X_test_signal_model)
        _write_prediction_reference(
            output_root,
            "test_reference_signal_region",
            tree_name,
            "signal_region",
            X_test_signal_model.columns,
            sample_labels_test_ref,
            y_test_ref,
            w_test_ref,
            proba_signal_test,
        )

        mass_thresholds, bdt_thresholds = _split_mass_thresholds(thresholds)
        log_message(
            f"Preparing qcd_est reference for tree = {tree_name}: "
            f"non_mass_thresholds={len(bdt_thresholds)}, mass_thresholds={len(mass_thresholds)}"
        )
        X_test_qcd_raw, y_test_qcd, w_test_qcd, sample_labels_test_qcd = filter_X(
            X_test_unfiltered,
            y_test_unfiltered,
            w_test_unfiltered,
            load_cols,
            bdt_thresholds,
            apply_to_sentinel=True,
            sample_labels=sample_labels_test_unfiltered,
        )
        X_test_qcd_ref, y_test_qcd_ref, w_test_qcd_ref, sample_labels_test_qcd_ref = filter_X(
            X_test_unfiltered.copy(),
            y_test_unfiltered.copy(),
            w_test_physics_unfiltered.copy(),
            load_cols,
            bdt_thresholds,
            apply_to_sentinel=True,
            sample_labels=sample_labels_test_unfiltered.copy(),
        )
        if not X_test_qcd_ref.index.equals(X_test_qcd_raw.index):
            raise RuntimeError("Filtered qcd_est model/reference test splits are misaligned")
        X_test_qcd_model = standardize_X(X_test_qcd_raw[branches].copy(), clip_ranges, log_tf)
        X_test_qcd_model = _drop_decorrelated_features(X_test_qcd_model, decorrelate)
        proba_qcd_test = _predict_proba(final_model, X_test_qcd_model)
        _write_prediction_reference(
            output_root,
            "test_reference_qcd_est",
            tree_name,
            "qcd_est",
            X_test_qcd_model.columns,
            sample_labels_test_qcd_ref,
            y_test_qcd_ref,
            w_test_qcd_ref,
            proba_qcd_test,
        )

        log_message(f"Plotting results for tree = {tree_name}")
        plot_results(
            stage1_model,
            stage2_model,
            splits,
            tree_name,
            output_root,
            loss_history,
            stage_boundary,
            decorrelate_feature_names=decorrelate,
        )
        log_message(f"Finished train.py for tree = {tree_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
