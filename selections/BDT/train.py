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

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
from typing import List

from model_io import (
    TorchModelHandle,
    build_torch_mlp,
    import_torch,
    model_type_from_config,
    predict_model_logits as _shared_predict_model_logits,
    predict_model_proba as _shared_predict_model_proba,
)

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
# Per-sample cap on TEST events too: the per-round custom metric (and ROC eval) runs
# over the full test set each boosting round, which dominates wall-clock for the huge
# QCD/Top classes.  Defaults to ENTRIES_PER_SAMPLE.  Small classes (e.g. VV) stay full.
TEST_ENTRIES_PER_SAMPLE = cfg.get("test_entries_per_sample", ENTRIES_PER_SAMPLE)
# Evaluate the (expensive) custom loss metric only every Nth boosting round. The stage-2
# decorrelation metric runs over the full train+test sets each round and dominates
# wall-clock; computing it every Nth round (default 1 = every round) cuts that cost ~N x.
# Early-stopping / lr-schedule patience is interpreted in actual rounds (scaled internally).
METRIC_EVAL_EVERY = max(1, int(cfg.get("metric_eval_every", 1)))
TRAIN_FRACTION     = float(cfg.get("train_fraction", 0.7))
DECOR_LAMBDA       = cfg.get("decor_lambda", 30)
DECOR_LOSS_MODE    = str(cfg.get("decor_loss_mode", "smooth_cvm")).strip().lower()
DECOR_N_BINS       = int(cfg.get("decor_n_bins", 5))
DECOR_N_THRESHOLDS = int(cfg.get("decor_n_thresholds", 31))
DECOR_SCORE_TAU    = float(cfg.get("decor_score_tau", 0.20))
DECOR_BIN_TAU_SCALE = float(cfg.get("decor_bin_tau_scale", 0.35))
DECOR_MSOFTDROP_TRAINING_MAX = float(cfg.get("decor_msoftdrop_training_max", 200.0))
# Tail-aware decorrelation (optional): also enforce score-vs-mass flatness deep in
# the score tails where the signal-region boxes live (default off => unchanged).
DECOR_TAIL_AWARE    = bool(cfg.get("decor_tail_aware", False))
DECOR_TAIL_MIN_PROB = float(cfg.get("decor_tail_min_prob", 0.002))
DECOR_TAIL_N        = int(cfg.get("decor_tail_n", 8))
SUBMIT_TREES       = cfg.get("submit_trees", ["fat2"])
INPUT_ROOT         = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN      = cfg["input_pattern"]
OUTPUT_ROOT_PATTERN = cfg.get("output_root", ".")
MODEL_PATTERN      = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
MODEL_TYPE         = model_type_from_config(cfg)

if not 0.0 < TRAIN_FRACTION < 1.0:
    raise ValueError(f"train_fraction must be in (0, 1), got {TRAIN_FRACTION}")
if not np.isfinite(DECOR_MSOFTDROP_TRAINING_MAX) or DECOR_MSOFTDROP_TRAINING_MAX <= 0.0:
    raise ValueError(
        "decor_msoftdrop_training_max must be a positive finite number, "
        f"got {DECOR_MSOFTDROP_TRAINING_MAX}"
    )

if DECOR_TAIL_AWARE:
    if not (0.0 < DECOR_TAIL_MIN_PROB < 0.02):
        raise ValueError(
            "decor_tail_min_prob must be in (0, 0.02) when decor_tail_aware is set, "
            f"got {DECOR_TAIL_MIN_PROB}"
        )
    if DECOR_TAIL_N < 0:
        raise ValueError(f"decor_tail_n must be >= 0, got {DECOR_TAIL_N}")

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


def _qcd_ht_bounds(sample_name):
    text = str(sample_name).strip().lower()
    prefix = "qcd_ht"
    if not text.startswith(prefix):
        return None
    suffix = text[len(prefix):]
    if "to" in suffix:
        low_text, high_text = suffix.split("to", 1)
    else:
        low_text, high_text = suffix, None
    if not low_text.isdigit() or (high_text is not None and not high_text.isdigit()):
        return None
    low = int(low_text)
    high = int(high_text) if high_text is not None else None
    return low, high


def _qcd_ht_sort_key(sample_name):
    bounds = _qcd_ht_bounds(sample_name)
    if bounds is None:
        return (float("inf"), float("inf"), str(sample_name))
    low, high = bounds
    return (low, high if high is not None else float("inf"), str(sample_name))


def _qcd_ht_training_weight_scales(tree_name):
    step = float(cfg.get(tree_name, {}).get("qcd_ht_training_weight_step", 0.0))
    if not np.isfinite(step) or step < 0.0:
        raise ValueError(
            f"qcd_ht_training_weight_step for tree '{tree_name}' must be a finite non-negative number, "
            f"got {step}"
        )
    if step <= 0.0:
        return {}

    qcd_ht_samples = []
    seen = set()
    for members in CLASS_GROUPS.values():
        for sample_name in members:
            if sample_name in seen:
                continue
            bounds = _qcd_ht_bounds(sample_name)
            if bounds is None:
                continue
            low, high = bounds
            qcd_ht_samples.append((low, high if high is not None else float("inf"), sample_name))
            seen.add(sample_name)

    if not qcd_ht_samples:
        log_warning(
            f"tree '{tree_name}' has qcd_ht_training_weight_step={step:g} "
            "but no qcd_ht samples in class_groups"
        )
        return {}

    qcd_ht_samples.sort(key=lambda item: (item[0], item[1], item[2]))
    return {
        sample_name: 1.0 + step * rank
        for rank, (_low, _high, sample_name) in enumerate(qcd_ht_samples)
    }


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


def _decor_efficiencies_for_tree(tree_name):
    values = cfg.get(tree_name, {}).get("decor_efficiencies", [1.0, 0.5, 0.1, 0.01])
    effs = []
    for value in values:
        eff = float(value)
        if not 0.0 < eff <= 1.0:
            raise ValueError(
                f"decor_efficiencies for tree '{tree_name}' must be fractions in (0, 1], got {eff}"
            )
        if not any(abs(eff - old) < 1e-12 for old in effs):
            effs.append(eff)
    return effs


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


def _rebalance_filtered_weights(split_name, y, w, sample_labels):
    balance_df = pd.DataFrame({
        "weight": np.asarray(w, dtype=float),
        "class_idx": np.asarray(y, dtype=int),
        "sample_name": np.asarray(sample_labels),
    })
    balance_df = _rebalance_class_weights(balance_df)
    _report_sample_weights(
        balance_df,
        f"Sample totals after thresholding and class balancing ({split_name})",
    )
    return balance_df["weight"].to_numpy(dtype=float, copy=True)


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
        raise RuntimeError(f"No ROOT files found for sample '{sample_name}' in tree '{tree_name}'")

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
        raise RuntimeError(f"Zero entries found for sample '{sample_name}' in tree '{tree_name}'")

    train_stop = int(total_entries * TRAIN_FRACTION)
    return {
        "sample_name": sample_name,
        "file_infos": file_infos,
        "total_entries": total_entries,
        "train_stop": train_stop,
        "test_start": train_stop,
        "train_segments_full": _build_segments(file_infos, 0, train_stop),
        "train_segments_read": _build_segments(file_infos, 0, train_stop, max_entries=ENTRIES_PER_SAMPLE),
        "test_segments": _build_segments(file_infos, train_stop, total_entries, max_entries=TEST_ENTRIES_PER_SAMPLE),
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

    if len(parts) == 1:
        return parts[0], n_read

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


def prepare_split_data(tree_name, branches, split_name, split_plans, shuffle, training_weight_scales=None):
    dfs = []
    sample_target_totals = {}
    if training_weight_scales is None:
        training_weight_scales = _qcd_ht_training_weight_scales(tree_name)

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
        training_weight_scale = float(training_weight_scales.get(sample_name, 1.0))
        if raw_entries <= 0:
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
            raise RuntimeError(
                f"Zero entries read for sample '{sample_name}' in split '{split_name}' of tree '{tree_name}'"
            )

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
        training_target_total = target_total * training_weight_scale

        if target_total <= 0.0:
            df["weight_physics"] = 0.0
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
        if "weight_physics" not in df.columns:
            df["weight_physics"] = 0.0
        del raw_w

        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        df["weight"] = df["weight_physics"] * training_weight_scale
        sample_target_totals[sample_name] = training_target_total
        dfs.append(df)

        log_message(
            f"  {sample_name}: split={split_name}, tree_entries={plan['total_entries']}, "
            f"split_entries={split_total_entries}, used_entries={n_read}, raw_entries={raw_entries}, "
            f"target_total={target_total:.6g}, training_scale={training_weight_scale:.6g}, "
            f"training_target_total={training_target_total:.6g}, "
            f"class={CLASS_NAMES[SAMPLE_TO_CLASS[sample_name]]}"
        )

    if not dfs:
        raise RuntimeError(f"No data loaded for split '{split_name}' in tree '{tree_name}'")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    _validate_sample_weight_totals(df_all, sample_target_totals)
    _report_sample_weights(df_all, f"Sample totals before thresholding ({split_name})")
    missing_classes = [
        cls_name for cls_idx, cls_name in enumerate(CLASS_NAMES)
        if float(df_all.loc[df_all["class_idx"] == cls_idx, "weight"].sum()) <= 0.0
    ]
    if missing_classes:
        raise RuntimeError(
            f"Missing positive-weight content for split '{split_name}' in classes: "
            + ", ".join(missing_classes)
        )

    if shuffle:
        df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df_all[branches].copy()
    y = df_all["class_idx"].to_numpy(dtype=int, copy=True)
    w = df_all["weight"].to_numpy(dtype=float, copy=True)
    w_physics = df_all["weight_physics"].to_numpy(dtype=float, copy=True)
    sample_labels = df_all["sample_name"].astype(str).to_numpy(copy=True)

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
             sample_labels=None, return_index: bool = False):
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
        kept_index = X.index.to_numpy(copy=True)
        if sample_labels is None:
            result = (X.copy(), y.copy(), w.copy())
        else:
            result = (X.copy(), y.copy(), w.copy(), np.asarray(sample_labels).copy())
        if return_index:
            return (*result, kept_index)
        return result

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
    kept_index = X_out.index.to_numpy(copy=True)
    if sample_labels is None:
        result = (X_out, y_out, w_out)
    else:
        result = (X_out, y_out, w_out, np.asarray(sample_labels)[mask.values].copy())
    if return_index:
        return (*result, kept_index)
    return result


# -------------------- Feature standardization --------------------
def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    """Clip values and apply log transform in-place; sentinel values (< -990) are untouched."""
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].to_numpy(copy=False)
        needs_assign = False
        if not arr.flags.writeable:
            arr = arr.copy()
            needs_assign = True
        changed = False
        mask = arr < -990   # Sentinel placeholder values.
        valid = ~mask
        if not valid.any():
            if needs_assign:
                X[col] = arr
            continue

        lo, hi = clip_ranges.get(col, (None, None))
        if lo is not None:
            low = valid & (arr < lo)
            if low.any():
                arr[low] = lo
                changed = True
        if hi is not None:
            high = valid & (arr > hi)
            if high.any():
                arr[high] = hi
                changed = True

        if col in log_set:
            pos = valid & (arr > 0)
            if pos.any():
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                    needs_assign = True
                arr[pos] = np.log(arr[pos])
                changed = True

        if needs_assign or changed:
            X[col] = arr
    return X


def _clip_only_X(X: pd.DataFrame, clip_ranges: dict) -> pd.DataFrame:
    """Apply clip_ranges only (no log_transform); sentinel values (< -990) untouched."""
    X = X.copy()
    for col in X.columns:
        arr = X[col].to_numpy(copy=False)
        needs_assign = False
        if not arr.flags.writeable:
            arr = arr.copy()
            needs_assign = True
        changed = False
        mask = arr < -990
        valid = ~mask
        if not valid.any():
            if needs_assign:
                X[col] = arr
            continue
        lo, hi = clip_ranges.get(col, (None, None))
        if lo is not None:
            low = valid & (arr < lo)
            if low.any():
                arr[low] = lo
                changed = True
        if hi is not None:
            high = valid & (arr > hi)
            if high.any():
                arr[high] = hi
                changed = True
        if needs_assign or changed:
            X[col] = arr
    return X


def _clipped_column_values(X: pd.DataFrame, col: str, clip_ranges: dict) -> np.ndarray:
    """Return one clipped column as float values without copying the full DataFrame."""
    arr = X[col].to_numpy(copy=True)
    mask = arr < -990
    valid = ~mask
    if not valid.any():
        return arr.astype(float, copy=False)
    lo, hi = clip_ranges.get(col, (None, None))
    if lo is not None:
        arr[valid & (arr < lo)] = lo
    if hi is not None:
        arr[valid & (arr > hi)] = hi
    return arr.astype(float, copy=False)


# -------------------- Input branch distribution plots --------------------
def _sample_color_palette(n):
    """Return up to ``n`` reasonably distinct colors for per-sample plots."""
    cmaps = ["tab20", "tab20b", "tab20c"]
    colors = []
    for cm in cmaps:
        colors.extend(list(plt.colormaps[cm].colors))
    if n <= 0:
        return []
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def plot_branch_distributions(output_root, branches, clip_ranges,
                              X_train, y_train, w_train, sample_labels_train,
                              X_test, y_test, w_test, sample_labels_test,
                              n_bins=200):
    """Plot normalized per-class and per-sample distributions for each training branch.

    Uses train+test samples combined. Values are after thresholds and
    clip_ranges, but BEFORE log_transform. For each branch, saves two PDFs
    under ``{output_root}/branches/``: ``{branch}.pdf`` (one curve per class)
    and ``{branch}_by_sample.pdf`` (one curve per sub-sample).
    """
    out_dir = os.path.join(output_root, "branches")
    os.makedirs(out_dir, exist_ok=True)

    y_all = np.concatenate([np.asarray(y_train, dtype=int),
                            np.asarray(y_test, dtype=int)])
    w_all = np.concatenate([np.asarray(w_train, dtype=float),
                            np.asarray(w_test, dtype=float)])
    s_all = np.concatenate([np.asarray(sample_labels_train, dtype=object),
                            np.asarray(sample_labels_test, dtype=object)])

    class_palette = plt.colormaps["tab10"].resampled(max(NUM_CLASSES, 2))(np.arange(max(NUM_CLASSES, 2)))

    # Keep TRAINING_SAMPLES ordering (which follows class_groups) so samples
    # from the same class sit together in the legend and color palette.
    sample_names_ordered = [s for s in TRAINING_SAMPLES if np.any(s_all == s)]
    sample_palette = _sample_color_palette(len(sample_names_ordered))

    for col in branches:
        v_all = np.concatenate([
            _clipped_column_values(X_train, col, clip_ranges),
            _clipped_column_values(X_test, col, clip_ranges),
        ])
        valid = v_all > -990
        if not np.any(valid):
            log_warning(f"branch '{col}' has no valid entries to plot, skipping")
            continue

        v_valid = v_all[valid]
        lo = float(np.min(v_valid))
        hi = float(np.max(v_valid))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            log_warning(f"branch '{col}' has degenerate range ({lo}, {hi}), skipping")
            continue
        bins = np.linspace(lo, hi, n_bins + 1)

        # -------- per-class view --------
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = valid & (y_all == cls_idx)
            if not np.any(mask):
                continue
            w_cls = w_all[mask]
            if float(np.sum(w_cls)) <= 0.0:
                continue
            ax.hist(
                v_all[mask],
                bins=bins,
                weights=w_cls,
                density=True,
                histtype="step",
                linewidth=2,
                color=class_palette[cls_idx],
                label=cls_name,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            log_warning(f"branch '{col}' has no positive-weight entries, skipping")
            continue

        ax.set_xlim(lo, hi)
        ax.set_xlabel(col)
        ax.set_ylabel("A.U.")
        ax.legend()
        path = os.path.join(out_dir, f"{col}.pdf")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        log_message(f"Wrote plot file: {path}")

        # -------- per-sample view --------
        fig, ax = plt.subplots(figsize=(9, 6))
        plotted_any_sample = False
        for i, sname in enumerate(sample_names_ordered):
            mask = valid & (s_all == sname)
            if not np.any(mask):
                continue
            w_s = w_all[mask]
            if float(np.sum(w_s)) <= 0.0:
                continue
            ax.hist(
                v_all[mask],
                bins=bins,
                weights=w_s,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=sample_palette[i],
                label=sname,
            )
            plotted_any_sample = True

        if not plotted_any_sample:
            plt.close(fig)
            continue

        ax.set_xlim(lo, hi)
        ax.set_xlabel(col)
        ax.set_ylabel("A.U.")
        ax.legend(fontsize=8, ncol=2, loc="best")
        path = os.path.join(out_dir, f"{col}_by_sample.pdf")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        log_message(f"Wrote plot file: {path}")
        log_message(f"Wrote plot file: {path}")


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


def _build_decor_prob_grid():
    """Probability grid at which smooth-CvM enforces score-vs-msoftdrop flatness.

    Default: uniform in probability on [0.02, 0.98] (DECOR_N_THRESHOLDS points).
    Tail-aware (opt-in via decor_tail_aware): additionally place DECOR_TAIL_N
    log-spaced thresholds in each tail, down to decor_tail_min_prob and up to
    1 - decor_tail_min_prob, so the deep score tails where the signal-region boxes
    live (e.g. an anti-QCD cut at QCD-score < 0.01) are decorrelated against the
    soft-drop mass too -- the region the default [0.02, 0.98] grid never touches.
    """
    base = np.linspace(0.02, 0.98, max(3, DECOR_N_THRESHOLDS))
    if not DECOR_TAIL_AWARE or DECOR_TAIL_N <= 0:
        return base
    lo = float(min(max(DECOR_TAIL_MIN_PROB, 1e-6), 0.02))
    low_tail = np.geomspace(lo, 0.02, DECOR_TAIL_N + 1)[:-1]
    high_tail = 1.0 - low_tail
    return np.unique(np.concatenate([low_tail, base, high_tail]))


def _build_decor_state(Z: np.ndarray, mode: str):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.size == 0 or Z.shape[1] == 0:
        return {"mode": "none"}
    if mode == "cvm":
        return {"mode": "cvm", "groups": _build_cvm_groups(Z)}
    if mode == "smooth_cvm":
        prob_grid = _build_decor_prob_grid()
        score_thresholds = np.log(prob_grid / (1.0 - prob_grid))
        if DECOR_TAIL_AWARE:
            log_message(
                f"Tail-aware decorrelation ON: {len(prob_grid)} thresholds, "
                f"prob in [{prob_grid.min():.4g}, {prob_grid.max():.4g}] "
                f"(+{DECOR_TAIL_N} log-spaced per tail down to {DECOR_TAIL_MIN_PROB:g})"
            )
        return {
            "mode": "smooth_cvm",
            "memberships": _build_smooth_cvm_memberships(Z),
            "score_thresholds": score_thresholds.astype(float),
            "score_tau": max(float(DECOR_SCORE_TAU), _EPS),
        }
    raise ValueError(f"Unsupported decorrelation loss mode: {mode}")


def _prepare_decor_state_for_labels(decor_state, labels, weights, num_class):
    mode = decor_state.get("mode", "none")
    if mode == "none":
        return decor_state

    labels = np.asarray(labels, dtype=int).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    class_states = []

    for cls_idx in range(int(num_class)):
        idx = np.nonzero(labels == cls_idx)[0]
        cls_weights = weights[idx]
        state = {"indices": idx, "weights": cls_weights}

        if mode == "smooth_cvm":
            feature_states = []
            for memberships in decor_state.get("memberships", []):
                feature_memberships = memberships[idx, :] if memberships is not None else None
                feature_states.append(
                    _prepare_smooth_cvm_feature_state(feature_memberships, cls_weights)
                )
            state["features"] = feature_states
        elif mode == "cvm":
            feature_groups = []
            for groups in decor_state.get("groups", []):
                if not groups:
                    feature_groups.append(None)
                    continue
                local_groups = []
                for group_idx in groups:
                    group_idx = np.asarray(group_idx, dtype=int)
                    cls_group_idx = group_idx[labels[group_idx] == cls_idx]
                    if cls_group_idx.size >= 3:
                        local_groups.append(np.searchsorted(idx, cls_group_idx))
                feature_groups.append(local_groups if local_groups else None)
            state["groups"] = feature_groups
        else:
            raise ValueError(f"Unsupported decorrelation loss mode: {mode}")

        class_states.append(state)

    prepared = dict(decor_state)
    prepared["class_states"] = class_states
    if mode == "smooth_cvm":
        prepared.pop("memberships", None)
    return prepared


def _prepare_smooth_cvm_feature_state(memberships, weights):
    if memberships is None:
        return None
    memberships = np.asarray(memberships, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if memberships.ndim != 2 or memberships.shape[0] != weights.size:
        raise ValueError("Smooth-CvM memberships and weights have incompatible shapes.")
    total_w = float(np.sum(weights))
    if memberships.shape[0] == 0 or total_w <= _EPS:
        return None

    weighted_memberships = weights[:, None] * memberships
    bin_totals = np.sum(weighted_memberships, axis=0)
    valid_bins = bin_totals > _EPS
    if not np.any(valid_bins):
        return None

    bin_totals_v = bin_totals[valid_bins]
    memberships_v = memberships[:, valid_bins]
    return {
        "weights": weights,
        "total_w": total_w,
        "weighted_memberships_t": np.ascontiguousarray(weighted_memberships[:, valid_bins].T),
        "bin_totals": bin_totals_v,
        "rho": bin_totals_v / total_w,
        "membership_over_bin_total": np.ascontiguousarray(memberships_v / bin_totals_v),
    }


def _smooth_cvm_value_and_grad_1d(score, feature_state, thresholds, score_tau,
                                  need_derivatives=True):
    score = np.asarray(score, dtype=float).ravel()
    n = score.size
    grad = np.zeros(n, dtype=float)
    hess = np.zeros(n, dtype=float)
    if feature_state is None or n == 0:
        return 0.0, grad, hess

    weights = feature_state["weights"]
    total_w = float(feature_state["total_w"])
    if weights.size != n:
        raise ValueError("Smooth-CvM feature state and score have incompatible shapes.")
    if total_w <= _EPS:
        return 0.0, grad, hess

    thresholds = np.asarray(thresholds, dtype=float).ravel()
    sig = _sigmoid((score[:, None] - thresholds[None, :]) / score_tau)

    global_eff = np.sum(weights[:, None] * sig, axis=0) / total_w
    local_eff = (feature_state["weighted_memberships_t"] @ sig) / feature_state["bin_totals"][:, None]
    delta = local_eff - global_eff[None, :]
    n_thr = float(sig.shape[1])

    weighted_delta = feature_state["rho"][:, None] * delta
    loss = float(np.sum(weighted_delta * delta) / n_thr)
    if not need_derivatives:
        return loss, grad, hess

    dsig = sig * (1.0 - sig) / score_tau
    local_term = feature_state["membership_over_bin_total"] @ weighted_delta
    global_term = np.sum(weighted_delta, axis=0) / total_w
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


def _decorrelation_loss_components(logits, labels, weights, decor_state, num_class,
                                   decor_scale=1.0, need_derivatives=True):
    labels = np.asarray(labels, dtype=int).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    logits = _reshape_multiclass_margin(logits, num_class, labels.size)
    mode = decor_state.get("mode", "none")
    grad = np.zeros_like(logits, dtype=float)
    hess = np.zeros_like(logits, dtype=float)
    loss = 0.0

    if mode == "none":
        return loss, grad, hess

    class_states = decor_state.get("class_states")
    if class_states is not None:
        for cls_idx in range(num_class):
            state = class_states[cls_idx]
            idx = state["indices"]
            if idx.size == 0:
                continue
            cls_weights = state["weights"]
            score = logits[idx, cls_idx]

            if mode == "cvm":
                groups_by_feature = state.get("groups", [])
                cls_loss = _cvm_flatness_value_from_groups(score, groups_by_feature, cls_weights, power=2.0)
                cls_grad = np.zeros_like(score)
                cls_hess = np.zeros_like(score)
                if need_derivatives:
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
                for feature_state in state.get("features", []):
                    part_loss, part_grad, part_hess = _smooth_cvm_value_and_grad_1d(
                        score,
                        feature_state,
                        decor_state["score_thresholds"],
                        decor_state["score_tau"],
                        need_derivatives=need_derivatives,
                    )
                    cls_loss += part_loss
                    if need_derivatives:
                        cls_grad += part_grad
                        cls_hess += part_hess
                if need_derivatives:
                    cls_hess = np.maximum(cls_hess, 1e-6)
            else:
                raise ValueError(f"Unsupported decorrelation loss mode: {mode}")

            loss += float(decor_scale * cls_loss)
            if need_derivatives:
                grad[idx, cls_idx] = decor_scale * cls_grad
                hess[idx, cls_idx] = decor_scale * cls_hess

        return float(loss), grad, hess

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
                feature_state = _prepare_smooth_cvm_feature_state(memberships, cls_weights)
                part_loss, part_grad, part_hess = _smooth_cvm_value_and_grad_1d(
                    score,
                    feature_state,
                    decor_state["score_thresholds"],
                    decor_state["score_tau"],
                    need_derivatives=need_derivatives,
                )
                cls_loss += part_loss
                if need_derivatives:
                    cls_grad += part_grad
                    cls_hess += part_hess
            if need_derivatives:
                cls_hess = np.maximum(cls_hess, 1e-6)
        else:
            raise ValueError(f"Unsupported decorrelation loss mode: {mode}")

        loss += float(decor_scale * cls_loss)
        if need_derivatives:
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
            predt, labels, weights, decor_state, num_class, decor_scale,
            need_derivatives=False,
        )
    decor_loss = float(lam * decor_loss_raw)
    return {
        "classification": float(cls_loss),
        "mlogloss": float(mlogloss),
        "decorrelation": decor_loss,
        "regularization": 0.0,
        "total": float(cls_loss + decor_loss),
    }


def _nn_loss_components_from_epoch_sums(epoch_sums, n_steps):
    if int(n_steps) <= 0:
        return {
            "classification": float("nan"),
            "mlogloss": float("nan"),
            "decorrelation": float("nan"),
            "regularization": 0.0,
            "objective": float("nan"),
            "total": float("nan"),
        }
    inv = 1.0 / float(n_steps)
    cls_loss = float(epoch_sums.get("classification", 0.0) * inv)
    decor_loss = float(epoch_sums.get("decorrelation", 0.0) * inv)
    reg_loss = float(epoch_sums.get("regularization", 0.0) * inv)
    objective_sum = epoch_sums.get("total")
    objective_loss = (
        float(objective_sum * inv)
        if objective_sum is not None
        else float(cls_loss + decor_loss)
    )
    return {
        "classification": cls_loss,
        "mlogloss": cls_loss,
        "decorrelation": decor_loss,
        "regularization": reg_loss,
        "objective": objective_loss,
        "total": objective_loss,
    }


def _nn_loss_components_from_values(cls_loss, decor_loss=0.0, reg_loss=0.0):
    cls_loss = float(cls_loss)
    decor_loss = float(decor_loss)
    reg_loss = float(reg_loss)
    objective_loss = float(cls_loss + decor_loss)
    return {
        "classification": cls_loss,
        "mlogloss": cls_loss,
        "decorrelation": decor_loss,
        "regularization": reg_loss,
        "objective": objective_loss,
        "total": objective_loss,
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
                 selection_metric_key="mlogloss", selection_metric_name=None, eval_every=1):
        self.datasets = list(datasets)
        self.eval_every = max(1, int(eval_every))
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
        n_sets = len(self.datasets)
        tag, labels, weights, decor_state = self.datasets[self._call_idx % n_sets]
        round_idx = self._call_idx // n_sets
        self._call_idx += 1
        keys = ("classification", "mlogloss", "decorrelation", "regularization", "total")
        # On non-evaluation rounds, skip the expensive full-loss recompute and repeat the
        # last evaluated value so the history stays index-aligned with the boosting round.
        if round_idx % self.eval_every != 0 and self.history[tag]["mlogloss"]:
            for key in keys:
                self.history[tag][key].append(self.history[tag][key][-1])
            return self.selection_metric_name, self.history[tag][self.selection_metric_key][-1]
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
        for key in keys:
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
                 monitor_metric_key="mlogloss", monitor_metric_label=None,
                 lr_reduce_patience=None, min_learning_rate=None, eval_every=1):
        self.recorder = recorder
        self.eval_every = max(1, int(eval_every))
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
        # Dynamic lr: when enabled, while current_lr > min_learning_rate the
        # early-stopping counter is suppressed; after lr_reduce_patience
        # consecutive stale rounds (no new best) lr halves (floored at
        # min_learning_rate) and both counters reset. Early stopping only takes
        # effect once current_lr has bottomed out at min_learning_rate.
        if lr_reduce_patience is not None and int(lr_reduce_patience) > 0 \
                and min_learning_rate is not None and float(min_learning_rate) > 0.0:
            self.lr_reduce_patience = int(lr_reduce_patience)
            self.min_learning_rate = float(min_learning_rate)
            self._dynamic_lr = True
        else:
            self.lr_reduce_patience = 0
            self.min_learning_rate = self.learning_rate
            self._dynamic_lr = False
        self.best_iteration = None  # stage-local best iteration
        self.best_score = float("inf")
        self._stale_rounds = 0
        self._lr_stale_rounds = 0

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
        # Only log / check early-stopping on rounds where the metric was actually
        # re-evaluated (every eval_every rounds); the regularization above is kept
        # accurate every round so the running total stays correct.
        if local_epoch % self.eval_every != 0:
            return False
        prefix = f"[{self.stage_label}]" if self.stage_label else ""
        log_message(_format_detailed_loss_line(
            local_epoch, self.recorder.history, prefix=prefix, compact=self.compact_log
        ))
        current_score = _loss_value_at(
            self.recorder.history, "test", self.monitor_metric_key, local_epoch
        )
        improved = np.isfinite(current_score) and (
            self.best_iteration is None or current_score < self.best_score - 1e-12
        )
        if improved:
            self.best_iteration = int(local_epoch)
            self.best_score = float(current_score)
            self._stale_rounds = 0
            self._lr_stale_rounds = 0
            model.set_attr(
                best_iteration=str(self.tree_offset + self.best_iteration),
                best_score=str(self.best_score),
            )
            return False

        # No new best this round.
        if self._dynamic_lr and self.learning_rate > self.min_learning_rate + 1e-12:
            # Still in the lr-reduction regime: early stopping is suppressed.
            self._lr_stale_rounds += 1
            if self._lr_stale_rounds * self.eval_every >= self.lr_reduce_patience:
                old_lr = self.learning_rate
                new_lr = max(old_lr * 0.5, self.min_learning_rate)
                try:
                    model.set_param({"learning_rate": new_lr, "eta": new_lr})
                except Exception:
                    model.set_param("learning_rate", new_lr)
                    model.set_param("eta", new_lr)
                self.learning_rate = float(new_lr)
                tag = f" ({self.stage_label})" if self.stage_label else ""
                log_message(
                    f"Info: lr reduced{tag} at epoch {local_epoch} "
                    f"(old_lr={old_lr:.6g}, new_lr={new_lr:.6g}, "
                    f"best_test_{self.monitor_metric_label}={self.best_score:.5f})"
                )
                self._lr_stale_rounds = 0
                self._stale_rounds = 0
            return False

        # Either dynamic lr disabled, or already at min_learning_rate.
        self._stale_rounds += 1
        if self._stale_rounds * self.eval_every >= self.early_stopping_rounds:
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
    diagnostic outputs shared across both stages. When configured,
    stage 1 and stage 2 both defer early stopping until the dynamic
    learning-rate schedule has reached ``min_learning_rate``.

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
    n_threads = max(1, min(32, os.cpu_count() or 1))
    n_estimators = int(hp.get("n_estimators", 200))
    n_estimators_decorr = int(hp.get("n_estimators_decorr", 1000))
    early_stopping_rounds = int(hp.get("early_stopping_rounds", 10))
    if early_stopping_rounds <= 0:
        raise ValueError(
            f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}"
        )
    learning_rate = float(hp.get("learning_rate", 0.1))
    learning_rate_decorr = float(hp.get("learning_rate_decorr", 0.01))
    lr_reduce_patience = int(hp.get("lr_reduce_patience", 0))
    min_learning_rate = float(hp.get("min_learning_rate", 0.0))
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
        subsample=hp.get("subsample", 1.0),
        colsample_bytree=hp.get("colsample_bytree", 1.0),
        nthread=n_threads,
        seed=RANDOM_STATE,
        disable_default_eval_metric=1,
        tree_method="hist",
    )
    for optional_param in ("colsample_bylevel", "colsample_bynode", "max_bin"):
        if optional_param in hp:
            base_params[optional_param] = hp[optional_param]

    # Build decor state on both splits when decor is enabled; stage-1 recorder
    # uses an explicit "none" decor_state so it contributes zero loss/grad/hess.
    if use_decor:
        train_decor_state = _prepare_decor_state_for_labels(
            _build_decor_state(Z_train, DECOR_LOSS_MODE), y_train, w_train, NUM_CLASSES
        )
        test_decor_state = _prepare_decor_state_for_labels(
            _build_decor_state(Z_test, DECOR_LOSS_MODE), y_test, w_test, NUM_CLASSES
        )
    else:
        train_decor_state = {"mode": "none"}
        test_decor_state = {"mode": "none"}
    del Z_train, Z_test
    gc.collect()

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
            eval_every=METRIC_EVAL_EVERY,
        )
        monitor = _DetailedLossMonitor(
            recorder,
            reg_lambda=stage1_params["reg_lambda"],
            reg_alpha=stage1_params["reg_alpha"],
            gamma=stage1_params["gamma"],
            learning_rate=stage1_params["eta"],
            early_stopping_rounds=early_stopping_rounds,
            stage_label="stage1",
            eval_every=METRIC_EVAL_EVERY,
            compact_log=True,
            monitor_metric_key="classification",
            monitor_metric_label="classification_loss",
            lr_reduce_patience=lr_reduce_patience,
            min_learning_rate=min_learning_rate,
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
    stage1_reg_loss = _loss_value_at(
        stage1_history, "train", "regularization", stage1_rounds - 1
    )

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
    reg_loss_ref = float(stage1_reg_loss)
    decor_loss_ref_raw, _, _ = _decorrelation_loss_components(
        stage1_logits, y_train, w_train, train_decor_state, NUM_CLASSES,
        decor_scale=1.0, need_derivatives=False,
    )
    numerator = max(float(cls_loss_ref) + float(reg_loss_ref), _EPS)
    if not np.isfinite(decor_loss_ref_raw) or decor_loss_ref_raw <= _EPS:
        raise RuntimeError(
            f"Stage-1 raw decorrelation loss is non-positive ({decor_loss_ref_raw:.6g}); "
            "cannot calibrate decorrelation scale"
        )
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
            eval_every=METRIC_EVAL_EVERY,
        )
        monitor = _DetailedLossMonitor(
            recorder,
            reg_lambda=stage2_params["reg_lambda"],
            reg_alpha=stage2_params["reg_alpha"],
            gamma=stage2_params["gamma"],
            learning_rate=stage2_params["eta"],
            early_stopping_rounds=early_stopping_rounds,
            stage_label="stage2",
            eval_every=METRIC_EVAL_EVERY,
            compact_log=False,
            initial_reg=reg_loss_ref,
            tree_offset=stage1_rounds,
            monitor_metric_key="total",
            monitor_metric_label="total_loss",
            lr_reduce_patience=lr_reduce_patience,
            min_learning_rate=min_learning_rate,
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


def _nn_config_for_tree(tree_name):
    hp = dict(cfg.get(f"{tree_name}_nn", {}))
    if not hp:
        raise RuntimeError(
            f"model_type='nn' requires a '{tree_name}_nn' configuration block"
        )
    return hp


def _torch_state_copy(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _torch_predict_logits_numpy(torch, model, device, X_np, batch_size):
    model.eval()
    out = np.empty((X_np.shape[0], NUM_CLASSES), dtype=np.float32)
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            end = min(start + batch_size, X_np.shape[0])
            xb = torch.as_tensor(
                X_np[start:end],
                dtype=torch.float32,
                device=device,
            )
            out[start:end] = model(xb).detach().cpu().numpy()
    return out


def _nn_build_decor_edges(torch, Z_train, device):
    edges = []
    for j in range(Z_train.shape[1]):
        z = np.asarray(Z_train[:, j], dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            edges.append(None)
            continue
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            edges.append(None)
            continue
        edge_np = np.linspace(z_min, z_max, max(2, DECOR_N_BINS) + 1, dtype=np.float32)
        edges.append(torch.as_tensor(edge_np, dtype=torch.float32, device=device))
    return edges


def _torch_smooth_decor_loss(torch, logits, labels, weights, Z, decor_edges,
                             score_thresholds, score_tau):
    if Z is None or Z.shape[1] == 0 or not decor_edges:
        return logits.new_tensor(0.0)

    total_loss = logits.new_tensor(0.0)
    eps = logits.new_tensor(_EPS)
    score_tau_t = logits.new_tensor(max(float(score_tau), _EPS))

    for cls_idx in range(NUM_CLASSES):
        cls_mask = labels == int(cls_idx)
        if int(cls_mask.sum().item()) < 3:
            continue
        score = logits[cls_mask, cls_idx]
        cls_weights = weights[cls_mask]
        total_w = torch.sum(cls_weights)
        if not bool((total_w > eps).item()):
            continue

        sig = torch.sigmoid(
            (score[:, None] - score_thresholds[None, :]) / score_tau_t
        )
        global_eff = torch.sum(cls_weights[:, None] * sig, dim=0) / torch.clamp(total_w, min=eps)

        for feat_idx, edges in enumerate(decor_edges):
            if edges is None:
                continue
            z = Z[cls_mask, feat_idx]
            if z.numel() < 3:
                continue
            width = torch.clamp(edges[1] - edges[0], min=eps)
            tau_z = torch.clamp(width * float(DECOR_BIN_TAU_SCALE), min=eps)
            left = torch.sigmoid((z[:, None] - edges[:-1][None, :]) / tau_z)
            right = torch.sigmoid((edges[1:][None, :] - z[:, None]) / tau_z)
            memberships = left * right
            row_sum = torch.sum(memberships, dim=1, keepdim=True)
            memberships = memberships / torch.clamp(row_sum, min=eps)

            weighted_memberships = cls_weights[:, None] * memberships
            bin_totals = torch.sum(weighted_memberships, dim=0)
            valid_bins = bin_totals > eps
            if not bool(torch.any(valid_bins).item()):
                continue
            weighted_memberships = weighted_memberships[:, valid_bins]
            bin_totals = bin_totals[valid_bins]
            local_eff = weighted_memberships.transpose(0, 1).matmul(sig) / torch.clamp(
                bin_totals[:, None], min=eps
            )
            rho = bin_totals / torch.clamp(total_w, min=eps)
            delta = local_eff - global_eff[None, :]
            total_loss = total_loss + torch.sum(rho[:, None] * delta * delta) / float(sig.shape[1])

    return total_loss


def _save_nn_checkpoint(torch, path, model, feature_names, hp, input_dim):
    checkpoint = {
        "model_type": "nn",
        "num_classes": int(NUM_CLASSES),
        "input_dim": int(input_dim),
        "feature_names": list(feature_names),
        "hidden_layers": [int(v) for v in hp.get("hidden_layers", [])],
        "activation": str(hp.get("activation", "silu")),
        "dropout": float(hp.get("dropout", 0.0)),
        "batch_norm": bool(hp.get("batch_norm", False)),
        "state_dict": _torch_state_copy(model),
        "training_config": dict(hp),
        "class_names": list(CLASS_NAMES),
    }
    torch.save(checkpoint, path)
    log_message(f"Wrote model file: {path}")
    return checkpoint


def train_nn_model(X_train_all, y_train, w_train, X_test_all, y_test, w_test,
                   model_name, tree_name, decorrelate_feature_names=None):
    """Two-stage PyTorch MLP training with one NN objective-loss definition.

    Stage 1 uses weighted CE. Stage 2 uses weighted CE plus scaled smooth-CvM
    decorrelation. AdamW weight decay keeps the native decoupled optimizer
    semantics and is not added to the objective loss.
    """
    torch, nn, F = import_torch()

    hp = _nn_config_for_tree(tree_name)
    requested_decor = bool(decorrelate_feature_names) and DECOR_LAMBDA > 0.0
    if requested_decor and DECOR_LOSS_MODE != "smooth_cvm":
        raise RuntimeError(
            "model_type='nn' currently supports decor_loss_mode='smooth_cvm' only"
        )

    torch.manual_seed(int(RANDOM_STATE))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(RANDOM_STATE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_all = X_train_all if isinstance(X_train_all, pd.DataFrame) else pd.DataFrame(X_train_all)
    X_test_all = X_test_all if isinstance(X_test_all, pd.DataFrame) else pd.DataFrame(X_test_all)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    w_train = np.asarray(w_train, dtype=float)
    w_test = np.asarray(w_test, dtype=float)

    decor_idx = _resolve_decor_indices(X_train_all, decorrelate_feature_names)
    all_idx = np.arange(len(X_train_all.columns))
    if decor_idx:
        keep_idx = np.setdiff1d(all_idx, decor_idx)
        if keep_idx.size == 0:
            raise ValueError("Decorrelation columns cover all features; nothing left to train on.")
        X_train = X_train_all.iloc[:, keep_idx]
        X_test = X_test_all.iloc[:, keep_idx]
        Z_train = X_train_all.iloc[:, decor_idx].to_numpy(dtype=np.float32, copy=True)
        Z_test = X_test_all.iloc[:, decor_idx].to_numpy(dtype=np.float32, copy=True)
    else:
        X_train = X_train_all
        X_test = X_test_all
        Z_train = np.zeros((len(X_train_all), 0), dtype=np.float32)
        Z_test = np.zeros((len(X_test_all), 0), dtype=np.float32)

    feature_names = list(X_train.columns)
    X_train_np = X_train.to_numpy(dtype=np.float32, copy=True)
    X_test_np = X_test.to_numpy(dtype=np.float32, copy=True)

    log_message(
        f"Training arrays: X_train={X_train_np.shape}, Z_train={Z_train.shape}, "
        f"decor_mode={DECOR_LOSS_MODE}"
    )

    hidden_layers = [int(v) for v in hp.get("hidden_layers", [256, 128, 64])]
    batch_size = max(1, int(hp.get("batch_size", 8192)))
    epochs = max(1, int(hp.get("epochs", 100)))
    epochs_decorr = max(1, int(hp.get("epochs_decorr", 50)))
    learning_rate = float(hp.get("learning_rate", 1e-3))
    learning_rate_decorr = float(hp.get("learning_rate_decorr", learning_rate * 0.5))
    min_learning_rate = float(hp.get("min_learning_rate", 0.0))
    lr_reduce_patience = int(hp.get("lr_reduce_patience", 0))
    early_stopping_rounds = int(hp.get("early_stopping_rounds", 10))
    weight_decay = float(hp.get("weight_decay", 0.0))
    grad_clip_norm = float(hp.get("grad_clip_norm", 0.0))
    batch_norm = bool(hp.get("batch_norm", False))
    if early_stopping_rounds <= 0:
        raise ValueError(
            f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}"
        )

    n_train = X_train_np.shape[0]
    X_train_t = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Z_train_t = torch.as_tensor(Z_train, dtype=torch.float32, device=device)

    use_decor = Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0
    if use_decor:
        decor_edges = _nn_build_decor_edges(torch, Z_train, device)
        prob_grid = np.linspace(0.02, 0.98, max(3, DECOR_N_THRESHOLDS))
        score_thresholds = torch.as_tensor(
            np.log(prob_grid / (1.0 - prob_grid)).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
    else:
        decor_edges = []
        score_thresholds = torch.zeros(1, dtype=torch.float32, device=device)

    model = build_torch_mlp(
        nn,
        input_dim=X_train_np.shape[1],
        hidden_layers=hidden_layers,
        output_dim=NUM_CLASSES,
        activation=hp.get("activation", "silu"),
        dropout=float(hp.get("dropout", 0.0)),
        batch_norm=batch_norm,
    ).to(device)

    log_message(
        f"Thread mode: PyTorch, device = {device}, batch_size = {batch_size}, "
        f"hidden_layers = {hidden_layers}"
    )
    log_message(
        "NN objective diagnostics: backward() uses class-balanced mini-batches "
        "with sampling-corrected event weights; train_objective_loss and "
        "test_objective_loss are epoch-end eval-mode full-split batch objectives "
        "using the same criterion formula. AdamW weight_decay remains a "
        "decoupled optimizer update, not a logged loss term."
    )

    splits = (X_train_all, X_test_all, y_train, y_test, w_train, w_test)

    if batch_size < NUM_CLASSES:
        raise ValueError(
            f"NN batch_size={batch_size} is smaller than NUM_CLASSES={NUM_CLASSES}; "
            "class-balanced batching needs at least one event per class."
        )
    class_indices_train = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        idx = np.flatnonzero(y_train == cls_idx).astype(np.int64, copy=False)
        if idx.size == 0:
            raise RuntimeError(
                f"Cannot build NN class-balanced batches: no training events for class '{cls_name}'"
            )
        class_indices_train.append(idx)
    class_batch_counts = np.full(NUM_CLASSES, batch_size // NUM_CLASSES, dtype=np.int64)
    remainder = int(batch_size - int(np.sum(class_batch_counts)))
    if remainder > 0:
        class_batch_counts[:remainder] += 1
    steps_per_epoch = max(1, int(np.ceil(float(n_train) / float(batch_size))))
    class_counts_train = np.asarray([idx.size for idx in class_indices_train], dtype=float)
    class_sampling_factors = class_counts_train / class_batch_counts.astype(float)
    mean_sampling_factor = float(np.mean(class_sampling_factors))
    if not np.isfinite(mean_sampling_factor) or mean_sampling_factor <= 0.0:
        raise RuntimeError("Invalid NN class-balanced sampling correction factors")
    class_sampling_factors /= mean_sampling_factor
    w_train_update = w_train * class_sampling_factors[y_train]
    w_train_update_t = torch.as_tensor(w_train_update, dtype=torch.float32, device=device)
    log_message(
        "NN batch sampler: class-balanced, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"per_class_batch={class_batch_counts.tolist()}, "
        f"sampling_weight_factors={class_sampling_factors.tolist()}"
    )

    def _set_lr(optimizer, value):
        for group in optimizer.param_groups:
            group["lr"] = float(value)

    def _make_optimizer(lr):
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)

    def _torch_batch_loss(logits, labels, weights, Z_batch, lam, decor_scale):
        ce = F.cross_entropy(logits, labels, reduction="none")
        weight_sum = torch.clamp(torch.sum(weights), min=logits.new_tensor(_EPS))
        cls_loss = torch.sum(weights * ce) / weight_sum
        decor_loss = logits.new_tensor(0.0)
        if lam > 0.0 and use_decor:
            decor_raw = _torch_smooth_decor_loss(
                torch,
                logits,
                labels,
                weights,
                Z_batch,
                decor_edges,
                score_thresholds,
                DECOR_SCORE_TAU,
            )
            decor_loss = float(lam) * float(decor_scale) * decor_raw
        return cls_loss + decor_loss, cls_loss, decor_loss

    def _evaluate_torch_split(X_np, y_np, w_np, Z_np, lam, decor_scale):
        model.eval()
        epoch_sums = {
            "classification": 0.0,
            "decorrelation": 0.0,
            "regularization": 0.0,
            "total": 0.0,
        }
        n_steps_eval = 0
        with torch.no_grad():
            for start in range(0, X_np.shape[0], batch_size):
                end = min(start + batch_size, X_np.shape[0])
                if end <= start:
                    continue
                xb = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_np[start:end], dtype=torch.long, device=device)
                wb = torch.as_tensor(w_np[start:end], dtype=torch.float32, device=device)
                if use_decor:
                    zb = torch.as_tensor(Z_np[start:end], dtype=torch.float32, device=device)
                else:
                    zb = None
                logits = model(xb)
                loss, cls_loss, decor_loss = _torch_batch_loss(
                    logits, yb, wb, zb, lam, decor_scale
                )
                epoch_sums["classification"] += float(cls_loss.detach().cpu().item())
                epoch_sums["decorrelation"] += float(decor_loss.detach().cpu().item())
                epoch_sums["total"] += float(loss.detach().cpu().item())
                n_steps_eval += 1
        return _nn_loss_components_from_epoch_sums(epoch_sums, n_steps_eval)

    def _class_balanced_epoch_batches(rng):
        shuffled = [rng.permutation(idx) for idx in class_indices_train]
        positions = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _ in range(steps_per_epoch):
            parts = []
            for cls_idx, need_raw in enumerate(class_batch_counts):
                need = int(need_raw)
                take_parts = []
                remaining = need
                while remaining > 0:
                    idx = shuffled[cls_idx]
                    pos = int(positions[cls_idx])
                    available = int(idx.size - pos)
                    if available <= 0:
                        shuffled[cls_idx] = rng.permutation(class_indices_train[cls_idx])
                        positions[cls_idx] = 0
                        continue
                    take_now = min(remaining, available)
                    take_parts.append(idx[pos:pos + take_now])
                    positions[cls_idx] = pos + take_now
                    remaining -= take_now
                parts.append(np.concatenate(take_parts))
            batch = np.concatenate(parts).astype(np.int64, copy=False)
            rng.shuffle(batch)
            yield batch

    def _batch_mean_ce_from_logits(logits_np, y_np, w_np):
        cls_loss_sum = 0.0
        n_steps_eval = 0
        for start in range(0, len(y_np), batch_size):
            end = min(start + batch_size, len(y_np))
            if end <= start:
                continue
            _, _, _, batch_cls_sum = _multiclass_classification_terms(
                y_np[start:end],
                logits_np[start:end],
                w_np[start:end],
                NUM_CLASSES,
            )
            batch_weight_sum = float(np.sum(w_np[start:end]))
            if not np.isfinite(batch_weight_sum) or batch_weight_sum <= 0.0:
                continue
            cls_loss_sum += float(batch_cls_sum) / batch_weight_sum
            n_steps_eval += 1
        if n_steps_eval <= 0:
            return float("nan")
        return cls_loss_sum / float(n_steps_eval)

    def _batch_mean_raw_decor_from_logits(logits_np, y_np, w_np, Z_np):
        if not use_decor:
            return 0.0
        raw_sum = 0.0
        n_steps_eval = 0
        with torch.no_grad():
            for start in range(0, len(y_np), batch_size):
                end = min(start + batch_size, len(y_np))
                if end <= start:
                    continue
                logits_t = torch.as_tensor(
                    logits_np[start:end], dtype=torch.float32, device=device
                )
                y_t = torch.as_tensor(y_np[start:end], dtype=torch.long, device=device)
                w_t = torch.as_tensor(w_np[start:end], dtype=torch.float32, device=device)
                Z_t = torch.as_tensor(Z_np[start:end], dtype=torch.float32, device=device)
                decor_raw = _torch_smooth_decor_loss(
                    torch,
                    logits_t,
                    y_t,
                    w_t,
                    Z_t,
                    decor_edges,
                    score_thresholds,
                    DECOR_SCORE_TAU,
                )
                raw_sum += float(decor_raw.detach().cpu().item())
                n_steps_eval += 1
        if n_steps_eval <= 0:
            return float("nan")
        return raw_sum / float(n_steps_eval)

    def _format_nn_loss_line(epoch, stage_label, train_comp, test_comp, compact=False):
        head = f"[{stage_label}][{epoch}]"
        if compact:
            return (
                f"{head}"
                f"\ttrain_objective_loss:{train_comp['objective']:.5f}"
                f"\ttest_objective_loss:{test_comp['objective']:.5f}"
                f"\ttrain_weighted_ce_loss:{train_comp['classification']:.5f}"
                f"\ttest_weighted_ce_loss:{test_comp['classification']:.5f}"
            )
        return (
            f"{head}"
            f"\ttrain_objective_loss:{train_comp['objective']:.5f}"
            f"\ttest_objective_loss:{test_comp['objective']:.5f}"
            f"\ttrain_weighted_ce_loss:{train_comp['classification']:.5f}"
            f"\ttrain_decorrelation_loss:{train_comp['decorrelation']:.5f}"
            f"\ttest_weighted_ce_loss:{test_comp['classification']:.5f}"
            f"\ttest_decorrelation_loss:{test_comp['decorrelation']:.5f}"
        )

    def _append_history(history, train_comp, test_comp):
        for key in ("classification", "mlogloss", "decorrelation", "regularization", "objective", "total"):
            history["train"][key].append(float(train_comp[key]))
            history["test"][key].append(float(test_comp[key]))

    def _run_stage(stage_label, n_epochs, lr, lam, decor_scale,
                   monitor_metric_key, monitor_metric_label, compact_log):
        optimizer = _make_optimizer(lr)
        current_lr = float(lr)
        dynamic_lr = (
            lr_reduce_patience > 0
            and min_learning_rate > 0.0
            and current_lr > min_learning_rate + 1e-12
        )
        history = {
            tag: {
                "classification": [],
                "mlogloss": [],
                "decorrelation": [],
                "regularization": [],
                "objective": [],
                "total": [],
            }
            for tag in ("train", "test")
        }
        best_state = None
        best_iteration = None
        best_score = float("inf")
        stale_rounds = 0
        lr_stale_rounds = 0
        rng = np.random.default_rng(int(RANDOM_STATE))

        for epoch in range(int(n_epochs)):
            model.train()
            for idx_np in _class_balanced_epoch_batches(rng):
                if idx_np.size == 0:
                    continue
                if batch_norm and idx_np.size < 2:
                    continue
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                xb = X_train_t.index_select(0, idx)
                yb = y_train_t.index_select(0, idx)
                wb = w_train_update_t.index_select(0, idx)
                logits = model(xb)
                if use_decor:
                    zb = Z_train_t.index_select(0, idx)
                else:
                    zb = None
                loss, _, _ = _torch_batch_loss(
                    logits, yb, wb, zb, lam, decor_scale
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            train_comp = _evaluate_torch_split(X_train_np, y_train, w_train, Z_train, lam, decor_scale)
            test_comp = _evaluate_torch_split(X_test_np, y_test, w_test, Z_test, lam, decor_scale)
            _append_history(history, train_comp, test_comp)
            log_message(_format_nn_loss_line(
                epoch, stage_label, train_comp, test_comp, compact=compact_log
            ))

            current_score = _loss_value_at(history, "test", monitor_metric_key, epoch)
            improved = np.isfinite(current_score) and (
                best_iteration is None or current_score < best_score - 1e-12
            )
            if improved:
                best_iteration = int(epoch)
                best_score = float(current_score)
                best_state = _torch_state_copy(model)
                stale_rounds = 0
                lr_stale_rounds = 0
                continue

            if dynamic_lr and current_lr > min_learning_rate + 1e-12:
                lr_stale_rounds += 1
                if lr_stale_rounds >= lr_reduce_patience:
                    old_lr = current_lr
                    current_lr = max(old_lr * 0.5, min_learning_rate)
                    _set_lr(optimizer, current_lr)
                    log_message(
                        f"Info: lr reduced ({stage_label}) at epoch {epoch} "
                        f"(old_lr={old_lr:.6g}, new_lr={current_lr:.6g}, "
                        f"best_test_{monitor_metric_label}={best_score:.5f})"
                    )
                    lr_stale_rounds = 0
                    stale_rounds = 0
                continue

            stale_rounds += 1
            if stale_rounds >= early_stopping_rounds:
                log_message(
                    f"Info: early stopping ({stage_label}) on {monitor_metric_label} "
                    f"(best_iteration={best_iteration}, "
                    f"best_test_{monitor_metric_label}={best_score:.5f})"
                )
                break

        if best_state is None:
            best_state = _torch_state_copy(model)
            best_iteration = len(history["test"][monitor_metric_key]) - 1
        model.load_state_dict(best_state)
        kept = int(best_iteration) + 1
        return _trim_loss_history(history, kept), kept

    base_path = model_name[:-3] if model_name.endswith(".pt") else model_name
    if base_path.endswith(".json"):
        base_path = base_path[:-5]

    # ---------- Stage 1: classification-only ----------
    log_message(
        f"Starting stage 1 (PyTorch MLP cls-only, epochs={epochs}, lr={learning_rate})"
    )
    stage1_history, stage1_rounds = _run_stage(
        "stage1",
        epochs,
        learning_rate,
        0.0,
        1.0,
        "objective",
        "objective_loss",
        True,
    )

    stage1_save_path = f"{base_path}_stage1.pt"
    stage1_checkpoint = _save_nn_checkpoint(
        torch,
        stage1_save_path,
        model,
        feature_names,
        hp,
        X_train_np.shape[1],
    )
    stage1_model_eval = build_torch_mlp(
        nn,
        input_dim=X_train_np.shape[1],
        hidden_layers=hidden_layers,
        output_dim=NUM_CLASSES,
        activation=hp.get("activation", "silu"),
        dropout=float(hp.get("dropout", 0.0)),
        batch_norm=batch_norm,
    ).to(device)
    stage1_model_eval.load_state_dict(stage1_checkpoint["state_dict"])
    stage1_model_eval.eval()
    stage1_handle = TorchModelHandle(
        stage1_model_eval, device, feature_names, NUM_CLASSES, stage1_checkpoint
    )

    if not use_decor:
        main_save_path = f"{base_path}.pt"
        main_checkpoint = _save_nn_checkpoint(
            torch,
            main_save_path,
            model,
            feature_names,
            hp,
            X_train_np.shape[1],
        )
        stage1_handle.checkpoint = main_checkpoint
        return stage1_handle, None, splits, stage1_history, stage1_rounds

    # ---------- Stage 2: classification + smooth-CvM decorrelation ----------
    stage1_logits = _torch_predict_logits_numpy(torch, model, device, X_train_np, batch_size)
    cls_loss_ref = _batch_mean_ce_from_logits(stage1_logits, y_train, w_train)
    decor_loss_ref_raw = _batch_mean_raw_decor_from_logits(stage1_logits, y_train, w_train, Z_train)
    if not np.isfinite(decor_loss_ref_raw) or decor_loss_ref_raw <= _EPS:
        raise RuntimeError(
            f"Stage-1 raw decorrelation loss is non-positive ({decor_loss_ref_raw:.6g}); "
            "cannot calibrate decorrelation scale"
        )
    decor_scale = max(float(cls_loss_ref), _EPS) / float(decor_loss_ref_raw)
    log_message(
        f"Stage-1 end: best_iter={stage1_rounds - 1}, cls_loss={cls_loss_ref:.6g}, "
        f"reg_loss=0, raw_decor_loss={decor_loss_ref_raw:.6g}"
    )
    log_message(f"Decorrelation scale: mode={DECOR_LOSS_MODE}, fixed_scale={decor_scale:.6g}")

    log_message(
        f"Starting stage 2 (PyTorch MLP cls+decor, epochs_decorr={epochs_decorr}, "
        f"lr={learning_rate_decorr})"
    )
    stage2_history, stage2_rounds = _run_stage(
        "stage2",
        epochs_decorr,
        learning_rate_decorr,
        DECOR_LAMBDA,
        decor_scale,
        "objective",
        "objective_loss",
        False,
    )

    combined_history = {
        "train": {
            k: list(stage1_history["train"].get(k, [])) + list(stage2_history["train"].get(k, []))
            for k in ("classification", "mlogloss", "decorrelation", "regularization", "objective", "total")
        },
        "test": {
            k: list(stage1_history["test"].get(k, [])) + list(stage2_history["test"].get(k, []))
            for k in ("classification", "mlogloss", "decorrelation", "regularization", "objective", "total")
        },
    }

    main_save_path = f"{base_path}.pt"
    main_checkpoint = _save_nn_checkpoint(
        torch,
        main_save_path,
        model,
        feature_names,
        hp,
        X_train_np.shape[1],
    )
    model.eval()
    stage2_handle = TorchModelHandle(model, device, feature_names, NUM_CLASSES, main_checkpoint)
    return stage1_handle, stage2_handle, splits, combined_history, stage1_rounds


# -------------------- Plotting --------------------
def _booster_from_model(model):
    if isinstance(model, TorchModelHandle):
        return None
    return model.get_booster() if hasattr(model, "get_booster") else model


def _predict_margins(model, Xlike):
    return _shared_predict_model_logits(model, Xlike, NUM_CLASSES)


def _predict_proba(model, Xlike):
    return _shared_predict_model_proba(model, Xlike, NUM_CLASSES)


def plot_results(stage1_model, stage2_model, splits, tree_name, output_root,
                 loss_history, stage_boundary, decorrelate_feature_names=None,
                 decor_plot_X_test=None, decor_efficiencies=None,
                 eval_splits=None):
    """ROC curves, feature importance, score distributions, loss curves, and decorrelation checks.

    Model-dependent plots (ROC, importance, score distributions, decor_corr,
    and decor diagnostic multipage PDFs) are saved twice with ``_cls`` /
    ``_decorr`` suffixes (for the stage-1 baseline and stage-2 final model
    respectively). ``feature_corr.pdf`` and the shared BDT
    ``loss_mlogloss.pdf`` / ``loss_classification.pdf`` / ``loss_total.pdf``
    or NN ``loss_weighted_ce.pdf`` / ``loss_objective.pdf`` plus
    ``loss_decorrelation.pdf`` files are saved once. ``stage_boundary`` is the
    number of stage-1 iterations kept; it is drawn as a dotted vertical line
    on the loss curves. ``splits`` are the training-objective splits; when
    ``eval_splits`` is provided, ordinary ROC, score, importance, and feature
    correlation plots use those full-threshold evaluation splits while
    decorrelation diagnostics keep using ``splits``.
    """
    (
        X_train_decor_full,
        X_test_decor_full,
        y_train_decor,
        y_test_decor,
        w_train_decor,
        w_test_decor,
    ) = splits
    if eval_splits is None:
        eval_splits = splits
    (
        X_train_eval_full,
        X_test_eval_full,
        y_train_eval,
        y_test_eval,
        w_train_eval,
        w_test_eval,
    ) = eval_splits

    full_feature_names = list(X_train_decor_full.columns) if hasattr(X_train_decor_full, "columns") \
        else [f"f{i}" for i in range(X_train_decor_full.shape[1])]

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

    X_train_decor_used = _slice(X_train_decor_full, keep_idx)
    X_test_decor_used = _slice(X_test_decor_full, keep_idx)
    X_train_eval_used = _slice(X_train_eval_full, keep_idx)
    X_test_eval_used = _slice(X_test_eval_full, keep_idx)
    feat_names_used = [full_feature_names[i] for i in keep_idx]

    booster_ref = _booster_from_model(stage1_model if stage1_model is not None else stage2_model)
    booster_features = (booster_ref.feature_names or []) if booster_ref is not None else []
    if booster_features and len(booster_features) == len(full_feature_names):
        X_train_decor_used, X_test_decor_used = X_train_decor_full, X_test_decor_full
        X_train_eval_used, X_test_eval_used = X_train_eval_full, X_test_eval_full
        feat_names_used = full_feature_names
        decor_idx_full = []

    n_classes = NUM_CLASSES
    class_names = CLASS_NAMES
    is_nn_model = isinstance(stage1_model, TorchModelHandle) or isinstance(stage2_model, TorchModelHandle)
    palette = plt.colormaps["tab10"].resampled(max(n_classes, 2))(np.arange(max(n_classes, 2)))

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
    if decor_efficiencies is None:
        decor_efficiencies = [1.0, 0.5, 0.1, 0.01]
    decor_efficiencies = [float(eff) for eff in decor_efficiencies if 0.0 < float(eff) <= 1.0]

    decor_plot_names = []
    decor_plot_df = None
    if decor_plot_X_test is not None and decorrelate_feature_names:
        if isinstance(decor_plot_X_test, pd.DataFrame):
            decor_plot_names = [
                name for name in decorrelate_feature_names
                if not isinstance(name, int) and name in decor_plot_X_test.columns
            ]
            if decor_plot_names:
                decor_plot_df = decor_plot_X_test[decor_plot_names].reset_index(drop=True)
        else:
            arr = np.asarray(decor_plot_X_test)
            names = [name for name in decorrelate_feature_names if not isinstance(name, int)]
            if arr.ndim == 2 and arr.shape[1] == len(names):
                decor_plot_names = list(names)
                decor_plot_df = pd.DataFrame(arr, columns=decor_plot_names)
    if decorrelate_feature_names and (decor_plot_df is None or not decor_plot_names):
        raise RuntimeError("No valid decorrelation plot data available for requested decorrelate branches.")

    def _weighted_bin_average(x, y_arr, wv, min_bins=200, max_bins=600):
        x = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y_arr, dtype=float).ravel()
        wv = np.asarray(wv, dtype=float).ravel()
        m = np.isfinite(x) & np.isfinite(y_arr) & np.isfinite(wv) & (wv > 0.0)
        x, y_arr, wv = x[m], y_arr[m], wv[m]
        if x.size < 2:
            return np.array([]), np.array([])
        lo, hi = float(np.min(x)), float(np.max(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.array([]), np.array([])
        n_bins = min(max_bins, max(min_bins, int(np.sqrt(float(x.size)))))
        n_bins = max(2, min(n_bins, max(2, x.size)))
        edges = np.linspace(lo, hi, n_bins + 1)
        sum_w, _ = np.histogram(x, bins=edges, weights=wv)
        sum_wy, _ = np.histogram(x, bins=edges, weights=wv * y_arr)
        valid = sum_w > 0.0
        if not np.any(valid):
            return np.array([]), np.array([])
        centers = 0.5 * (edges[:-1] + edges[1:])
        avg = np.full_like(centers, np.nan, dtype=float)
        avg[valid] = sum_wy[valid] / sum_w[valid]
        return centers[valid], avg[valid]

    def _hist_edges(x, n_bins=60):
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size < 2:
            return None
        lo, hi = float(np.min(x)), float(np.max(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return np.linspace(lo, hi, int(n_bins) + 1)

    def _weighted_score_threshold(score, wv, efficiency):
        score = np.asarray(score, dtype=float).ravel()
        wv = np.asarray(wv, dtype=float).ravel()
        m = np.isfinite(score) & np.isfinite(wv) & (wv > 0.0)
        score, wv = score[m], wv[m]
        if score.size == 0:
            return float("nan")
        if efficiency >= 1.0:
            return -float("inf")
        order = np.argsort(score)
        score_s = score[order]
        w_s = wv[order]
        total_w = float(np.sum(w_s))
        if total_w <= 0.0:
            return float("nan")
        target_below = max(0.0, min(1.0, 1.0 - float(efficiency))) * total_w
        idx = int(np.searchsorted(np.cumsum(w_s), target_below, side="left"))
        idx = max(0, min(idx, score_s.size - 1))
        return float(score_s[idx])

    def _format_eff(eff):
        if eff >= 0.999999:
            return "100%"
        if eff >= 0.01:
            return f"{100.0 * eff:.0f}%"
        return f"{100.0 * eff:.2g}%"

    def _save_pdf_page(pdf_state, path, fig):
        if pdf_state["pdf"] is None:
            pdf_state["pdf"] = PdfPages(path)
        pdf_state["pdf"].savefig(fig, bbox_inches="tight")
        pdf_state["pages"] += 1
        plt.close(fig)

    def _close_pdf(pdf_state, path):
        if pdf_state["pdf"] is None:
            return
        pdf_state["pdf"].close()
        log_message(f"Wrote plot file: {path}")

    def _plot_score_vs_decor_pdf(scores, suffix, stage_tag):
        if not decorrelate_feature_names:
            return
        if decor_plot_df is None or not decor_plot_names:
            raise RuntimeError(
                f"{tree_name} [{stage_tag}] no valid decorrelation plot data for score-vs-branch PDF"
            )
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_test_decor, dtype=int)
        wv_all = np.asarray(w_test_decor, dtype=float)
        path = _figure_path(output_root, f"decor_score_vs_branch{suffix}")
        pdf_state = {"pdf": None, "pages": 0}
        for cls_idx, cls_name in enumerate(class_names):
            class_mask = y_true == cls_idx
            if not np.any(class_mask):
                continue
            for branch_name in decor_plot_names:
                x = decor_plot_df[branch_name].to_numpy(dtype=float, copy=False)
                s = scores[:, cls_idx]
                mask = (
                    class_mask
                    & np.isfinite(x)
                    & (x > -990.0)
                    & np.isfinite(s)
                    & np.isfinite(wv_all)
                    & (wv_all > 0.0)
                )
                if not np.any(mask):
                    raise RuntimeError(
                        f"{tree_name} [{stage_tag}] no valid test events for "
                        f"class '{cls_name}' and decorrelate branch '{branch_name}'"
                    )
                fig, ax = plt.subplots(figsize=(8.5, 6.2))
                ax.scatter(
                    x[mask],
                    s[mask],
                    s=2.0,
                    alpha=0.12,
                    color=palette[cls_idx],
                    edgecolors="none",
                    rasterized=True,
                    label=f"Test {cls_name}",
                )
                centers, avg = _weighted_bin_average(x[mask], s[mask], wv_all[mask])
                if centers.size > 0:
                    ax.plot(
                        centers,
                        avg,
                        color="black",
                        linewidth=1.8,
                        label="weighted average",
                    )
                ax.set_title(f"{tree_name} {stage_tag}: {cls_name}", fontsize=15)
                ax.set_xlabel(branch_name)
                ax.set_ylabel(f"p({cls_name})")
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(loc="best", fontsize=10)
                _save_pdf_page(pdf_state, path, fig)
        if pdf_state["pages"] == 0:
            raise RuntimeError(f"{tree_name} [{stage_tag}] no pages written for score-vs-branch PDF")
        _close_pdf(pdf_state, path)

    def _plot_decor_shape_by_score_pdf(scores, suffix, stage_tag):
        if not decorrelate_feature_names:
            return
        if decor_plot_df is None or not decor_plot_names:
            raise RuntimeError(
                f"{tree_name} [{stage_tag}] no valid decorrelation plot data for signal-score-shape PDF"
            )
        if not SIGNAL_CLASS_INDICES:
            raise RuntimeError(
                f"{tree_name} [{stage_tag}] cannot build signal-score-shape PDF without signal classes"
            )
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_test_decor, dtype=int)
        wv_all = np.asarray(w_test_decor, dtype=float)
        path = _figure_path(output_root, f"decor_branch_shapes_by_signal_score{suffix}")
        pdf_state = {"pdf": None, "pages": 0}
        colors = plt.colormaps["viridis"].resampled(max(len(decor_efficiencies), 2))(
            np.arange(max(len(decor_efficiencies), 2))
        )

        for cls_idx, cls_name in enumerate(class_names):
            class_mask = y_true == cls_idx
            if not np.any(class_mask):
                continue
            for branch_name in decor_plot_names:
                x = decor_plot_df[branch_name].to_numpy(dtype=float, copy=False)
                base_x_mask = class_mask & np.isfinite(x) & (x > -990.0) & np.isfinite(wv_all) & (wv_all > 0.0)
                edges = _hist_edges(x[base_x_mask])
                if edges is None:
                    raise RuntimeError(
                        f"{tree_name} [{stage_tag}] no valid histogram range for "
                        f"class '{cls_name}' and decorrelate branch '{branch_name}'"
                    )
                for sig_idx in SIGNAL_CLASS_INDICES:
                    sig_name = class_names[sig_idx]
                    sig_score = scores[:, sig_idx]
                    base = base_x_mask & np.isfinite(sig_score)
                    if not np.any(base):
                        raise RuntimeError(
                            f"{tree_name} [{stage_tag}] no valid events for class '{cls_name}', "
                            f"decorrelate branch '{branch_name}', signal score '{sig_name}'"
                        )

                    fig, ax = plt.subplots(figsize=(8.5, 6.2))
                    plotted_any = False
                    for eff_i, eff in enumerate(decor_efficiencies):
                        if eff >= 0.999999:
                            cut = base
                            label = f"eff={_format_eff(eff)} (no cut)"
                        else:
                            thr = _weighted_score_threshold(sig_score[base], wv_all[base], eff)
                            if not np.isfinite(thr):
                                plt.close(fig)
                                raise RuntimeError(
                                    f"{tree_name} [{stage_tag}] invalid weighted threshold for "
                                    f"class '{cls_name}', signal score '{sig_name}', efficiency={eff}"
                                )
                            cut = base & (sig_score > thr)
                            label = f"eff={_format_eff(eff)}, p({sig_name})>{thr:.3f}"
                        if not np.any(cut) or float(np.sum(wv_all[cut])) <= 0.0:
                            plt.close(fig)
                            raise RuntimeError(
                                f"{tree_name} [{stage_tag}] no positive-weight events after "
                                f"efficiency cut {eff} for class '{cls_name}', "
                                f"decorrelate branch '{branch_name}', signal score '{sig_name}'"
                            )
                        ax.hist(
                            x[cut],
                            bins=edges,
                            weights=wv_all[cut],
                            density=True,
                            histtype="step",
                            linewidth=2.0,
                            color=colors[eff_i],
                            label=label,
                        )
                        plotted_any = True
                    if not plotted_any:
                        plt.close(fig)
                        raise RuntimeError(
                            f"{tree_name} [{stage_tag}] no page data for class '{cls_name}', "
                            f"decorrelate branch '{branch_name}', signal score '{sig_name}'"
                        )
                    ax.set_title(
                        f"{tree_name} {stage_tag}: {cls_name}, cut on p({sig_name})",
                        fontsize=15,
                    )
                    ax.set_xlabel(branch_name)
                    ax.set_ylabel("A.U.")
                    ax.grid(True, linestyle="--", alpha=0.35)
                    ax.legend(loc="best", fontsize=9)
                    _save_pdf_page(pdf_state, path, fig)
        if pdf_state["pages"] == 0:
            raise RuntimeError(f"{tree_name} [{stage_tag}] no pages written for signal-score-shape PDF")
        _close_pdf(pdf_state, path)

    def _classification_loss_from_probs(probs, labels, weights):
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)
        weights = np.asarray(weights, dtype=float)
        return float(np.sum(weights * (-np.log(probs[np.arange(labels.size), labels] + _EPS))))

    def _subset_for_permutation_importance(Xlike, labels, weights):
        max_events = int(cfg.get(f"{tree_name}_nn", {}).get("permutation_importance_events", 50000))
        n_events = len(labels)
        if max_events > 0 and n_events > max_events:
            rng = np.random.default_rng(int(RANDOM_STATE))
            idx = np.sort(rng.choice(n_events, size=max_events, replace=False))
            if hasattr(Xlike, "iloc"):
                return Xlike.iloc[idx].reset_index(drop=True), labels[idx], weights[idx]
            return np.asarray(Xlike)[idx], labels[idx], weights[idx]
        if hasattr(Xlike, "reset_index"):
            return Xlike.reset_index(drop=True), labels, weights
        return Xlike, labels, weights

    def _permutation_importance(model, Xlike, labels, weights):
        X_ref, y_ref, w_ref = _subset_for_permutation_importance(Xlike, labels, weights)
        if len(y_ref) == 0:
            return np.zeros(Xlike.shape[1], dtype=float)
        if isinstance(model, TorchModelHandle):
            X_perm = (
                X_ref.to_numpy(dtype=np.float32, copy=True)
                if hasattr(X_ref, "to_numpy")
                else np.asarray(X_ref, dtype=np.float32).copy()
            )
            baseline = _classification_loss_from_probs(_predict_proba(model, X_perm), y_ref, w_ref)
            importances = []
            rng = np.random.default_rng(int(RANDOM_STATE) + 17)
            for col_idx in range(X_perm.shape[1]):
                values = X_perm[:, col_idx].copy()
                shuffled = values.copy()
                rng.shuffle(shuffled)
                X_perm[:, col_idx] = shuffled
                loss = _classification_loss_from_probs(_predict_proba(model, X_perm), y_ref, w_ref)
                importances.append(max(0.0, float(loss - baseline)))
                X_perm[:, col_idx] = values
            return np.asarray(importances, dtype=float)

        baseline = _classification_loss_from_probs(_predict_proba(model, X_ref), y_ref, w_ref)
        importances = []
        rng = np.random.default_rng(int(RANDOM_STATE) + 17)
        for col_idx in range(X_ref.shape[1]):
            if hasattr(X_ref, "iloc"):
                X_perm = X_ref.copy()
                values = X_perm.iloc[:, col_idx].to_numpy(copy=True)
                rng.shuffle(values)
                X_perm.iloc[:, col_idx] = values
            else:
                X_perm = np.asarray(X_ref).copy()
                values = X_perm[:, col_idx].copy()
                rng.shuffle(values)
                X_perm[:, col_idx] = values
            loss = _classification_loss_from_probs(_predict_proba(model, X_perm), y_ref, w_ref)
            importances.append(max(0.0, float(loss - baseline)))
        return np.asarray(importances, dtype=float)

    def _plot_for_model(model, suffix, stage_tag):
        if model is None:
            return
        probs_train = _predict_proba(model, X_train_eval_used)
        probs_test = _predict_proba(model, X_test_eval_used)
        probs_test_decor = (
            _predict_proba(model, X_test_decor_used)
            if decorrelate_feature_names else None
        )
        margins_train_decor = margins_test_decor = None
        if decor_idx_full:
            margins_train_decor = _predict_margins(model, X_train_decor_used)
            margins_test_decor = _predict_margins(model, X_test_decor_used)
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
                mask_train = (y_train_eval == sig_idx) | (y_train_eval == bkg_idx)
                mask_test = (y_test_eval == sig_idx) | (y_test_eval == bkg_idx)
                color = palette[bkg_idx]
                r_tst = _roc_binary(mask_test, score_test, y_test_eval, w_test_eval, sig_idx)
                r_trn = _roc_binary(mask_train, score_train, y_train_eval, w_train_eval, sig_idx)
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
        if isinstance(model, TorchModelHandle):
            importances = _permutation_importance(model, X_test_eval_used, y_test_eval, w_test_eval)
            importance_label = "Permutation loss increase"
        else:
            score_map = booster.get_score(importance_type="gain")
            importances = []
            for i, name in enumerate(feat_names_used):
                importances.append(float(score_map.get(name, score_map.get(f"f{i}", 0.0))))
            importances = np.asarray(importances, dtype=float)
            importance_label = "Gain"
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
        ax.set_xlabel(importance_label, fontsize=12)
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
            mask_train = (y_train_eval == sig_idx) | (y_train_eval == bkg_idx)
            mask_test = (y_test_eval == sig_idx) | (y_test_eval == bkg_idx)
            bins = np.linspace(0, 1, 31)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 1)
            ax.hist(
                score_train[mask_train & (y_train_eval == bkg_idx)],
                bins=bins,
                weights=w_train_eval[mask_train & (y_train_eval == bkg_idx)],
                density=True,
                histtype="bar",
                alpha=0.5,
                label=f"Train {bkg_name}",
            )
            ax.hist(
                score_train[mask_train & (y_train_eval == sig_idx)],
                bins=bins,
                weights=w_train_eval[mask_train & (y_train_eval == sig_idx)],
                density=True,
                histtype="bar",
                alpha=0.5,
                label=f"Train {sig_name}",
            )
            ax.hist(
                score_test[mask_test & (y_test_eval == bkg_idx)],
                bins=bins,
                weights=w_test_eval[mask_test & (y_test_eval == bkg_idx)],
                density=True,
                histtype="step",
                linewidth=2,
                color="lime",
                label=f"Test {bkg_name}",
            )
            ax.hist(
                score_test[mask_test & (y_test_eval == sig_idx)],
                bins=bins,
                weights=w_test_eval[mask_test & (y_test_eval == sig_idx)],
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
                ("train", margins_train_decor, X_train_decor_full, y_train_decor, w_train_decor),
                ("test", margins_test_decor, X_test_decor_full, y_test_decor, w_test_decor),
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

        _plot_score_vs_decor_pdf(probs_test_decor, suffix, stage_tag)
        _plot_decor_shape_by_score_pdf(probs_test_decor, suffix, stage_tag)

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
        ax.set_xlabel("Epoch" if is_nn_model else "Boosting Round")
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

    if is_nn_model:
        _plot_loss_metric("classification", "weighted_ce_loss", "loss_weighted_ce")
        _plot_loss_metric("objective", "objective_loss", "loss_objective")
    else:
        _plot_loss_metric("classification", "classification_loss", "loss_classification")
        _plot_loss_metric("mlogloss", "mlogloss", "loss_mlogloss")
        _plot_loss_metric("total", "total_loss", "loss_total")
    _plot_loss_metric("decorrelation", "decorrelation_loss", "loss_decorrelation")

    X_tr_df = (
        X_train_eval_used
        if isinstance(X_train_eval_used, pd.DataFrame)
        else pd.DataFrame(_as_array(X_train_eval_used), columns=feat_names_used)
    )
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
        if _is_msoftdrop_branch(name):
            mass_thresholds[name] = cond
        else:
            other_thresholds[name] = cond
    return mass_thresholds, other_thresholds


def _is_msoftdrop_branch(name):
    return str(name).startswith("ScoutingFatPFJetRecluster_msoftdrop_")


def _split_training_thresholds(thresholds, decorrelate_feature_names):
    decor_threshold_names = {
        str(name) for name in decorrelate_feature_names
        if not isinstance(name, (int, np.integer)) and str(name) in thresholds
    }
    training_thresholds = {}
    decor_thresholds = {}
    decor_training_overrides = {}
    for name, cond in thresholds.items():
        if name not in decor_threshold_names:
            training_thresholds[name] = cond
            continue
        decor_thresholds[name] = cond
        if _is_msoftdrop_branch(name):
            decor_training_overrides[name] = (0.0, DECOR_MSOFTDROP_TRAINING_MAX)
            training_thresholds[name] = decor_training_overrides[name]
    return training_thresholds, decor_thresholds, decor_training_overrides


def _drop_decorrelated_features(X, decorrelate_feature_names):
    if not decorrelate_feature_names:
        return X
    drop_cols = [name for name in decorrelate_feature_names if name in X.columns]
    if drop_cols:
        return X.drop(columns=drop_cols)
    return X


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
        training_thresholds, decor_thresholds, decor_training_overrides = _split_training_thresholds(
            thresholds, decorrelate
        )
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
            f"Running train.py for tree = {tree_name}, output = {output_root}, "
            f"classes = {NUM_CLASSES}, model_type = {MODEL_TYPE}"
        )
        training_weight_scales = _qcd_ht_training_weight_scales(tree_name)
        if training_weight_scales:
            formatted_scales = ", ".join(
                f"{sample_name}={scale:.6g}"
                for sample_name, scale in sorted(
                    training_weight_scales.items(),
                    key=lambda item: _qcd_ht_sort_key(item[0]),
                )
            )
            log_message(
                f"QCD HT training weight scales for tree = {tree_name}: {formatted_scales}"
            )
        omitted_decor_thresholds = [
            name for name in decor_thresholds.keys()
            if name not in decor_training_overrides
        ]
        if omitted_decor_thresholds:
            log_message(
                f"Training threshold override for tree = {tree_name}: excluding decorrelate "
                f"threshold(s) from training/eval loss only: {', '.join(omitted_decor_thresholds)}"
            )
        if decor_training_overrides:
            formatted_overrides = [
                f"{name}=(0, {upper:g})"
                for name, (_lower, upper) in decor_training_overrides.items()
            ]
            log_message(
                f"Training threshold override for tree = {tree_name}: using loose "
                f"decorrelate msoftdrop threshold(s) in training/eval loss only: "
                f"{', '.join(formatted_overrides)}"
            )
        split_plans = build_split_plans(tree_name)
        write_config_copy(output_root)
        write_branch_copy(output_root)
        write_selection_copy(output_root)
        write_split_metadata(output_root, tree_name, split_plans)

        log_message(f"Loading training split for tree = {tree_name}")
        X_train, y_train, w_train, sample_labels_train, _ = prepare_split_data(
            tree_name, load_cols, "train", split_plans, shuffle=True,
            training_weight_scales=training_weight_scales,
        )
        check_weights(w_train, f"{tree_name}_train_weight_before_filter")

        log_message(f"Loading test split for tree = {tree_name}")
        X_test, y_test, w_test, sample_labels_test, w_test_physics = prepare_split_data(
            tree_name, load_cols, "test", split_plans, shuffle=False,
            training_weight_scales=training_weight_scales,
        )
        check_weights(w_test, f"{tree_name}_test_weight_before_filter")
        check_weights(w_test_physics, f"{tree_name}_test_physics_weight_before_filter")
        X_test_unfiltered = X_test
        y_test_unfiltered = y_test
        w_test_physics_unfiltered = w_test_physics
        sample_labels_test_unfiltered = np.asarray(sample_labels_test)
        del w_test_physics

        log_message(f"Applying training thresholds for training split of tree = {tree_name}")
        X_train, y_train, w_train, sample_labels_train = filter_X(
            X_train, y_train, w_train, load_cols, training_thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_train
        )
        _validate_filtered_split(
            tree_name,
            "train (training thresholds)",
            y_train,
            w_train,
            sample_labels_train,
        )

        X_train_eval, y_train_eval, w_train_eval, sample_labels_train_eval = filter_X(
            X_train, y_train, w_train, load_cols, decor_thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_train
        )
        _validate_filtered_split(
            tree_name,
            "train (full evaluation thresholds)",
            y_train_eval,
            w_train_eval,
            sample_labels_train_eval,
        )
        w_train = _rebalance_filtered_weights("train", y_train, w_train, sample_labels_train)
        w_train_eval = _rebalance_filtered_weights(
            "train (full evaluation thresholds)",
            y_train_eval,
            w_train_eval,
            sample_labels_train_eval,
        )
        check_weights(w_train, f"{tree_name}_train_weight_after_filter")
        check_weights(w_train_eval, f"{tree_name}_train_eval_weight_after_filter")

        log_message(f"Applying training thresholds for test split of tree = {tree_name}")
        X_test, y_test, w_test, sample_labels_test = filter_X(
            X_test, y_test, w_test, load_cols, training_thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_test
        )
        X_test_eval, y_test_eval, w_test_eval, sample_labels_test_eval, test_eval_index = filter_X(
            X_test, y_test, w_test, load_cols, decor_thresholds, apply_to_sentinel=True,
            sample_labels=sample_labels_test,
            return_index=True,
        )
        test_eval_pos = np.asarray(test_eval_index, dtype=np.int64)
        y_test_ref = y_test_eval.copy()
        w_test_ref = w_test_physics_unfiltered[test_eval_pos].copy()
        sample_labels_test_ref = np.asarray(sample_labels_test_eval).copy()
        _validate_filtered_split(
            tree_name,
            "test (training thresholds)",
            y_test,
            w_test,
            sample_labels_test,
        )
        _validate_filtered_split(
            tree_name,
            "test (full evaluation thresholds)",
            y_test_eval,
            w_test_eval,
            sample_labels_test_eval,
        )
        w_test = _rebalance_filtered_weights("test", y_test, w_test, sample_labels_test)
        w_test_eval = _rebalance_filtered_weights(
            "test (full evaluation thresholds)",
            y_test_eval,
            w_test_eval,
            sample_labels_test_eval,
        )
        check_weights(w_test, f"{tree_name}_test_weight_after_filter")
        check_weights(w_test_eval, f"{tree_name}_test_eval_weight_after_filter")
        check_weights(w_test_ref, f"{tree_name}_test_physics_weight_after_filter")

        decor_plot_cols = [c for c in decorrelate if c in X_test.columns]
        X_test_decor_plot = (
            _clip_only_X(X_test[decor_plot_cols], clip_ranges)
            if decor_plot_cols else None
        )

        if drop_after_filter:
            X_train = X_train.drop(columns=drop_after_filter, errors="ignore")
            X_test = X_test.drop(columns=drop_after_filter, errors="ignore")
            X_train_eval = X_train_eval.drop(columns=drop_after_filter, errors="ignore")
            X_test_eval = X_test_eval.drop(columns=drop_after_filter, errors="ignore")

        log_message(f"Plotting input branch distributions for tree = {tree_name}")
        plot_branch_distributions(
            output_root, branches, clip_ranges,
            X_train_eval, y_train_eval, w_train_eval, sample_labels_train_eval,
            X_test_eval, y_test_eval, w_test_eval, sample_labels_test_eval,
        )

        log_message(f"Standardising training split for tree = {tree_name}")
        X_train_std = standardize_X(X_train.copy(), clip_ranges, log_tf)
        log_message(f"Standardising test split for tree = {tree_name}")
        X_test_std = standardize_X(X_test.copy(), clip_ranges, log_tf)
        log_message(f"Standardising full-threshold evaluation splits for tree = {tree_name}")
        X_train_eval_std = standardize_X(X_train_eval.copy(), clip_ranges, log_tf)
        X_test_eval_std = standardize_X(X_test_eval.copy(), clip_ranges, log_tf)
        del X_train, X_test, X_train_eval, X_test_eval
        del sample_labels_train, sample_labels_test, sample_labels_train_eval, sample_labels_test_eval
        gc.collect()

        log_message(f"Training model for tree = {tree_name}")
        if MODEL_TYPE == "nn":
            stage1_model, stage2_model, splits, loss_history, stage_boundary = train_nn_model(
                X_train_std, y_train, w_train,
                X_test_std, y_test, w_test,
                model_path, tree_name,
                decorrelate_feature_names=decorrelate
            )
        else:
            stage1_model, stage2_model, splits, loss_history, stage_boundary = train_multi_model(
                X_train_std, y_train, w_train,
                X_test_std, y_test, w_test,
                model_path, tree_name,
                decorrelate_feature_names=decorrelate
            )
        final_model = stage2_model if stage2_model is not None else stage1_model

        X_test_qcd_full_model = standardize_X(X_test_unfiltered[branches].copy(), clip_ranges, log_tf)
        X_test_qcd_full_model = _drop_decorrelated_features(X_test_qcd_full_model, decorrelate)
        proba_qcd_full_test = _predict_proba(final_model, X_test_qcd_full_model)
        _write_prediction_reference(
            output_root,
            "test_reference_qcd_est_full",
            tree_name,
            "qcd_est_full",
            X_test_qcd_full_model.columns,
            sample_labels_test_unfiltered,
            y_test_unfiltered,
            w_test_physics_unfiltered,
            proba_qcd_full_test,
        )
        del X_test_qcd_full_model, proba_qcd_full_test
        gc.collect()

        X_test_signal_model = _drop_decorrelated_features(X_test_eval_std, decorrelate)
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
        del X_test_signal_model, proba_signal_test

        mass_thresholds, bdt_thresholds = _split_mass_thresholds(thresholds)
        log_message(
            f"Preparing qcd_est reference for tree = {tree_name}: "
            f"non_mass_thresholds={len(bdt_thresholds)}, mass_thresholds={len(mass_thresholds)}"
        )
        X_test_qcd_raw, y_test_qcd_ref, w_test_qcd_ref, sample_labels_test_qcd_ref = filter_X(
            X_test_unfiltered,
            y_test_unfiltered,
            w_test_physics_unfiltered,
            load_cols,
            bdt_thresholds,
            apply_to_sentinel=True,
            sample_labels=sample_labels_test_unfiltered,
        )
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
        del X_test_qcd_raw, X_test_qcd_model, proba_qcd_test
        del y_test_ref, w_test_ref, sample_labels_test_ref
        del y_test_qcd_ref, w_test_qcd_ref, sample_labels_test_qcd_ref
        del X_test_unfiltered, y_test_unfiltered, w_test_physics_unfiltered, sample_labels_test_unfiltered
        gc.collect()

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
            decor_plot_X_test=X_test_decor_plot,
            decor_efficiencies=_decor_efficiencies_for_tree(tree_name),
            eval_splits=(
                X_train_eval_std,
                X_test_eval_std,
                y_train_eval,
                y_test_eval,
                w_train_eval,
                w_test_eval,
            ),
        )
        log_message(f"Finished train.py for tree = {tree_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
