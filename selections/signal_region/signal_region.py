import os
import json
import gc
import pickle
import time
import ctypes
import subprocess
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import xgboost as xgb
from itertools import product as iproduct
from matplotlib.lines import Line2D

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)

# _SCRIPT_DIR points to selections/signal_region/.
# The copied BDT configs still store paths relative to selections/BDT/, where train.py runs.
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_SELECTIONS_DIR = os.path.dirname(_SCRIPT_DIR)
_BDT_DIR        = os.path.join(_SELECTIONS_DIR, "BDT")


# -------------------- Logging --------------------
def log_message(message):
    print(message, flush=True)

def log_warning(message):
    log_message(f"Warning: {message}")

def log_info(message):
    log_message(f"Info: {message}")

def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# -------------------- Config loading --------------------
_scan_cfg_path = os.environ.get("SCAN_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
if not os.path.isabs(_scan_cfg_path):
    _scan_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _scan_cfg_path))

scan_cfg = _load_json(_scan_cfg_path)

LUMI             = float(scan_cfg["lumi"])
N_SIGNAL_REGIONS = int(scan_cfg.get("n_signal_regions", scan_cfg.get("N", 4)))
BDT_ROOT         = scan_cfg.get("bdt_root", scan_cfg.get("output_root"))
OUTPUT_DIR       = scan_cfg.get("output_dir", BDT_ROOT)
MIN_BKG_WEIGHT   = float(scan_cfg.get("min_bkg_weight", 5.0))
MIN_SIGNAL_WEIGHT = float(scan_cfg.get("min_signal_weight", 0.0))
MIN_SIGNAL_ENTRIES = max(0, int(scan_cfg.get("min_signal_entries", 0)))
MIN_BKG_ENTRIES = max(0, int(scan_cfg.get("min_bkg_entries", 0)))
MAX_EDGE_CANDIDATES_PER_AXIS = max(8, int(scan_cfg.get("max_edge_candidates_per_axis", 120)))
BEAM_WIDTH = max(1, int(scan_cfg.get("beam_width", 48)))
TOP_INTERVALS_PER_AXIS = max(1, int(scan_cfg.get("top_intervals_per_axis", 8)))
COORDINATE_ROUNDS = max(1, int(scan_cfg.get("coordinate_rounds", 8)))
LOCAL_REFINE_ROUNDS = max(0, int(scan_cfg.get("local_refine_rounds", 3)))
LOCAL_REFINE_NEIGHBOR_EDGES = max(1, int(scan_cfg.get("local_refine_neighbor_edges", 48)))
LOCAL_REFINE_TOP_CANDIDATES = max(0, int(scan_cfg.get("local_refine_top_candidates", 512)))
CANDIDATE_POOL_LIMIT = max(N_SIGNAL_REGIONS, int(scan_cfg.get("candidate_pool_limit", 20000)))
PROGRESS_EVERY_SECONDS = float(scan_cfg.get("progress_every_seconds", 30.0))
GLOBAL_SELECTION_CANDIDATES = max(
    N_SIGNAL_REGIONS,
    int(scan_cfg.get(
        "global_selection_candidates",
        scan_cfg.get("final_selection_candidates", 5000),
    )),
)
GLOBAL_BEAM_WIDTH = max(
    1,
    int(scan_cfg.get("global_beam_width", scan_cfg.get("selection_beam_width", 512))),
)
BRANCH_BOUND_MAX_NODES = max(0, int(scan_cfg.get("branch_bound_max_nodes", 0)))
BRANCH_BOUND_TIME_LIMIT_SECONDS = max(
    0.0, float(scan_cfg.get("branch_bound_time_limit_seconds", 0.0))
)
DEDUPLICATE_EVENT_MASKS = bool(scan_cfg.get("deduplicate_event_masks", True))
REQUIRE_EXACT_N_REGIONS = bool(scan_cfg.get("require_exact_n_regions", True))
MAX_THREADS = max(1, int(scan_cfg.get("max_threads", 8)))
SEED_QUANTILES = [
    float(q) for q in scan_cfg.get(
        "seed_quantiles", [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
    )
]

if BDT_ROOT is None:
    raise KeyError("signal_region config requires 'bdt_root'")

if not os.path.isabs(BDT_ROOT):
    BDT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, BDT_ROOT))
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, OUTPUT_DIR))


# -------------------- BDT config copies --------------------
cfg       = _load_json(os.path.join(BDT_ROOT, "config.json"))
br_cfg    = _load_json(os.path.join(BDT_ROOT, "branch.json"))
sel_cfg   = _load_json(os.path.join(BDT_ROOT, "selection.json"))
test_meta = _load_json(os.path.join(BDT_ROOT, "test_ranges.json"))

# sample_config paths in config.json are relative to _BDT_DIR, where train.py runs.
_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_BDT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

TREE_NAME     = test_meta["tree_name"]
MODEL_PATTERN = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
TEST_REFERENCE_SIGNAL_REGION = os.path.join(BDT_ROOT, "test_reference_signal_region.npz")


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

CLASS_GROUPS = cfg["class_groups"]
CLASS_NAMES  = list(CLASS_GROUPS.keys())
NUM_CLASSES  = len(CLASS_NAMES)

SIGNAL_CLASS_INDICES     = []
BACKGROUND_CLASS_INDICES = []
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    _flags = [SAMPLE_INFO[_s]["is_signal"] for _s in _members]
    if _flags and all(_flags):
        SIGNAL_CLASS_INDICES.append(_idx)
    else:
        BACKGROUND_CLASS_INDICES.append(_idx)

SAMPLE_TO_CLASS = {}
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    for _s in _members:
        SAMPLE_TO_CLASS[_s] = _idx


# -------------------- Test data loading --------------------
def load_test_data(branches):
    """Load test events from test_ranges.json with physics-normalised weights.

    For each sample:
      raw_w            = product of event_reweight_branches (per event)
      total_weight     = lumi * xsec * total_tree_entries / raw_entries
      per_event_weight = raw_w * total_weight / sum(raw_w_loaded)

    The reweight branches come from the BDT config copy in ``bdt_root`` and are
    read on raw values (before any clip/log/threshold). Weights are fixed here;
    threshold filtering later does NOT rescale them.
    """
    log_message(f"Loading test data from: {os.path.join(BDT_ROOT, 'test_ranges.json')}")
    dfs = []

    reweight_branches = list(cfg.get(TREE_NAME, {}).get("event_reweight_branches", []))
    load_branches = list(branches)
    for rb in reweight_branches:
        if rb not in load_branches:
            load_branches.append(rb)

    for sample_name, sample_meta in test_meta["samples"].items():
        info = SAMPLE_INFO.get(sample_name)
        if info is None:
            raise RuntimeError(f"Sample '{sample_name}' not in sample config")
        if sample_name not in SAMPLE_TO_CLASS:
            raise RuntimeError(f"Sample '{sample_name}' not in any class group")

        xsec          = float(info["xsection"])
        raw_entries   = int(info["raw_entries"])
        total_entries = int(sample_meta["total_entries"])
        if raw_entries <= 0:
            raise RuntimeError(
                f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json"
            )

        parts = []
        for seg in sample_meta["test_segments"]:
            fpath = seg["file"]
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Test split file not found: {fpath}")
            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        raise KeyError(f"Tree '{TREE_NAME}' not in {fpath}")
                    tree = uf[TREE_NAME]
                    available = set(tree.keys())
                    missing = [b for b in load_branches if b not in available]
                    if missing:
                        raise KeyError(
                            f"Missing branches in {fpath}:{TREE_NAME}: "
                            f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                        )
                    df_part = tree.arrays(
                        load_branches,
                        library="pd",
                        entry_start=int(seg["entry_start"]),
                        entry_stop=int(seg["entry_stop"]),
                    )
                    parts.append(df_part)
            except Exception as exc:
                raise RuntimeError(f"Failed to read test split file {fpath}: {exc}") from exc

        if not parts:
            raise RuntimeError(f"No data loaded for sample '{sample_name}'")

        df      = pd.concat(parts, ignore_index=True)
        n_loaded = len(df)

        if reweight_branches:
            raw_w = np.ones(n_loaded, dtype=float)
            for rb in reweight_branches:
                raw_w *= df[rb].to_numpy(dtype=float, copy=False)
            df = df.drop(columns=reweight_branches)
        else:
            raw_w = np.ones(n_loaded, dtype=float)

        if xsec <= 0.0:
            target_total = 0.0
            df["weight"] = 0.0
            log_warning(
                f"  {sample_name}: non-positive xsec={xsec}, zero weight"
            )
        else:
            # Normalize the sample total weight to lumi * xsec * total_tree_entries / raw_entries,
            # then shape per-event weights by raw_w so sum(weight) stays at target_total.
            target_total  = LUMI * xsec * total_entries / raw_entries
            raw_w_sum = float(raw_w.sum())
            if raw_w_sum <= 0.0:
                raise RuntimeError(
                    f"Sample '{sample_name}' has non-positive raw weight sum {raw_w_sum:.6g}"
                )
            df["weight"] = raw_w * (target_total / raw_w_sum)

        df["class_idx"]   = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        dfs.append(df)

        log_message(
            f"  {sample_name}: n_loaded={n_loaded}, total_entries={total_entries}, "
            f"raw_entries={raw_entries}, xsec={xsec:.6g}, target_total={target_total:.6g}, "
            f"class={CLASS_NAMES[SAMPLE_TO_CLASS[sample_name]]}"
        )

    if not dfs:
        raise RuntimeError("No test data loaded")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df_all


# -------------------- Feature standardization --------------------
def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    log_set = set(log_transform)
    for col in X.columns:
        arr   = X[col].values.copy()
        mask  = arr < -990
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


def _reshape_multiclass_margin(predt, num_class):
    predt = np.asarray(predt, dtype=float)
    if predt.ndim == 2:
        if predt.shape[1] == num_class:
            return predt
        if predt.shape[0] == num_class:
            return predt.T
    rows = predt.size // num_class
    return predt.reshape(rows, num_class)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / (np.sum(exp_v, axis=1, keepdims=True) + 1e-12)


def _predict_model_proba(model, X):
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X, feature_names=list(X.columns) if hasattr(X, "columns") else None)
        margins = model.predict(dmat, output_margin=True)
        return _softmax_rows(_reshape_multiclass_margin(margins, NUM_CLASSES))
    return model.predict_proba(X)


def _compare_prediction_reference(path, feature_names, sample_labels, class_idx, weights, proba):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction reference not found: {path}. Re-run train.py before signal_region.py."
        )

    ref = np.load(path, allow_pickle=False)
    ref_features = ref["feature_names"].astype(str).tolist()
    cur_features = list(feature_names)
    if cur_features != ref_features:
        raise RuntimeError(
            "Prediction reference mismatch for signal_region model features: "
            f"current={cur_features}, reference={ref_features}"
        )

    ref_samples = ref["sample_name"].astype(str)
    cur_samples = np.asarray(sample_labels, dtype=str)
    if not np.array_equal(cur_samples, ref_samples):
        raise RuntimeError("Prediction reference mismatch for signal_region sample order/content")

    ref_class_idx = ref["class_idx"].astype(int)
    cur_class_idx = np.asarray(class_idx, dtype=int)
    if not np.array_equal(cur_class_idx, ref_class_idx):
        raise RuntimeError("Prediction reference mismatch for signal_region class labels")

    ref_weights = ref["weight"].astype(float) * LUMI
    cur_weights = np.asarray(weights, dtype=float)
    weight_rtol = float(ref["weight_rtol"])
    weight_atol = float(ref["weight_atol"])
    if not np.allclose(cur_weights, ref_weights, rtol=weight_rtol, atol=weight_atol):
        diff = float(np.max(np.abs(cur_weights - ref_weights)))
        raise RuntimeError(
            "Prediction reference mismatch for signal_region weights: "
            f"max_abs_diff={diff:.6g}, rtol={weight_rtol}, atol={weight_atol}"
        )

    ref_proba = ref["proba"].astype(float)
    cur_proba = np.asarray(proba, dtype=float)
    proba_rtol = float(ref["proba_rtol"])
    proba_atol = float(ref["proba_atol"])
    if cur_proba.shape != ref_proba.shape:
        raise RuntimeError(
            "Prediction reference mismatch for signal_region probabilities shape: "
            f"current={cur_proba.shape}, reference={ref_proba.shape}"
        )
    if not np.allclose(cur_proba, ref_proba, rtol=proba_rtol, atol=proba_atol):
        diff = float(np.max(np.abs(cur_proba - ref_proba)))
        raise RuntimeError(
            "Prediction reference mismatch for signal_region probabilities: "
            f"max_abs_diff={diff:.6g}, rtol={proba_rtol}, atol={proba_atol}"
        )

    log_message(f"Validated prediction reference: {path}")


# -------------------- Threshold filtering --------------------
def filter_X(X: pd.DataFrame, y, w, branch: list,
             thresholds: dict = None, apply_to_sentinel: bool = True,
             sample_labels=None):
    """Apply per-branch threshold cuts.

    Only branches that appear as keys in ``thresholds`` are inspected: for each
    such branch, events with sentinel values (< -990) are dropped (when
    ``apply_to_sentinel`` is True) and the threshold condition is enforced.
    Branches not listed in ``thresholds`` are left untouched, so an event with
    a sentinel value in some other branch is still kept. The ``branch``
    argument is retained for backward compatibility and is not used to drive
    filtering.
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
        if isinstance(cond, (list, tuple)) and len(cond) == 2 and \
                not isinstance(cond[0], (list, dict, tuple)):
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


# -------------------- Plotting helpers --------------------
def _slugify(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")

def _savefig(stem):
    path = os.path.join(OUTPUT_DIR, f"{stem}.pdf")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log_message(f"Wrote plot file: {path}")


def write_signal_region_csv(result):
    axis_names = [CLASS_NAMES[d] for d in range(max(1, NUM_CLASSES - 1))]
    if result["top_bins"]:
        axis_names = list(result["top_bins"][0]["axis_names"])

    base_columns = [
        "bin_index",
        "significance",
        "significance_error",
        "S",
        "S_err",
        "S_entries",
        "B",
        "B_err",
        "B_entries",
    ]
    axis_columns = []
    for axis_name in axis_names:
        axis_columns.extend([f"{axis_name}_low", f"{axis_name}_high"])

    rows = []
    for b in result["top_bins"]:
        row = {
            "bin_index": b["bin_index"],
            "significance": b["significance"],
            "significance_error": b["significance_error"],
            "S": b["S"],
            "S_err": b["S_err"],
            "S_entries": b["S_entries"],
            "B": b["B"],
            "B_err": b["B_err"],
            "B_entries": b["B_entries"],
        }
        for dim, axis_name in enumerate(b["axis_names"]):
            row[f"{axis_name}_low"] = float(b["thr_low"][dim])
            row[f"{axis_name}_high"] = float(b["thr_high"][dim])
        rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "signal_region.csv")
    pd.DataFrame(rows, columns=base_columns + axis_columns).to_csv(csv_path, index=False)
    log_message(f"Wrote signal region file: {csv_path}")


def plot_score_distributions(proba, y, w):
    """Weighted BDT score distributions per scan axis (one plot per axis)."""
    D = max(1, proba.shape[1] - 1)
    palette = plt.cm.get_cmap("tab10", max(NUM_CLASSES, 3))(np.arange(max(NUM_CLASSES, 3)))
    bins = np.linspace(0.0, 1.0, 51)

    for d in range(D):
        axis_name = CLASS_NAMES[d]
        score = proba[:, d]
        plt.figure(figsize=(8, 6))
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            m = (y == cls_i)
            if not np.any(m):
                continue
            plt.hist(score[m], bins=bins, weights=w[m],
                     histtype="step", linewidth=2,
                     color=palette[cls_i], label=cls_name)
        plt.xlabel(f"p({axis_name})")
        plt.ylabel(f"Events / {LUMI:.0f} fb$^{{-1}}$")
        plt.yscale("log")
        plt.xlim(0, 1)
        plt.legend(fontsize=12)
        _savefig(f"sr_score_{_slugify(axis_name)}")


def plot_signal_regions_2d(result, proba, y, w):
    """Regular-polygon projection of multiclass scores, with optional SR outlines."""
    del w
    n_classes = int(proba.shape[1])
    if n_classes < 2:
        return

    def _simplex_vertices(n):
        angles = (np.pi / 2.0) + 2.0 * np.pi * np.arange(n, dtype=float) / float(n)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def _convex_hull(points):
        pts = sorted(set((float(x), float(y)) for x, y in np.asarray(points, dtype=float)))
        if len(pts) <= 2:
            return np.asarray(pts, dtype=float)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)
        return np.asarray(lower[:-1] + upper[:-1], dtype=float)

    vertices = _simplex_vertices(n_classes)
    coords = np.asarray(proba, dtype=float) @ vertices

    def _project_region(bin_info):
        D = len(bin_info["thr_low"])
        if D != n_classes - 1:
            return None
        lo = np.clip(np.asarray(bin_info["thr_low"], dtype=float), 0.0, 1.0)
        hi = np.clip(np.asarray(bin_info["thr_high"], dtype=float), 0.0, 1.0)
        eps = 1e-12
        candidates = []

        def add_candidate(p_first):
            p_first = np.asarray(p_first, dtype=float)
            p_last = 1.0 - float(np.sum(p_first))
            if p_last < -eps:
                return
            full = np.r_[p_first, max(0.0, p_last)]
            if np.all(full >= -eps) and np.all(full <= 1.0 + eps):
                candidates.append(np.clip(full, 0.0, 1.0))

        for fixed in iproduct([0, 1], repeat=D):
            p = np.array([hi[d] if fixed[d] else lo[d] for d in range(D)], dtype=float)
            if float(np.sum(p)) <= 1.0 + eps:
                add_candidate(p)

        for free_dim in range(D):
            fixed_dims = [d for d in range(D) if d != free_dim]
            for fixed in iproduct([0, 1], repeat=max(0, D - 1)):
                p = np.zeros(D, dtype=float)
                for bit, dim in zip(fixed, fixed_dims):
                    p[dim] = hi[dim] if bit else lo[dim]
                p[free_dim] = 1.0 - float(np.sum(p[fixed_dims]))
                if lo[free_dim] - eps <= p[free_dim] <= hi[free_dim] + eps:
                    add_candidate(p)

        if not candidates:
            return None
        projected = np.asarray(candidates, dtype=float) @ vertices
        return _convex_hull(projected)

    def _draw(show_regions, stem):
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        class_cmap = "tab10" if NUM_CLASSES <= 10 else "tab20"
        class_palette = plt.cm.get_cmap(class_cmap, max(NUM_CLASSES, 3))(
            np.arange(max(NUM_CLASSES, 3))
        )

        for cls_i, cls_name in enumerate(CLASS_NAMES):
            mask = y == cls_i
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=0.8,
                alpha=0.07,
                color=class_palette[cls_i],
                edgecolors="none",
                rasterized=True,
            )

        closed_vertices = np.vstack([vertices, vertices[0]])
        ax.plot(closed_vertices[:, 0], closed_vertices[:, 1],
                color="0.45", linewidth=1.1, alpha=0.8)
        ax.scatter(vertices[:, 0], vertices[:, 1], s=18, color="0.25", zorder=5)
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            vx, vy = vertices[cls_i]
            ax.text(1.12 * vx, 1.12 * vy, cls_name, ha="center", va="center", fontsize=11)

        handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=class_palette[i],
                markeredgecolor="none",
                markersize=6,
                label=CLASS_NAMES[i],
            )
            for i in range(NUM_CLASSES)
        ]

        if show_regions and result["top_bins"]:
            sr_palette = plt.cm.get_cmap("Set1", max(len(result["top_bins"]), 3))
            for i, b in enumerate(result["top_bins"]):
                poly = _project_region(b)
                if poly is None or len(poly) < 2:
                    continue
                color = sr_palette(i)
                if len(poly) >= 3:
                    poly_draw = np.vstack([poly, poly[0]])
                    ax.plot(poly_draw[:, 0], poly_draw[:, 1], color=color, linewidth=2.0)
                else:
                    ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2.0)
                handles.append(
                    Line2D(
                        [0], [0],
                        color=color,
                        linewidth=2.0,
                        label=f"SR{b['bin_index']} (Z={b['significance']:.2f})",
                    )
                )

        ax.set_xlabel("simplex projection x")
        ax.set_ylabel("simplex projection y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.95)
        _savefig(stem)

    _draw(False, "scores_no_regions")
    _draw(True, "scores")


# -------------------- Signal-region scan --------------------
def find_signal_regions(proba, y, w, forbidden_regions=None, target_regions=None):
    """Find a global set of non-overlapping high-D score rectangles.

    The candidate generator uses the same score-rectangle cuts as before. The
    final selection is global: it maximizes the existing combined objective
    ``sum(Z_i^2)`` over mutually non-overlapping candidates, using branch and
    bound when the OpenMP helper is available.
    """
    forbidden_regions = forbidden_regions or []
    target_regions = max(1, int(target_regions or N_SIGNAL_REGIONS))
    n_events, n_cls = proba.shape
    D = max(1, n_cls - 1)

    score_axes = np.column_stack([proba[:, d] for d in range(D)])  # (N, D)
    axis_names = [CLASS_NAMES[d] for d in range(D)]

    is_sig = np.isin(y, SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, BACKGROUND_CLASS_INDICES)
    w_sig = np.where(is_sig, w, 0.0)
    w_bkg = np.where(is_bkg, w, 0.0)

    S_total = float(w_sig.sum())
    B_total = float(w_bkg.sum())

    log_message(f"  S_total={S_total:.4g}, B_total={B_total:.4g}")
    log_message(f"  Scan dimensions D={D}, axes={axis_names}")
    log_message(
        "  Optimizer: "
        f"max_edges={MAX_EDGE_CANDIDATES_PER_AXIS}, "
        f"beam_width={BEAM_WIDTH}, top_intervals={TOP_INTERVALS_PER_AXIS}, "
        f"rounds={COORDINATE_ROUNDS}, local_refine_rounds={LOCAL_REFINE_ROUNDS}, "
        f"global_candidates={GLOBAL_SELECTION_CANDIDATES}, "
        f"global_beam_width={GLOBAL_BEAM_WIDTH}"
    )

    scan_t0 = time.monotonic()
    last_progress = [scan_t0]

    def _elapsed():
        return time.monotonic() - scan_t0

    def _progress(message, force=False):
        now = time.monotonic()
        if force or PROGRESS_EVERY_SECONDS <= 0.0 or now - last_progress[0] >= PROGRESS_EVERY_SECONDS:
            log_message(f"  [{_elapsed():.1f}s] {message}")
            last_progress[0] = now

    def _calc_Z(S, B, sS, sB):
        if S <= 0.0 or B <= 0.0:
            return 0.0, 0.0
        f = (S + B) * np.log(1.0 + S / B) - S
        if f <= 0.0:
            return 0.0, 0.0
        Z = float(np.sqrt(2.0 * f))
        ln1sb = np.log(1.0 + S / B)
        dZ_dS = ln1sb / Z
        dZ_dB = (ln1sb - S / B) / Z
        sZ = float(np.sqrt((dZ_dS * sS) ** 2 + (dZ_dB * sB) ** 2))
        return Z, sZ

    def _calc_Z_val(S, B):
        if S <= 0.0 or B <= 0.0:
            return 0.0
        f = (S + B) * np.log(1.0 + S / B) - S
        return float(np.sqrt(2.0 * max(0.0, f)))

    EPS = 1e-12

    def _hi_to_open(h):
        return float(h) >= 1.0 - EPS

    def _rect_mask(lo, hi):
        m = np.ones(n_events, dtype=bool)
        for d in range(D):
            v = score_axes[:, d]
            if _hi_to_open(hi[d]):
                m &= v >= lo[d]
            else:
                m &= (v >= lo[d]) & (v < hi[d])
        return m

    def _rect_SB(lo, hi):
        m = _rect_mask(lo, hi)
        return float(w_sig[m].sum()), float(w_bkg[m].sum())

    def _rect_stats(lo, hi):
        m = _rect_mask(lo, hi)
        ms = m & is_sig
        mb = m & is_bkg
        return (
            float(w[ms].sum()),
            float(w[mb].sum()),
            int(ms.sum()),
            int(mb.sum()),
        )

    def _overlap(lo1, hi1, lo2, hi2):
        for d in range(D):
            if not (lo1[d] < hi2[d] - EPS and lo2[d] < hi1[d] - EPS):
                return False
        return True

    forbidden_boxes = []
    for region in forbidden_regions:
        if "lo" in region and "hi" in region:
            lo_prev = region["lo"]
            hi_prev = region["hi"]
        else:
            lo_prev = region["thr_low"]
            hi_prev = region["thr_high"]
        forbidden_boxes.append((
            [float(v) for v in lo_prev],
            [float(v) for v in hi_prev],
        ))
    if forbidden_boxes:
        log_message(f"  Excluding {len(forbidden_boxes)} previously selected signal regions")

    def _overlaps_forbidden(lo, hi):
        return any(_overlap(lo, hi, flo, fhi) for flo, fhi in forbidden_boxes)

    def _region_key(lo, hi):
        return (tuple(round(float(v), 10) for v in lo),
                tuple(round(float(v), 10) for v in hi))

    def _valid_region(lo, hi):
        return all(float(lo[d]) < float(hi[d]) - EPS for d in range(D))

    def _quantile_levels(n_levels):
        n_levels = max(4, int(n_levels))
        n_uniform = max(4, n_levels // 2)
        n_tail = max(2, n_levels // 4)
        tail = np.geomspace(1.0e-4, 0.25, n_tail)
        qs = np.unique(np.r_[0.0, 1.0, np.linspace(0.0, 1.0, n_uniform), tail, 1.0 - tail])
        if qs.size > n_levels:
            keep = np.unique(np.rint(np.linspace(0, qs.size - 1, n_levels)).astype(int))
            qs = qs[keep]
        return np.clip(qs, 0.0, 1.0)

    def _weighted_quantiles(values, weights, qs):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(mask):
            return np.array([], dtype=float)
        values = values[mask]
        weights = weights[mask]
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cw = np.cumsum(weights)
        if cw[-1] <= 0.0:
            return np.array([], dtype=float)
        return np.interp(np.clip(qs, 0.0, 1.0), cw / cw[-1], values)

    def _cap_edges(edge_values, max_count):
        edges = np.unique(np.clip(np.asarray(edge_values, dtype=float), 0.0, 1.0))
        edges = edges[np.isfinite(edges)]
        edges = np.unique(np.r_[0.0, edges, 1.0])
        if edges.size <= max_count:
            return edges.astype(float)
        keep = np.unique(np.rint(np.linspace(0, edges.size - 1, max_count)).astype(int))
        return edges[keep].astype(float)

    # One-dimensional tail scan for reference efficiencies only.
    T_REF = 200
    p_exp = 0.005
    thr_1d = np.clip(np.linspace(0.0, 1.0, T_REF) ** p_exp, 0.0, 1.0)
    S_tail_by_dim = np.zeros((D, T_REF))
    B_tail_by_dim = np.zeros((D, T_REF))
    for d in range(D):
        s = score_axes[:, d]
        order = np.argsort(s)
        s_sorted = s[order]
        cw_sig = np.cumsum(w_sig[order])
        cw_bkg = np.cumsum(w_bkg[order])
        idx = np.searchsorted(s_sorted, thr_1d, side="left")
        S_tail_by_dim[d] = (cw_sig[-1] if cw_sig.size else 0.0) - np.where(
            idx > 0, cw_sig[np.clip(idx - 1, 0, cw_sig.size - 1)], 0.0
        )
        B_tail_by_dim[d] = (cw_bkg[-1] if cw_bkg.size else 0.0) - np.where(
            idx > 0, cw_bkg[np.clip(idx - 1, 0, cw_bkg.size - 1)], 0.0
        )

    # Dense but bounded per-axis edge candidates.
    max_edges = max(8, MAX_EDGE_CANDIDATES_PER_AXIS)
    total_q = _quantile_levels(max(8, max_edges // 2))
    sig_q = _quantile_levels(max(8, max_edges // 3))
    bkg_q = _quantile_levels(max(8, max_edges // 3))
    all_q = _quantile_levels(max(6, max_edges // 6))
    edges_per_axis = []
    for d in range(D):
        v = score_axes[:, d]
        edge_values = [0.0, 1.0]
        edge_values.extend(_weighted_quantiles(v, w_sig + w_bkg, total_q))
        edge_values.extend(_weighted_quantiles(v, w_sig, sig_q))
        edge_values.extend(_weighted_quantiles(v, w_bkg, bkg_q))
        finite_v = v[np.isfinite(v)]
        if finite_v.size:
            edge_values.extend(np.quantile(finite_v, all_q))
        edges_per_axis.append(_cap_edges(edge_values, max_edges))
    axis_unique_values = [
        np.unique(score_axes[np.isfinite(score_axes[:, d]), d])
        for d in range(D)
    ]

    log_message(
        f"  Edges per axis: " +
        ", ".join(f"{axis_names[d]}={len(edges_per_axis[d])}" for d in range(D))
    )

    pool = {}  # key -> {"lo", "hi", "S", "B", "Z"}

    def _add_to_pool(lo, hi, S_v=None, B_v=None):
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        if not _valid_region(lo, hi):
            return None
        if _overlaps_forbidden(lo, hi):
            return None
        key = _region_key(lo, hi)
        if key in pool:
            return pool[key]
        if S_v is None or B_v is None:
            S_v, B_v = _rect_SB(lo, hi)
        else:
            S_v, B_v = float(S_v), float(B_v)
        if B_v < MIN_BKG_WEIGHT or S_v <= MIN_SIGNAL_WEIGHT:
            return None
        if MIN_SIGNAL_ENTRIES > 0 or MIN_BKG_ENTRIES > 0:
            S_check, B_check, S_entries, B_entries = _rect_stats(lo, hi)
            S_v, B_v = S_check, B_check
            if S_entries < MIN_SIGNAL_ENTRIES or B_entries < MIN_BKG_ENTRIES:
                return None
            if B_v < MIN_BKG_WEIGHT or S_v <= MIN_SIGNAL_WEIGHT:
                return None
        Z = _calc_Z_val(S_v, B_v)
        if Z <= 0.0:
            return None
        pool[key] = {"lo": [float(v) for v in lo],
                     "hi": [float(v) for v in hi],
                     "S": S_v, "B": B_v, "Z": Z}
        return pool[key]

    def _trim_pool():
        if len(pool) <= CANDIDATE_POOL_LIMIT:
            return
        before = len(pool)
        keep = sorted(pool.items(), key=lambda kv: -kv[1]["Z"])[:CANDIDATE_POOL_LIMIT]
        pool.clear()
        pool.update(keep)
        log_warning(
            f"Candidate pool trimmed from {before} to {len(pool)} by Z; "
            "increase candidate_pool_limit for a wider final search"
        )

    def _top_intervals_on_axis(d, lo, hi, edges, top_n):
        """Top [edges[a], edges[b]) intervals on axis d with other axes fixed."""
        m = np.ones(n_events, dtype=bool)
        for dd in range(D):
            if dd == d:
                continue
            v = score_axes[:, dd]
            if _hi_to_open(hi[dd]):
                m &= v >= lo[dd]
            else:
                m &= (v >= lo[dd]) & (v < hi[dd])
        if not m.any():
            return []
        edges = _cap_edges(edges, max(2, len(edges)))
        if edges.size < 2:
            return []
        v_d = score_axes[:, d]
        hS, _ = np.histogram(v_d[m], bins=edges, weights=w_sig[m])
        hB, _ = np.histogram(v_d[m], bins=edges, weights=w_bkg[m])
        pS = np.r_[0.0, np.cumsum(hS)]
        pB = np.r_[0.0, np.cumsum(hB)]
        K = pS.size
        a_idx = np.arange(K).reshape(-1, 1)
        b_idx = np.arange(K).reshape(1, -1)
        mask = b_idx > a_idx
        S_mat = pS[b_idx] - pS[a_idx]
        B_mat = pB[b_idx] - pB[a_idx]
        valid = mask & (B_mat >= MIN_BKG_WEIGHT) & (S_mat > 0.0)
        if not valid.any():
            return []
        Bsafe = np.where(valid, B_mat, 1.0)
        Ssafe = np.where(valid, S_mat, 0.0)
        f = (Ssafe + Bsafe) * np.log1p(Ssafe / Bsafe) - Ssafe
        f = np.where(valid & (f > 0.0), f, 0.0)
        Z2 = np.where(valid, 2.0 * f, -np.inf)
        valid_count = int(np.count_nonzero(np.isfinite(Z2) & (Z2 > 0.0)))
        if valid_count == 0:
            return []
        take = min(max(1, int(top_n)), valid_count)
        flat_scores = Z2.ravel()
        if take >= flat_scores.size:
            flat_idx = np.argsort(flat_scores)[::-1]
        else:
            flat_idx = np.argpartition(flat_scores, -take)[-take:]
            flat_idx = flat_idx[np.argsort(flat_scores[flat_idx])[::-1]]

        intervals = []
        seen = set()
        for flat in flat_idx:
            if not np.isfinite(flat_scores[flat]) or flat_scores[flat] <= 0.0:
                continue
            a_best = int(flat // K)
            b_best = int(flat % K)
            if not valid[a_best, b_best]:
                continue
            key = (a_best, b_best)
            if key in seen:
                continue
            seen.add(key)
            intervals.append((
                float(edges[a_best]),
                float(edges[b_best]),
                float(S_mat[a_best, b_best]),
                float(B_mat[a_best, b_best]),
                float(np.sqrt(Z2[a_best, b_best])),
            ))
            if len(intervals) >= take:
                break
        return intervals

    def _select_beam(candidates):
        unique = {}
        for item in candidates:
            if item is None:
                continue
            key = _region_key(item["lo"], item["hi"])
            if key not in unique or item["Z"] > unique[key]["Z"]:
                unique[key] = item
        ordered = sorted(unique.values(), key=lambda item: -item["Z"])
        if len(ordered) <= BEAM_WIDTH:
            return ordered

        selected = []
        selected_keys = set()
        diverse_target = max(1, BEAM_WIDTH // 2)
        for item in ordered:
            if len(selected) >= diverse_target:
                break
            if all(not _overlap(item["lo"], item["hi"], prev["lo"], prev["hi"]) for prev in selected):
                selected.append(item)
                selected_keys.add(_region_key(item["lo"], item["hi"]))

        for item in ordered:
            if len(selected) >= BEAM_WIDTH:
                break
            key = _region_key(item["lo"], item["hi"])
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
        return selected

    # ---- Build seeds ----
    seeds = []
    seed_keys = set()

    def _add_seed(lo, hi):
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        if not _valid_region(lo, hi):
            return
        key = _region_key(lo, hi)
        if key in seed_keys:
            return
        seed_keys.add(key)
        seeds.append((lo, hi))

    _add_seed([0.0] * D, [1.0] * D)  # full box
    for d in range(D):
        edges = edges_per_axis[d]
        for q in SEED_QUANTILES:
            qc = float(np.clip(q, 0.0, 1.0))
            idx = int(np.clip(round(qc * (len(edges) - 1)), 0, len(edges) - 1))
            lo = [0.0] * D
            hi = [1.0] * D
            lo[d] = float(edges[idx])
            _add_seed(lo, hi)
            if 0 < idx < len(edges) - 1:
                lo2 = [0.0] * D
                hi2 = [1.0] * D
                hi2[d] = float(edges[idx])
                _add_seed(lo2, hi2)

    if forbidden_boxes:
        axis_segments = []
        for d in range(D):
            vals = [0.0, 1.0]
            for flo, fhi in forbidden_boxes:
                vals.extend([flo[d], fhi[d]])
            vals = np.unique(np.clip(np.asarray(vals, dtype=float), 0.0, 1.0))
            segs = []
            for a, b in zip(vals[:-1], vals[1:]):
                if float(a) < float(b) - EPS:
                    segs.append((float(a), float(b)))
            axis_segments.append(segs)
            for a, b in segs:
                lo = [0.0] * D
                hi = [1.0] * D
                lo[d] = a
                hi[d] = b
                _add_seed(lo, hi)

        pair_seeds = []
        for d1 in range(D):
            for d2 in range(d1 + 1, D):
                for a1, b1 in axis_segments[d1]:
                    for a2, b2 in axis_segments[d2]:
                        lo = [0.0] * D
                        hi = [1.0] * D
                        lo[d1], hi[d1] = a1, b1
                        lo[d2], hi[d2] = a2, b2
                        if _overlaps_forbidden(lo, hi):
                            continue
                        volume = (b1 - a1) * (b2 - a2)
                        pair_seeds.append((volume, lo, hi))
        pair_seeds.sort(key=lambda x: -x[0])
        for _volume, lo, hi in pair_seeds[:max(BEAM_WIDTH * 8, 64)]:
            _add_seed(lo, hi)

    initial = []
    for seed_lo, seed_hi in seeds:
        item = _add_to_pool(seed_lo, seed_hi)
        if item is not None:
            initial.append(item)

    if not initial:
        raise RuntimeError(
            "No seed signal region passed min_bkg_weight; "
            "lower min_bkg_weight or check inputs"
        )

    beam = _select_beam(initial)
    _progress(
        f"Beam search start: seeds={len(seeds)}, valid_seeds={len(initial)}, "
        f"beam={len(beam)}, pool={len(pool)}, best_Z={beam[0]['Z']:.4f}",
        force=True,
    )

    for r in range(COORDINATE_ROUNDS):
        before_pool = len(pool)
        produced = []
        for ib, item in enumerate(beam):
            for d in range(D):
                intervals = _top_intervals_on_axis(
                    d, item["lo"], item["hi"], edges_per_axis[d], TOP_INTERVALS_PER_AXIS
                )
                for low_d, high_d, S_v, B_v, _Z in intervals:
                    lo = list(item["lo"])
                    hi = list(item["hi"])
                    lo[d] = low_d
                    hi[d] = high_d
                    new_item = _add_to_pool(lo, hi, S_v, B_v)
                    if new_item is not None:
                        produced.append(new_item)
            _progress(
                f"Beam round {r + 1}/{COORDINATE_ROUNDS}: "
                f"processed {ib + 1}/{len(beam)} beam states, pool={len(pool)}"
            )
        _trim_pool()
        pool_top = sorted(pool.values(), key=lambda item: -item["Z"])[:BEAM_WIDTH]
        beam = _select_beam(beam + produced + pool_top)
        best_Z = beam[0]["Z"] if beam else 0.0
        _progress(
            f"Beam round {r + 1}/{COORDINATE_ROUNDS} done: "
            f"new={len(pool) - before_pool}, produced={len(produced)}, "
            f"pool={len(pool)}, beam={len(beam)}, best_Z={best_Z:.4f}",
            force=True,
        )
        if len(pool) == before_pool:
            break

    if not pool:
        raise RuntimeError(
            "No candidate signal region passed min_bkg_weight; "
            "lower min_bkg_weight or check inputs"
        )

    def _local_edges_for_axis(d, lo, hi):
        edge_values = [0.0, 1.0, lo[d], hi[d]]
        edge_values.extend(edges_per_axis[d])
        vals = axis_unique_values[d]
        if vals.size:
            for boundary in (lo[d], hi[d]):
                idx = int(np.searchsorted(vals, boundary, side="left"))
                left = max(0, idx - LOCAL_REFINE_NEIGHBOR_EDGES)
                right = min(vals.size, idx + LOCAL_REFINE_NEIGHBOR_EDGES + 1)
                edge_values.extend(vals[left:right])
        return np.unique(np.clip(np.asarray(edge_values, dtype=float), 0.0, 1.0))

    if LOCAL_REFINE_ROUNDS > 0 and LOCAL_REFINE_TOP_CANDIDATES > 0:
        ordered_for_refine = sorted(pool.values(), key=lambda x: -x["Z"])
        refine_items = ordered_for_refine[:LOCAL_REFINE_TOP_CANDIDATES]
        _progress(
            f"Local event-threshold refinement start: candidates={len(refine_items)}, "
            f"rounds={LOCAL_REFINE_ROUNDS}",
            force=True,
        )
        for ic, item in enumerate(refine_items):
            lo = list(item["lo"])
            hi = list(item["hi"])
            prev_Z = item["Z"]
            for rr in range(LOCAL_REFINE_ROUNDS):
                changed = False
                for d in range(D):
                    local_edges = _local_edges_for_axis(d, lo, hi)
                    intervals = _top_intervals_on_axis(
                        d, lo, hi, local_edges, max(1, min(TOP_INTERVALS_PER_AXIS, 4))
                    )
                    if not intervals:
                        continue
                    for low_d, high_d, S_v, B_v, _Z in intervals:
                        lo_alt = list(lo)
                        hi_alt = list(hi)
                        lo_alt[d] = low_d
                        hi_alt[d] = high_d
                        _add_to_pool(lo_alt, hi_alt, S_v, B_v)
                    low_d, high_d, S_v, B_v, Z_axis = intervals[0]
                    boundary_changed = (
                        abs(low_d - lo[d]) > 1e-12 or abs(high_d - hi[d]) > 1e-12
                    )
                    if boundary_changed and Z_axis >= prev_Z - 1e-10:
                        lo[d] = low_d
                        hi[d] = high_d
                        prev_Z = Z_axis
                        changed = True
                refined = _add_to_pool(lo, hi)
                if refined is not None:
                    prev_Z = refined["Z"]
                if not changed:
                    break
            _progress(
                f"Local refinement: processed {ic + 1}/{len(refine_items)}, "
                f"pool={len(pool)}"
            )
        _trim_pool()
        _progress(f"Local refinement done: pool={len(pool)}", force=True)

    all_items = sorted(pool.values(), key=lambda x: -x["Z"])
    log_message(f"  Candidate pool size: {len(all_items)}")
    n_items_total = len(all_items)
    n_limited = min(n_items_total, GLOBAL_SELECTION_CANDIDATES)
    if n_limited < n_items_total:
        log_warning(
            f"Global selection uses top {n_limited} of {n_items_total} candidates for speed; "
            "increase global_selection_candidates for a wider exact search"
        )

    def _dedupe_by_event_mask(candidate_items):
        if not DEDUPLICATE_EVENT_MASKS:
            return candidate_items
        seen_masks = set()
        deduped = []
        duplicate_count = 0
        for i, item in enumerate(candidate_items):
            packed = np.packbits(_rect_mask(item["lo"], item["hi"])).tobytes()
            if packed in seen_masks:
                duplicate_count += 1
                continue
            seen_masks.add(packed)
            deduped.append(item)
            _progress(
                f"Event-mask dedupe: processed {i + 1}/{len(candidate_items)}, "
                f"kept={len(deduped)}"
            )
        log_message(
            f"  Event-mask dedupe: input={len(candidate_items)}, "
            f"duplicates={duplicate_count}, kept={len(deduped)}"
        )
        return deduped

    items = _dedupe_by_event_mask(all_items[:n_limited])
    if len(items) < target_regions:
        raise RuntimeError(
            f"Only {len(items)} unique candidate signal regions are available; "
            f"requested {target_regions}"
        )

    n_items = len(items)
    Z_arr = np.array([it["Z"] for it in items], dtype=float)
    Z2 = Z_arr ** 2
    los = [it["lo"] for it in items]
    his = [it["hi"] for it in items]
    target_n = target_regions

    def _compatible_with_picks(i, picks):
        return all(not _overlap(los[i], his[i], los[j], his[j]) for j in picks)

    def _prune_state_bucket(bucket, width):
        bucket.sort(key=lambda state: (-state[0], state[1]))
        if len(bucket) > width:
            del bucket[width:]

    def _select_regions_beam_python(n_select, beam_width):
        states = [[] for _ in range(target_n + 1)]
        states[0] = [(0.0, tuple())]
        for i in range(n_select):
            max_count = min(i, target_n - 1)
            for count in range(max_count, -1, -1):
                additions = []
                for score, picks in states[count]:
                    if _compatible_with_picks(i, picks):
                        additions.append((score + Z2[i], picks + (i,)))
                if additions:
                    states[count + 1].extend(additions)
                    _prune_state_bucket(states[count + 1], beam_width)
            _progress(
                f"Global incumbent beam: processed {i + 1}/{n_select}, "
                f"best_count={max(c for c, bucket in enumerate(states) if bucket)}"
            )
        for count in range(target_n, 0, -1):
            if states[count]:
                _prune_state_bucket(states[count], beam_width)
                return list(states[count][0][1])
        return []

    def _score_picks(picks):
        return float(np.sum(Z2[list(picks)])) if picks else 0.0

    def _selection_result(picks, score, upper_bound, completed, nodes, selector_name):
        score = float(score)
        upper_bound = float(max(score, upper_bound))
        return {
            "picks": list(picks),
            "score": score,
            "upper_bound": upper_bound,
            "completed": bool(completed),
            "nodes": int(nodes),
            "selector": selector_name,
        }

    def _select_regions_branch_bound_python(n_select):
        seed = _select_regions_beam_python(n_select, GLOBAL_BEAM_WIDTH)
        best_picks = tuple(seed) if len(seed) == target_n else tuple()
        best_score = _score_picks(best_picks)
        nodes = 0
        stopped = False
        start_time = time.monotonic()

        def _limit_hit():
            nonlocal stopped
            if stopped:
                return True
            if BRANCH_BOUND_MAX_NODES > 0 and nodes >= BRANCH_BOUND_MAX_NODES:
                stopped = True
            if (
                BRANCH_BOUND_TIME_LIMIT_SECONDS > 0.0 and
                time.monotonic() - start_time >= BRANCH_BOUND_TIME_LIMIT_SECONDS
            ):
                stopped = True
            return stopped

        def _optimistic_bound(start, picks, score):
            remaining = target_n - len(picks)
            if remaining <= 0:
                return float(score)
            bound = float(score)
            count = 0
            for cand in range(start, n_select):
                if _compatible_with_picks(cand, picks):
                    bound += Z2[cand]
                    count += 1
                    if count >= remaining:
                        return float(bound)
            return -np.inf

        root_bound = _optimistic_bound(0, tuple(), 0.0)
        if not np.isfinite(root_bound):
            return _selection_result([], 0.0, 0.0, True, nodes, "Python branch-and-bound")

        def _better_picks(a, b):
            return tuple(a) < tuple(b)

        def _update_best(picks, score):
            nonlocal best_picks, best_score
            picks = tuple(picks)
            if (
                score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and picks and _better_picks(picks, best_picks))
            ):
                best_score = float(score)
                best_picks = picks

        def _dfs(start, picks, score):
            nonlocal nodes
            if stopped:
                return
            nodes += 1
            if nodes % 4096 == 0 and _limit_hit():
                return
            if len(picks) == target_n:
                _update_best(picks, score)
                return
            bound = _optimistic_bound(start, picks, score)
            if not np.isfinite(bound) or bound <= best_score + 1e-12:
                return
            remaining = target_n - len(picks)
            for cand in range(start, n_select - remaining + 1):
                if _limit_hit():
                    return
                if not _compatible_with_picks(cand, picks):
                    continue
                _dfs(cand + 1, picks + (cand,), score + Z2[cand])

        _progress(
            f"Python branch-and-bound start: candidates={n_select}, target_bins={target_n}, "
            f"incumbent_Z={np.sqrt(best_score):.4f}",
            force=True,
        )
        _dfs(0, tuple(), 0.0)
        completed = not stopped
        upper = best_score if completed else root_bound
        return _selection_result(
            best_picks,
            best_score,
            upper,
            completed,
            nodes,
            "Python branch-and-bound",
        )

    def _build_openmp_selector():
        src = os.path.join(_SCRIPT_DIR, "openmp_region_select.cpp")
        if not os.path.exists(src):
            return None
        build_dir = os.path.join(OUTPUT_DIR, ".openmp")
        os.makedirs(build_dir, exist_ok=True)
        lib = os.path.join(build_dir, "openmp_region_select.so")
        needs_build = (
            not os.path.exists(lib) or
            os.path.getmtime(lib) < os.path.getmtime(src)
        )
        if needs_build:
            base_cmd = ["c++", "-O3", "-std=c++17", "-fPIC", "-shared"]
            attempts = [
                (
                    "Homebrew libomp",
                    [
                        "-Xpreprocessor", "-fopenmp", "-D_OPENMP=201511",
                        "-I/opt/homebrew/opt/libomp/include",
                    ],
                    ["-L/opt/homebrew/opt/libomp/lib", "-lomp"],
                ),
                (
                    "Homebrew libomp (/usr/local)",
                    [
                        "-Xpreprocessor", "-fopenmp", "-D_OPENMP=201511",
                        "-I/usr/local/opt/libomp/include",
                    ],
                    ["-L/usr/local/opt/libomp/lib", "-lomp"],
                ),
                ("generic -fopenmp", ["-fopenmp"], []),
            ]
            errors = []
            built = False
            for label, cflags, ldflags in attempts:
                cmd = base_cmd + cflags + [src, "-o", lib] + ldflags
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                except OSError as exc:
                    errors.append(f"{label}: {exc}")
                    continue
                if proc.returncode == 0:
                    log_message(f"  OpenMP selector built with {label}")
                    built = True
                    break
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                if detail:
                    errors.append(f"{label}: {detail[-1]}")
                else:
                    errors.append(f"{label}: compiler returned {proc.returncode}")
            if not built:
                log_warning(
                    "OpenMP selector build failed; falling back to Python branch-and-bound. "
                    + " | ".join(errors)
                )
                return None
        try:
            helper = ctypes.CDLL(lib)
            beam_fn = helper.select_regions_beam_openmp
            beam_fn.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            beam_fn.restype = ctypes.c_int

            bnb_fn = helper.select_regions_branch_bound_openmp
            bnb_fn.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_longlong,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
            ]
            bnb_fn.restype = ctypes.c_int
            return {"beam": beam_fn, "branch_bound": bnb_fn}
        except Exception as exc:
            log_warning(f"OpenMP selector load failed; falling back to Python branch-and-bound: {exc}")
            return None

    def _select_regions_branch_bound_openmp(n_select):
        if MAX_THREADS <= 1 or target_n > 16:
            return None
        helper = _build_openmp_selector()
        if helper is None:
            return None
        fn = helper["branch_bound"]
        lows_arr = np.ascontiguousarray(np.asarray(los[:n_select], dtype=np.float64))
        highs_arr = np.ascontiguousarray(np.asarray(his[:n_select], dtype=np.float64))
        z2_arr = np.ascontiguousarray(np.asarray(Z2[:n_select], dtype=np.float64))
        out = np.full(target_n, -1, dtype=np.int32)
        stats = np.zeros(6, dtype=np.float64)
        ret = fn(
            ctypes.c_int(n_select),
            ctypes.c_int(D),
            ctypes.c_int(target_n),
            ctypes.c_int(GLOBAL_BEAM_WIDTH),
            ctypes.c_longlong(BRANCH_BOUND_MAX_NODES),
            ctypes.c_double(BRANCH_BOUND_TIME_LIMIT_SECONDS),
            lows_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            highs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(MAX_THREADS),
        )
        if ret < 0:
            log_warning(f"OpenMP branch-and-bound returned {ret}; falling back to Python")
            return None
        picks = [int(v) for v in out[:ret] if int(v) >= 0]
        return _selection_result(
            picks,
            stats[0],
            stats[1],
            stats[3] > 0.5,
            int(round(stats[2])),
            "OpenMP branch-and-bound",
        )

    # ---- Global selection over the candidate pool. ----
    n_select = n_items
    _progress(
        f"Global branch-and-bound start: candidates={n_select}, target_bins={target_n}, "
        f"beam_width={GLOBAL_BEAM_WIDTH}, max_threads={MAX_THREADS}",
        force=True,
    )

    selection = _select_regions_branch_bound_openmp(n_select)
    if selection is None:
        selection = _select_regions_branch_bound_python(n_select)
    best_picks = selection["picks"]
    best_score = selection["score"]

    if not best_picks:
        raise RuntimeError("Global selection found no valid signal-region set")
    if len(best_picks) < target_regions:
        message = (
            f"Global selection found only {len(best_picks)} non-overlapping "
            f"regions; requested {target_regions}"
        )
        if REQUIRE_EXACT_N_REGIONS:
            raise RuntimeError(message)
        log_warning(message)

    selected_masks = [_rect_mask(los[idx], his[idx]) for idx in best_picks]
    geometry_overlap_pairs = 0
    event_overlap_pairs = 0
    for ia in range(len(best_picks)):
        for ib in range(ia + 1, len(best_picks)):
            if _overlap(
                los[best_picks[ia]], his[best_picks[ia]],
                los[best_picks[ib]], his[best_picks[ib]],
            ):
                geometry_overlap_pairs += 1
            if np.any(selected_masks[ia] & selected_masks[ib]):
                event_overlap_pairs += 1
    if geometry_overlap_pairs or event_overlap_pairs:
        raise RuntimeError(
            "Selected signal-region definitions overlap: "
            f"geometry_pairs={geometry_overlap_pairs}, event_pairs={event_overlap_pairs}"
        )

    upper_score = selection["upper_bound"]
    z_best = float(np.sqrt(best_score))
    z_upper = float(np.sqrt(max(best_score, upper_score)))
    if selection["completed"]:
        log_message(
            f"  Global selection completed exactly over {n_select} candidates: "
            f"sum(Z^2)={best_score:.6g}, Z_comb={z_best:.4f}, "
            f"nodes={selection['nodes']}, selector={selection['selector']}"
        )
    else:
        log_warning(
            f"Global selection stopped before exhausting the search: "
            f"Z_best={z_best:.4f}, Z_upper_bound<={z_upper:.4f}, "
            f"delta_Z<={max(0.0, z_upper - z_best):.4f}, "
            f"nodes={selection['nodes']}, selector={selection['selector']}"
        )

    log_message(
        f"  Selected {len(best_picks)} signal regions, "
        f"sum(Z^2)={best_score:.6g}, Z_comb={z_best:.4f}"
    )
    log_message(
        f"  Non-overlap check passed: geometry_pairs={geometry_overlap_pairs}, "
        f"event_pairs={event_overlap_pairs}"
    )

    # ---- Build per-bin reports for the chosen rectangles. ----
    top_bins = []
    for k, idx in enumerate(best_picks):
        thr_low_vec = list(map(float, los[idx]))
        thr_high_vec = list(map(float, his[idx]))

        m_bin = _rect_mask(thr_low_vec, thr_high_vec)
        wS = w[m_bin & is_sig]
        wB = w[m_bin & is_bkg]
        S_bin = float(wS.sum())
        B_bin = float(wB.sum())
        sS_bin = float(np.sqrt((wS ** 2).sum()))
        sB_bin = float(np.sqrt((wB ** 2).sum()))
        S_e = int((m_bin & is_sig).sum())
        B_e = int((m_bin & is_bkg).sum())
        Z_bin, sZ_bin = _calc_Z(S_bin, B_bin, sS_bin, sB_bin)

        # Per-class break-down.
        W_bin = S_bin + B_bin
        w2_bin = sS_bin ** 2 + sB_bin ** 2
        cat_data = []
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            mC = (y == cls_i) & m_bin
            wC = w[mC]
            S_j = float(wC.sum())
            sS_j = float(np.sqrt((wC ** 2).sum()))
            B_j = W_bin - S_j
            sB_j = float(np.sqrt(max(0.0, w2_bin - sS_j ** 2)))
            Z_j, sZ_j = _calc_Z(S_j, B_j, sS_j, sB_j)
            cat_data.append({
                "name": cls_name,
                "S": S_j, "S_err": sS_j,
                "B": B_j, "B_err": sB_j,
                "Z": Z_j, "Z_err": sZ_j,
            })

        bkg_data = []
        for bkg_i in BACKGROUND_CLASS_INDICES:
            mC = (y == bkg_i) & m_bin
            wC = w[mC]
            bkg_data.append({
                "name": CLASS_NAMES[bkg_i],
                "B": float(wC.sum()),
                "B_err": float(np.sqrt((wC ** 2).sum())),
            })

        bin_sig_eff = (S_bin / S_total) if S_total > 0 else float("nan")
        bin_bkg_eff = (B_bin / B_total) if B_total > 0 else float("nan")

        tail_sig_eff, tail_bkg_eff = [], []
        for d in range(D):
            tidx = max(0, min(
                int(np.searchsorted(thr_1d, thr_low_vec[d], side="right") - 1),
                T_REF - 1,
            ))
            tail_sig_eff.append(
                (S_tail_by_dim[d, tidx] / S_total) if S_total > 0 else float("nan")
            )
            tail_bkg_eff.append(
                (B_tail_by_dim[d, tidx] / B_total) if B_total > 0 else float("nan")
            )

        top_bins.append({
            "bin_index":                 k + 1,
            "thr_low":                   np.array(thr_low_vec),
            "thr_high":                  np.array(thr_high_vec),
            "axis_names":                axis_names,
            "significance":              Z_bin,
            "significance_error":        sZ_bin,
            "S":                         S_bin,  "S_err": sS_bin, "S_entries": S_e,
            "B":                         B_bin,  "B_err": sB_bin, "B_entries": B_e,
            "categories":                cat_data,
            "backgrounds":               bkg_data,
            "bin_signal_efficiency":     bin_sig_eff,
            "bin_background_efficiency": bin_bkg_eff,
            "tail_signal_efficiency":    tail_sig_eff,
            "tail_background_efficiency": tail_bkg_eff,
        })

        log_message(
            f"  Bin {k + 1}: Z={Z_bin:.4f}±{sZ_bin:.4f}, "
            f"S={S_bin:.4g}±{sS_bin:.4g}, B={B_bin:.4g}±{sB_bin:.4g}"
        )

    selection_summary = {
        "selector": selection["selector"],
        "completed": selection["completed"],
        "nodes": selection["nodes"],
        "objective_sum_z2": best_score,
        "objective_upper_bound_sum_z2": upper_score,
        "geometry_overlap_pairs": geometry_overlap_pairs,
        "event_overlap_pairs": event_overlap_pairs,
        "candidate_count": n_select,
    }
    return _make_signal_region_result(top_bins, S_total, B_total, selection_summary)


def _make_signal_region_result(top_bins, S_total, B_total, selection_summary=None):
    # Combine the per-bin significances as Z_comb = sqrt(sum Z_i^2).
    if top_bins:
        S_comb  = float(sum(b["S"]       for b in top_bins))
        B_comb  = float(sum(b["B"]       for b in top_bins))
        sS_comb = float(np.sqrt(sum(b["S_err"] ** 2 for b in top_bins)))
        sB_comb = float(np.sqrt(sum(b["B_err"] ** 2 for b in top_bins)))
        Z_vec   = np.array([b["significance"]       for b in top_bins])
        sZ_vec  = np.array([b["significance_error"] for b in top_bins])
        Z_comb  = float(np.sqrt(np.sum(Z_vec ** 2)))
        sZ_comb = (float(np.sqrt(np.sum((Z_vec * sZ_vec) ** 2))) / Z_comb
                   if Z_comb > 0 else float(np.sqrt(np.sum(sZ_vec ** 2))))
    else:
        S_comb = B_comb = sS_comb = sB_comb = Z_comb = sZ_comb = 0.0

    return {
        "top_bins":                    top_bins,
        "combined_S":                  S_comb,
        "combined_B":                  B_comb,
        "combined_S_err":              sS_comb,
        "combined_B_err":              sB_comb,
        "combined_significance":       Z_comb,
        "combined_significance_error": sZ_comb,
        "S_total":                     S_total,
        "B_total":                     B_total,
        "selection":                   selection_summary or {},
    }


# -------------------- Result printing --------------------
def print_results(result):
    top_bins = result["top_bins"]

    for b in top_bins:
        log_message(f"\n-- Bin {b['bin_index']} --")
        region_parts = [
            f"{b['axis_names'][d]}: [{b['thr_low'][d]:.6f}, {b['thr_high'][d]:.6f})"
            for d in range(len(b["axis_names"]))
        ]
        log_message(f"  Region: " + "  ".join(region_parts))
        log_message(f"  Significance: {b['significance']:.4f} ± {b['significance_error']:.4f}")
        log_message(
            f"  S (weighted): {b['S']:.4g} ± {b['S_err']:.4g}  |  S (entries): {b['S_entries']}"
        )
        log_message(
            f"  B (weighted): {b['B']:.4g} ± {b['B_err']:.4g}  |  B (entries): {b['B_entries']}"
        )
        log_message(
            f"  Signal eff. (bin): {b['bin_signal_efficiency']:.6f}  |  "
            f"Bkg eff. (bin): {b['bin_background_efficiency']:.6f}"
        )
        if b["tail_signal_efficiency"]:
            tse = "  ".join(
                f"{b['axis_names'][d]}={b['tail_signal_efficiency'][d]:.6f}"
                for d in range(len(b["axis_names"]))
            )
            tbe = "  ".join(
                f"{b['axis_names'][d]}={b['tail_background_efficiency'][d]:.6f}"
                for d in range(len(b["axis_names"]))
            )
            log_message(f"  Tail signal eff.:   {tse}")
            log_message(f"  Tail bkg eff.:      {tbe}")

        log_message("  Per-class significance:")
        for cat in b["categories"]:
            log_message(f"    {cat['name']}: {cat['Z']:.4f} ± {cat['Z_err']:.4f}")

        log_message("  Per-class breakdown (weighted ± error):")
        for cat in b["categories"]:
            log_message(
                f"    {cat['name']}: S={cat['S']:.4g} ± {cat['S_err']:.4g}  |  "
                f"B(rest)={cat['B']:.4g} ± {cat['B_err']:.4g}"
            )

        log_message("  Background breakdown (weighted ± error):")
        for bkg in b["backgrounds"]:
            log_message(f"    {bkg['name']}: B={bkg['B']:.4g} ± {bkg['B_err']:.4g}")

    log_message(f"\n==== Combined (top {len(top_bins)} bins) ====")
    log_message(f"  S total:               {result['combined_S']:.4g} ± {result['combined_S_err']:.4g}")
    log_message(f"  B total:               {result['combined_B']:.4g} ± {result['combined_B_err']:.4g}")
    log_message(
        f"  Combined significance: {result['combined_significance']:.4f} ± "
        f"{result['combined_significance_error']:.4f}"
    )
    selection = result.get("selection") or {}
    if selection:
        upper = float(selection.get("objective_upper_bound_sum_z2", 0.0))
        best = float(selection.get("objective_sum_z2", 0.0))
        log_message(
            f"  Global selector:        {selection.get('selector', 'unknown')} "
            f"(completed={selection.get('completed', False)}, nodes={selection.get('nodes', 0)})"
        )
        log_message(
            f"  Search certificate:     Z_best={np.sqrt(max(0.0, best)):.4f}, "
            f"Z_upper_bound<={np.sqrt(max(best, upper, 0.0)):.4f}"
        )
        log_message(
            f"  Non-overlap pairs:      geometry={selection.get('geometry_overlap_pairs', 0)}, "
            f"events={selection.get('event_overlap_pairs', 0)}"
        )


# -------------------- Main --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_message(
        f"Running signal_region.py: tree={TREE_NAME}, lumi={LUMI} fb^-1, "
        f"n_signal_regions={N_SIGNAL_REGIONS}, bdt_root={BDT_ROOT}, output_dir={OUTPUT_DIR}"
    )

    sel         = sel_cfg[TREE_NAME]
    branches    = [b["name"] for b in br_cfg[TREE_NAME]]
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_tf      = sel.get("log_transform", [])
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    decorrelate = cfg.get(TREE_NAME, {}).get("decorrelate", [])

    # Threshold and decorrelate branches that are NOT declared in branch.json
    # still need to be read from the ROOT files so filter_X can cut on them and
    # the decorrelation step can reference them. They are removed from X before
    # model inference so the BDT input feature set stays strictly defined by
    # branch.json (mirrors train.py).
    extra_cols = []
    for c in list(thresholds.keys()) + list(decorrelate):
        if c not in branches and c not in extra_cols:
            extra_cols.append(c)
    load_cols = branches + extra_cols
    drop_after_filter = [c for c in extra_cols if c not in decorrelate]

    # Load the test data once; the physics weights stay fixed afterwards.
    df_all = load_test_data(load_cols)
    X             = df_all[load_cols].copy()
    y             = df_all["class_idx"].values.astype(int)
    w             = df_all["weight"].values.astype(float)
    sample_labels = df_all["sample_name"].values
    del df_all
    gc.collect()

    # Apply threshold filtering without changing the fixed weights.
    log_message("Applying thresholds")
    X, y, w, sample_labels = filter_X(
        X, y, w, load_cols, thresholds, apply_to_sentinel=True, sample_labels=sample_labels
    )
    log_message(f"After filtering: {len(X)} events")

    # Apply the same feature standardization used in train.py.
    log_message("Standardising features")
    X = standardize_X(X, clip_ranges, log_tf)

    if drop_after_filter:
        X = X.drop(columns=drop_after_filter, errors="ignore")

    # Drop the decorrelated features before model inference, exactly as in training.
    all_feature_names = list(X.columns)
    if decorrelate:
        name_to_idx = {c: i for i, c in enumerate(all_feature_names)}
        decor_idx   = sorted(name_to_idx[k] for k in decorrelate if k in name_to_idx)
        keep_idx    = [i for i in range(len(all_feature_names)) if i not in decor_idx]
        X_model     = X.iloc[:, keep_idx]
        log_message(f"Removed decorrelated features: {decorrelate}")
    else:
        X_model = X

    # Load the trained model.
    model_base = MODEL_PATTERN.format(output_root=BDT_ROOT, tree_name=TREE_NAME)
    if os.path.exists(model_base + ".json"):
        model_path = model_base + ".json"
        clf = xgb.Booster()
        clf.load_model(model_path)
        log_message(f"Loaded model: {model_path}")
    elif os.path.exists(model_base + ".pkl"):
        model_path = model_base + ".pkl"
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        log_message(f"Loaded model: {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_base}(.json/.pkl)")

    # Run the BDT prediction.
    log_message("Running BDT prediction")
    proba = _predict_model_proba(clf, X_model)
    log_message(f"Predicted probabilities shape: {proba.shape}")
    log_message("Validating test-set prediction reference")
    _compare_prediction_reference(
        TEST_REFERENCE_SIGNAL_REGION,
        X_model.columns if hasattr(X_model, "columns") else [f"f{i}" for i in range(X_model.shape[1])],
        sample_labels,
        y,
        w,
        proba,
    )

    # Plot the weighted score distributions.
    log_message("Plotting score distributions")
    plot_score_distributions(proba, y, w)

    # Scan once and globally select K mutually non-overlapping regions.
    log_message(f"Scanning globally for {N_SIGNAL_REGIONS} signal regions")
    result = find_signal_regions(
        proba,
        y,
        w,
        target_regions=N_SIGNAL_REGIONS,
    )

    # Plot the first two scan axes when D >= 2.
    log_message("Plotting signal regions")
    plot_signal_regions_2d(result, proba, y, w)

    # Print the text summary.
    print_results(result)
    write_signal_region_csv(result)

    log_message(f"Finished signal_region.py for tree={TREE_NAME}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
