#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data vs MC histogram comparison plotter.

Reads convert_branch.C output ROOT files directly for ordinary branches,
including branches that are not BDT inputs, applies the trained-model
selection.json clip/threshold cuts (no log transform), and draws a stacked MC +
data panel with a Data/MC ratio sub-panel. Derived model score branches use the
saved MC test split and optionally validate against the trained-model prediction
reference.
"""

import gc
import os
import sys
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import mplhep as hep
import uproot


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_BDT_DIR = os.path.join(_ROOT_DIR, "selections", "BDT")
if _BDT_DIR not in sys.path:
    sys.path.insert(0, _BDT_DIR)

from model_io import (
    load_model as _shared_load_model,
    predict_model_proba as _shared_predict_model_proba,
)

# -------------------- Style --------------------
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)


# -------------------- Helpers --------------------
def log_message(msg):
    print(msg, flush=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(path, base):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def _score_branch_name(class_name):
    return f"score_{class_name}"


# -------------------- Config loading --------------------
_cfg_path = os.environ.get("PLOT_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
_cfg_path = _resolve(_cfg_path, _SCRIPT_DIR)

plot_cfg             = _load_json(_cfg_path)
branch_overrides_cfg = _load_json(os.path.join(_SCRIPT_DIR, "branch.json"))

SUBMIT_TREES     = plot_cfg.get("submit_trees", ["fat2", "fat3"])
DATA_SAMPLES     = list(plot_cfg.get("data_samples", []))
DEFAULT_BINS     = int(plot_cfg.get("default_bins", 10))
OUTPUT_ROOT_PATT = plot_cfg.get("output_root", "./pre-selection/{tree_name}")
BDT_ROOT_PATT    = plot_cfg["bdt_root"]
VALIDATE_BDT_TEST_SCORES = bool(plot_cfg.get("validate_bdt_test_scores", True))
PLOT_BDT_SCORES = bool(plot_cfg.get("plot_bdt_scores", True))
STREAM_STEP_SIZE = "100 MB"

SAMPLE_CFG_PATH         = _resolve(plot_cfg["sample_config"], _SCRIPT_DIR)
CONVERT_BRANCH_CFG_PATH = _resolve(plot_cfg["convert_branch_config"], _SCRIPT_DIR)

sample_cfg         = _load_json(SAMPLE_CFG_PATH)
convert_branch_cfg = _load_json(CONVERT_BRANCH_CFG_PATH)

SAMPLE_INFO = {s["name"]: s for s in sample_cfg["sample"]}


def _compute_lumi_total():
    total = 0.0
    for name in DATA_SAMPLES:
        if name not in SAMPLE_INFO:
            raise RuntimeError(f"Data sample '{name}' not found in sample.json")
        info = SAMPLE_INFO[name]
        if info.get("is_MC", True):
            raise RuntimeError(f"Sample '{name}' is flagged as MC in sample.json but listed as data")
        total += float(info.get("lumi", 0.0))
    return total


LUMI_TOTAL = _compute_lumi_total()


# -------------------- Branch discovery --------------------
def _tree_plot_cfg(tree_name):
    if not isinstance(branch_overrides_cfg, dict):
        return {}
    tree_cfg = branch_overrides_cfg.get(tree_name, {})
    return tree_cfg if isinstance(tree_cfg, dict) else {}


def _skip_branches_for_tree(tree_name):
    skip = _tree_plot_cfg(tree_name).get("skip_branches", [])
    if not isinstance(skip, list):
        raise TypeError(f"plotting/branch.json:{tree_name}.skip_branches must be a list")
    return set(skip)


def _tree_output_entry(tree_name):
    for tree in convert_branch_cfg["output"]["trees"]:
        if tree["name"] == tree_name:
            return tree
    raise KeyError(f"Tree '{tree_name}' not in convert branch config")


def _plot_branches_for_tree(tree_name):
    """Return branch names to plot (onlyMC=false, not skipped, slots expanded)."""
    tree    = _tree_output_entry(tree_name)
    skip    = _skip_branches_for_tree(tree_name)
    scalars = tree.get("scalars", {})
    entries = list(scalars.get("regular", [])) + list(scalars.get("extrema", []))
    out, seen = [], set()
    for e in entries:
        if e.get("onlyMC", False):
            continue
        name = e["name"]
        slots = e.get("slots")
        if slots:
            for i in range(int(slots)):
                n = f"{name}_{i + 1}"
                if n in skip or n in seen:
                    continue
                seen.add(n)
                out.append(n)
        else:
            if name in skip or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


# -------------------- Trained-model config copies --------------------
def _bdt_root_for_tree(tree_name):
    return _resolve(BDT_ROOT_PATT.format(tree_name=tree_name), _SCRIPT_DIR)


def _bdt_configs_for_tree(tree_name, load_test_ranges=True):
    bdt_root = _bdt_root_for_tree(tree_name)
    cfg = _load_json(os.path.join(bdt_root, "config.json"))
    br = _load_json(os.path.join(bdt_root, "branch.json"))
    sel = _load_json(os.path.join(bdt_root, "selection.json"))
    meta = None
    if load_test_ranges:
        meta = _load_json(os.path.join(bdt_root, "test_ranges.json"))
    return cfg, br, sel, meta


# -------------------- Input file resolution --------------------
def _sample_group(info):
    if not info.get("is_MC", True):
        return "data"
    return "signal" if info.get("is_signal", False) else "bkg"


def _input_files(sample_name, input_root, input_pattern):
    info = SAMPLE_INFO[sample_name]
    sg   = _sample_group(info)
    pattern = input_pattern
    if not info.get("is_MC", True):
        pattern = pattern.replace("{sample_group}_mixed", "{sample_group}")
    base = pattern.format(input_root=input_root, sample_group=sg, sample=sample_name)
    stem = base[:-5] if base.endswith(".root") else base
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


def _tree_entries_total(files, tree_name):
    total = 0
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            total += int(uf[tree_name].num_entries)
    return total


def _concat_parts(parts):
    if not parts:
        return None
    if len(parts) == 1:
        df = parts[0].reset_index(drop=True)
        parts.clear()
        gc.collect()
        return df
    df = pd.concat(parts, ignore_index=True)
    parts.clear()
    gc.collect()
    return df


def _iter_tree_chunks(files, tree_name, branches):
    branches = list(branches)
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree = uf[tree_name]
            avail = set(tree.keys())
            missing = [b for b in branches if b not in avail]
            if missing:
                raise KeyError(
                    f"Missing branches in {fpath}:{tree_name}: "
                    f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                )
            for df_part in tree.iterate(
                branches,
                library="pd",
                step_size=STREAM_STEP_SIZE,
            ):
                if df_part is None or len(df_part) == 0:
                    continue
                yield df_part


# -------------------- Threshold and clip filtering --------------------
def _missing_value_mask(arr):
    return np.asarray(arr, dtype=float) <= -99.0


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
        masks = [_mask_from_cond(col, item) for item in cond]
        out = pd.Series(False, index=idx)
        for mask in masks:
            out |= mask
        return out
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key not in cond:
                continue
            items = cond[key]
            out = pd.Series(True if is_and else False, index=idx)
            for item in items:
                mask = _mask_from_cond(col, item)
                out = (out & mask) if is_and else (out | mask)
            return out
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported threshold condition: {cond!r}")


def _threshold_mask(df, thresholds):
    if not thresholds or df is None or len(df) == 0:
        return pd.Series(True, index=df.index if df is not None else None)
    mask = pd.Series(True, index=df.index)
    for b, cond in thresholds.items():
        if b not in df.columns:
            raise KeyError(f"Column {b!r} not found in DataFrame")
        col = df[b]
        sentinel = _missing_value_mask(col)
        mask &= ~sentinel
        mask &= _mask_from_cond(col, cond)
    return mask


def _apply_thresholds(df, thresholds):
    if not thresholds or df is None or len(df) == 0:
        return df
    mask = _threshold_mask(df, thresholds)
    return df.loc[mask].reset_index(drop=True)


def _apply_clip(df, clip_ranges):
    if not clip_ranges or df is None or len(df) == 0:
        return df
    for col, rng in clip_ranges.items():
        if col not in df.columns:
            continue
        arr   = df[col].values.astype(float, copy=True)
        valid = ~_missing_value_mask(arr)
        lo, hi = rng
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        df[col] = arr
    return df


def _standardize_model_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        sentinel = _missing_value_mask(arr)
        valid = ~sentinel
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


def _drop_decorrelated_features(X, decorrelate):
    if not decorrelate:
        return X
    drop_cols = [name for name in decorrelate if name in X.columns]
    if drop_cols:
        return X.drop(columns=drop_cols)
    return X


def _predict_model_proba(model, X, num_classes):
    return _shared_predict_model_proba(model, X, num_classes)


def _load_score_model(bdt_root, bdt_cfg, tree_name):
    model_pattern = bdt_cfg.get("model_pattern", "{output_root}/{tree_name}_model")
    model_base = model_pattern.format(output_root=bdt_root, tree_name=tree_name)
    class_groups = bdt_cfg["class_groups"]
    return _shared_load_model(
        model_base,
        bdt_cfg,
        len(class_groups),
        log_message=log_message,
    )


def _compare_score_reference(path, feature_names, sample_labels, class_idx, weights, proba):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction reference not found: {path}. Re-run train.py before data_mc.py."
        )

    ref = np.load(path, allow_pickle=False)
    ref_features = ref["feature_names"].astype(str).tolist()
    cur_features = list(feature_names)
    if cur_features != ref_features:
        raise RuntimeError(
            "Prediction reference mismatch for score model features: "
            f"current={cur_features}, reference={ref_features}"
        )

    ref_samples = ref["sample_name"].astype(str)
    cur_samples = np.asarray(sample_labels, dtype=str)
    if not np.array_equal(cur_samples, ref_samples):
        raise RuntimeError("Prediction reference mismatch for score sample order/content")

    ref_class_idx = ref["class_idx"].astype(int)
    cur_class_idx = np.asarray(class_idx, dtype=int)
    if not np.array_equal(cur_class_idx, ref_class_idx):
        raise RuntimeError("Prediction reference mismatch for score class labels")

    ref_weights = ref["weight"].astype(float) * LUMI_TOTAL
    cur_weights = np.asarray(weights, dtype=float)
    weight_rtol = float(ref["weight_rtol"])
    weight_atol = float(ref["weight_atol"])
    if not np.allclose(cur_weights, ref_weights, rtol=weight_rtol, atol=weight_atol):
        diff = float(np.max(np.abs(cur_weights - ref_weights)))
        raise RuntimeError(
            "Prediction reference mismatch for score weights: "
            f"max_abs_diff={diff:.6g}, rtol={weight_rtol}, atol={weight_atol}"
        )

    ref_proba = ref["proba"].astype(float)
    cur_proba = np.asarray(proba, dtype=float)
    proba_rtol = float(ref["proba_rtol"])
    proba_atol = float(ref["proba_atol"])
    if cur_proba.shape != ref_proba.shape:
        raise RuntimeError(
            "Prediction reference mismatch for score probabilities shape: "
            f"current={cur_proba.shape}, reference={ref_proba.shape}"
        )
    if not np.allclose(cur_proba, ref_proba, rtol=proba_rtol, atol=proba_atol):
        diff = float(np.max(np.abs(cur_proba - ref_proba)))
        raise RuntimeError(
            "Prediction reference mismatch for score probabilities: "
            f"max_abs_diff={diff:.6g}, rtol={proba_rtol}, atol={proba_atol}"
        )
    log_message(f"Validated score prediction reference: {path}")


# -------------------- Weight assignment --------------------
def _load_test_segments(tree_name, branches, sample_meta):
    parts = []
    for seg in sample_meta["test_segments"]:
        fpath = seg["file"]
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Test split file not found: {fpath}")
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                raise KeyError(f"Tree '{tree_name}' not in {fpath}")
            tree = uf[tree_name]
            avail = set(tree.keys())
            missing = [branch for branch in branches if branch not in avail]
            if missing:
                raise KeyError(
                    f"Missing branches in {fpath}:{tree_name}: "
                    f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                )
            parts.append(
                tree.arrays(
                    branches,
                    library="pd",
                    entry_start=int(seg["entry_start"]),
                    entry_stop=int(seg["entry_stop"]),
                )
            )
    return _concat_parts(parts)


def _assign_test_split_mc_weight(df, sample_name, total_entries, reweight_branches=None):
    reweight_branches = list(reweight_branches or [])
    n_loaded = len(df)
    if reweight_branches:
        missing = [rb for rb in reweight_branches if rb not in df.columns]
        if missing:
            raise KeyError(
                f"Sample '{sample_name}' missing score reweight branches: {', '.join(missing)}"
            )
        raw_w = np.ones(n_loaded, dtype=float)
        for rb in reweight_branches:
            raw_w *= df[rb].to_numpy(dtype=float, copy=False)
        df = df.drop(columns=reweight_branches)
    else:
        raw_w = np.ones(n_loaded, dtype=float)

    info = SAMPLE_INFO[sample_name]
    xsec = float(info.get("xsection", 0.0))
    raw_entries = float(info.get("raw_entries", 0.0))
    if raw_entries <= 0.0:
        raise RuntimeError(f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json")
    if n_loaded == 0 or total_entries == 0 or xsec <= 0.0:
        df["weight"] = 0.0
        return df
    raw_w_sum = float(raw_w.sum())
    if raw_w_sum <= 0.0:
        raise RuntimeError(
            f"Sample '{sample_name}' has non-positive score raw weight sum {raw_w_sum:.6g}"
        )
    target_total = LUMI_TOTAL * xsec * float(total_entries) / raw_entries
    df["weight"] = raw_w * (target_total / raw_w_sum)
    return df


def _add_score_columns(df, proba, class_names):
    out = pd.DataFrame({"weight": df["weight"].to_numpy(dtype=float, copy=False)})
    for idx, class_name in enumerate(class_names):
        out[_score_branch_name(class_name)] = proba[:, idx]
    return out


def _mc_target_total(sample_name, tree_entries_total):
    info        = SAMPLE_INFO[sample_name]
    xsec        = float(info.get("xsection", 0.0))
    raw_entries = float(info.get("raw_entries", 0.0))
    if raw_entries <= 0.0:
        raise RuntimeError(f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json")
    return LUMI_TOTAL * xsec * float(tree_entries_total) / raw_entries


def _raw_weight_array(df, reweight_branches, sample_name):
    n_loaded = len(df)
    if not reweight_branches:
        return np.ones(n_loaded, dtype=float)
    missing = [rb for rb in reweight_branches if rb not in df.columns]
    if missing:
        raise KeyError(
            f"Sample '{sample_name}' missing reweight branches: {', '.join(missing)}"
        )
    raw_w = np.ones(n_loaded, dtype=float)
    for rb in reweight_branches:
        raw_w *= df[rb].to_numpy(dtype=float, copy=False)
    return raw_w


# -------------------- Binning --------------------
def _branch_override(tree_name, branch):
    tree_ov = _tree_plot_cfg(tree_name)
    branches = tree_ov.get("branches", {})
    if isinstance(branches, dict) and branch in branches:
        override = branches.get(branch, {})
        return override if isinstance(override, dict) else {}
    override = tree_ov.get(branch, {})
    return override if isinstance(override, dict) else {}


def _branch_binning_settings(tree_name, branch, log_tf_set):
    override = _branch_override(tree_name, branch)
    bins     = int(override.get("bins", DEFAULT_BINS))
    logx     = bool(override.get("logx", False if branch.startswith("score_") else branch in log_tf_set))
    logy     = bool(override.get("logy", True))
    y_range  = tuple(override["y_range"]) if "y_range" in override else None

    if "x_range" in override:
        x_lo, x_hi = override["x_range"]
        x_range = (float(x_lo), float(x_hi))
    elif branch.startswith("score_"):
        x_range = (0.0, 1.0)
    else:
        x_range = None
    return bins, x_range, logx, logy, y_range


def _auto_range(arrs, logx):
    state = {"lo": None, "hi": None, "has_missing": False}
    for arr in arrs:
        if arr is None:
            continue
        _update_range_state_for_array(state, arr, logx)
    if state["lo"] is None and not state["has_missing"]:
        return None
    return state["lo"], state["hi"], state["has_missing"]


def _make_range_state(branches):
    return {branch: {"lo": None, "hi": None, "has_missing": False} for branch in branches}


def _update_range_state_for_array(state, arr, logx):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return
    missing = _missing_value_mask(a)
    if missing.any():
        state["has_missing"] = True
    valid = ~missing
    if logx:
        valid &= a > 0
    a = a[valid]
    if a.size == 0:
        return
    lo = float(a.min())
    hi = float(a.max())
    state["lo"] = lo if state["lo"] is None else min(state["lo"], lo)
    state["hi"] = hi if state["hi"] is None else max(state["hi"], hi)


def _update_range_state_from_df(range_state, df, branches, log_tf_set, tree_name):
    if df is None or len(df) == 0:
        return
    for branch in branches:
        if branch not in df.columns:
            continue
        _, _, logx, _, _ = _branch_binning_settings(tree_name, branch, log_tf_set)
        _update_range_state_for_array(range_state[branch], df[branch].to_numpy(copy=False), logx)


def _regular_bin_edges(bins, x_range, logx):
    lo, hi = x_range
    if logx:
        if lo <= 0:
            lo = 1e-9
        return np.logspace(math.log10(lo), math.log10(hi), bins + 1)
    return np.linspace(lo, hi, bins + 1)


def _plot_edges_with_missing_bin(edges, logx):
    if len(edges) < 2:
        raise ValueError("Cannot build missing-value bin without at least one regular bin")
    first_width = edges[1] - edges[0]
    if logx:
        ratio = edges[1] / edges[0] if edges[0] > 0 else 10.0
        if not np.isfinite(ratio) or ratio <= 1.0:
            ratio = 10.0
        left = edges[0] / ratio
    else:
        left = edges[0] - first_width
    return np.concatenate([[left], edges])


def _build_binning(bins, x_range, logx, logy, y_range, has_missing):
    lo, hi = x_range
    if lo >= hi:
        hi = lo + 1.0
    regular_edges = _regular_bin_edges(bins, (lo, hi), logx)
    if has_missing:
        edges = _plot_edges_with_missing_bin(regular_edges, logx)
        regular_offset = 1
        x_range = (float(edges[0]), float(edges[-1]))
    else:
        edges = regular_edges
        regular_offset = 0
        x_range = (float(regular_edges[0]), float(regular_edges[-1]))
    return {
        "bins": int(bins),
        "edges": edges,
        "regular_edges": regular_edges,
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "bin_widths": edges[1:] - edges[:-1],
        "x_range": x_range,
        "logx": logx,
        "logy": logy,
        "y_range": y_range,
        "has_missing_bin": bool(has_missing),
        "regular_offset": regular_offset,
        "missing_center": 0.5 * (edges[0] + edges[1]) if has_missing else None,
        "missing_separator": float(edges[1]) if has_missing else None,
        "regular_x_range": (float(regular_edges[0]), float(regular_edges[-1])),
    }


def _resolve_binning(tree_name, branch, arrs, log_tf_set):
    bins, x_range, logx, logy, y_range = _branch_binning_settings(tree_name, branch, log_tf_set)
    scanned = _auto_range(arrs, logx)
    has_missing = scanned[2] if scanned is not None else False
    if x_range is None:
        if scanned is None or scanned[0] is None:
            return None
        x_range = (scanned[0], scanned[1])
    return _build_binning(bins, x_range, logx, logy, y_range, has_missing)


def _resolve_binning_from_state(tree_name, branch, range_state, log_tf_set):
    bins, x_range, logx, logy, y_range = _branch_binning_settings(tree_name, branch, log_tf_set)
    state = range_state.get(branch, {})
    has_missing = bool(state.get("has_missing", False))
    if x_range is None:
        if state.get("lo") is None:
            return None
        x_range = (state["lo"], state["hi"])
    return _build_binning(bins, x_range, logx, logy, y_range, has_missing)


def _weighted_hist(vals, weights, binning):
    v = np.asarray(vals,    dtype=float)
    w = np.asarray(weights, dtype=float)
    edges = binning["edges"]
    h = np.zeros(len(edges) - 1, dtype=float)
    h2 = np.zeros(len(edges) - 1, dtype=float)

    missing = _missing_value_mask(v)
    if binning["has_missing_bin"] and missing.any():
        mw = w[missing]
        h[0] = float(np.sum(mw))
        h2[0] = float(np.sum(mw * mw))

    valid = ~missing
    if binning["logx"]:
        valid &= v > 0
    v = v[valid]
    w = w[valid]
    if v.size:
        rh,  _ = np.histogram(v, bins=binning["regular_edges"], weights=w)
        rh2, _ = np.histogram(v, bins=binning["regular_edges"], weights=w * w)
        offset = int(binning["regular_offset"])
        h[offset:offset + len(rh)] += rh.astype(float)
        h2[offset:offset + len(rh2)] += rh2.astype(float)
    return h, h2


def _prepare_ordinary_chunk(df, thresholds, clip_ranges):
    if df is None or len(df) == 0:
        return df
    df = _apply_thresholds(df, thresholds)
    df = _apply_clip(df, clip_ranges)
    return df


def _prepare_ordinary_chunk_with_weights(df, weights, thresholds, clip_ranges):
    if df is None or len(df) == 0:
        return df, weights
    if thresholds:
        mask = _threshold_mask(df, thresholds)
        keep = mask.to_numpy(dtype=bool, copy=False)
        df = df.loc[mask].reset_index(drop=True)
        weights = weights[keep]
    df = _apply_clip(df, clip_ranges)
    return df, weights


def _scan_ordinary_inputs(
    *,
    tree_name,
    root_plot_branches,
    log_tf_set,
    thresholds,
    clip_ranges,
    mc_sources,
    data_sources,
    mc_need_load,
    data_need_load,
    reweight_branches,
):
    range_state = _make_range_state(root_plot_branches)
    mc_weight_scales = {}

    log_message(f"Streaming MC range scan for {len(mc_sources)} samples")
    for source in mc_sources:
        sample_name = source["sample"]
        raw_w_sum = 0.0
        n_loaded = 0
        n_after = 0
        for df in _iter_tree_chunks(source["files"], tree_name, mc_need_load):
            raw_w = _raw_weight_array(df, reweight_branches, sample_name)
            raw_w_sum += float(raw_w.sum())
            n_loaded += len(df)
            df = _prepare_ordinary_chunk(df, thresholds, clip_ranges)
            n_after += 0 if df is None else len(df)
            _update_range_state_from_df(range_state, df, root_plot_branches, log_tf_set, tree_name)
            del df, raw_w
        if n_loaded == 0 or source["entries"] == 0:
            mc_weight_scales[sample_name] = 0.0
        elif raw_w_sum <= 0.0:
            raise RuntimeError(
                f"Sample '{sample_name}' has non-positive raw weight sum {raw_w_sum:.6g}"
            )
        else:
            mc_weight_scales[sample_name] = _mc_target_total(sample_name, source["entries"]) / raw_w_sum
        log_message(
            f"  {sample_name}: class={source['class']}, tree_entries={source['entries']}, "
            f"loaded={n_loaded}, after_filter={n_after}, "
            f"weight_sum={raw_w_sum * mc_weight_scales[sample_name]:.6g}"
        )
        gc.collect()

    log_message(f"Streaming data range scan for {len(data_sources)} samples")
    for source in data_sources:
        n_loaded = 0
        n_after = 0
        for df in _iter_tree_chunks(source["files"], tree_name, data_need_load):
            n_loaded += len(df)
            df = _prepare_ordinary_chunk(df, thresholds, clip_ranges)
            n_after += 0 if df is None else len(df)
            _update_range_state_from_df(range_state, df, root_plot_branches, log_tf_set, tree_name)
            del df
        if n_loaded == 0:
            log_message(f"  [WARN] data sample '{source['sample']}' has zero entries in tree '{tree_name}'")
        log_message(f"  data {source['sample']}: loaded={n_loaded}, after_filter={n_after}")
        gc.collect()

    return range_state, mc_weight_scales


def _book_ordinary_histograms(class_names, binnings):
    mc_hists = {}
    data_hists = {}
    for branch, binning in binnings.items():
        n_bins = len(binning["edges"]) - 1
        mc_hists[branch] = {
            cls: (np.zeros(n_bins, dtype=float), np.zeros(n_bins, dtype=float))
            for cls in class_names
        }
        data_hists[branch] = (np.zeros(n_bins, dtype=float), np.zeros(n_bins, dtype=float))
    return mc_hists, data_hists


def _fill_hist_pair(target, vals, weights, binning):
    h, h2 = _weighted_hist(vals, weights, binning)
    target[0][:] += h
    target[1][:] += h2


def _fill_ordinary_histograms(
    *,
    tree_name,
    root_plot_branches,
    thresholds,
    clip_ranges,
    mc_sources,
    data_sources,
    mc_need_load,
    data_need_load,
    reweight_branches,
    mc_weight_scales,
    binnings,
    mc_hists,
    data_hists,
):
    if not binnings:
        return

    log_message(f"Streaming MC histogram fill for {len(mc_sources)} samples")
    for source in mc_sources:
        sample_name = source["sample"]
        cls_name = source["class"]
        scale = float(mc_weight_scales.get(sample_name, 0.0))
        n_after = 0
        for df in _iter_tree_chunks(source["files"], tree_name, mc_need_load):
            raw_w = _raw_weight_array(df, reweight_branches, sample_name) * scale
            df, raw_w = _prepare_ordinary_chunk_with_weights(df, raw_w, thresholds, clip_ranges)
            if df is None or len(df) == 0:
                del df, raw_w
                continue
            n_after += len(df)
            for branch in root_plot_branches:
                if branch not in binnings or branch not in df.columns:
                    continue
                _fill_hist_pair(
                    mc_hists[branch][cls_name],
                    df[branch].to_numpy(copy=False),
                    raw_w,
                    binnings[branch],
                )
            del df, raw_w
        log_message(f"  filled MC {sample_name}: class={cls_name}, after_filter={n_after}")
        gc.collect()

    log_message(f"Streaming data histogram fill for {len(data_sources)} samples")
    for source in data_sources:
        n_after = 0
        for df in _iter_tree_chunks(source["files"], tree_name, data_need_load):
            df = _prepare_ordinary_chunk(df, thresholds, clip_ranges)
            if df is None or len(df) == 0:
                del df
                continue
            weights = np.ones(len(df), dtype=float)
            n_after += len(df)
            for branch in root_plot_branches:
                if branch not in binnings or branch not in df.columns:
                    continue
                _fill_hist_pair(
                    data_hists[branch],
                    df[branch].to_numpy(copy=False),
                    weights,
                    binnings[branch],
                )
            del df, weights
        log_message(f"  filled data {source['sample']}: after_filter={n_after}")
        gc.collect()


# -------------------- Ratio --------------------
def _ratio_data_over_mc(data_vals, data_vars, mc_vals, mc_vars):
    with np.errstate(divide="ignore", invalid="ignore"):
        r  = np.where(mc_vals > 0, data_vals / mc_vals, np.nan)
        data_sigma = np.where(mc_vals > 0, np.sqrt(np.maximum(data_vars, 0.0)) / mc_vals, np.nan)
        mc_sigma = np.where(mc_vals > 0, np.sqrt(np.maximum(mc_vars, 0.0)) / mc_vals, np.nan)
    return r, data_sigma, mc_sigma


def _smallest_stack_component_peak(mc_per_cls):
    peaks = []
    for h, _ in mc_per_cls.values():
        positive = np.asarray(h, dtype=float)
        positive = positive[positive > 0]
        if positive.size:
            peaks.append(float(np.max(positive)))
    return min(peaks) if peaks else None


def _unit_normalized_histograms(mc_per_cls, mc_total_v, mc_total_w2, data_v, data_w2):
    mc_sum = float(np.sum(mc_total_v))
    data_sum = float(np.sum(data_v))
    mc_scale = 1.0 / mc_sum if mc_sum > 0.0 else 0.0
    data_scale = 1.0 / data_sum if data_sum > 0.0 else 0.0

    mc_per_cls_norm = {
        cls: (h * mc_scale, h2 * (mc_scale ** 2))
        for cls, (h, h2) in mc_per_cls.items()
    }
    return (
        mc_per_cls_norm,
        mc_total_v * mc_scale,
        mc_total_w2 * (mc_scale ** 2),
        data_v * data_scale,
        data_w2 * (data_scale ** 2),
    )


def _draw_data_mc_plot(
    *,
    class_names,
    color_map,
    edges,
    bin_centers,
    bin_widths,
    mc_per_cls,
    mc_total_v,
    mc_total_w2,
    data_v,
    data_w2,
    branch,
    x_range,
    logx,
    logy,
    y_range,
    y_label,
    out_path,
    logy_floor=0.1,
    missing_center=None,
    missing_separator=None,
    regular_x_range=None,
):
    bins = len(bin_centers)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    mc_yields = {cls: float(mc_per_cls[cls][0].sum()) for cls in class_names}
    order = np.argsort([mc_yields[c] for c in class_names])
    bottom = np.zeros(bins)
    for idx in order:
        cls = class_names[idx]
        h, _ = mc_per_cls[cls]
        ax.bar(
            edges[:-1], h, width=bin_widths, bottom=bottom,
            align="edge", color=color_map[cls], edgecolor="none",
            linewidth=0, antialiased=False, alpha=0.9, label=cls,
        )
        bottom += h
    ax.margins(x=0)

    mc_sigma = np.sqrt(np.maximum(mc_total_w2, 0.0))
    lower = np.clip(mc_total_v - mc_sigma, 1e-12, None)
    upper = np.clip(mc_total_v + mc_sigma, 1e-12, None)
    ax.fill_between(
        bin_centers, lower, upper, step="mid",
        facecolor="none", edgecolor="gray", hatch="///", linewidth=0,
    )

    data_sigma = np.sqrt(np.maximum(data_w2, 0.0))
    y_plot = np.where(data_v > 0, data_v, np.nan)
    ax.errorbar(
        bin_centers, y_plot, yerr=data_sigma,
        fmt="o", ms=7.6, color="black", mfc="black", mec="black",
        elinewidth=1.5, capsize=0, label="Data",
    )

    if logx:
        ax.set_xscale("log")
        axr.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(*x_range)
    axr.set_xlim(*x_range)
    if missing_separator is not None:
        ax.axvline(missing_separator, color="black", linestyle="-", linewidth=1.5)
        axr.axvline(missing_separator, color="black", linestyle="-", linewidth=1.5)

    if y_range is not None:
        ax.set_ylim(*y_range)
    else:
        vis = (mc_total_v > 0) | (data_v > 0)
        if np.any(vis):
            ymax = max(float(np.max(mc_total_v[vis])), float(np.max(data_v[vis])))
        else:
            ymax = 1.0
        if logy:
            if logy_floor is None:
                positive = np.concatenate([mc_total_v[mc_total_v > 0], data_v[data_v > 0]])
                ymin = max(float(np.min(positive)) / 5.0, 1e-12) if positive.size else 1e-6
                smallest_component_peak = _smallest_stack_component_peak(mc_per_cls)
                if smallest_component_peak is not None:
                    ymin = min(ymin, smallest_component_peak / 5.0)
            else:
                ymin = float(logy_floor)
            ax.set_ylim(ymin, max(ymin * 10.0, ymax * 5.0))
        else:
            ax.set_ylim(0.0, max(1.0, ymax * 1.3))

    ax.set_ylabel(y_label, fontsize=24)
    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=int(LUMI_TOTAL), ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if "Data" in labels:
        i = labels.index("Data")
        handles.append(handles.pop(i))
        labels.append(labels.pop(i))
    ax.legend(handles, labels, loc="best", fontsize=17, frameon=False, ncol=2)

    ratio, data_r_err, mc_r_err = _ratio_data_over_mc(data_v, data_w2, mc_total_v, mc_total_w2)
    mc_band_low = 1.0 - mc_r_err
    mc_band_high = 1.0 + mc_r_err
    axr.fill_between(
        bin_centers, mc_band_low, mc_band_high, step="mid",
        color="gray", alpha=0.35, linewidth=0,
    )
    axr.errorbar(
        bin_centers, ratio, yerr=data_r_err,
        fmt="o", ms=7.6, color="black", mfc="black", mec="black",
        elinewidth=1.5, capsize=0,
    )
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)

    finite = np.isfinite(ratio)
    band_finite = np.isfinite(mc_band_low) & np.isfinite(mc_band_high)
    if np.any(finite) or np.any(band_finite):
        ratio_high = ratio[finite] + np.nan_to_num(data_r_err[finite], nan=0.0)
        ratio_low = ratio[finite] - np.nan_to_num(data_r_err[finite], nan=0.0)
        high_values = [ratio_high, mc_band_high[band_finite]]
        low_values = [ratio_low, mc_band_low[band_finite]]
        high_values = [values for values in high_values if values.size]
        low_values = [values for values in low_values if values.size]
        rmax = float(np.nanmax(np.concatenate(high_values))) if high_values else 1.0
        rmin = float(np.nanmin(np.concatenate(low_values))) if low_values else 0.0
        if not np.isfinite(rmax) or rmax <= 0:
            rmax = 1.0
        if rmax < 5.0:
            axr.set_ylim(max(0.0, 0.8 * rmin), 1.2 * rmax)
        else:
            axr.set_ylim(0.0, 5.0)
    else:
        axr.set_ylim(0.0, 2.0)

    axr.set_ylabel(r"$\frac{Data}{MC}$", fontsize=26)
    axr.yaxis.set_label_coords(-0.05, 0.6)
    axr.set_xlabel(branch, fontsize=24)
    if missing_center is not None and regular_x_range is not None:
        base_formatter = axr.xaxis.get_major_formatter()
        lo, hi = regular_x_range
        ticks = np.asarray(axr.get_xticks(), dtype=float)
        ticks = ticks[np.isfinite(ticks) & (ticks >= lo) & (ticks <= hi)]
        ticks = np.concatenate([[float(missing_center)], ticks])

        def _fmt_tick(value, pos):
            if np.isclose(value, missing_center):
                return "-99"
            return base_formatter(value, pos)

        axr.xaxis.set_major_locator(FixedLocator(ticks))
        axr.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------- Per-tree processing --------------------
def _process_tree(tree_name):
    log_message(f"Running data_mc.py: tree={tree_name}")

    log_message("Loading trained-model config copies")
    bdt_cfg, bdt_br, bdt_sel, test_meta = _bdt_configs_for_tree(
        tree_name,
        load_test_ranges=PLOT_BDT_SCORES,
    )
    class_groups = (
        bdt_cfg["class_groups"]
        if PLOT_BDT_SCORES
        else plot_cfg.get("class_groups", bdt_cfg["class_groups"])
    )
    class_names      = list(class_groups.keys())
    model_branches   = [item["name"] for item in bdt_br[tree_name]] if PLOT_BDT_SCORES else []
    score_branches   = [_score_branch_name(class_name) for class_name in class_names] if PLOT_BDT_SCORES else []

    # Resolve input_root relative to the BDT script directory used by train.py.
    bdt_root_dir   = _bdt_root_for_tree(tree_name)
    bdt_script_dir = os.path.dirname(bdt_root_dir)
    input_root     = _resolve(bdt_cfg["input_root"], bdt_script_dir)
    input_pattern  = bdt_cfg["input_pattern"]

    sel         = bdt_sel.get(tree_name, {})
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    log_tf_set  = set(sel.get("log_transform", []))

    skip_score = _skip_branches_for_tree(tree_name)
    branches_to_plot = _plot_branches_for_tree(tree_name)
    score_branches = [branch for branch in score_branches if branch not in skip_score]
    for branch in score_branches:
        if branch not in branches_to_plot:
            branches_to_plot.append(branch)
    root_plot_branches = [branch for branch in branches_to_plot if branch not in score_branches]
    need_load        = sorted(set(root_plot_branches)
                              | set(thresholds.keys())
                              | set(clip_ranges.keys()))
    reweight_cfg      = plot_cfg.get("event_reweight_branches", {})
    reweight_branches = list(reweight_cfg.get(tree_name, []))
    mc_need_load     = sorted(set(need_load) | set(reweight_branches))
    score_reweight_branches = list(bdt_cfg.get(tree_name, {}).get("event_reweight_branches", []))
    log_message(
        f"Resolved plotting config: branches={len(branches_to_plot)}, "
        f"threshold_branches={len(thresholds)}, clip_branches={len(clip_ranges)}, "
        f"reweight_branches={len(reweight_branches)}, score_branches={len(score_branches)}"
    )
    log_message("Ordinary MC branch entry cap per sample: none")

    out_dir = _resolve(OUTPUT_ROOT_PATT.format(tree_name=tree_name), _SCRIPT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    log_message(f"Output directory: {out_dir}")

    # Ordinary ROOT branches are streamed into histograms instead of held as
    # per-sample DataFrames, which keeps large data samples bounded by one
    # uproot chunk plus the small histogram arrays.
    mc_sources = []
    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = _input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            n_total = _tree_entries_total(files, tree_name)
            if n_total <= 0:
                raise RuntimeError(f"Empty tree '{tree_name}' for MC sample '{sname}'")
            mc_sources.append({
                "class": cls_name,
                "sample": sname,
                "files": files,
                "entries": n_total,
            })
    if not mc_sources and root_plot_branches:
        raise RuntimeError(f"No MC samples available for ordinary plots in tree '{tree_name}'")

    data_sources = []
    for sname in DATA_SAMPLES:
        files = _input_files(sname, input_root, input_pattern)
        if not files:
            raise RuntimeError(f"No ROOT files found for data sample '{sname}'")
        data_sources.append({
            "sample": sname,
            "files": files,
        })

    ordinary_binnings = {}
    ordinary_mc_hists = {}
    ordinary_data_hists = {}
    if root_plot_branches:
        range_state, mc_weight_scales = _scan_ordinary_inputs(
            tree_name=tree_name,
            root_plot_branches=root_plot_branches,
            log_tf_set=log_tf_set,
            thresholds=thresholds,
            clip_ranges=clip_ranges,
            mc_sources=mc_sources,
            data_sources=data_sources,
            mc_need_load=mc_need_load,
            data_need_load=need_load,
            reweight_branches=reweight_branches,
        )
        for branch in root_plot_branches:
            binning = _resolve_binning_from_state(tree_name, branch, range_state, log_tf_set)
            if binning is None:
                log_message(f"  [WARN] no data for {tree_name}:{branch}, skipping")
                continue
            ordinary_binnings[branch] = binning
        ordinary_mc_hists, ordinary_data_hists = _book_ordinary_histograms(
            class_names,
            ordinary_binnings,
        )
        _fill_ordinary_histograms(
            tree_name=tree_name,
            root_plot_branches=root_plot_branches,
            thresholds=thresholds,
            clip_ranges=clip_ranges,
            mc_sources=mc_sources,
            data_sources=data_sources,
            mc_need_load=mc_need_load,
            data_need_load=need_load,
            reweight_branches=reweight_branches,
            mc_weight_scales=mc_weight_scales,
            binnings=ordinary_binnings,
            mc_hists=ordinary_mc_hists,
            data_hists=ordinary_data_hists,
        )

    # Build derived model score branches. MC scores use the saved test split;
    # data scores use the full configured data input, matching ordinary plots.
    score_class_dfs = {}
    score_binnings = {}
    score_data_hists = {}
    if score_branches:
        log_message("Preparing model score branches")
        clf = _load_score_model(bdt_root_dir, bdt_cfg, tree_name)
        decorrelate = list(bdt_cfg.get(tree_name, {}).get("decorrelate", []))
        score_load = sorted(set(model_branches) | set(thresholds.keys()) | set(score_reweight_branches))
        sample_to_class_name = {}
        sample_to_class_idx = {}
        for idx, (cls_name, samples) in enumerate(class_groups.items()):
            for sample_name in samples:
                sample_to_class_name[sample_name] = cls_name
                sample_to_class_idx[sample_name] = idx

        score_parts_by_class = {cls_name: [] for cls_name in class_names}
        ref_sample_labels = [] if VALIDATE_BDT_TEST_SCORES else None
        ref_class_idx = [] if VALIDATE_BDT_TEST_SCORES else None
        ref_weights = [] if VALIDATE_BDT_TEST_SCORES else None
        ref_proba_parts = [] if VALIDATE_BDT_TEST_SCORES else None
        ref_feature_names = None

        log_message(f"Loading MC score test split samples: n={len(test_meta['samples'])}")
        for sample_name, sample_meta in test_meta["samples"].items():
            if sample_name not in sample_to_class_name:
                raise RuntimeError(f"Test split sample '{sample_name}' is not in class_groups")
            df = _load_test_segments(tree_name, score_load, sample_meta)
            if df is None or len(df) == 0:
                raise RuntimeError(f"No test split events loaded for sample '{sample_name}'")
            df = _assign_test_split_mc_weight(
                df,
                sample_name,
                int(sample_meta["total_entries"]),
                score_reweight_branches,
            )
            mask = _threshold_mask(df, thresholds)
            df = df.loc[mask].reset_index(drop=True)
            if len(df) == 0:
                log_message(f"  [WARN] score sample '{sample_name}' has zero events after filtering")
                continue
            X_model = _standardize_model_X(df[model_branches].copy(), clip_ranges, list(log_tf_set))
            X_model = _drop_decorrelated_features(X_model, decorrelate)
            proba = _predict_model_proba(clf, X_model, len(class_names))
            if VALIDATE_BDT_TEST_SCORES and ref_feature_names is None:
                ref_feature_names = list(X_model.columns)
            score_df = _add_score_columns(df, proba, class_names)
            cls_name = sample_to_class_name[sample_name]
            score_parts_by_class[cls_name].append(score_df)
            if VALIDATE_BDT_TEST_SCORES:
                ref_sample_labels.extend([sample_name] * len(df))
                ref_class_idx.extend([sample_to_class_idx[sample_name]] * len(df))
                ref_weights.extend(score_df["weight"].to_numpy(dtype=float, copy=False))
                ref_proba_parts.append(proba)
            log_message(
                f"  score {sample_name}: class={cls_name}, test_loaded={len(df)}, "
                f"weight_sum={float(score_df['weight'].sum()):.6g}"
            )

        for cls_name, parts in score_parts_by_class.items():
            if parts:
                score_class_dfs[cls_name] = _concat_parts(parts)

        if not score_class_dfs:
            raise RuntimeError(f"No MC score events after filtering for tree '{tree_name}'")
        if VALIDATE_BDT_TEST_SCORES:
            score_proba_ref = np.concatenate(ref_proba_parts, axis=0)
            _compare_score_reference(
                os.path.join(bdt_root_dir, "test_reference_signal_region.npz"),
                ref_feature_names,
                ref_sample_labels,
                ref_class_idx,
                ref_weights,
                score_proba_ref,
            )
            del ref_sample_labels, ref_class_idx, ref_weights, ref_proba_parts, score_proba_ref
        else:
            log_message("Skipping score prediction reference validation")
        gc.collect()

        for branch in score_branches:
            binning = _resolve_binning(tree_name, branch, [], log_tf_set)
            if binning is None:
                log_message(f"  [WARN] no score binning for {tree_name}:{branch}, skipping")
                continue
            score_binnings[branch] = binning
            n_bins = len(binning["edges"]) - 1
            score_data_hists[branch] = (
                np.zeros(n_bins, dtype=float),
                np.zeros(n_bins, dtype=float),
            )

        if DATA_SAMPLES:
            score_data_load = sorted(set(model_branches) | set(thresholds.keys()))
            log_message(f"Loading data score samples: n={len(DATA_SAMPLES)}")
            total_score_data = 0
            for sname in DATA_SAMPLES:
                files = _input_files(sname, input_root, input_pattern)
                if not files:
                    raise RuntimeError(f"No ROOT files found for data score sample '{sname}'")
                sample_score_data = 0
                loaded_any = False
                for df in _iter_tree_chunks(files, tree_name, score_data_load):
                    loaded_any = True
                    df = _apply_thresholds(df, thresholds)
                    if df is None or len(df) == 0:
                        continue
                    weights = np.ones(len(df), dtype=float)
                    X_model = _standardize_model_X(df[model_branches].copy(), clip_ranges, list(log_tf_set))
                    X_model = _drop_decorrelated_features(X_model, decorrelate)
                    proba = _predict_model_proba(clf, X_model, len(class_names))
                    score_df = _add_score_columns(df, proba, class_names)
                    for branch in score_branches:
                        if branch not in score_binnings:
                            continue
                        _fill_hist_pair(
                            score_data_hists[branch],
                            score_df[branch].to_numpy(copy=False),
                            weights,
                            score_binnings[branch],
                        )
                    sample_score_data += len(df)
                    del df, weights, X_model, proba, score_df
                if not loaded_any:
                    log_message(f"  [WARN] data score sample '{sname}' has zero entries")
                    continue
                if sample_score_data == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero events after filtering")
                    continue
                total_score_data += sample_score_data
                log_message(f"  data score {sname}: events={sample_score_data}")
                gc.collect()
            log_message(f"Loaded data score events: {total_score_data}")

    # Plot each requested branch.
    log_message(f"Plotting branches: total={len(branches_to_plot)}")
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}

    for idx, branch in enumerate(branches_to_plot, start=1):
        log_message(f"Plotting branch {idx}/{len(branches_to_plot)}: {branch}")
        is_score_branch = branch in score_branches
        if is_score_branch:
            if branch not in score_binnings:
                continue
            plot_class_dfs = score_class_dfs
            binning = score_binnings[branch]

            mc_total_v  = np.zeros(len(binning["edges"]) - 1)
            mc_total_w2 = np.zeros(len(binning["edges"]) - 1)
            mc_per_cls  = {}
            for cls in class_names:
                if cls in plot_class_dfs and branch in plot_class_dfs[cls].columns:
                    h, h2 = _weighted_hist(
                        plot_class_dfs[cls][branch].values,
                        plot_class_dfs[cls]["weight"].values,
                        binning,
                    )
                else:
                    h  = np.zeros(len(binning["edges"]) - 1)
                    h2 = np.zeros(len(binning["edges"]) - 1)
                mc_per_cls[cls] = (h, h2)
                mc_total_v  += h
                mc_total_w2 += h2

            data_v, data_w2 = score_data_hists[branch]
        else:
            if branch not in ordinary_binnings:
                continue
            binning = ordinary_binnings[branch]
            mc_per_cls = ordinary_mc_hists[branch]
            mc_total_v = np.zeros(len(binning["edges"]) - 1)
            mc_total_w2 = np.zeros(len(binning["edges"]) - 1)
            for cls in class_names:
                h, h2 = mc_per_cls[cls]
                mc_total_v += h
                mc_total_w2 += h2
            data_v, data_w2 = ordinary_data_hists[branch]

        out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
        _draw_data_mc_plot(
            class_names=class_names,
            color_map=color_map,
            edges=binning["edges"],
            bin_centers=binning["bin_centers"],
            bin_widths=binning["bin_widths"],
            mc_per_cls=mc_per_cls,
            mc_total_v=mc_total_v,
            mc_total_w2=mc_total_w2,
            data_v=data_v,
            data_w2=data_w2,
            branch=branch,
            x_range=binning["x_range"],
            logx=binning["logx"],
            logy=binning["logy"],
            y_range=binning["y_range"],
            y_label="Events",
            out_path=out_path,
            logy_floor=0.1,
            missing_center=binning["missing_center"],
            missing_separator=binning["missing_separator"],
            regular_x_range=binning["regular_x_range"],
        )
        log_message(f"Wrote plot file: {out_path}")

        (
            mc_per_cls_norm,
            mc_total_v_norm,
            mc_total_w2_norm,
            data_v_norm,
            data_w2_norm,
        ) = _unit_normalized_histograms(mc_per_cls, mc_total_v, mc_total_w2, data_v, data_w2)
        out_path_normal = os.path.join(out_dir, f"{tree_name}_{branch}_normal.pdf")
        _draw_data_mc_plot(
            class_names=class_names,
            color_map=color_map,
            edges=binning["edges"],
            bin_centers=binning["bin_centers"],
            bin_widths=binning["bin_widths"],
            mc_per_cls=mc_per_cls_norm,
            mc_total_v=mc_total_v_norm,
            mc_total_w2=mc_total_w2_norm,
            data_v=data_v_norm,
            data_w2=data_w2_norm,
            branch=branch,
            x_range=binning["x_range"],
            logx=binning["logx"],
            logy=binning["logy"],
            y_range=binning["y_range"],
            y_label="A.U.",
            out_path=out_path_normal,
            logy_floor=None,
            missing_center=binning["missing_center"],
            missing_separator=binning["missing_separator"],
            regular_x_range=binning["regular_x_range"],
        )
        log_message(f"Wrote plot file: {out_path_normal}")
    log_message(f"Finished data_mc.py for tree={tree_name}")


def main():
    log_message(
        f"Running data_mc.py: trees={','.join(SUBMIT_TREES)}, "
        f"bdt_root={BDT_ROOT_PATT}, output_root={OUTPUT_ROOT_PATT}"
    )
    for tree_name in SUBMIT_TREES:
        _process_tree(tree_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
