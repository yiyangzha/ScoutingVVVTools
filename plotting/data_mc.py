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


def _load_tree(files, tree_name, branches, max_entries=None):
    parts = []
    remaining = None
    if max_entries is not None:
        remaining = max(0, int(max_entries))
        if remaining == 0:
            return None
    for fpath in files:
        if remaining is not None and remaining <= 0:
            break
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree  = uf[tree_name]
            avail = set(tree.keys())
            missing = [b for b in branches if b not in avail]
            if missing:
                raise KeyError(
                    f"Missing branches in {fpath}:{tree_name}: "
                    f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                )
            entry_stop = int(tree.num_entries)
            if remaining is not None:
                entry_stop = min(entry_stop, remaining)
            if entry_stop <= 0:
                continue
            df_part = tree.arrays(
                branches,
                library="pd",
                entry_start=0,
                entry_stop=entry_stop,
            )
            parts.append(df_part)
            if remaining is not None:
                remaining -= len(df_part)
    return _concat_parts(parts)


# -------------------- Threshold and clip filtering --------------------
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
            continue
        col = df[b]
        sentinel = col < -990
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
        valid = arr >= -990
        lo, hi = rng
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        df[col] = arr
    return df


def _drop_unneeded_columns(df, keep_columns):
    if df is None or len(df) == 0:
        return df
    keep = set(keep_columns)
    drop_cols = [col for col in df.columns if col not in keep]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        gc.collect()
    return df


def _standardize_model_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        sentinel = arr < -990
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
def _assign_mc_weight(df, sample_name, tree_entries_total, n_loaded, reweight_branches=None):
    """Assign per-event weight for an MC sample.

    Per event:
        raw_w  = product of reweight_branches (1.0 if empty)
        target_total = lumi_total * xsection * tree_entries_total / raw_entries
        weight = raw_w * target_total / sum(raw_w_loaded)

    So the sample's total weight sums to ``target_total`` regardless of raw_w's
    magnitude; raw_w only shapes the per-event distribution inside the sample.

    Reweight branches are read on raw values (before clip/log/threshold) and
    dropped from ``df`` once raw_w is computed. Computed before any filtering;
    the weights are unchanged afterwards.
    """
    reweight_branches = list(reweight_branches or [])
    if reweight_branches:
        missing = [rb for rb in reweight_branches if rb not in df.columns]
        if missing:
            raise KeyError(
                f"Sample '{sample_name}' missing reweight branches: {', '.join(missing)}"
            )
        raw_w = np.ones(n_loaded, dtype=float)
        for rb in reweight_branches:
            raw_w *= df[rb].to_numpy(dtype=float, copy=False)
        df = df.drop(columns=reweight_branches)
    else:
        raw_w = np.ones(n_loaded, dtype=float)

    info        = SAMPLE_INFO[sample_name]
    xsec        = float(info.get("xsection", 0.0))
    raw_entries = float(info.get("raw_entries", 0.0))
    if raw_entries <= 0.0:
        raise RuntimeError(f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json")
    if n_loaded == 0 or tree_entries_total == 0:
        df["weight"] = 0.0
        return df
    target_total = LUMI_TOTAL * xsec * float(tree_entries_total) / raw_entries
    raw_w_sum = float(raw_w.sum())
    if raw_w_sum <= 0.0:
        raise RuntimeError(
            f"Sample '{sample_name}' has non-positive raw weight sum {raw_w_sum:.6g}"
        )
    df["weight"] = raw_w * (target_total / raw_w_sum)
    return df


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


# -------------------- Binning --------------------
def _branch_override(tree_name, branch):
    tree_ov = _tree_plot_cfg(tree_name)
    branches = tree_ov.get("branches", {})
    if isinstance(branches, dict) and branch in branches:
        override = branches.get(branch, {})
        return override if isinstance(override, dict) else {}
    override = tree_ov.get(branch, {})
    return override if isinstance(override, dict) else {}


def _auto_range(arrs, logx):
    mins, maxs = [], []
    for arr in arrs:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        valid = a[a >= -990]
        if logx:
            valid = valid[valid > 0]
        if valid.size == 0:
            continue
        mins.append(float(valid.min()))
        maxs.append(float(valid.max()))
    if not mins:
        return None
    lo, hi = min(mins), max(maxs)
    if lo >= hi:
        hi = lo + 1.0
    return lo, hi


def _resolve_binning(tree_name, branch, arrs, log_tf_set):
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
        x_range = _auto_range(arrs, logx)
        if x_range is None:
            return None
    return bins, x_range, logx, logy, y_range


def _bin_edges(bins, x_range, logx):
    lo, hi = x_range
    if logx:
        if lo <= 0:
            lo = 1e-9
        return np.logspace(math.log10(lo), math.log10(hi), bins + 1)
    return np.linspace(lo, hi, bins + 1)


def _weighted_hist(vals, weights, edges):
    v = np.asarray(vals,    dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = v >= -990
    v = v[valid]
    w = w[valid]
    h,  _ = np.histogram(v, bins=edges, weights=w)
    h2, _ = np.histogram(v, bins=edges, weights=w * w)
    return h.astype(float), h2.astype(float)


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

    # Load the MC events for each class.
    log_message(f"Loading MC samples for {len(class_names)} classes")
    class_dfs = {}
    for cls_name, samples in class_groups.items():
        log_message(f"  Loading class '{cls_name}' with {len(samples)} samples")
        dfs = []
        for sname in samples:
            if sname not in SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = _input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            n_total = _tree_entries_total(files, tree_name)
            if n_total <= 0:
                raise RuntimeError(f"Empty tree '{tree_name}' for MC sample '{sname}'")
            df = _load_tree(files, tree_name, mc_need_load)
            if df is None or len(df) == 0:
                raise RuntimeError(f"No events loaded for MC sample '{sname}' in tree '{tree_name}'")
            df = _assign_mc_weight(df, sname, n_total, len(df), reweight_branches)
            dfs.append(df)
            log_message(
                f"  {sname}: class={cls_name}, tree_entries={n_total}, "
                f"loaded={len(df)}, entry_cap=none, "
                f"weight_sum={float(df['weight'].sum()):.6g}"
            )
        if dfs:
            class_dfs[cls_name] = _concat_parts(dfs)
            log_message(f"  Loaded class '{cls_name}': events={len(class_dfs[cls_name])}")
        else:
            raise RuntimeError(f"MC class '{cls_name}' has no usable events")

    # Load the data events.
    log_message(f"Loading data samples: n={len(DATA_SAMPLES)}")
    data_dfs = []
    for sname in DATA_SAMPLES:
        files = _input_files(sname, input_root, input_pattern)
        if not files:
            raise RuntimeError(f"No ROOT files found for data sample '{sname}'")
        df = _load_tree(files, tree_name, need_load)
        if df is None or len(df) == 0:
            log_message(f"  [WARN] data sample '{sname}' has zero entries in tree '{tree_name}'")
            continue
        df["weight"] = 1.0
        data_dfs.append(df)
        log_message(f"  data {sname}: loaded={len(df)}")
    data_df = _concat_parts(data_dfs) if data_dfs else None
    if data_df is None:
        log_message("Loaded data events: 0")
    else:
        log_message(f"Loaded data events: {len(data_df)}")

    # Build derived model score branches. MC scores use the saved test split;
    # data scores use the full configured data input, matching ordinary plots.
    score_class_dfs = {}
    score_data_df = None
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

        if DATA_SAMPLES:
            score_data_load = sorted(set(model_branches) | set(thresholds.keys()))
            score_data_parts = []
            log_message(f"Loading data score samples: n={len(DATA_SAMPLES)}")
            for sname in DATA_SAMPLES:
                files = _input_files(sname, input_root, input_pattern)
                if not files:
                    raise RuntimeError(f"No ROOT files found for data score sample '{sname}'")
                df = _load_tree(files, tree_name, score_data_load)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero entries")
                    continue
                df = _apply_thresholds(df, thresholds)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero events after filtering")
                    continue
                df["weight"] = 1.0
                X_model = _standardize_model_X(df[model_branches].copy(), clip_ranges, list(log_tf_set))
                X_model = _drop_decorrelated_features(X_model, decorrelate)
                proba = _predict_model_proba(clf, X_model, len(class_names))
                score_data_parts.append(_add_score_columns(df, proba, class_names))
                log_message(f"  data score {sname}: events={len(df)}")
            if score_data_parts:
                score_data_df = _concat_parts(score_data_parts)
                log_message(f"Loaded data score events: {len(score_data_df)}")
            else:
                log_message("Loaded data score events: 0")

    # Apply thresholds and then clip ranges; the weights stay fixed.
    def _prepare(df):
        if df is None or len(df) == 0:
            return df
        df = _apply_thresholds(df, thresholds)
        df = _apply_clip(df, clip_ranges)
        df = _drop_unneeded_columns(df, set(root_plot_branches) | {"weight"})
        return df

    log_message("Applying thresholds and clip ranges")
    for cls in list(class_dfs.keys()):
        class_dfs[cls] = _prepare(class_dfs[cls])
        if class_dfs[cls] is None or len(class_dfs[cls]) == 0:
            class_dfs.pop(cls)
            log_message(f"  [WARN] class '{cls}' became empty after filtering")
        else:
            log_message(f"  class '{cls}' after filtering: events={len(class_dfs[cls])}")
    if data_df is not None:
        data_df = _prepare(data_df)
        if data_df is None or len(data_df) == 0:
            data_df = None
            log_message("  data after filtering: 0 events")
        else:
            log_message(f"  data after filtering: events={len(data_df)}")

    # Plot each requested branch.
    log_message(f"Plotting branches: total={len(branches_to_plot)}")
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}

    for idx, branch in enumerate(branches_to_plot, start=1):
        log_message(f"Plotting branch {idx}/{len(branches_to_plot)}: {branch}")
        is_score_branch = branch in score_branches
        plot_class_dfs = score_class_dfs if is_score_branch else class_dfs
        plot_data_df = score_data_df if is_score_branch else data_df
        arrs = []
        for cls in class_names:
            if cls in plot_class_dfs and branch in plot_class_dfs[cls].columns:
                arrs.append(plot_class_dfs[cls][branch].values)
        if plot_data_df is not None and branch in plot_data_df.columns:
            arrs.append(plot_data_df[branch].values)

        binning = _resolve_binning(tree_name, branch, arrs, log_tf_set)
        if binning is None:
            log_message(f"  [WARN] no data for {tree_name}:{branch}, skipping")
            continue
        bins, x_range, logx, logy, y_range = binning
        edges       = _bin_edges(bins, x_range, logx)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths  = edges[1:] - edges[:-1]

        mc_total_v  = np.zeros(bins)
        mc_total_w2 = np.zeros(bins)
        mc_per_cls  = {}
        for cls in class_names:
            if cls in plot_class_dfs and branch in plot_class_dfs[cls].columns:
                h, h2 = _weighted_hist(
                    plot_class_dfs[cls][branch].values,
                    plot_class_dfs[cls]["weight"].values, edges
                )
            else:
                h  = np.zeros(bins)
                h2 = np.zeros(bins)
            mc_per_cls[cls] = (h, h2)
            mc_total_v  += h
            mc_total_w2 += h2

        if plot_data_df is not None and branch in plot_data_df.columns:
            data_v, data_w2 = _weighted_hist(
                plot_data_df[branch].values, plot_data_df["weight"].values, edges
            )
        else:
            data_v  = np.zeros(bins)
            data_w2 = np.zeros(bins)

        out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
        _draw_data_mc_plot(
            class_names=class_names,
            color_map=color_map,
            edges=edges,
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            mc_per_cls=mc_per_cls,
            mc_total_v=mc_total_v,
            mc_total_w2=mc_total_w2,
            data_v=data_v,
            data_w2=data_w2,
            branch=branch,
            x_range=x_range,
            logx=logx,
            logy=logy,
            y_range=y_range,
            y_label="Events",
            out_path=out_path,
            logy_floor=0.1,
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
            edges=edges,
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            mc_per_cls=mc_per_cls_norm,
            mc_total_v=mc_total_v_norm,
            mc_total_w2=mc_total_w2_norm,
            data_v=data_v_norm,
            data_w2=data_w2_norm,
            branch=branch,
            x_range=x_range,
            logx=logx,
            logy=logy,
            y_range=y_range,
            y_label="A.U.",
            out_path=out_path_normal,
            logy_floor=None,
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
