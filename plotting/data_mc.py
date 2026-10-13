#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data vs MC histogram comparison plotter.

Two stages. The fill stage reads every entry of every input file once, applies the
trained-model selection.json threshold and clip cuts (no log transform), computes the model
scores, removes the data events of the signal regions (blinding; --unblind keeps them) and
stores every histogram on a fine grid in
{output_root}/{tree}_hists.npz. The plot stage (--plot-only to run it alone) rebins those
histograms and draws a stacked MC + data panel with a Data/MC ratio sub-panel, without reading
the inputs again.

Pass --no-selection to skip threshold and clip-range cuts on plot variables
(BDT model clip ranges are still applied internally for score computation).
"""

import argparse
import fnmatch
import gc
import hashlib
import os
import sys
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import multiprocessing
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
import kfold

sys.path.insert(0, os.path.join(_ROOT_DIR, "selections", "mc_weight_common"))
import mc_weights  # noqa: E402

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

SUBMIT_TREES          = plot_cfg.get("submit_trees", ["fat2", "fat3"])
DATA_SAMPLES          = list(plot_cfg.get("data_samples", []))
DEFAULT_BINS          = int(plot_cfg.get("default_bins", 10))
OUTPUT_ROOT_PATT      = plot_cfg.get("output_root", "./pre-selection/{tree_name}")
OUTPUT_ROOT_NOSEL_PATT = plot_cfg.get("output_root_nosel", "./no-selection/{tree_name}")
BDT_ROOT_PATT         = plot_cfg["bdt_root"]

SAMPLE_CFG_PATH         = _resolve(plot_cfg["sample_config"], _SCRIPT_DIR)
CONVERT_BRANCH_CFG_PATH = _resolve(plot_cfg["convert_branch_config"], _SCRIPT_DIR)

sample_cfg         = _load_json(SAMPLE_CFG_PATH)
convert_branch_cfg = _load_json(CONVERT_BRANCH_CFG_PATH)

_CONVERT_CFG_PATH = os.path.join(os.path.dirname(CONVERT_BRANCH_CFG_PATH), "config.json")
convert_cfg       = _load_json(_CONVERT_CFG_PATH)

SAMPLE_INFO = {s["name"]: s for s in sample_cfg["sample"]}

# Optional theory-syst sidecar (produced by mode 8 / theory_syst.py).
# If configured, the JSON is read once here and a flat fractional theory
# uncertainty is drawn as a separate hatched band on each Data/MC plot.
_theory_syst_json_path = plot_cfg.get("theory_syst_json", None)
if _theory_syst_json_path is not None:
    _theory_syst_json_path = _resolve(_theory_syst_json_path, _SCRIPT_DIR)
THEORY_SYST_YIELDS = None
if _theory_syst_json_path is not None and os.path.exists(_theory_syst_json_path):
    THEORY_SYST_YIELDS = _load_json(_theory_syst_json_path)
    log_message(f"Loaded theory syst yields from {_theory_syst_json_path}")
elif _theory_syst_json_path is not None:
    log_message(f"[WARN] theory_syst_json not found at {_theory_syst_json_path}; run mode 8 first")
# Cross-section uncertainties of combine's xsec_<process> rows (combine/xsec_unc.json): the theory
# ratios of mode 8 hold the acceptance change only, so the band adds the normalization in quadrature.
_xsec_unc_json_path = plot_cfg.get("xsec_unc_json", None)
XSEC_UNC = (_load_json(_resolve(_xsec_unc_json_path, _SCRIPT_DIR))["uncertainties"]
            if _xsec_unc_json_path is not None else None)


def _sample_xsec_unc(sample):
    """Cross-section uncertainty of a sample: the xsec_unc_json pattern(s) matching it (fnmatch, as
    combine.C); every matching pattern must give the same value."""
    value = None
    for pattern, unc in XSEC_UNC.items():
        if not fnmatch.fnmatchcase(sample, pattern):
            continue
        if value is not None and float(unc) != value:
            raise ValueError(f"xsec_unc_json: patterns with different values match sample '{sample}'")
        value = float(unc)
    if value is None:
        raise KeyError(f"xsec_unc_json has no cross-section uncertainty for sample '{sample}'")
    return value


def _theory_frac_uncertainty(tree_name, class_groups, mc_target_totals):
    """Return per-class fractional theory uncertainty (flat normalization).

    Per sample, PDF, scale, PS_ISR and PS_FSR (mode 8, acceptance only) and the cross-section
    uncertainty (xsec_unc_json) are added in quadrature; the samples of a class are combined
    weighted by their yields. Returns a dict {class_name: fractional_uncertainty}. Samples
    without theory weights have no acceptance part.

    mc_target_totals: {sample_name: target_total} pre-computed by the caller.
    """
    if THEORY_SYST_YIELDS is None and XSEC_UNC is None:
        return {}

    result = {}
    for cls_name, samples in class_groups.items():
        num2 = 0.0   # (weighted fractional uncertainty)^2 × class_yield^2
        denom = 0.0  # total class yield (sum of target_totals)
        for sname in samples:
            target = mc_target_totals.get(sname, 0.0)
            if target <= 0.0:
                continue
            denom += target
            delta2 = _sample_xsec_unc(sname) ** 2 if XSEC_UNC is not None else 0.0
            sdata = (THEORY_SYST_YIELDS.get(sname, {}).get(tree_name)
                     if THEORY_SYST_YIELDS is not None else None)
            if sdata is not None:
                # Fractional uncertainty for each source: half-width of the band
                # relative to central_ratio (≈ 1.0 for well-behaved MC).
                def _half(up_key, dn_key):
                    u = float(sdata.get(up_key, 1.0))
                    d = float(sdata.get(dn_key, 1.0))
                    return 0.5 * abs(u - d)

                delta2 += (
                    _half("pdf_up",     "pdf_down"    ) ** 2
                    + _half("scale_up",   "scale_down"  ) ** 2
                    + _half("ps_isr_up",  "ps_isr_down" ) ** 2
                    + _half("ps_fsr_up",  "ps_fsr_down" ) ** 2
                )
            num2 += target * target * delta2

        if denom > 0.0:
            result[cls_name] = math.sqrt(num2) / denom
    return result


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

# Workers for multiprocessing pools: configurable via config.json "n_workers" (0 = cpu_count).
_n_workers_cfg = plot_cfg.get("n_workers", None)
_N_WORKERS = int(_n_workers_cfg) if _n_workers_cfg is not None else (os.cpu_count() or 4)


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
    """Return branch names to plot (onlyMC=false, not skipped, slots expanded).

    If ``config.json["plot_branches"][tree_name]`` is a non-null list, only
    those names are included (after the normal skip/slot expansion).  A null
    value or an absent key keeps the current "plot everything" behaviour.
    """
    tree    = _tree_output_entry(tree_name)
    skip    = _skip_branches_for_tree(tree_name)
    only_cfg = plot_cfg.get("plot_branches", {})
    only    = only_cfg.get(tree_name) if isinstance(only_cfg, dict) else None
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
                if only is not None and n not in only:
                    continue
                seen.add(n)
                out.append(n)
        else:
            if name in skip or name in seen:
                continue
            if only is not None and name not in only:
                continue
            seen.add(name)
            out.append(name)
    return out


# -------------------- Trained-model config copies --------------------
def _bdt_root_for_tree(tree_name):
    return _resolve(BDT_ROOT_PATT.format(tree_name=tree_name), _SCRIPT_DIR)


def _bdt_configs_for_tree(tree_name, load_test_meta=True):
    bdt_root = _bdt_root_for_tree(tree_name)
    if kfold.is_kfold_root(bdt_root):
        bdt_root = os.path.join(bdt_root, "fold0")   # k-fold rotations share one configuration
    cfg = _load_json(os.path.join(bdt_root, "config.json"))
    br = _load_json(os.path.join(bdt_root, "branch.json"))
    sel = _load_json(os.path.join(bdt_root, "selection.json"))
    meta = _load_json(os.path.join(bdt_root, "test_ranges.json")) if load_test_meta else None
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
    print (base)
    stem = base[:-5] if base.endswith(".root") else base
    print(glob.glob(base) + glob.glob(stem + "_*.root"))
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


def _tree_entries_total(files, tree_name):
    total = 0
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            total += int(uf[tree_name].num_entries)
    return total


_CHUNK_SIZE = "200 MB"


def _iter_chunks(files, tree_name, branches, strict=True):
    """Yield DataFrame chunks from a list of files without loading all events at once."""
    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree  = uf[tree_name]
            avail = set(tree.keys())
            missing = [b for b in branches if b not in avail]
            if missing:
                if strict:
                    raise KeyError(
                        f"Missing branches in {fpath}:{tree_name}: "
                        f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                    )
                log_message(
                    f"  [WARN] skipping {len(missing)} missing branch(es) in "
                    f"{os.path.basename(fpath)}:{tree_name}: "
                    f"{', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else "")
                )
            load = [b for b in branches if b in avail]
            if not load:
                continue
            for chunk in tree.iterate(expressions=load, step_size=_CHUNK_SIZE, library="pd"):
                yield chunk


def _prepass(files, tree_name, auto_range_branches, reweight_branches, branch_logx, strict):
    """Single streaming pass to collect per-branch value ranges and the raw-weight sum.

    Returns (raw_w_sum, ranges) where ranges = {branch: (lo, hi)}.
    raw_w_sum is the sum of the product of reweight_branches over all events (1.0 each
    if reweight_branches is empty), used later to normalise MC event weights.
    """
    prepass_cols = sorted(set(auto_range_branches) | set(reweight_branches))
    raw_w_sum = 0.0
    mins = {b:  np.inf for b in auto_range_branches}
    maxs = {b: -np.inf for b in auto_range_branches}

    for chunk in _iter_chunks(files, tree_name, prepass_cols, strict=strict):
        raw_w = np.ones(len(chunk), dtype=float)
        for rb in reweight_branches:
            if rb not in chunk.columns:
                raise KeyError(f"MC tree '{tree_name}' has no reweight branch {rb}")
            raw_w *= chunk[rb].to_numpy(dtype=float, copy=False)
        raw_w_sum += float(raw_w.sum())

        for b in auto_range_branches:
            if b not in chunk.columns:
                continue
            arr = chunk[b].to_numpy(dtype=float, copy=False)
            valid = arr[arr >= -990]
            if branch_logx.get(b, False):
                valid = valid[valid > 0]
            if valid.size == 0:
                continue
            mins[b] = min(mins[b], float(valid.min()))
            maxs[b] = max(maxs[b], float(valid.max()))

    ranges = {}
    for b in auto_range_branches:
        lo, hi = mins[b], maxs[b]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            ranges[b] = (lo, hi)

    return raw_w_sum, ranges


def _prepass_worker(args):
    files, tree_name, auto_range_branches, reweight_branches, branch_logx, strict = args
    return _prepass(files, tree_name, auto_range_branches, reweight_branches, branch_logx, strict)


def _stream_hists(files, tree_name, branch_edges, target_total, raw_w_sum,
                  reweight_branches, plot_thresholds, plot_clip_ranges, strict):
    """Stream files and accumulate weighted histograms for all branches in branch_edges.

    Returns {branch: (h, h2)} where h = sum-of-weights and h2 = sum-of-weights-squared.
    Every event gets raw_w * target_total / raw_w_sum (raw_w = product of reweight_branches).
    For data pass target_total=1.0, raw_w_sum=1.0 and reweight_branches=[]; for MC
    (genWeight * weight_pu) pass target_total = lumi * xsec / S and raw_w_sum = 1.0
    (selections/mc_weight_common/mc_weights.py).
    """
    load_cols = sorted(set(branch_edges.keys())
                       | set(plot_thresholds.keys())
                       | set(plot_clip_ranges.keys())
                       | set(reweight_branches))

    hists = {b: [np.zeros(len(edges) - 1, dtype=float),
                 np.zeros(len(edges) - 1, dtype=float)]
             for b, edges in branch_edges.items()}

    for chunk in _iter_chunks(files, tree_name, load_cols, strict=strict):
        # Compute per-event weights before any filtering.
        raw_w = np.ones(len(chunk), dtype=float)
        for rb in reweight_branches:
            if rb not in chunk.columns:
                raise KeyError(f"MC tree '{tree_name}' has no reweight branch {rb}")
            raw_w *= chunk[rb].to_numpy(dtype=float, copy=False)
        weight = raw_w * (target_total / raw_w_sum) if raw_w_sum > 0 else np.zeros(len(chunk))

        # Drop reweight columns — they are not plot variables.
        drop = [rb for rb in reweight_branches if rb in chunk.columns]
        if drop:
            chunk = chunk.drop(columns=drop)

        # Apply threshold cuts and clip ranges.
        if plot_thresholds:
            mask = _threshold_mask(chunk, plot_thresholds).to_numpy(dtype=bool)
            chunk  = chunk[mask].reset_index(drop=True)
            weight = weight[mask]

        if len(chunk) == 0:
            continue

        if plot_clip_ranges:
            _apply_clip(chunk, plot_clip_ranges)

        # Accumulate histograms.
        for b, edges in branch_edges.items():
            if b not in chunk.columns:
                continue
            h, h2 = _weighted_hist(chunk[b].to_numpy(dtype=float, copy=False), weight, edges)
            hists[b][0] += h
            hists[b][1] += h2

    return {b: (hists[b][0], hists[b][1]) for b in hists}


def _stream_hists_worker(args):
    return _stream_hists(*args)


# -------------------- Theory shape variations (per-bin) --------------------
# Member layout of the per-event theory weights, matched to the convert output:
#   PDF[101] (member 0 = central), alpha_s[2], scale[9], PS[4].
_TH_N_PDF       = 101
_TH_N_ALPHAS    = 2
_TH_N_SCALE     = 9
_TH_N_PS        = 4
_TH_N_MEMBERS   = _TH_N_PDF + _TH_N_ALPHAS + _TH_N_SCALE + _TH_N_PS
_TH_OFF_PDF     = 0
_TH_OFF_ALPHAS  = _TH_N_PDF
_TH_OFF_SCALE   = _TH_N_PDF + _TH_N_ALPHAS
_TH_OFF_PS      = _TH_N_PDF + _TH_N_ALPHAS + _TH_N_SCALE
_TH_SCALE_VALID = [0, 1, 3, 4, 5, 7, 8]   # drop anti-correlated corners 2 and 6
_TH_LAYOUT      = [("LHEPdfWeight", _TH_OFF_PDF, _TH_N_PDF), ("LHEPdfWeightAlphaS", _TH_OFF_ALPHAS, _TH_N_ALPHAS),
                   ("LHEScaleWeight", _TH_OFF_SCALE, _TH_N_SCALE), ("PSWeight", _TH_OFF_PS, _TH_N_PS)]
_TH_ARRAY_BRANCHES = [name for name, _, _ in _TH_LAYOUT]


def _theory_shape_band(member_hist):
    """(band_up, band_down) absolute per-bin theory uncertainty from member hists.

    PDF: symmetric-Hessian quadrature over the 100 members; alpha_s: half the
    up/down spread; scale and PS(ISR/FSR): per-bin envelopes (asymmetric).  All
    sources combined per bin in quadrature.  Column 0 (central PDF member) is the
    nominal yield of the theory samples.
    """
    central = member_hist[:, _TH_OFF_PDF]
    pdf = member_hist[:, _TH_OFF_PDF + 1:_TH_OFF_PDF + _TH_N_PDF]
    sig_pdf2 = np.sum((pdf - central[:, None]) ** 2, axis=1)

    as_dn = member_hist[:, _TH_OFF_ALPHAS]
    as_up = member_hist[:, _TH_OFF_ALPHAS + 1]
    sig_as2 = (0.5 * np.abs(as_up - as_dn)) ** 2

    scale = member_hist[:, [_TH_OFF_SCALE + i for i in _TH_SCALE_VALID]]
    scale_up = np.clip(scale.max(axis=1) - central, 0.0, None)
    scale_dn = np.clip(central - scale.min(axis=1), 0.0, None)

    # PSWeight: [0] isr.murfac=2.0; [1] fsr.murfac=2.0; [2] isr.murfac=0.5; [3] fsr.murfac=0.5.
    isr = member_hist[:, [_TH_OFF_PS + 0, _TH_OFF_PS + 2]]
    fsr = member_hist[:, [_TH_OFF_PS + 1, _TH_OFF_PS + 3]]
    isr_up = np.clip(isr.max(axis=1) - central, 0.0, None)
    isr_dn = np.clip(central - isr.min(axis=1), 0.0, None)
    fsr_up = np.clip(fsr.max(axis=1) - central, 0.0, None)
    fsr_dn = np.clip(central - fsr.min(axis=1), 0.0, None)

    band_up = np.sqrt(sig_pdf2 + sig_as2 + scale_up ** 2 + isr_up ** 2 + fsr_up ** 2)
    band_dn = np.sqrt(sig_pdf2 + sig_as2 + scale_dn ** 2 + isr_dn ** 2 + fsr_dn ** 2)
    return band_up, band_dn


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


def _apply_clip(df, clip_ranges):
    if not clip_ranges or df is None or len(df) == 0:
        return df
    for col, rng in clip_ranges.items():
        if col not in df.columns:
            continue
        arr = df[col].to_numpy(dtype=float, copy=False)
        valid = arr >= -990
        lo, hi = rng
        lo_mask = (lo is not None) and np.any(valid & (arr < lo))
        hi_mask = (hi is not None) and np.any(valid & (arr > hi))
        if not lo_mask and not hi_mask:
            continue
        arr = arr.copy()
        if lo_mask:
            arr[valid & (arr < lo)] = lo
        if hi_mask:
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
    return df


def _standardize_model_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    cols_to_modify = [col for col in X.columns if col in clip_ranges or col in log_set]
    if not cols_to_modify:
        return X
    X = X.copy()
    for col in cols_to_modify:
        # In the stored dtype, as train.py and qcd_est.py: a float64 log shifts float32 inputs by
        # rounding, which moves rare events across a tree split (seen on the k-fold toy: 1 in ~2000
        # events, score changes up to 0.009).
        arr = X[col].to_numpy(copy=True)
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
    if arrs is None:
        # Branch is declared in the convert branch.json but absent from every input file
        # (e.g. a schema change not yet re-converted).  Caller treats None as "skip branch".
        return None
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
    elif isinstance(arrs, tuple):
        # Pre-computed (lo, hi) from the streaming pre-pass.
        lo, hi = arrs
        x_range = (lo, lo + 1.0) if lo >= hi else (lo, hi)
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
        tm = np.where(mc_vals   > 0, mc_vars   / np.maximum(mc_vals,   1e-300) ** 2, 0.0)
        td = np.where(data_vals > 0, data_vars / np.maximum(data_vals, 1e-300) ** 2, 0.0)
        sigma = np.abs(r) * np.sqrt(tm + td)
    return r, sigma


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
    theory_fracs=None,
    theory_shape_up=None,
    theory_shape_down=None,
    note=None,
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

    # Theory uncertainty band.  Prefer the per-bin SHAPE band (computed from the
    # LHE weight variations); otherwise fall back to a per-bin flat-normalization
    # band built from the integrated per-class fractional uncertainties.
    theory_up = theory_dn = None
    if theory_shape_up is not None and theory_shape_down is not None:
        up = np.asarray(theory_shape_up, dtype=float)
        dn = np.asarray(theory_shape_down, dtype=float)
        if up.shape == mc_total_v.shape and (np.any(up > 0.0) or np.any(dn > 0.0)):
            theory_up, theory_dn = up, dn
    if theory_up is None and theory_fracs:
        # Flat-normalization fallback: scale each class's per-bin yield by its
        # fractional uncertainty and combine classes in quadrature (symmetric).
        theory_var = np.zeros(bins)
        for cls_name in class_names:
            delta_cls = float(theory_fracs.get(cls_name, 0.0))
            if delta_cls > 0.0:
                theory_var += (mc_per_cls[cls_name][0] * delta_cls) ** 2
        band = np.sqrt(theory_var)
        if np.any(band > 0.0):
            theory_up = theory_dn = band
    if theory_up is not None:
        th_lower = np.clip(mc_total_v - theory_dn, 1e-12, None)
        th_upper = mc_total_v + theory_up
        ax.fill_between(
            bin_centers, th_lower, th_upper, step="mid",
            facecolor="none", edgecolor="#e07b39", hatch="\\\\\\", linewidth=0,
            label="Theory unc.",
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
            else:
                ymin = float(logy_floor)
            ax.set_ylim(ymin, max(ymin * 10.0, ymax * 5.0))
        else:
            ax.set_ylim(0.0, max(1.0, ymax * 1.3))

    ax.set_ylabel(y_label, fontsize=24)
    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=int(LUMI_TOTAL), ax=ax)
    if note:
        ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=17)

    handles, labels = ax.get_legend_handles_labels()
    if "Data" in labels:
        i = labels.index("Data")
        handles.append(handles.pop(i))
        labels.append(labels.pop(i))
    ax.legend(handles, labels, loc="best", fontsize=17, frameon=False, ncol=2)

    ratio, r_err = _ratio_data_over_mc(data_v, data_w2, mc_total_v, mc_total_w2)
    # MC uncertainty bands around 1.0 in the ratio panel (where Data/MC
    # agreement is read off against the systematics): gray /// = MC stat,
    # orange \\\ = theory.  Drawn under the data points.
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_mc = np.where(mc_total_v > 0.0, mc_total_v, np.nan)
        stat_rel = np.abs(mc_sigma / safe_mc)
        axr.fill_between(
            bin_centers, 1.0 - stat_rel, 1.0 + stat_rel, step="mid",
            facecolor="none", edgecolor="gray", hatch="///", linewidth=0,
        )
        if theory_up is not None:
            th_rel_up = np.abs(theory_up / safe_mc)
            th_rel_dn = np.abs(theory_dn / safe_mc)
            axr.fill_between(
                bin_centers, 1.0 - th_rel_dn, 1.0 + th_rel_up, step="mid",
                facecolor="none", edgecolor="#e07b39", hatch="\\\\\\", linewidth=0,
            )

    axr.errorbar(
        bin_centers, ratio, yerr=r_err,
        fmt="o", ms=7.6, color="black", mfc="black", mec="black",
        elinewidth=1.5, capsize=0,
    )
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)

    finite = np.isfinite(ratio)
    if np.any(finite):
        safe_err = np.nan_to_num(r_err[finite], nan=0.0)
        rmax = float(np.nanmax(ratio[finite] + safe_err))
        rmin = float(np.nanmin(ratio[finite] - safe_err))
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


# -------------------- Signal regions (blinding) --------------------
def _signal_region_csv_for_tree(tree_name):
    cfg = plot_cfg.get("signal_region_csv")
    raw = cfg.get(tree_name) if isinstance(cfg, dict) else cfg
    if not raw:
        return None
    return _resolve(raw.format(tree_name=tree_name), _SCRIPT_DIR)


def _load_signal_region_boxes(path, class_names):
    """Score boxes of a signal_region.csv as (axis indices, lows, highs), with the half-open
    [low, high) convention of qcd_est._region_mask (high = 1 is open-ended)."""
    df = pd.read_csv(path)
    axes = [col[:-4] for col in df.columns
            if col.endswith("_low") and f"{col[:-4]}_high" in df.columns]
    unknown = [axis for axis in axes if axis not in class_names]
    if not axes or unknown or df.empty:
        raise RuntimeError(f"Signal region CSV {path} has no usable score axes "
                           f"(axes={axes}, not in the classes: {unknown})")
    index = np.array([class_names.index(axis) for axis in axes], dtype=int)
    lows = df[[f"{axis}_low" for axis in axes]].to_numpy(dtype=float)
    highs = df[[f"{axis}_high" for axis in axes]].to_numpy(dtype=float)
    return index, lows, highs


def _in_signal_regions(proba, boxes):
    index, lows, highs = boxes
    scores = proba[:, index]
    inside = np.zeros(len(proba), dtype=bool)
    for low, high in zip(lows, highs):
        upper = np.where(high < 1.0 - 1e-12, scores < high, True)
        inside |= np.all((scores >= low) & upper, axis=1)
    return inside


# -------------------- Fill stage: all histograms in one pass --------------------
# Every branch is filled on a fine grid of HIST_REFINE x (its plot bins) bins plus an underflow
# (index 0) and an overflow bin (last index), so the plot stage can rebin, trim the range and
# restyle without reading the inputs again. Ranges come from the branch.json x_range, [0, 1] for
# model scores, else from a scan of the first RANGE_SCAN_ENTRIES entries of every sample.
HIST_REFINE = int(plot_cfg.get("hist_refine", 10))
# MC samples left out of every plotted yield (exclude_samples, as in qcd_est.py and the SR scan:
# the HT < 400 GeV QCD samples, pileup double counting).
EXCLUDE_SAMPLES = set(plot_cfg.get("exclude_samples", []))
RANGE_SCAN_ENTRIES = int(plot_cfg.get("range_scan_entries", 200000))
_FILL_CTX = {}
_FILL_MODEL = []


def _fine_hist(vals, weights, edges, logx):
    """(sum w, sum w^2) on edges plus under/overflow; sentinels (< -990) and, for log-x
    branches, non-positive values are dropped."""
    v = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & (v >= -990)
    if logx:
        keep &= v > 0
    idx = np.searchsorted(edges, v[keep], side="right")
    n = len(edges) + 1
    w = w[keep]
    return idx, np.bincount(idx, weights=w, minlength=n), np.bincount(idx, weights=w * w, minlength=n)


def _scan_ranges_task(task):
    """Min/max of the auto-range branches in the first entries of one sample."""
    files, starts, tree_name, branches, logx_map, max_entries = task
    lo = {b: np.inf for b in branches}
    hi = {b: -np.inf for b in branches}
    remaining = max_entries
    for fpath, start in zip(files, starts):
        if remaining <= 0:
            break
        with uproot.open(fpath) as uf:
            tree = uf[tree_name]
            stop = min(int(tree.num_entries), start + remaining)
            if stop <= start:
                continue
            arrays = tree.arrays([b for b in branches if b in tree], entry_start=start,
                                 entry_stop=stop, library="np")
            remaining -= stop - start
        for b, arr in arrays.items():
            v = np.asarray(arr, dtype=float)
            v = v[np.isfinite(v) & (v >= -990)]
            if logx_map[b]:
                v = v[v > 0]
            if v.size:
                lo[b] = min(lo[b], float(v.min()))
                hi[b] = max(hi[b], float(v.max()))
    return lo, hi


def _init_fill_worker(ctx):
    _FILL_CTX.clear()
    _FILL_CTX.update(ctx)
    _FILL_MODEL.clear()


def _fill_model():
    if not _FILL_MODEL:
        model = _load_score_model(_FILL_CTX["bdt_root"], _FILL_CTX["bdt_cfg"], _FILL_CTX["tree_name"])
        if hasattr(model, "set_param"):
            model.set_param({"nthread": 1})   # one thread per pool worker
        _FILL_MODEL.append(model)
    return _FILL_MODEL[0]


def _fill_kfold():
    """The worker's k-fold models (one XGBoost thread per pool worker)."""
    if not _FILL_MODEL:
        kf = kfold.KFoldModels(_FILL_CTX["bdt_root"], log=log_message)
        for fold in range(kf.k):
            model = kf.model(fold)
            if hasattr(model, "set_param"):
                model.set_param({"nthread": 1})
        _FILL_MODEL.append(kf)
    return _FILL_MODEL[0]


def _fill_task(task):
    """Histogram one entry range of one input file: returns the summed fine histograms."""
    ctx = _FILL_CTX
    is_mc = task["kind"] == "mc"
    edges = ctx["edges"]
    hists = {b: [np.zeros(len(e) + 1), np.zeros(len(e) + 1)] for b, e in edges.items()}
    theory = None
    counts = {"read": 0, "selected": 0, "blinded": 0, "weight": 0.0}
    scalars = list(ctx["columns"])
    if is_mc:
        scalars = sorted(set(scalars) | set(ctx["reweight_branches"]))
    if ctx["kfold"]:
        scalars = sorted(set(scalars) | {"event"})
    with uproot.open(task["file"]) as uf:
        tree = uf[ctx["tree_name"]]
        avail = set(tree.keys())
        missing = [b for b in scalars if b not in avail]
        if missing:
            raise KeyError(f"Missing branches in {task['file']}:{ctx['tree_name']}: "
                           f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
        arrays_in = [b for b in _TH_ARRAY_BRANCHES if b in avail] if task["theory"] else []
        if task["theory"] and "LHEPdfWeight" in arrays_in:
            theory = {b: np.zeros((len(e) + 1, _TH_N_MEMBERS)) for b, e in edges.items()}
        else:
            arrays_in = []
        for chunk, report in tree.iterate(expressions=scalars + arrays_in, entry_start=task["start"],
                                          entry_stop=task["stop"], step_size=_CHUNK_SIZE, library="np",
                                          report=True):
            df = pd.DataFrame({c: np.asarray(chunk[c]) for c in scalars})
            n = len(df)
            counts["read"] += n
            if is_mc:
                # lumi * xsec * genWeight * weight_pu / S (selections/mc_weight_common/mc_weights.py)
                weight = task["scale"] * mc_weights.event_weight_product(df, ctx["reweight_branches"])
            else:
                weight = np.ones(n, dtype=float)
            proba = None
            if ctx["scoring"]:
                X = _standardize_model_X(df[ctx["model_branches"]], ctx["clip_ranges"], ctx["log_transform"])
                X = _drop_decorrelated_features(X, ctx["decorrelate"])
                if ctx["kfold"]:
                    # k-fold: a trained MC event with the model of its block (task offset = global
                    # index of the file's first entry), data and untrained samples by event number.
                    kf = _fill_kfold()
                    folds = (kf.fold_of_index(task["sample"], task["offset"] + report.tree_entry_start
                                              + np.arange(n))
                             if task.get("offset") is not None else kf.data_fold(df["event"].to_numpy()))
                    proba = kf.predict(X, folds)
                else:
                    proba = _predict_model_proba(_fill_model(), X, ctx["n_classes"])
            sel = _threshold_mask(df, ctx["plot_thresholds"]).to_numpy(dtype=bool)
            if ctx["blind"] and not is_mc:
                # Blinding: drop every data event of a signal region (score box and all its
                # selection thresholds); MC is kept.
                in_sr = (_in_signal_regions(proba, ctx["sr_boxes"])
                         & _threshold_mask(df, ctx["sr_thresholds"]).to_numpy(dtype=bool))
                counts["blinded"] += int(np.count_nonzero(sel & in_sr))
                sel &= ~in_sr
            counts["selected"] += int(np.count_nonzero(sel))
            if not sel.any():
                continue
            dsel = df.loc[sel].reset_index(drop=True)
            if ctx["plot_clip_ranges"]:
                _apply_clip(dsel, ctx["plot_clip_ranges"])
            wsel = weight[sel]
            counts["weight"] += float(wsel.sum())
            members = None
            if theory is not None:
                members = np.ones((len(wsel), _TH_N_MEMBERS), dtype=float)
                for name, offset, count in _TH_LAYOUT:
                    if name in chunk:
                        a = np.asarray(chunk[name], dtype=float)[sel].reshape(len(wsel), -1)
                        c = min(count, a.shape[1])
                        members[:, offset:offset + c] = a[:, :c]
                members *= wsel[:, None]
            for b, e in edges.items():
                if b in ctx["score_index"]:
                    vals = proba[sel, ctx["score_index"][b]]
                else:
                    vals = dsel[b].to_numpy(dtype=float)
                idx, h, h2 = _fine_hist(vals, wsel, e, ctx["logx"][b])
                hists[b][0] += h
                hists[b][1] += h2
                if members is not None:
                    keep = np.isfinite(vals) & (vals >= -990)
                    if ctx["logx"][b]:
                        keep &= vals > 0
                    kept = members[keep]
                    for m in range(_TH_N_MEMBERS):
                        theory[b][:, m] += np.bincount(idx, weights=kept[:, m], minlength=len(e) + 1)
    return {"kind": task["kind"], "cls": task["cls"], "sample": task["sample"],
            "hists": hists, "theory": theory, "counts": counts}


def _sample_entry_ranges(files, tree_name):
    """Per-file (file, 0, entries) of a sample's non-empty trees, and the total entry count."""
    out, total = [], 0
    for fpath in files:
        with uproot.open(fpath) as uf:
            n = int(uf[tree_name].num_entries) if tree_name in uf else 0
        if n > 0:
            out.append((fpath, 0, n))
        total += n
    return out, total


def _hists_path(tree_name, no_selection):
    out_patt = OUTPUT_ROOT_NOSEL_PATT if no_selection else OUTPUT_ROOT_PATT
    out_dir = _resolve(out_patt.format(tree_name=tree_name), _SCRIPT_DIR)
    return out_dir, os.path.join(out_dir, f"{tree_name}_hists.npz")


def _fill_tree(tree_name, no_selection=False, unblind=False, n_workers=None):
    """Fill stage: reads every input once and writes {out_dir}/{tree}_hists.npz."""
    n_workers = n_workers or _N_WORKERS
    log_message(f"Running data_mc.py fill: tree={tree_name}, no_selection={no_selection}, "
                f"unblind={unblind}, n_workers={n_workers}")
    bdt_cfg, bdt_br, bdt_sel, _ = _bdt_configs_for_tree(tree_name, load_test_meta=False)
    _cg_override = plot_cfg.get("class_groups")
    if isinstance(_cg_override, dict) and tree_name in _cg_override and isinstance(_cg_override[tree_name], dict):
        _cg_override = _cg_override[tree_name]
    class_groups = _cg_override or bdt_cfg["class_groups"]
    _mc_whitelist = plot_cfg.get("mc_samples")
    if isinstance(_mc_whitelist, dict):
        _mc_whitelist = _mc_whitelist.get(tree_name)
    if _mc_whitelist is not None:
        _wl = set(_mc_whitelist)
        class_groups = {cls: [s for s in samples if s in _wl] for cls, samples in class_groups.items()}
        class_groups = {cls: samples for cls, samples in class_groups.items() if samples}
    _unknown_excluded = EXCLUDE_SAMPLES - {s for samples in class_groups.values() for s in samples}
    if _unknown_excluded:
        raise RuntimeError(f"exclude_samples names samples not in class_groups: {sorted(_unknown_excluded)}")
    if EXCLUDE_SAMPLES:
        log_message(f"  excluding MC samples {sorted(EXCLUDE_SAMPLES)}")
        class_groups = {cls: [s for s in samples if s not in EXCLUDE_SAMPLES] for cls, samples in class_groups.items()}
        class_groups = {cls: samples for cls, samples in class_groups.items() if samples}
    class_names = list(class_groups.keys())
    model_classes = list(bdt_cfg["class_groups"].keys())
    model_branches = [item["name"] for item in bdt_br.get(tree_name, [])]
    bdt_root_dir = _bdt_root_for_tree(tree_name)
    scoring = bool(model_branches)
    kf = kfold.KFoldModels(bdt_root_dir, log=log_message) if scoring and kfold.is_kfold_root(bdt_root_dir) else None

    bdt_script_dir = os.path.dirname(bdt_root_dir)
    if no_selection:
        input_root = _resolve(convert_cfg["output_root"], os.path.dirname(_CONVERT_CFG_PATH))
        input_pattern = convert_cfg["output_pattern"].replace("{output_root}", "{input_root}")
    else:
        input_root = _resolve(bdt_cfg["input_root"], bdt_script_dir)
        input_pattern = bdt_cfg["input_pattern"]

    sel = bdt_sel.get(tree_name, {})
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    thresholds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sel.get("thresholds", {}).items()}
    log_tf_set = set(sel.get("log_transform", []))
    extra_cfg = plot_cfg.get("plot_thresholds", {}).get(tree_name, {}) or {}
    extra_plot_thresholds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in extra_cfg.items()}
    plot_thresholds = {**({} if no_selection else thresholds), **extra_plot_thresholds}
    plot_clip_ranges = {} if no_selection else clip_ranges

    # Blinding (default): needs the tree's signal regions and the model scores.
    unblinded_trees = set(plot_cfg.get("unblinded_trees", []))
    blind = bool(DATA_SAMPLES) and not unblind and tree_name not in unblinded_trees
    sr_csv = _signal_region_csv_for_tree(tree_name)
    sr_boxes = None
    sr_csv_hash = None
    if blind:
        if not scoring or sr_csv is None or not os.path.exists(sr_csv):
            raise RuntimeError(
                f"Tree '{tree_name}' cannot be blinded: it needs a trained model in {bdt_root_dir} and "
                f"signal_region_csv['{tree_name}'] in the plotting config (found: {sr_csv}). Pass "
                f"--unblind, or list the tree in unblinded_trees if it has no signal regions.")
        sr_boxes = _load_signal_region_boxes(sr_csv, model_classes)
        with open(sr_csv, "rb") as fh:
            sr_csv_hash = hashlib.sha256(fh.read()).hexdigest()[:13]
        log_message(f"Blinding: removing the data events in the {len(sr_boxes[1])} signal regions of {sr_csv}")
    else:
        log_message("Blinding: OFF" + (" (--unblind)" if unblind else " (tree listed in unblinded_trees)"
                                        if tree_name in unblinded_trees else " (no data samples)"))

    skip = _skip_branches_for_tree(tree_name)
    branches = _plot_branches_for_tree(tree_name)
    score_branches = ([_score_branch_name(c) for c in model_classes if _score_branch_name(c) not in skip]
                      if scoring else [])
    branches += [b for b in score_branches if b not in branches]
    root_branches = [b for b in branches if b not in score_branches]
    reweight_branches = list(plot_cfg.get("event_reweight_branches", {}).get(tree_name, []))
    decorrelate = list(bdt_cfg.get(tree_name, {}).get("decorrelate", []))
    out_dir, hists_path = _hists_path(tree_name, no_selection)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Inputs: every entry of every MC and data file (MC normalised to its full sample) ----
    mc_tasks, theory_samples = [], []
    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = _input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            ranges, n_total = _sample_entry_ranges(files, tree_name)
            if n_total <= 0:
                log_message(f"  [WARN] skipping MC sample '{sname}': no entries for tree '{tree_name}'")
                continue
            info = SAMPLE_INFO[sname]
            # Every entry is read: per-event weight lumi * xsec * raw_w / S (mc_weights.py).
            scale = (LUMI_TOTAL * float(info.get("xsection", 0.0))
                     / mc_weights.generated_weight_sum(info, reweight_branches))
            has_theory = bool(info.get("has_theory_weights", False))
            if has_theory:
                theory_samples.append(sname)
            offsets = {}
            if kf is not None and sname in kf.samples:
                kf.check_files(sname, files)
                offsets = kf.file_offsets(sname)
            for fpath, lo, hi in ranges:
                mc_tasks.append({"kind": "mc", "cls": cls_name, "sample": sname, "file": fpath, "start": lo,
                                 "stop": hi, "scale": scale, "theory": has_theory,
                                 "offset": offsets.get(fpath)})
            log_message(f"  MC {sname} (class={cls_name}): {n_total} entries in {len(files)} files, "
                        f"weight scale lumi*xsec/S={scale:.6g}")
    data_tasks = []
    for sname in DATA_SAMPLES:
        files = _input_files(sname, input_root, input_pattern)
        if not files:
            raise RuntimeError(f"No ROOT files found for data sample '{sname}'")
        ranges, n_total = _sample_entry_ranges(files, tree_name)
        data_tasks += [{"kind": "data", "cls": None, "sample": sname, "file": f, "start": lo, "stop": hi,
                        "scale": 1.0, "theory": False} for f, lo, hi in ranges]
        log_message(f"  data {sname}: {n_total} entries in {len(files)} files")

    # ---- Fine binning: branch.json ranges, [0, 1] for scores, else a range scan ----
    logx = {}
    for b in branches:
        override = _branch_override(tree_name, b)
        logx[b] = bool(override.get("logx", False if b in score_branches else b in log_tf_set))
    auto = [b for b in root_branches if "x_range" not in _branch_override(tree_name, b)]
    scanned = {}
    if auto:
        by_sample = {}
        for t in mc_tasks + data_tasks:
            by_sample.setdefault(t["sample"], ([], []))
            by_sample[t["sample"]][0].append(t["file"])
            by_sample[t["sample"]][1].append(t["start"])
        scan_tasks = [(f, s, tree_name, auto, logx, RANGE_SCAN_ENTRIES) for f, s in by_sample.values()]
        log_message(f"Range scan: {len(auto)} branches, first {RANGE_SCAN_ENTRIES} entries of {len(scan_tasks)} samples")
        with multiprocessing.Pool(processes=max(1, min(n_workers, len(scan_tasks)))) as pool:
            for lo, hi in pool.imap_unordered(_scan_ranges_task, scan_tasks):
                for b in auto:
                    old = scanned.get(b, (np.inf, -np.inf))
                    scanned[b] = (min(old[0], lo[b]), max(old[1], hi[b]))
    edges = {}
    for b in branches:
        override = _branch_override(tree_name, b)
        bins = int(override.get("bins", DEFAULT_BINS)) * HIST_REFINE
        if "x_range" in override:
            x_range = (float(override["x_range"][0]), float(override["x_range"][1]))
        elif b in score_branches:
            x_range = (0.0, 1.0)
        else:
            lo, hi = scanned.get(b, (np.inf, -np.inf))
            if not (np.isfinite(lo) and np.isfinite(hi)):
                log_message(f"  [WARN] no values for {tree_name}:{b}, skipping")
                continue
            x_range = (lo, hi if hi > lo else lo + 1.0)
        edges[b] = _bin_edges(bins, x_range, logx[b])

    columns = sorted((set(root_branches) & set(edges)) | set(plot_thresholds) | set(plot_clip_ranges)
                     | (set(model_branches) if scoring else set()) | (set(thresholds) if blind else set()))
    ctx = {
        "tree_name": tree_name, "bdt_root": bdt_root_dir, "bdt_cfg": bdt_cfg, "scoring": scoring,
        "kfold": kf is not None,
        "columns": columns, "reweight_branches": reweight_branches, "model_branches": model_branches,
        "clip_ranges": clip_ranges, "log_transform": list(log_tf_set), "decorrelate": decorrelate,
        "n_classes": len(model_classes), "plot_thresholds": plot_thresholds,
        "plot_clip_ranges": plot_clip_ranges, "blind": blind, "sr_boxes": sr_boxes,
        "sr_thresholds": thresholds, "edges": edges, "logx": logx,
        "score_index": {_score_branch_name(c): i for i, c in enumerate(model_classes)
                        if _score_branch_name(c) in edges},
    }

    # ---- One pass over all inputs ----
    tasks = data_tasks + mc_tasks
    mc_h = {cls: {b: [np.zeros(len(e) + 1), np.zeros(len(e) + 1)] for b, e in edges.items()} for cls in class_names}
    data_h = {b: [np.zeros(len(e) + 1), np.zeros(len(e) + 1)] for b, e in edges.items()}
    theory_h = {b: np.zeros((len(e) + 1, _TH_N_MEMBERS)) for b, e in edges.items()}
    have_theory = False
    counts = {}
    log_message(f"Filling: {len(tasks)} tasks ({len(data_tasks)} data, {len(mc_tasks)} MC), {len(edges)} branches, "
                f"{n_workers} workers")
    done = 0
    with multiprocessing.Pool(processes=max(1, min(n_workers, len(tasks))), initializer=_init_fill_worker,
                              initargs=(ctx,)) as pool:
        for res in pool.imap_unordered(_fill_task, tasks):
            target = data_h if res["kind"] == "data" else mc_h[res["cls"]]
            for b, (h, h2) in res["hists"].items():
                target[b][0] += h
                target[b][1] += h2
            if res["theory"] is not None:
                have_theory = True
                for b, m in res["theory"].items():
                    theory_h[b] += m
            c = counts.setdefault(res["sample"], {"read": 0, "selected": 0, "blinded": 0, "weight": 0.0})
            for k, v in res["counts"].items():
                c[k] += v
            done += 1
            if done % 50 == 0 or done == len(tasks):
                log_message(f"  filled {done}/{len(tasks)} tasks")
    for sname, c in counts.items():
        log_message(f"  {sname}: read={c['read']}, selected={c['selected']}, blinded={c['blinded']}, "
                    f"selected weight={c['weight']:.6g}")
    # Selected MC yields of every sample: the weights of the classes' flat theory fractions.
    mc_target_totals = {t["sample"]: counts[t["sample"]]["weight"] for t in mc_tasks if t["sample"] in counts}

    meta = {
        "tree_name": tree_name, "no_selection": no_selection, "blinded": blind, "sr_csv": sr_csv,
        "sr_csv_hash": sr_csv_hash, "class_names": class_names, "branches": list(edges),
        "score_branches": [b for b in score_branches if b in edges], "logx": {b: logx[b] for b in edges},
        "lumi": LUMI_TOTAL, "theory_fracs": _theory_frac_uncertainty(tree_name, class_groups, mc_target_totals),
        "theory_samples": theory_samples if have_theory else [], "counts": counts, "hist_refine": HIST_REFINE,
    }
    arrays = {"meta": np.array(json.dumps(meta))}
    for b, e in edges.items():
        arrays[f"{b}::edges"] = e
        arrays[f"{b}::data::w"], arrays[f"{b}::data::w2"] = data_h[b]
        for cls in class_names:
            arrays[f"{b}::mc::{cls}::w"], arrays[f"{b}::mc::{cls}::w2"] = mc_h[cls][b]
        if have_theory:
            arrays[f"{b}::theory"] = theory_h[b]
    tmp_path = hists_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, hists_path)
    log_message(f"Wrote histogram file: {hists_path}")
    return hists_path


# -------------------- Plot stage: from the stored histograms --------------------
def _plot_binning(tree_name, branch, fine_edges, fine_logx, is_score, populated):
    """Coarse binning of one branch and the map fine bin (incl. under/overflow) -> coarse bin.
    Without an x_range override the populated fine bins are grouped into about `bins` bins on the
    fine edges (exact); an explicit range maps the fine bins by their centres (exact when the
    coarse edges lie on fine edges, as for the ranges known at fill time)."""
    override = _branch_override(tree_name, branch)
    bins = int(override.get("bins", DEFAULT_BINS))
    logx = bool(override.get("logx", fine_logx))
    logy = bool(override.get("logy", True))
    y_range = tuple(override["y_range"]) if "y_range" in override else None
    n_fine = len(fine_edges) - 1
    if "x_range" in override or is_score:
        x_range = tuple(float(v) for v in override["x_range"]) if "x_range" in override else (0.0, 1.0)
        coarse = _bin_edges(bins, x_range, logx)
        tol = 1e-9 * max(1.0, float(np.max(np.abs(fine_edges))))
        inside = coarse[(coarse >= fine_edges[0] - tol) & (coarse <= fine_edges[-1] + tol)]
        if logx != fine_logx or not all(np.min(np.abs(fine_edges - e)) <= tol for e in inside):
            log_message(f"  [WARN] {tree_name}:{branch}: plot binning is not aligned with the stored fine "
                        f"bins (approximate rebinning); rerun the fill stage for exact bins")
    else:
        if populated.size == 0:
            return None
        i0, i1 = int(populated[0]), int(populated[-1]) + 1
        step = max(1, int(math.ceil((i1 - i0) / bins)))
        idx = list(range(i0, i1, step)) + [i1]
        coarse = fine_edges[idx]
        x_range = (float(coarse[0]), float(coarse[-1]))
    centres = (np.sqrt(fine_edges[:-1] * fine_edges[1:]) if fine_logx and fine_edges[0] > 0
               else 0.5 * (fine_edges[:-1] + fine_edges[1:]))
    n_coarse = len(coarse) - 1
    mapping = np.empty(n_fine + 2, dtype=int)
    mapping[0], mapping[-1] = 0, n_coarse - 1   # under/overflow folded into the edge bins
    mapping[1:-1] = np.clip(np.searchsorted(coarse, centres, side="right") - 1, 0, n_coarse - 1)
    return coarse, mapping, x_range, logx, logy, y_range


def _rebin(values, mapping, n_coarse):
    if values.ndim == 1:
        return np.bincount(mapping, weights=values, minlength=n_coarse)
    out = np.zeros((n_coarse, values.shape[1]))
    np.add.at(out, mapping, values)
    return out


def _render_branch(job):
    """Absolute and unit-normalised PDFs of one branch (pool worker)."""
    common = dict(class_names=job["class_names"], color_map=job["color_map"], edges=job["edges"],
                  bin_centers=job["bin_centers"], bin_widths=job["bin_widths"], branch=job["branch"],
                  x_range=job["x_range"], logx=job["logx"], logy=job["logy"], y_range=job["y_range"],
                  note=job["note"])
    out_path = os.path.join(job["out_dir"], f"{job['tree_name']}_{job['branch']}.pdf")
    _draw_data_mc_plot(
        mc_per_cls=job["mc_per_cls"], mc_total_v=job["mc_total_v"], mc_total_w2=job["mc_total_w2"],
        data_v=job["data_v"], data_w2=job["data_w2"], y_label="Events", out_path=out_path, logy_floor=0.1,
        theory_fracs=job["theory_fracs"], theory_shape_up=job["theory_up"], theory_shape_down=job["theory_dn"],
        **common)
    norm = _unit_normalized_histograms(job["mc_per_cls"], job["mc_total_v"], job["mc_total_w2"],
                                       job["data_v"], job["data_w2"])
    out_path_normal = os.path.join(job["out_dir"], f"{job['tree_name']}_{job['branch']}_normal.pdf")
    _draw_data_mc_plot(mc_per_cls=norm[0], mc_total_v=norm[1], mc_total_w2=norm[2], data_v=norm[3],
                       data_w2=norm[4], y_label="A.U.", out_path=out_path_normal, logy_floor=None, **common)
    return out_path, out_path_normal


def _plot_tree(tree_name, no_selection=False, n_workers=None):
    """Plot stage: renders every branch from {out_dir}/{tree}_hists.npz without reading inputs."""
    n_workers = n_workers or _N_WORKERS
    out_dir, hists_path = _hists_path(tree_name, no_selection)
    if not os.path.exists(hists_path):
        raise FileNotFoundError(f"No histogram file {hists_path}; run the fill stage first")
    store = np.load(hists_path, allow_pickle=False)
    meta = json.loads(str(store["meta"]))
    class_names = meta["class_names"]
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}
    note = "Data in signal regions removed" if meta["blinded"] else None
    log_message(f"Running data_mc.py plot: tree={tree_name}, file={hists_path}, blinded={meta['blinded']}")
    jobs = []
    for b in meta["branches"]:
        fine = store[f"{b}::edges"]
        mc_w = {c: store[f"{b}::mc::{c}::w"] for c in class_names}
        mc_w2 = {c: store[f"{b}::mc::{c}::w2"] for c in class_names}
        data_w, data_w2 = store[f"{b}::data::w"], store[f"{b}::data::w2"]
        total = sum(mc_w.values())
        populated = np.nonzero((total[1:-1] != 0) | (data_w[1:-1] != 0))[0]
        binning = _plot_binning(tree_name, b, fine, meta["logx"][b], b in meta["score_branches"], populated)
        if binning is None:
            log_message(f"  [WARN] {tree_name}:{b} is empty, skipping")
            continue
        coarse, mapping, x_range, logx, logy, y_range = binning
        n = len(coarse) - 1
        mc_per_cls = {c: (_rebin(mc_w[c], mapping, n), _rebin(mc_w2[c], mapping, n)) for c in class_names}
        theory_up = theory_dn = None
        if f"{b}::theory" in store.files:
            theory_up, theory_dn = _theory_shape_band(_rebin(store[f"{b}::theory"], mapping, n))
        jobs.append({
            "out_dir": out_dir, "tree_name": tree_name, "branch": b, "class_names": class_names,
            "color_map": color_map, "edges": coarse, "bin_centers": 0.5 * (coarse[:-1] + coarse[1:]),
            "bin_widths": coarse[1:] - coarse[:-1], "mc_per_cls": mc_per_cls,
            "mc_total_v": sum(v[0] for v in mc_per_cls.values()),
            "mc_total_w2": sum(v[1] for v in mc_per_cls.values()),
            "data_v": _rebin(data_w, mapping, n), "data_w2": _rebin(data_w2, mapping, n),
            "x_range": x_range, "logx": logx, "logy": logy, "y_range": y_range,
            "theory_fracs": meta["theory_fracs"], "theory_up": theory_up, "theory_dn": theory_dn, "note": note,
        })
    log_message(f"Rendering {len(jobs) * 2} PDFs ({n_workers} workers)")
    if n_workers > 1 and len(jobs) > 1:
        with multiprocessing.Pool(processes=min(len(jobs), n_workers)) as pool:
            pool.map(_render_branch, jobs)
    else:
        for job in jobs:
            _render_branch(job)
    log_message(f"Finished data_mc.py plot for tree={tree_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Data vs MC comparison plotter: a fill stage that reads every input once and stores "
                    "all histograms, and a plot stage that renders them.")
    parser.add_argument("--no-selection", action="store_true",
                        help="Skip the selection.json threshold and clip-range cuts on the plot variables "
                             "(writes to output_root_nosel).")
    parser.add_argument("--unblind", action="store_true",
                        help="Fill without removing the events of the signal regions (default: blinded).")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--fill-only", action="store_true", help="Only fill and store the histograms.")
    stage.add_argument("--plot-only", "--restyle", action="store_true",
                       help="Only plot from the stored histograms (no inputs are read).")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Parallel worker processes (default: n_workers from config.json, "
                             f"or {os.cpu_count() or 4} = cpu_count).")
    args = parser.parse_args()
    n_workers = args.workers if args.workers is not None else _N_WORKERS
    log_message(f"Running data_mc.py: trees={','.join(SUBMIT_TREES)}, bdt_root={BDT_ROOT_PATT}, "
                f"no_selection={args.no_selection}, unblind={args.unblind}, fill={not args.plot_only}, "
                f"plot={not args.fill_only}, n_workers={n_workers}")
    for tree_name in SUBMIT_TREES:
        if not args.plot_only:
            _fill_tree(tree_name, no_selection=args.no_selection, unblind=args.unblind, n_workers=n_workers)
        if not args.fill_only:
            _plot_tree(tree_name, no_selection=args.no_selection, n_workers=n_workers)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
