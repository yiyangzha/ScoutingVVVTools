#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data vs MC histogram comparison plotter.

Reads convert_branch.C output ROOT files directly (per-sample trees), optionally
applies the trained-model selection.json clip/threshold cuts (no log transform),
and draws a stacked MC + data panel with a Data/MC ratio sub-panel.

Pass --no-selection to skip threshold and clip-range cuts on plot variables
(BDT model clip ranges are still applied internally for score computation).
"""

import argparse
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


def _theory_frac_uncertainty(tree_name, class_groups, mc_target_totals):
    """Return per-class fractional theory uncertainty (flat normalization).

    Combines PDF, scale, PS_ISR, PS_FSR in quadrature.  Returns a dict
    {class_name: fractional_uncertainty}.  Samples without theory weights
    are treated as having zero theory uncertainty.

    mc_target_totals: {sample_name: target_total} pre-computed by the caller.
    """
    if THEORY_SYST_YIELDS is None:
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
            sdata = THEORY_SYST_YIELDS.get(sname, {}).get(tree_name)
            if sdata is None:
                continue
            # Fractional uncertainty for each source: half-width of the band
            # relative to central_ratio (≈ 1.0 for well-behaved MC).
            def _half(up_key, dn_key):
                u = float(sdata.get(up_key, 1.0))
                d = float(sdata.get(dn_key, 1.0))
                return 0.5 * abs(u - d)

            delta = math.sqrt(
                _half("pdf_up",     "pdf_down"    ) ** 2
                + _half("scale_up",   "scale_down"  ) ** 2
                + _half("ps_isr_up",  "ps_isr_down" ) ** 2
                + _half("ps_fsr_up",  "ps_fsr_down" ) ** 2
            )
            num2 += (target * delta) ** 2

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


def _concat_parts(parts):
    if not parts:
        return None
    if len(parts) == 1:
        df = parts[0].reset_index(drop=True)
        parts.clear()
        return df
    df = pd.concat(parts, ignore_index=True)
    parts.clear()
    return df


def _load_tree(files, tree_name, branches, strict=True, max_entries=None):
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
            entry_stop = int(tree.num_entries)
            if remaining is not None:
                entry_stop = min(entry_stop, remaining)
            if entry_stop <= 0:
                continue
            df_part = tree.arrays(
                load,
                library="pd",
                entry_start=0,
                entry_stop=entry_stop,
            )
            parts.append(df_part)
            if remaining is not None:
                remaining -= len(df_part)
    return _concat_parts(parts)


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
            if rb in chunk.columns:
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
    For data pass target_total=1.0, raw_w_sum=1.0, reweight_branches=[].
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
            if rb in chunk.columns:
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
    files, tree_name, branch_edges, target_total, raw_w_sum, \
        reweight_branches, plot_thresholds, plot_clip_ranges, strict = args
    return _stream_hists(files, tree_name, branch_edges, target_total, raw_w_sum,
                         reweight_branches, plot_thresholds, plot_clip_ranges, strict)


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


def _stream_theory_shape(files, tree_name, branch_edges, target_total, raw_w_sum,
                         reweight_branches, plot_thresholds, plot_clip_ranges, strict):
    """Per-bin theory member histograms for one MC sample (shape variations).

    Reads with library='np' so the array-valued theory weight branches load.
    Returns {branch: member_hist} with member_hist of shape (nbins,
    _TH_N_MEMBERS): each column is the branch histogram filled with
    weight = nominal_weight * theory_member_weight, so deviations from column 0
    (the central PDF member, == nominal) give the per-bin theory variation.
    Same nominal weight / threshold / clip handling as _stream_hists.  Returns {}
    when the sample's files carry no theory weights.
    """
    scalar_cols = sorted(set(branch_edges.keys())
                         | set(plot_thresholds.keys())
                         | set(plot_clip_ranges.keys())
                         | set(reweight_branches))
    theory_arr_branches = ["LHEPdfWeight", "LHEPdfWeightAlphaS", "LHEScaleWeight", "PSWeight"]
    member_hists = {b: np.zeros((len(edges) - 1, _TH_N_MEMBERS), dtype=float)
                    for b, edges in branch_edges.items()}
    found = False

    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                continue
            tree = uf[tree_name]
            avail = set(tree.keys())
            if "LHEPdfWeight" not in avail:
                continue   # no theory weights in this file
            found = True
            load = ([c for c in scalar_cols if c in avail]
                    + [b for b in theory_arr_branches if b in avail])
            for chunk in tree.iterate(expressions=load, step_size=_CHUNK_SIZE, library="np"):
                n = len(chunk["LHEPdfWeight"])
                raw_w = np.ones(n, dtype=float)
                for rb in reweight_branches:
                    if rb in chunk:
                        raw_w *= np.asarray(chunk[rb], dtype=float)
                weight = raw_w * (target_total / raw_w_sum) if raw_w_sum > 0 else np.zeros(n)

                # Reuse the pandas threshold/clip helpers on the scalar columns.
                df = pd.DataFrame({c: np.asarray(chunk[c]) for c in scalar_cols if c in chunk})
                mask = np.ones(n, dtype=bool)
                if plot_thresholds:
                    mask = _threshold_mask(df, plot_thresholds).to_numpy(dtype=bool)
                if not mask.any():
                    continue
                df = df[mask].reset_index(drop=True)
                weight = weight[mask]
                if plot_clip_ranges:
                    _apply_clip(df, plot_clip_ranges)

                nsel = len(weight)
                M = np.ones((nsel, _TH_N_MEMBERS), dtype=float)

                def _fill(name, off, count):
                    if name in chunk:
                        a = np.asarray(chunk[name], dtype=float)[mask].reshape(nsel, -1)
                        c = min(count, a.shape[1])
                        M[:, off:off + c] = a[:, :c]

                _fill("LHEPdfWeight",       _TH_OFF_PDF,    _TH_N_PDF)
                _fill("LHEPdfWeightAlphaS", _TH_OFF_ALPHAS, _TH_N_ALPHAS)
                _fill("LHEScaleWeight",     _TH_OFF_SCALE,  _TH_N_SCALE)
                _fill("PSWeight",           _TH_OFF_PS,     _TH_N_PS)
                WM = weight[:, None] * M

                for b, edges in branch_edges.items():
                    if b not in df.columns:
                        continue
                    vals = df[b].to_numpy(dtype=float)
                    nb = len(edges) - 1
                    in_range = np.isfinite(vals) & (vals >= edges[0]) & (vals <= edges[-1])
                    if not in_range.any():
                        continue
                    bi = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nb - 1)
                    np.add.at(member_hists[b], bi[in_range], WM[in_range])

    return member_hists if found else {}


def _stream_theory_shape_worker(args):
    return _stream_theory_shape(*args)


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

    isr = member_hist[:, [_TH_OFF_PS + 0, _TH_OFF_PS + 1]]
    fsr = member_hist[:, [_TH_OFF_PS + 2, _TH_OFF_PS + 3]]
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
        arr = X[col].to_numpy(dtype=float, copy=True)
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


# -------------------- Histogram cache (for --restyle) --------------------
def _save_branch_hists(out_dir, tree_name, branch, edges, mc_per_cls, data_v, data_w2,
                       logx, logy, x_range, y_range, class_names, color_map, theory_fracs,
                       theory_shape_up=None, theory_shape_down=None):
    """Persist histogram arrays to JSON so plots can be restyled without re-reading ROOT files."""
    payload = {
        "branch":      branch,
        "tree_name":   tree_name,
        "class_names": class_names,
        "color_map":   color_map,
        "edges":       edges.tolist(),
        "mc_h":  {cls: mc_per_cls[cls][0].tolist() for cls in class_names},
        "mc_h2": {cls: mc_per_cls[cls][1].tolist() for cls in class_names},
        "data_v":  data_v.tolist(),
        "data_w2": data_w2.tolist(),
        "logx":    logx,
        "logy":    logy,
        "x_range": list(x_range),
        "y_range": list(y_range) if y_range is not None else None,
        "theory_fracs": theory_fracs if theory_fracs else {},
        "theory_shape_up":   theory_shape_up.tolist()   if theory_shape_up   is not None else None,
        "theory_shape_down": theory_shape_down.tolist() if theory_shape_down is not None else None,
    }
    path = os.path.join(out_dir, f"{tree_name}_{branch}_hists.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _restyle_from_hists(hists_path, out_dir):
    """Re-render the two PDFs for one branch from a saved histogram JSON."""
    with open(hists_path, encoding="utf-8") as fh:
        d = json.load(fh)
    branch      = d["branch"]
    tree_name   = d["tree_name"]
    class_names = d["class_names"]
    color_map   = d["color_map"]
    edges       = np.array(d["edges"])
    mc_per_cls  = {cls: (np.array(d["mc_h"][cls]), np.array(d["mc_h2"][cls]))
                   for cls in class_names}
    data_v      = np.array(d["data_v"])
    data_w2     = np.array(d["data_w2"])
    logx        = d["logx"]
    logy        = d["logy"]
    x_range     = tuple(d["x_range"])
    y_range     = tuple(d["y_range"]) if d["y_range"] is not None else None
    theory_fracs = d.get("theory_fracs", {})
    sh_up = np.array(d["theory_shape_up"])   if d.get("theory_shape_up")   is not None else None
    sh_dn = np.array(d["theory_shape_down"]) if d.get("theory_shape_down") is not None else None

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths  = edges[1:] - edges[:-1]
    mc_total_v  = np.zeros(len(bin_centers))
    mc_total_w2 = np.zeros(len(bin_centers))
    for cls in class_names:
        mc_total_v  += mc_per_cls[cls][0]
        mc_total_w2 += mc_per_cls[cls][1]

    out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
    _draw_data_mc_plot(
        class_names=class_names, color_map=color_map,
        edges=edges, bin_centers=bin_centers, bin_widths=bin_widths,
        mc_per_cls=mc_per_cls, mc_total_v=mc_total_v, mc_total_w2=mc_total_w2,
        data_v=data_v, data_w2=data_w2,
        branch=branch, x_range=x_range, logx=logx, logy=logy, y_range=y_range,
        y_label="Events", out_path=out_path, logy_floor=0.1,
        theory_fracs=theory_fracs, theory_shape_up=sh_up, theory_shape_down=sh_dn,
    )
    log_message(f"Wrote plot file: {out_path}")

    (mc_per_cls_norm, mc_total_v_norm, mc_total_w2_norm, data_v_norm, data_w2_norm) = \
        _unit_normalized_histograms(mc_per_cls, mc_total_v, mc_total_w2, data_v, data_w2)
    out_path_normal = os.path.join(out_dir, f"{tree_name}_{branch}_normal.pdf")
    _draw_data_mc_plot(
        class_names=class_names, color_map=color_map,
        edges=edges, bin_centers=bin_centers, bin_widths=bin_widths,
        mc_per_cls=mc_per_cls_norm, mc_total_v=mc_total_v_norm, mc_total_w2=mc_total_w2_norm,
        data_v=data_v_norm, data_w2=data_w2_norm,
        branch=branch, x_range=x_range, logx=logx, logy=logy, y_range=y_range,
        y_label="A.U.", out_path=out_path_normal, logy_floor=None,
    )
    log_message(f"Wrote plot file: {out_path_normal}")


def _restyle_tree(tree_name, no_selection=False, n_workers=None):
    """Re-render all cached branch plots for one tree without reading ROOT files."""
    if n_workers is None:
        n_workers = _N_WORKERS
    out_patt = OUTPUT_ROOT_NOSEL_PATT if no_selection else OUTPUT_ROOT_PATT
    out_dir  = _resolve(out_patt.format(tree_name=tree_name), _SCRIPT_DIR)
    pattern  = os.path.join(out_dir, f"{tree_name}_*_hists.json")
    hists_files = sorted(glob.glob(pattern))
    if not hists_files:
        log_message(f"[WARN] No histogram cache files found in {out_dir} for tree '{tree_name}'")
        return
    log_message(f"Restyling {len(hists_files)} cached branches for tree '{tree_name}' ({n_workers} workers)")
    if n_workers > 1 and len(hists_files) > 1:
        with multiprocessing.Pool(processes=min(len(hists_files), n_workers)) as pool:
            pool.map(_restyle_worker, [(h, out_dir) for h in hists_files])
    else:
        for hpath in hists_files:
            _restyle_from_hists(hpath, out_dir)
    log_message(f"Finished restyling for tree={tree_name}")


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


# -------------------- Parallel workers --------------------
def _draw_plot_job(job):
    """Render the absolute and unit-normalised PDFs for one branch (pool worker)."""
    out_dir      = job["out_dir"]
    tree_name    = job["tree_name"]
    branch       = job["branch"]
    class_names  = job["class_names"]
    color_map    = job["color_map"]
    edges        = job["edges"]
    bin_centers  = job["bin_centers"]
    bin_widths   = job["bin_widths"]
    mc_per_cls   = job["mc_per_cls"]
    mc_total_v   = job["mc_total_v"]
    mc_total_w2  = job["mc_total_w2"]
    data_v       = job["data_v"]
    data_w2      = job["data_w2"]
    x_range      = job["x_range"]
    logx         = job["logx"]
    logy         = job["logy"]
    y_range      = job["y_range"]
    theory_fracs = job.get("theory_fracs", {})
    sh_up        = job.get("theory_shape_up", None)
    sh_dn        = job.get("theory_shape_down", None)

    out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
    _draw_data_mc_plot(
        class_names=class_names, color_map=color_map,
        edges=edges, bin_centers=bin_centers, bin_widths=bin_widths,
        mc_per_cls=mc_per_cls, mc_total_v=mc_total_v, mc_total_w2=mc_total_w2,
        data_v=data_v, data_w2=data_w2,
        branch=branch, x_range=x_range, logx=logx, logy=logy, y_range=y_range,
        y_label="Events", out_path=out_path, logy_floor=0.1,
        theory_fracs=theory_fracs, theory_shape_up=sh_up, theory_shape_down=sh_dn,
    )
    (mc_per_cls_norm, mc_total_v_norm, mc_total_w2_norm, data_v_norm, data_w2_norm) = \
        _unit_normalized_histograms(mc_per_cls, mc_total_v, mc_total_w2, data_v, data_w2)
    out_path_normal = os.path.join(out_dir, f"{tree_name}_{branch}_normal.pdf")
    _draw_data_mc_plot(
        class_names=class_names, color_map=color_map,
        edges=edges, bin_centers=bin_centers, bin_widths=bin_widths,
        mc_per_cls=mc_per_cls_norm, mc_total_v=mc_total_v_norm, mc_total_w2=mc_total_w2_norm,
        data_v=data_v_norm, data_w2=data_w2_norm,
        branch=branch, x_range=x_range, logx=logx, logy=logy, y_range=y_range,
        y_label="A.U.", out_path=out_path_normal, logy_floor=None,
    )
    return out_path, out_path_normal


def _restyle_worker(args):
    hists_path, out_dir = args
    _restyle_from_hists(hists_path, out_dir)


# -------------------- Per-tree processing --------------------
def _process_tree(tree_name, no_selection=False, use_cached_ranges=False, save_hists=False, n_workers=None):
    if n_workers is None:
        n_workers = _N_WORKERS
    log_message(f"Running data_mc.py: tree={tree_name}, no_selection={no_selection}, n_workers={n_workers}")

    log_message("Loading trained-model config copies")
    bdt_cfg, bdt_br, bdt_sel, test_meta = _bdt_configs_for_tree(tree_name, load_test_meta=not no_selection)
    # Optional plot-config override of the stack grouping.  Lets a reused BDT config (e.g. the ttbar
    # control region borrowing fat2's trained model) define its own classes — including backgrounds
    # absent from the signal-region model, such as W/Z+jets — without editing the shared BDT config.
    # Accepts a flat {class: [samples]} mapping or a {tree_name: {class: [samples]}} mapping.
    _cg_override = plot_cfg.get("class_groups")
    if isinstance(_cg_override, dict) and tree_name in _cg_override and isinstance(_cg_override[tree_name], dict):
        _cg_override = _cg_override[tree_name]
    class_groups     = _cg_override or bdt_cfg["class_groups"]
    # Optional MC whitelist: restrict the stack to samples actually produced for this tree.  A reused
    # BDT config (e.g. the ttbar control region borrowing fat2's class_groups) lists signal-region
    # samples that the CR convert never produced; intersecting with the whitelist drops them and any
    # now-empty class.  Accepts a flat list or a {tree_name: [...]} mapping; absent = keep all.
    _mc_whitelist = plot_cfg.get("mc_samples")
    if isinstance(_mc_whitelist, dict):
        _mc_whitelist = _mc_whitelist.get(tree_name)
    if _mc_whitelist is not None:
        _wl = set(_mc_whitelist)
        class_groups = {cls: [s for s in samples if s in _wl]
                        for cls, samples in class_groups.items()}
        class_groups = {cls: samples for cls, samples in class_groups.items() if samples}
    class_names      = list(class_groups.keys())
    # bdt_br is keyed by the trained-model trees (e.g. fat2/fat3); a reused BDT config (such as the
    # ttbar control region borrowing fat2's class_groups) has no entry for this tree.  model_branches
    # is only consumed by the scoring blocks below, which are skipped in --no-selection mode, so an
    # empty list is safe here.
    model_branches   = [item["name"] for item in bdt_br.get(tree_name, [])]
    # Score branches require a trained model; skip them in --no-selection mode.
    score_branches   = [] if no_selection else [_score_branch_name(class_name) for class_name in class_names]

    bdt_root_dir   = _bdt_root_for_tree(tree_name)
    bdt_script_dir = os.path.dirname(bdt_root_dir)
    if no_selection:
        # Read directly from the convert output (no mixing step required).
        _conv_dir     = os.path.dirname(_CONVERT_CFG_PATH)
        input_root    = _resolve(convert_cfg["output_root"], _conv_dir)
        # convert uses {output_root} as placeholder; _input_files expects {input_root}
        input_pattern = convert_cfg["output_pattern"].replace("{output_root}", "{input_root}")
    else:
        input_root    = _resolve(bdt_cfg["input_root"], bdt_script_dir)
        input_pattern = bdt_cfg["input_pattern"]

    sel              = bdt_sel.get(tree_name, {})
    clip_ranges      = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    thresholds       = {k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in sel.get("thresholds", {}).items()}
    log_tf_set       = set(sel.get("log_transform", []))
    # When --no-selection is active, skip event-level cuts and plot-variable clipping.
    # clip_ranges is still passed to _standardize_model_X so BDT scores stay valid.
    # Extra plot-only thresholds from config.json["plot_thresholds"][tree_name] are always
    # applied (even in --no-selection mode) and do not affect BDT training/selection.json.
    _extra_thresh_cfg = plot_cfg.get("plot_thresholds", {}).get(tree_name, {}) or {}
    extra_plot_thresholds = {k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in _extra_thresh_cfg.items()}
    plot_thresholds  = {**({} if no_selection else thresholds), **extra_plot_thresholds}
    plot_clip_ranges = {} if no_selection else clip_ranges

    skip_score = _skip_branches_for_tree(tree_name)
    branches_to_plot = _plot_branches_for_tree(tree_name)
    score_branches = [branch for branch in score_branches if branch not in skip_score]
    for branch in score_branches:
        if branch not in branches_to_plot:
            branches_to_plot.append(branch)
    root_plot_branches = [branch for branch in branches_to_plot if branch not in score_branches]
    reweight_cfg      = plot_cfg.get("event_reweight_branches", {})
    reweight_branches = list(reweight_cfg.get(tree_name, []))
    score_reweight_branches = list(bdt_cfg.get(tree_name, {}).get("event_reweight_branches", []))
    log_message(
        f"Resolved plotting config: branches={len(branches_to_plot)}, "
        f"threshold_branches={len(thresholds)}, clip_branches={len(clip_ranges)}, "
        f"reweight_branches={len(reweight_branches)}, score_branches={len(score_branches)}"
    )
    out_patt = OUTPUT_ROOT_NOSEL_PATT if no_selection else OUTPUT_ROOT_PATT
    out_dir = _resolve(out_patt.format(tree_name=tree_name), _SCRIPT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    log_message(f"Output directory: {out_dir}")

    # Determine which root branches need auto-ranging (no explicit x_range override).
    branch_logx = {b: bool(_branch_override(tree_name, b).get("logx", b in log_tf_set))
                   for b in root_plot_branches}
    auto_range_branches = [b for b in root_plot_branches
                           if "x_range" not in _branch_override(tree_name, b)
                           and not b.startswith("score_")]

    # Ranges cache: auto-saved after every full run; loaded with --use-cached-ranges
    # to skip the range-collection pre-pass (MC pre-pass still runs for reweight sums).
    _ranges_cache_path = os.path.join(out_dir, ".ranges_cache.json")
    _using_range_cache = use_cached_ranges and os.path.exists(_ranges_cache_path)
    if _using_range_cache:
        _cached_ranges = _load_json(_ranges_cache_path)
        merged_ranges = {b: tuple(v) for b, v in _cached_ranges.items()}
        auto_range_branches = []  # skip range collection; pre-pass only reads reweight cols
        log_message(f"Loaded ranges cache: {_ranges_cache_path} ({len(merged_ranges)} branches)")

    # ---- Step 1: Discover files and count entries (serial, fast metadata reads) ----
    n_mc_samples = sum(len(s) for s in class_groups.values())
    log_message(f"Pre-pass: {n_mc_samples} MC samples, {len(DATA_SAMPLES)} data samples")
    mc_n_totals     = {}
    mc_sample_files = {}
    mc_class_map    = {}  # sname -> cls_name

    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = _input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            n_total = _tree_entries_total(files, tree_name)
            if n_total <= 0:
                log_message(f"  [WARN] skipping MC sample '{sname}': no entries for tree '{tree_name}'")
                continue
            mc_n_totals[sname]     = n_total
            mc_sample_files[sname] = files
            mc_class_map[sname]    = cls_name

    data_sample_files = {}
    for sname in DATA_SAMPLES:
        files = _input_files(sname, input_root, input_pattern)
        if not files:
            raise RuntimeError(f"No ROOT files found for data sample '{sname}'")
        data_sample_files[sname] = files

    # ---- Step 2: Parallel pre-pass (range collection + reweight sums) ----
    mc_raw_w_sums = {}
    prepass_mins  = {b:  np.inf for b in auto_range_branches}
    prepass_maxs  = {b: -np.inf for b in auto_range_branches}

    need_prepass = bool(auto_range_branches or reweight_branches)
    if not need_prepass:
        for sname in mc_sample_files:
            mc_raw_w_sums[sname] = float(mc_n_totals[sname])
            log_message(f"  {sname}: n_total={mc_n_totals[sname]}, raw_w_sum={mc_raw_w_sums[sname]:.6g}")
        for sname in DATA_SAMPLES:
            log_message(f"  data {sname}: pre-pass skipped (no auto-range branches, no reweighting)")
    else:
        prepass_tasks  = []
        prepass_labels = []
        for sname in mc_sample_files:
            prepass_tasks.append((mc_sample_files[sname], tree_name, auto_range_branches,
                                   reweight_branches, branch_logx, not no_selection))
            prepass_labels.append(('mc', sname))
        if auto_range_branches:
            for sname in DATA_SAMPLES:
                prepass_tasks.append((data_sample_files[sname], tree_name, auto_range_branches,
                                       [], branch_logx, not no_selection))
                prepass_labels.append(('data', sname))

        log_message(f"  Running {len(prepass_tasks)} pre-pass tasks ({n_workers} workers)")
        if n_workers > 1 and len(prepass_tasks) > 1:
            with multiprocessing.Pool(processes=min(len(prepass_tasks), n_workers)) as pool:
                prepass_results = pool.map(_prepass_worker, prepass_tasks)
        else:
            prepass_results = [_prepass(*t) for t in prepass_tasks]

        for (kind, sname), (rw_sum, ranges) in zip(prepass_labels, prepass_results):
            for b, (lo, hi) in ranges.items():
                prepass_mins[b] = min(prepass_mins[b], lo)
                prepass_maxs[b] = max(prepass_maxs[b], hi)
            if kind == 'mc':
                mc_raw_w_sums[sname] = rw_sum if rw_sum > 0 else float(mc_n_totals[sname])
                log_message(f"  {sname}: n_total={mc_n_totals[sname]}, raw_w_sum={mc_raw_w_sums[sname]:.6g}")
            else:
                log_message(f"  data {sname}: pre-pass done")

        for sname in DATA_SAMPLES:
            if not auto_range_branches:
                log_message(f"  data {sname}: pre-pass skipped (no auto-range branches)")

    if not _using_range_cache:
        merged_ranges = {}
        for b in auto_range_branches:
            lo, hi = prepass_mins[b], prepass_maxs[b]
            if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
                merged_ranges[b] = (lo, hi)
        if auto_range_branches:
            with open(_ranges_cache_path, "w", encoding="utf-8") as fh:
                json.dump({b: list(rng) for b, rng in merged_ranges.items()}, fh, indent=2)
            log_message(f"Saved ranges cache: {_ranges_cache_path}")

    # Build derived model score branches. MC scores use the saved test split and
    # are validated against train.py's signal-region reference; data scores use
    # the full configured data input, matching the ordinary branch plots.
    score_class_dfs = {}
    score_data_df = None
    if score_branches:
        log_message("Preparing model score branches")
        clf = _load_score_model(bdt_root_dir, bdt_cfg, tree_name)
        decorrelate = list(bdt_cfg.get(tree_name, {}).get("decorrelate", []))
        score_load = sorted(set(model_branches) | set(thresholds.keys()) | set(score_reweight_branches) | set(extra_plot_thresholds.keys()))
        sample_to_class_name = {}
        sample_to_class_idx = {}
        for idx, (cls_name, samples) in enumerate(class_groups.items()):
            for sample_name in samples:
                sample_to_class_name[sample_name] = cls_name
                sample_to_class_idx[sample_name] = idx

        score_parts_by_class = {cls_name: [] for cls_name in class_names}
        ref_sample_labels = []
        ref_class_idx = []
        ref_weights = []
        ref_proba_parts = []
        ref_feature_names = None

        # Load + preprocess all MC test-split samples; defer inference until after concat.
        log_message(f"Loading MC score test split samples: n={len(test_meta['samples'])}")
        all_X_parts = []
        batch_meta  = []  # (sample_name, cls_name, cls_idx, weight_arr, n_rows)

        for sample_name, sample_meta in test_meta["samples"].items():
            if sample_name not in sample_to_class_name:
                raise RuntimeError(f"Test split sample '{sample_name}' is not in class_groups")
            df = _load_test_segments(tree_name, score_load, sample_meta)
            if df is None or len(df) == 0:
                raise RuntimeError(f"No test split events loaded for sample '{sample_name}'")
            df = _assign_test_split_mc_weight(
                df, sample_name, int(sample_meta["total_entries"]), score_reweight_branches,
            )
            mask = _threshold_mask(df, plot_thresholds)
            df = df.loc[mask].reset_index(drop=True)
            if len(df) == 0:
                log_message(f"  [WARN] score sample '{sample_name}' has zero events after filtering")
                continue
            X_part = _standardize_model_X(df[model_branches], clip_ranges, list(log_tf_set))
            X_part = _drop_decorrelated_features(X_part, decorrelate)
            if ref_feature_names is None:
                ref_feature_names = list(X_part.columns)
            all_X_parts.append(X_part)
            batch_meta.append((
                sample_name,
                sample_to_class_name[sample_name],
                sample_to_class_idx[sample_name],
                df["weight"].to_numpy(dtype=float, copy=False),
                len(df),
            ))

        if not all_X_parts:
            raise RuntimeError(f"No MC score events after filtering for tree '{tree_name}'")

        # Single batched inference across all samples.
        n_score_events = sum(m[4] for m in batch_meta)
        log_message(f"  Batched BDT inference: {n_score_events} events across {len(batch_meta)} samples")
        X_all     = pd.concat(all_X_parts, ignore_index=True)
        proba_all = _predict_model_proba(clf, X_all, len(class_names))
        del X_all, all_X_parts

        # Split predictions back to per-sample; accumulate score DataFrames + reference arrays.
        offset = 0
        for sample_name, cls_name, cls_idx, weights, n_rows in batch_meta:
            proba    = proba_all[offset : offset + n_rows]
            score_df = pd.DataFrame({"weight": weights})
            for i, cname in enumerate(class_names):
                score_df[_score_branch_name(cname)] = proba[:, i]
            score_parts_by_class[cls_name].append(score_df)
            ref_sample_labels.extend([sample_name] * n_rows)
            ref_class_idx.extend([cls_idx] * n_rows)
            ref_weights.extend(weights)
            ref_proba_parts.append(proba)
            offset += n_rows
            log_message(
                f"  score {sample_name}: class={cls_name}, test_loaded={n_rows}, "
                f"weight_sum={float(weights.sum()):.6g}"
            )

        for cls_name, parts in score_parts_by_class.items():
            if parts:
                score_class_dfs[cls_name] = _concat_parts(parts)

        if not ref_proba_parts:
            raise RuntimeError(f"No MC score events after filtering for tree '{tree_name}'")
        score_proba_ref = np.concatenate(ref_proba_parts, axis=0)
        #_compare_score_reference(
        #    os.path.join(bdt_root_dir, "test_reference_signal_region.npz"),
        #    ref_feature_names,
        #    ref_sample_labels,
        #    ref_class_idx,
        #    ref_weights,
        #    score_proba_ref,
        #)
        del ref_sample_labels, ref_class_idx, ref_weights, ref_proba_parts, score_proba_ref, proba_all

        if DATA_SAMPLES:
            score_data_load = sorted(set(model_branches) | set(thresholds.keys()))
            log_message(f"Loading data score samples: n={len(DATA_SAMPLES)}")
            data_X_parts = []
            data_w_parts = []
            for sname in DATA_SAMPLES:
                files = _input_files(sname, input_root, input_pattern)
                if not files:
                    raise RuntimeError(f"No ROOT files found for data score sample '{sname}'")
                df = _load_tree(files, tree_name, score_data_load)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero entries")
                    continue
                df = _apply_thresholds(df, plot_thresholds)
                if df is None or len(df) == 0:
                    log_message(f"  [WARN] data score sample '{sname}' has zero events after filtering")
                    continue
                df["weight"] = 1.0
                X_part = _standardize_model_X(df[model_branches], clip_ranges, list(log_tf_set))
                X_part = _drop_decorrelated_features(X_part, decorrelate)
                data_X_parts.append(X_part)
                data_w_parts.append(df["weight"].to_numpy(dtype=float, copy=False))
                log_message(f"  data score {sname}: events={len(df)}")
            if data_X_parts:
                X_data_all  = pd.concat(data_X_parts, ignore_index=True)
                proba_data  = _predict_model_proba(clf, X_data_all, len(class_names))
                del X_data_all, data_X_parts
                score_data_df = pd.DataFrame({"weight": np.concatenate(data_w_parts)})
                for i, cname in enumerate(class_names):
                    score_data_df[_score_branch_name(cname)] = proba_data[:, i]
                log_message(f"Loaded data score events: {len(score_data_df)}")
            else:
                log_message("Loaded data score events: 0")

    # Resolve binning for all root branches using pre-pass ranges.
    log_message("Resolving binning for all branches")
    branch_binning = {}
    for b in root_plot_branches:
        precomp = merged_ranges.get(b)   # None for explicit-range branches (override takes priority)
        binning = _resolve_binning(tree_name, b, precomp, log_tf_set)
        if binning is None:
            log_message(f"  [WARN] no range for {tree_name}:{b}, will skip")
        else:
            branch_binning[b] = binning
    branch_edges = {b: _bin_edges(bins, x_range, logx)
                    for b, (bins, x_range, logx, logy, y_range) in branch_binning.items()}
    log_message(f"  {len(branch_binning)}/{len(root_plot_branches)} root branches have valid binning")

    # ---- Step 5: Parallel histogram streaming (MC + data) ----
    mc_target_totals = {}
    stream_tasks  = []
    stream_labels = []  # ('mc', cls_name, sname, target_total) or ('data', sname, ...)

    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in mc_sample_files:
                continue
            info = SAMPLE_INFO[sname]
            xsec = float(info.get("xsection", 0.0))
            raw_entries = float(info.get("raw_entries", 0.0))
            n_total  = mc_n_totals[sname]
            raw_w_sum = mc_raw_w_sums[sname]
            target_total = (LUMI_TOTAL * xsec * float(n_total) / raw_entries
                            if raw_entries > 0 else 0.0)
            mc_target_totals[sname] = target_total
            stream_tasks.append((mc_sample_files[sname], tree_name, branch_edges,
                                   target_total, raw_w_sum, reweight_branches,
                                   plot_thresholds, plot_clip_ranges, not no_selection))
            stream_labels.append(('mc', cls_name, sname, target_total))

    for sname in DATA_SAMPLES:
        stream_tasks.append((data_sample_files[sname], tree_name, branch_edges,
                              1.0, 1.0, [], plot_thresholds, plot_clip_ranges, not no_selection))
        stream_labels.append(('data', sname, sname, 1.0))

    log_message(f"Streaming histograms: {len(stream_tasks)} samples ({n_workers} workers)")
    if n_workers > 1 and len(stream_tasks) > 1:
        with multiprocessing.Pool(processes=min(len(stream_tasks), n_workers)) as pool:
            stream_results = pool.map(_stream_hists_worker, stream_tasks)
    else:
        stream_results = [_stream_hists(*t) for t in stream_tasks]

    mc_hists   = {cls: {} for cls in class_names}
    data_hists = {}
    for label, sample_hists in zip(stream_labels, stream_results):
        kind = label[0]
        if kind == 'mc':
            _, cls_name, sname, target_total = label
            log_message(f"  MC {sname} (class={cls_name}): target_total={target_total:.6g}")
            for b, (h, h2) in sample_hists.items():
                if b not in mc_hists[cls_name]:
                    mc_hists[cls_name][b] = [h, h2]
                else:
                    mc_hists[cls_name][b][0] += h
                    mc_hists[cls_name][b][1] += h2
        else:
            _, sname, _, _ = label
            log_message(f"  data {sname}: streaming done")
            for b, (h, h2) in sample_hists.items():
                if b not in data_hists:
                    data_hists[b] = [h, h2]
                else:
                    data_hists[b][0] += h
                    data_hists[b][1] += h2

    mc_hists   = {cls: {b: (v[0], v[1]) for b, v in mc_hists[cls].items()} for cls in class_names}
    data_hists = {b: (v[0], v[1]) for b, v in data_hists.items()}
    theory_fracs = _theory_frac_uncertainty(tree_name, class_groups, mc_target_totals)

    # ---- Step 5b: Per-bin theory SHAPE variations from the LHE weights ----
    # A separate pass over the theory-weight samples (read with library='np' so
    # the array branches load) accumulates per-bin member histograms; the band
    # is the per-bin spread (PDF Hessian + alpha_s + scale/PS envelopes) summed
    # in quadrature, placed around the total MC.  Falls back to the flat
    # theory_fracs band for branches it cannot cover (e.g. score branches).
    theory_shape = {}   # branch -> (band_up, band_down) absolute per-bin arrays
    theory_samples = [sname for _cls, samples in class_groups.items()
                      for sname in samples
                      if sname in mc_sample_files
                      and SAMPLE_INFO.get(sname, {}).get("has_theory_weights", False)]
    if theory_samples and branch_edges:
        th_tasks = [
            (mc_sample_files[sname], tree_name, branch_edges,
             mc_target_totals.get(sname, 0.0), mc_raw_w_sums[sname],
             reweight_branches, plot_thresholds, plot_clip_ranges, not no_selection)
            for sname in theory_samples
        ]
        log_message(f"Theory-shape pass: {len(th_tasks)} theory samples ({n_workers} workers)")
        if n_workers > 1 and len(th_tasks) > 1:
            with multiprocessing.Pool(processes=min(len(th_tasks), n_workers)) as pool:
                th_results = pool.map(_stream_theory_shape_worker, th_tasks)
        else:
            th_results = [_stream_theory_shape(*t) for t in th_tasks]
        summed = {}
        for res in th_results:
            for b, mh in res.items():
                if b in summed:
                    summed[b] += mh
                else:
                    summed[b] = mh.copy()
        for b, mh in summed.items():
            up, dn = _theory_shape_band(mh)
            if np.any(up > 0.0) or np.any(dn > 0.0):
                theory_shape[b] = (up, dn)
        log_message(f"Theory-shape pass: per-bin bands for {len(theory_shape)} branches")

    # ---- Step 6: Collect plot jobs ----
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}

    log_message(f"Collecting plot data: {len(branches_to_plot)} branches")
    plot_jobs = []
    for branch in branches_to_plot:
        is_score_branch = branch in score_branches

        if is_score_branch:
            arrs = []
            for cls in class_names:
                if cls in score_class_dfs and branch in score_class_dfs[cls].columns:
                    arrs.append(score_class_dfs[cls][branch].values)
            if score_data_df is not None and branch in score_data_df.columns:
                arrs.append(score_data_df[branch].values)
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
                if cls in score_class_dfs and branch in score_class_dfs[cls].columns:
                    h, h2 = _weighted_hist(
                        score_class_dfs[cls][branch].values,
                        score_class_dfs[cls]["weight"].values, edges,
                    )
                else:
                    h, h2 = np.zeros(bins), np.zeros(bins)
                mc_per_cls[cls] = (h, h2)
                mc_total_v  += h
                mc_total_w2 += h2
            if score_data_df is not None and branch in score_data_df.columns:
                data_v, data_w2 = _weighted_hist(
                    score_data_df[branch].values, score_data_df["weight"].values, edges,
                )
            else:
                data_v, data_w2 = np.zeros(bins), np.zeros(bins)
        else:
            if branch not in branch_binning:
                log_message(f"  [WARN] no range for {tree_name}:{branch}, skipping")
                continue
            bins, x_range, logx, logy, y_range = branch_binning[branch]
            edges       = branch_edges[branch]
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            bin_widths  = edges[1:] - edges[:-1]
            mc_total_v  = np.zeros(bins)
            mc_total_w2 = np.zeros(bins)
            mc_per_cls  = {}
            for cls in class_names:
                h, h2 = mc_hists[cls].get(branch, (np.zeros(bins), np.zeros(bins)))
                mc_per_cls[cls] = (h, h2)
                mc_total_v  += h
                mc_total_w2 += h2
            data_v, data_w2 = data_hists.get(branch, (np.zeros(bins), np.zeros(bins)))

        sh_up, sh_dn = theory_shape.get(branch, (None, None))

        if save_hists:
            _save_branch_hists(
                out_dir, tree_name, branch, edges, mc_per_cls, data_v, data_w2,
                logx, logy, x_range, y_range, class_names, color_map, theory_fracs,
                sh_up, sh_dn,
            )

        plot_jobs.append({
            "out_dir": out_dir, "tree_name": tree_name, "branch": branch,
            "class_names": class_names, "color_map": color_map,
            "edges": edges, "bin_centers": bin_centers, "bin_widths": bin_widths,
            "mc_per_cls": mc_per_cls, "mc_total_v": mc_total_v, "mc_total_w2": mc_total_w2,
            "data_v": data_v, "data_w2": data_w2,
            "x_range": x_range, "logx": logx, "logy": logy, "y_range": y_range,
            "theory_fracs": theory_fracs,
            "theory_shape_up": sh_up, "theory_shape_down": sh_dn,
        })

    # ---- Step 7: Parallel plot rendering ----
    log_message(f"Rendering {len(plot_jobs) * 2} PDFs ({n_workers} workers)")
    if n_workers > 1 and len(plot_jobs) > 1:
        with multiprocessing.Pool(processes=min(len(plot_jobs), n_workers)) as pool:
            pool.map(_draw_plot_job, plot_jobs)
    else:
        for job in plot_jobs:
            _draw_plot_job(job)
    log_message(f"Finished data_mc.py for tree={tree_name}")


def main():
    parser = argparse.ArgumentParser(description="Data vs MC comparison plotter")
    parser.add_argument(
        "--no-selection", action="store_true",
        help="Skip threshold and clip-range cuts on plot variables (writes to output_root_nosel).",
    )
    parser.add_argument(
        "--use-cached-ranges", action="store_true",
        help=(
            "Load axis ranges from .ranges_cache.json (written by a previous run) and skip "
            "the range-collection pre-pass.  The MC pre-pass still runs to compute reweight "
            "branch sums when event_reweight_branches are configured."
        ),
    )
    parser.add_argument(
        "--save-hists", action="store_true",
        help=(
            "Save per-branch histogram arrays to {tree}_{branch}_hists.json alongside the PDFs. "
            "Use --restyle on a later run to re-render plots from these files without reading ROOT files."
        ),
    )
    parser.add_argument(
        "--restyle", action="store_true",
        help=(
            "Re-render all plots from previously saved *_hists.json files.  "
            "No ROOT files are read.  Requires a prior run with --save-hists."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            f"Number of parallel worker processes (default: n_workers from config.json, "
            f"or {os.cpu_count() or 4} = cpu_count).  Pass 1 to disable parallelism."
        ),
    )
    args = parser.parse_args()
    n_workers = args.workers if args.workers is not None else _N_WORKERS

    out_patt = OUTPUT_ROOT_NOSEL_PATT if args.no_selection else OUTPUT_ROOT_PATT

    if args.restyle:
        log_message(
            f"Restyling from histogram cache: trees={','.join(SUBMIT_TREES)}, "
            f"output_root={out_patt}, no_selection={args.no_selection}, n_workers={n_workers}"
        )
        for tree_name in SUBMIT_TREES:
            _restyle_tree(tree_name, no_selection=args.no_selection, n_workers=n_workers)
        return

    log_message(
        f"Running data_mc.py: trees={','.join(SUBMIT_TREES)}, "
        f"bdt_root={BDT_ROOT_PATT}, output_root={out_patt}, "
        f"no_selection={args.no_selection}, "
        f"use_cached_ranges={args.use_cached_ranges}, save_hists={args.save_hists}, "
        f"n_workers={n_workers}"
    )
    for tree_name in SUBMIT_TREES:
        _process_tree(
            tree_name,
            no_selection=args.no_selection,
            use_cached_ranges=args.use_cached_ranges,
            save_hists=args.save_hists,
            n_workers=n_workers,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
