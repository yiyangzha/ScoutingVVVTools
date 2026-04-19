#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data vs MC histogram comparison plotter.

Reads convert_branch.C output ROOT files directly (per-sample trees), applies
the BDT selection.json clip/threshold cuts (no log transform), and draws a
stacked MC + data panel with a Data/MC ratio sub-panel.
"""

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


# -------------------- BDT config copies --------------------
def _bdt_root_for_tree(tree_name):
    return _resolve(BDT_ROOT_PATT.format(tree_name=tree_name), _SCRIPT_DIR)


def _bdt_configs_for_tree(tree_name):
    bdt_root = _bdt_root_for_tree(tree_name)
    cfg = _load_json(os.path.join(bdt_root, "config.json"))
    sel = _load_json(os.path.join(bdt_root, "selection.json"))
    return cfg, sel


# -------------------- Input file resolution --------------------
def _sample_group(info):
    if not info.get("is_MC", True):
        return "data"
    return "signal" if info.get("is_signal", False) else "bkg"


def _input_files(sample_name, input_root, input_pattern):
    info = SAMPLE_INFO[sample_name]
    sg   = _sample_group(info)
    base = input_pattern.format(input_root=input_root, sample_group=sg, sample=sample_name)
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


def _load_tree(files, tree_name, branches):
    parts = []
    for fpath in files:
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
            parts.append(tree.arrays(branches, library="pd"))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


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
    raise TypeError(f"Unsupported threshold condition: {cond!r}")


def _apply_thresholds(df, thresholds):
    if not thresholds or df is None or len(df) == 0:
        return df
    mask = pd.Series(True, index=df.index)
    for b, cond in thresholds.items():
        if b not in df.columns:
            continue
        col = df[b]
        sentinel = col < -990
        mask &= ~sentinel
        mask &= _mask_from_cond(col, cond)
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
    if raw_entries == 0.0 or n_loaded == 0 or tree_entries_total == 0:
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
    logx     = bool(override.get("logx", branch in log_tf_set))
    logy     = bool(override.get("logy", True))
    y_range  = tuple(override["y_range"]) if "y_range" in override else None

    if "x_range" in override:
        x_lo, x_hi = override["x_range"]
        x_range = (float(x_lo), float(x_hi))
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


# -------------------- Per-tree processing --------------------
def _process_tree(tree_name):
    log_message(f"Running data_mc.py: tree={tree_name}")

    log_message("Loading BDT config copies")
    bdt_cfg, bdt_sel = _bdt_configs_for_tree(tree_name)
    class_groups     = bdt_cfg["class_groups"]
    class_names      = list(class_groups.keys())

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

    branches_to_plot = _plot_branches_for_tree(tree_name)
    need_load        = sorted(set(branches_to_plot)
                              | set(thresholds.keys())
                              | set(clip_ranges.keys()))
    reweight_cfg      = plot_cfg.get("event_reweight_branches", {})
    reweight_branches = list(reweight_cfg.get(tree_name, []))
    mc_need_load     = sorted(set(need_load) | set(reweight_branches))
    log_message(
        f"Resolved plotting config: branches={len(branches_to_plot)}, "
        f"threshold_branches={len(thresholds)}, clip_branches={len(clip_ranges)}, "
        f"reweight_branches={len(reweight_branches)}"
    )

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
                log_message(f"  [WARN] no files for '{sname}', skipping")
                continue
            n_total = _tree_entries_total(files, tree_name)
            if n_total <= 0:
                log_message(f"  [WARN] empty tree '{tree_name}' for '{sname}', skipping")
                continue
            df = _load_tree(files, tree_name, mc_need_load)
            if df is None or len(df) == 0:
                continue
            df = _assign_mc_weight(df, sname, n_total, len(df), reweight_branches)
            dfs.append(df)
            log_message(
                f"  {sname}: class={cls_name}, tree_entries={n_total}, "
                f"loaded={len(df)}, weight_sum={float(df['weight'].sum()):.6g}"
            )
        if dfs:
            class_dfs[cls_name] = pd.concat(dfs, ignore_index=True)
            log_message(f"  Loaded class '{cls_name}': events={len(class_dfs[cls_name])}")
        else:
            log_message(f"  [WARN] class '{cls_name}' has no usable events")

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
    data_df = pd.concat(data_dfs, ignore_index=True) if data_dfs else None
    if data_df is None:
        log_message("Loaded data events: 0")
    else:
        log_message(f"Loaded data events: {len(data_df)}")

    # Apply thresholds and then clip ranges; the weights stay fixed.
    def _prepare(df):
        if df is None or len(df) == 0:
            return df
        df = _apply_thresholds(df, thresholds)
        df = _apply_clip(df, clip_ranges)
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
        arrs = []
        for cls in class_names:
            if cls in class_dfs and branch in class_dfs[cls].columns:
                arrs.append(class_dfs[cls][branch].values)
        if data_df is not None and branch in data_df.columns:
            arrs.append(data_df[branch].values)

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
        mc_yields   = {}
        for cls in class_names:
            if cls in class_dfs and branch in class_dfs[cls].columns:
                h, h2 = _weighted_hist(
                    class_dfs[cls][branch].values,
                    class_dfs[cls]["weight"].values, edges
                )
            else:
                h  = np.zeros(bins)
                h2 = np.zeros(bins)
            mc_per_cls[cls] = (h, h2)
            mc_total_v  += h
            mc_total_w2 += h2
            mc_yields[cls] = float(h.sum())

        if data_df is not None and branch in data_df.columns:
            data_v, data_w2 = _weighted_hist(
                data_df[branch].values, data_df["weight"].values, edges
            )
        else:
            data_v  = np.zeros(bins)
            data_w2 = np.zeros(bins)

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
            sharex=True,
        )

        # Draw the stacked MC histograms from low to high total yield.
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

        # Draw the total MC uncertainty band.
        mc_sigma = np.sqrt(np.maximum(mc_total_w2, 0.0))
        lower = np.clip(mc_total_v - mc_sigma, 1e-12, None)
        upper = np.clip(mc_total_v + mc_sigma, 1e-12, None)
        ax.fill_between(
            bin_centers, lower, upper, step="mid",
            facecolor="none", edgecolor="gray", hatch="///", linewidth=0,
        )

        # Draw the data points.
        data_sigma = np.sqrt(np.maximum(data_w2, 0.0))
        y_plot = np.where(data_v > 0, data_v, np.nan)
        ax.errorbar(
            bin_centers, y_plot, yerr=data_sigma,
            fmt="o", ms=7.6, color="black", mfc="black", mec="black",
            elinewidth=1.5, capsize=0, label="Data",
        )

        # Configure the axes.
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
                ax.set_ylim(0.1, max(1.0, ymax * 5.0))
            else:
                ax.set_ylim(0.0, max(1.0, ymax * 1.3))

        ax.set_ylabel("Events", fontsize=24)
        hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=LUMI_TOTAL, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        if "Data" in labels:
            i = labels.index("Data")
            handles.append(handles.pop(i))
            labels.append(labels.pop(i))
        ax.legend(handles, labels, loc="best", fontsize=17, frameon=False, ncol=2)

        # Draw the Data/MC ratio panel.
        ratio, r_err = _ratio_data_over_mc(data_v, data_w2, mc_total_v, mc_total_w2)
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

        out_path = os.path.join(out_dir, f"{tree_name}_{branch}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log_message(f"Wrote plot file: {out_path}")
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
