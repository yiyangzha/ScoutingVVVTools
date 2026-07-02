#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Per-class shape-comparison plotter.

For each tree (fat2/fat3) and each plotted convert-branch variable, overlays the
distributions of the BDT process classes (class_groups), with EACH class histogram
individually normalized to unit integral ("A.U.").  This is a shape comparison:
MC-only, no data, no stacking, no ratio panel.

Reuses plotting/data_mc.py as an imported module for all I/O, MC weighting, the
parallel range pre-pass, per-sample weighted-histogram streaming, and binning.
The only new pieces here are (1) per-class unit normalization and (2) the overlaid
step-histogram draw.

The 3 QCD HT-bin classes from the active BDT config (QCD_HTLOW/MID/HIGH) are merged
into a single physics-weighted "QCD" class; single-top stays as its own "VT".

Usage (standalone):  cd plotting && python3 class_shapes.py [--no-selection] [--stat-band] [--workers N]
Or via run.py mode 10:  python3 run.py 10 [--no-selection]
"""

import argparse
import os
import sys
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Import the Data/MC plotter as a module to reuse its infrastructure.  Its module-
# level code only reads config (read-only) and applies the CMS style; no data is read.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import data_mc as dm  # noqa: E402

log_message = dm.log_message

# New output-dir config keys (mirror data_mc's output_root / output_root_nosel pair).
OUTPUT_SHAPES_PATT       = dm.plot_cfg.get("output_shapes", "./shapes/{tree_name}")
OUTPUT_SHAPES_NOSEL_PATT = dm.plot_cfg.get("output_shapes_nosel", "./shapes_nosel/{tree_name}")


# -------------------- Class grouping --------------------
def _shape_class_groups(tree_name, bdt_cfg):
    """Build the {class: [samples]} grouping for the shape overlay.

    Priority:
      1. config.json["shapes_class_groups"] (flat or {tree_name: {...}}), used as-is.
      2. Otherwise the BDT config's class_groups, with an optional mc_samples whitelist
         applied, and all classes whose name starts with "QCD" merged into one "QCD".
    """
    override = dm.plot_cfg.get("shapes_class_groups")
    if isinstance(override, dict) and tree_name in override and isinstance(override[tree_name], dict):
        override = override[tree_name]
    if isinstance(override, dict) and override and not any(isinstance(v, dict) for v in override.values()):
        return {cls: list(samples) for cls, samples in override.items()}

    class_groups = bdt_cfg["class_groups"]

    # Optional MC whitelist (restrict to samples actually produced), same as data_mc.
    whitelist = dm.plot_cfg.get("mc_samples")
    if isinstance(whitelist, dict):
        whitelist = whitelist.get(tree_name)
    if whitelist is not None:
        wl = set(whitelist)
        class_groups = {cls: [s for s in samples if s in wl] for cls, samples in class_groups.items()}

    # Merge QCD HT-bin classes (QCD_HTLOW/MID/HIGH, etc.) into a single "QCD".
    merged = {}
    for cls, samples in class_groups.items():
        if not samples:
            continue
        key = "QCD" if cls.upper().startswith("QCD") else cls
        merged.setdefault(key, [])
        merged[key].extend(samples)
    return merged


# -------------------- Per-class unit normalization --------------------
def _unit_normalize_per_class(mc_hists, class_names, branch, nbins):
    """Return {cls: (h_norm, h2_norm)} for one branch, each scaled to unit integral.

    Differs from data_mc._unit_normalized_histograms, which scales by the *total* MC
    sum; here each class is divided by its OWN integral so shapes are directly
    comparable.  The physics weights already encode the intra-class sample mixture.
    """
    out = {}
    for cls in class_names:
        h, h2 = mc_hists.get(cls, {}).get(branch, (np.zeros(nbins), np.zeros(nbins)))
        total = float(np.sum(h))
        s = 1.0 / total if total > 0.0 else 0.0
        out[cls] = (np.asarray(h, dtype=float) * s, np.asarray(h2, dtype=float) * (s * s))
    return out


# -------------------- Draw --------------------
def _draw_shapes_plot(*, class_names, color_map, edges, mc_per_cls_norm,
                      branch, x_range, logx, logy, out_path, stat_band):
    fig, ax = plt.subplots(figsize=(10, 8))
    ymax = 0.0
    ymin_pos = np.inf
    drawn = False
    for cls in class_names:
        h, h2 = mc_per_cls_norm[cls]
        if not np.any(h > 0):
            continue
        drawn = True
        color = color_map[cls]
        hep.histplot(h, bins=edges, ax=ax, color=color, histtype="step",
                     linewidth=2, label=cls)
        ymax = max(ymax, float(np.max(h)))
        pos = h[h > 0]
        if pos.size:
            ymin_pos = min(ymin_pos, float(np.min(pos)))
        if stat_band:
            sigma = np.sqrt(np.maximum(h2, 0.0))
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_between(centers, np.maximum(h - sigma, 0.0), h + sigma,
                            step="mid", color=color, alpha=0.15, linewidth=0)
    if not drawn:
        plt.close(fig)
        return

    ax.margins(x=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(*x_range)
    if logy:
        ax.set_yscale("log")
        floor = ymin_pos * 0.5 if np.isfinite(ymin_pos) and ymin_pos > 0 else 1e-5
        ax.set_ylim(floor, ymax * 5.0 if ymax > 0 else 1.0)
    else:
        ax.set_ylim(0.0, ymax * 1.35 if ymax > 0 else 1.0)

    ax.set_ylabel("A.U.", fontsize=24)
    ax.set_xlabel(branch, fontsize=20)
    hep.cms.label(data=False, com=13.6, ax=ax)
    ax.legend(loc="best", fontsize=16, frameon=False, ncol=2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_shapes_job(job):
    """Pool worker: render one branch's overlay PDF.  Module-level for picklability."""
    _draw_shapes_plot(
        class_names=job["class_names"],
        color_map=job["color_map"],
        edges=job["edges"],
        mc_per_cls_norm=job["mc_per_cls_norm"],
        branch=job["branch"],
        x_range=job["x_range"],
        logx=job["logx"],
        logy=job["logy"],
        out_path=job["out_path"],
        stat_band=job["stat_band"],
    )
    return job["out_path"]


# -------------------- Per-tree orchestration --------------------
def _process_tree_shapes(tree_name, no_selection=False, stat_band=False, n_workers=None):
    if n_workers is None:
        n_workers = dm._N_WORKERS
    log_message(f"Running class_shapes.py: tree={tree_name}, no_selection={no_selection}, "
                f"stat_band={stat_band}, n_workers={n_workers}")

    bdt_cfg, _bdt_br, bdt_sel, _meta = dm._bdt_configs_for_tree(tree_name, load_test_meta=False)
    class_groups = _shape_class_groups(tree_name, bdt_cfg)
    class_groups = {cls: samples for cls, samples in class_groups.items() if samples}
    class_names = list(class_groups.keys())
    log_message(f"Shape classes ({len(class_names)}): {', '.join(class_names)}")

    # ---- Inputs + selection (mirror data_mc._process_tree) ----
    bdt_root_dir   = dm._bdt_root_for_tree(tree_name)
    bdt_script_dir = os.path.dirname(bdt_root_dir)
    if no_selection:
        conv_dir      = os.path.dirname(dm._CONVERT_CFG_PATH)
        input_root    = dm._resolve(dm.convert_cfg["output_root"], conv_dir)
        input_pattern = dm.convert_cfg["output_pattern"].replace("{output_root}", "{input_root}")
    else:
        input_root    = dm._resolve(bdt_cfg["input_root"], bdt_script_dir)
        input_pattern = bdt_cfg["input_pattern"]

    sel         = bdt_sel.get(tree_name, {})
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    log_tf_set  = set(sel.get("log_transform", []))
    _extra_thresh_cfg = dm.plot_cfg.get("plot_thresholds", {}).get(tree_name, {}) or {}
    extra_plot_thresholds = {k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in _extra_thresh_cfg.items()}
    plot_thresholds  = {**({} if no_selection else thresholds), **extra_plot_thresholds}
    plot_clip_ranges = {} if no_selection else clip_ranges

    # Branches: convert-output variables only (skip BDT score_* — out of scope).
    branches = [b for b in dm._plot_branches_for_tree(tree_name) if not b.startswith("score_")]
    reweight_branches = list(dm.plot_cfg.get("event_reweight_branches", {}).get(tree_name, []))
    log_message(f"Branches to plot: {len(branches)}; reweight_branches={reweight_branches}")

    out_patt = OUTPUT_SHAPES_NOSEL_PATT if no_selection else OUTPUT_SHAPES_PATT
    out_dir  = dm._resolve(out_patt.format(tree_name=tree_name), _SCRIPT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    log_message(f"Output directory: {out_dir}")

    branch_logx = {b: bool(dm._branch_override(tree_name, b).get("logx", b in log_tf_set))
                   for b in branches}
    auto_range_branches = [b for b in branches
                           if "x_range" not in dm._branch_override(tree_name, b)]

    # ---- Step 1: discover files + entries (MC only) ----
    mc_n_totals, mc_sample_files = {}, {}
    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in dm.SAMPLE_INFO:
                raise RuntimeError(f"MC sample '{sname}' not found in sample.json")
            files = dm._input_files(sname, input_root, input_pattern)
            if not files:
                raise RuntimeError(f"No ROOT files found for MC sample '{sname}'")
            n_total = dm._tree_entries_total(files, tree_name)
            if n_total <= 0:
                log_message(f"  [WARN] skipping '{sname}': no entries for tree '{tree_name}'")
                continue
            mc_n_totals[sname]     = n_total
            mc_sample_files[sname] = files

    # ---- Step 2: parallel pre-pass (ranges + reweight sums) ----
    mc_raw_w_sums = {}
    prepass_mins  = {b:  np.inf for b in auto_range_branches}
    prepass_maxs  = {b: -np.inf for b in auto_range_branches}
    tasks, labels = [], []
    for sname in mc_sample_files:
        tasks.append((mc_sample_files[sname], tree_name, auto_range_branches,
                      reweight_branches, branch_logx, not no_selection))
        labels.append(sname)
    log_message(f"Pre-pass: {len(tasks)} MC samples ({n_workers} workers)")
    if n_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=min(len(tasks), n_workers)) as pool:
            results = pool.map(dm._prepass_worker, tasks)
    else:
        results = [dm._prepass(*t) for t in tasks]
    for sname, (rw_sum, ranges) in zip(labels, results):
        for b, (lo, hi) in ranges.items():
            prepass_mins[b] = min(prepass_mins[b], lo)
            prepass_maxs[b] = max(prepass_maxs[b], hi)
        mc_raw_w_sums[sname] = rw_sum if rw_sum > 0 else float(mc_n_totals[sname])

    merged_ranges = {}
    for b in auto_range_branches:
        lo, hi = prepass_mins[b], prepass_maxs[b]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            merged_ranges[b] = (lo, hi)

    # ---- Step 4: resolve binning ----
    branch_binning = {}
    for b in branches:
        binning = dm._resolve_binning(tree_name, b, merged_ranges.get(b), log_tf_set)
        if binning is None:
            log_message(f"  [WARN] no range for {tree_name}:{b}, skipping")
        else:
            branch_binning[b] = binning
    branch_edges = {b: dm._bin_edges(bins, x_range, logx)
                    for b, (bins, x_range, logx, logy, y_range) in branch_binning.items()}
    log_message(f"  {len(branch_binning)}/{len(branches)} branches have valid binning")

    # ---- Step 5: parallel streaming, accumulate per class ----
    stream_tasks, stream_labels = [], []
    for cls_name, samples in class_groups.items():
        for sname in samples:
            if sname not in mc_sample_files:
                continue
            info = dm.SAMPLE_INFO[sname]
            xsec = float(info.get("xsection", 0.0))
            raw_entries = float(info.get("raw_entries", 0.0))
            n_total = mc_n_totals[sname]
            target_total = (dm.LUMI_TOTAL * xsec * float(n_total) / raw_entries
                            if raw_entries > 0 else 0.0)
            stream_tasks.append((mc_sample_files[sname], tree_name, branch_edges,
                                 target_total, mc_raw_w_sums[sname], reweight_branches,
                                 plot_thresholds, plot_clip_ranges, not no_selection))
            stream_labels.append((cls_name, sname))
    log_message(f"Streaming histograms: {len(stream_tasks)} samples ({n_workers} workers)")
    if n_workers > 1 and len(stream_tasks) > 1:
        with multiprocessing.Pool(processes=min(len(stream_tasks), n_workers)) as pool:
            stream_results = pool.map(dm._stream_hists_worker, stream_tasks)
    else:
        stream_results = [dm._stream_hists(*t) for t in stream_tasks]

    mc_hists = {cls: {} for cls in class_names}
    for (cls_name, sname), sample_hists in zip(stream_labels, stream_results):
        for b, (h, h2) in sample_hists.items():
            if b not in mc_hists[cls_name]:
                mc_hists[cls_name][b] = [h.copy(), h2.copy()]
            else:
                mc_hists[cls_name][b][0] += h
                mc_hists[cls_name][b][1] += h2

    # ---- Color map (same construction as data_mc) ----
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(class_names)}

    # ---- Step 6/7: per-class unit normalize + render (parallel) ----
    jobs = []
    for b, (bins, x_range, logx, logy_default, y_range) in branch_binning.items():
        nbins = len(branch_edges[b]) - 1
        norm = _unit_normalize_per_class(mc_hists, class_names, b, nbins)
        if not any(np.any(h > 0) for h, _ in norm.values()):
            continue
        # Shapes are unit-area (bins <= 1): default to linear y unless an explicit override.
        logy = bool(dm._branch_override(tree_name, b).get("logy", False))
        jobs.append({
            "class_names": class_names,
            "color_map": color_map,
            "edges": branch_edges[b],
            "mc_per_cls_norm": norm,
            "branch": b,
            "x_range": x_range,
            "logx": logx,
            "logy": logy,
            "out_path": os.path.join(out_dir, f"{tree_name}_{b}_shapes.pdf"),
            "stat_band": stat_band,
        })

    log_message(f"Rendering {len(jobs)} shape plots ({n_workers} workers)")
    if n_workers > 1 and len(jobs) > 1:
        with multiprocessing.Pool(processes=min(len(jobs), n_workers)) as pool:
            pool.map(_draw_shapes_job, jobs)
    else:
        for job in jobs:
            _draw_shapes_job(job)
    log_message(f"Done {tree_name}: wrote {len(jobs)} PDFs to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Per-class unit-normalized shape-overlay plotter")
    p.add_argument("--no-selection", action="store_true",
                   help="Skip threshold/clip cuts; read directly from convert output "
                        "(writes to output_shapes_nosel).")
    p.add_argument("--stat-band", action="store_true",
                   help="Draw a light per-bin stat-uncertainty band per class (default off).")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (default: n_workers from config / cpu_count).")
    args = p.parse_args()
    n_workers = args.workers if args.workers is not None else dm._N_WORKERS
    for tree_name in dm.SUBMIT_TREES:
        _process_tree_shapes(tree_name, no_selection=args.no_selection,
                             stat_band=args.stat_band, n_workers=n_workers)


if __name__ == "__main__":
    main()
