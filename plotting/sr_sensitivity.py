"""Sensitivity illustration plots for the VVV scouting analysis.

1. ``sr_yield_summary.pdf``: stacked MC yields in every signal region of every
   configured category (read from the qcd_est.py ROOT outputs), with the
   per-SR Asimov significance in the lower panel.
2. For every configured target SR: stacked MC distributions of selected
   variables for events in that SR (score box + all selection.json thresholds,
   incl. the ABCD msoftdrop window). A variable that is itself cut on is drawn
   N-1 with its cut values marked. Lower panel: per-bin S/sqrt(B).

The event loading, weighting, model inference and SR definitions are reused
from background_estimation/qcd_est.py (imported once per target config).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_QCD_EST_PATH = os.path.join(_ROOT_DIR, "background_estimation", "qcd_est.py")
sys.path.insert(0, os.path.join(_ROOT_DIR, "selections", "BDT"))
import kfold  # noqa: E402


def log_message(message: str) -> None:
    print(message, flush=True)


def _resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _load_qcd_est(config_path: str, tag: str):
    """Import qcd_est.py as a fresh module bound to ``config_path``."""
    os.environ["QCD_EST_CONFIG_PATH"] = config_path
    spec = importlib.util.spec_from_file_location(f"qcd_est_{tag}", _QCD_EST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Match XGBoost threads to the CPUs actually allocated (e.g. by SLURM).
    module.INFERENCE_THREADS = max(1, len(os.sched_getaffinity(0)))
    return module


def _asimov_z(s: float, b: float) -> float:
    if b <= 0.0 or s <= 0.0:
        return 0.0
    return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))


def _signal_classes(q) -> list[str]:
    return [
        name for name, members in q.CLASS_GROUPS.items()
        if all(q.SAMPLE_INFO[s]["is_signal"] for s in members)
    ]


def _cms_label(ax, lumi: float) -> None:
    hep.cms.label("Private Work", data=False, com=13.6, year="2024", lumi=round(lumi, 1), ax=ax)


# -------------------- SR yield summary --------------------
def plot_sr_summary(categories: list[dict], output_root: str) -> None:
    log_message("Running SR yield summary")
    labels, yields, lumi, class_names, signal_names, colors = [], [], None, None, None, None
    for cat in categories:
        cfg_path = _resolve(cat["qcd_est_config"], _SCRIPT_DIR)
        q = _load_qcd_est(cfg_path, f"summary_{cat['name']}")
        if class_names is None:
            class_names, signal_names, colors = q.CLASS_NAMES, _signal_classes(q), q._group_color_map()
            lumi = q.LUMI
        elif q.CLASS_NAMES != class_names or q.LUMI != lumi:
            raise RuntimeError(f"Category {cat['name']} has different class_groups or lumi")
        root_path = os.path.join(q.OUTPUT_DIR, q.ROOT_FILE_NAME)
        with uproot.open(root_path) as f:
            bins = f["metadata/signal_regions"]["bin_index"].array(library="np")
            for b in bins:
                labels.append(f"{cat['name']} SR{int(b)}")
                yields.append([
                    float(f[f"groups/{q._slugify(c)}/sr{int(b)}/yield"].values()[0])
                    for c in class_names
                ])
    yields = np.asarray(yields)
    sig_idx = [class_names.index(c) for c in signal_names]
    bkg_idx = [i for i in range(len(class_names)) if i not in sig_idx]
    s_tot = yields[:, sig_idx].sum(axis=1)
    b_tot = yields[:, bkg_idx].sum(axis=1)
    z = np.array([_asimov_z(s, b) for s, b in zip(s_tot, b_tot)])
    for lab, s, b, zi in zip(labels, s_tot, b_tot, z):
        log_message(f"  {lab}: S={s:.3g}, B={b:.4g}, Z={zi:.3f}")
    log_message(f"  Combined Z = sqrt(sum Z^2) = {math.sqrt(float(np.sum(z ** 2))):.3f}")

    n = len(labels)
    edges = np.arange(n + 1, dtype=float)
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(max(12, 0.7 * n), 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
    )
    order = sorted(bkg_idx, key=lambda i: yields[:, i].sum(), reverse=True) + sig_idx
    hep.histplot(
        [yields[:, i] for i in order], edges, stack=True, histtype="fill", ax=ax,
        label=[class_names[i] for i in order], color=[colors[class_names[i]] for i in order],
    )
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.set_ylim(max(1e-2, 0.3 * float(np.min(yields.sum(axis=1)))), 50 * float(np.max(yields.sum(axis=1))))
    ax.legend(ncol=3, fontsize=16, loc="upper right")
    _cms_label(ax, lumi)
    rax.bar(edges[:-1] + 0.5, z, width=0.8, color="black")
    rax.set_ylabel("Asimov Z")
    rax.set_xticks(edges[:-1] + 0.5)
    rax.set_xticklabels(labels, rotation=60, ha="right", fontsize=12)
    rax.set_xlim(0, n)
    out = os.path.join(output_root, "sr_yield_summary.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Wrote {out}")


# -------------------- Per-SR distributions --------------------
def _cut_lines(cond) -> list[float]:
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, tuple, dict)):
        return [float(v) for v in cond if v is not None]
    if isinstance(cond, (int, float)):
        return [float(cond)]
    return []


def _predict(q, X_raw, model_branches, clip_ranges, log_transform, decorrelate, model) -> np.ndarray:
    X_model = q.standardize_X(X_raw[model_branches].copy(), clip_ranges, log_transform)
    X_model = q._drop_decorrelated_features(X_model, decorrelate)
    return q._predict_model_proba(model, X_model, context="sr_sensitivity")


def _load_kfold_events(q, branches, thresholds):
    """k-fold models: every MC event of the class_groups samples passing `thresholds`, each with
    its out-of-fold class probabilities (all entries read, kfold.KFoldModels.read_sample).
    Returns (the DataFrame of qcd_est.load_test_data with `branches`, the probabilities)."""
    models = kfold.KFoldModels(q.BDT_ROOT, log=log_message)
    frames, probas = [], []
    for name in [s for members in q.CLASS_GROUPS.values() for s in members if s not in q.EXCLUDE_SAMPLES]:
        r = models.read_sample(name, q.SAMPLE_INFO[name]["rule"], branches, thresholds)
        df = pd.DataFrame(r["columns"])
        df["weight"] = r["weight"] * q.LUMI
        df["sample_name"] = name
        df["group_name"] = q.SAMPLE_TO_GROUP[name]
        frames.append(df)
        probas.append(r["proba"])
        log_message(f"  {name}: {len(df)} events (all entries, k-fold scores)")
    return pd.concat(frames, ignore_index=True), np.concatenate(probas)


def _build_cut_masks(q, X_raw, proba, thresholds, row, axis_names):
    """One boolean mask per threshold cut and per SR score axis, so any cut can be dropped."""
    cut_masks = {
        name: q._threshold_mask(X_raw, {name: cond}).to_numpy(dtype=bool)
        for name, cond in thresholds.items()
    }
    score_cuts = {}
    for axis in axis_names:
        low, high = float(row[f"{axis}_low"]), float(row[f"{axis}_high"])
        scores = proba[:, q.CLASS_NAMES.index(axis)]
        if high < 1.0 - 1e-12:
            cut_masks[f"score_{axis}"] = (scores >= low) & (scores < high)
            score_cuts[f"score_{axis}"] = [low, high]
        else:
            cut_masks[f"score_{axis}"] = scores >= low
            score_cuts[f"score_{axis}"] = [low]
    return cut_masks, score_cuts


def _data_plot_mask(cut_masks, X_raw, var, plot_cfg, window, is_data) -> np.ndarray:
    """Selection for a data-overlay plot of ``var``: all cuts except ``drop_cuts``; events
    with any ``blind`` branch inside the window are removed (for MC, ``var`` itself is
    kept unblinded so the MC shape is shown under the blinded data region)."""
    mask = np.ones(len(X_raw), dtype=bool)
    for name, m in cut_masks.items():
        if name not in plot_cfg["drop_cuts"]:
            mask &= m
    lo, hi = window
    for b in plot_cfg["blind"]:
        vals = X_raw[b].to_numpy(dtype=float)
        mask &= vals > -990
        if is_data or b != var:
            mask &= ~((vals >= lo) & (vals <= hi))
    return mask & (X_raw[var].to_numpy(dtype=float) > -990)


def _hist_edges(vc) -> np.ndarray:
    lo, hi = vc["range"]
    return np.linspace(lo, hi, int(vc["bins"]) + 1)


def _fold(values, edges):
    # Fold under/overflow into the first/last bin.
    return np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))


def _data_files(q, sample: str) -> list[str]:
    pattern = q.cfg["input_pattern"].replace("{sample_group}_mixed", "{sample_group}")
    input_root = q._resolve(q.cfg["input_root"], q._BDT_DIR)
    base = pattern.format(input_root=input_root, sample_group="data", sample=sample)
    files = sorted(set(glob.glob(base) + glob.glob(base[:-len(".root")] + "_*.root")))
    if not files:
        raise FileNotFoundError(f"No data files for {sample}: {base}")
    return files


def _data_hist_dir(output_root, tree, bin_index) -> str:
    return os.path.join(output_root, "data_hists", f"{tree}_sr{bin_index}")


def fill_data_hists(target: dict, cfg: dict, output_root: str, file_index: int) -> None:
    """Fill blinded data histograms for one data file (one SLURM array task).
    Only already-blinded histograms are written; no event-level data is stored."""
    overlay = cfg["data_overlay"]
    plots = {v: p for v, p in overlay["plots"].items() if v in target["variables"]}
    if not plots:
        return
    cfg_path = _resolve(target["qcd_est_config"], _SCRIPT_DIR)
    bin_index = int(target["bin_index"])
    q = _load_qcd_est(cfg_path, f"data_{bin_index}_{os.path.basename(cfg_path)}")
    tree = q.TREE_NAME
    files = _data_files(q, overlay["sample"])
    path = files[file_index]
    log_message(f"Running data histograms: tree={tree}, SR{bin_index}, file {file_index + 1}/{len(files)}: {path}")

    model_branches = [item["name"] for item in q.br_cfg[tree]]
    selection = q.sel_cfg[tree]
    clip_ranges = {k: tuple(v) for k, v in selection.get("clip_ranges", {}).items()}
    log_transform = list(selection.get("log_transform", []))
    thresholds = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in selection.get("thresholds", {}).items()
    }
    decorrelate = q.cfg.get(tree, {}).get("decorrelate", [])
    blind_vars = {b for p in plots.values() for b in p["blind"]}
    load_branches = sorted(set(model_branches) | set(thresholds) | set(decorrelate) | set(plots) | blind_vars
                           | ({"event"} if q.KFOLD else set()))
    # Cuts that every data plot keeps can be applied before the (expensive) BDT inference.
    dropped = {c for p in plots.values() for c in p["drop_cuts"]}
    prefilter = {k: v for k, v in thresholds.items() if k not in dropped}

    signal_regions, axis_names = q._load_signal_regions()
    row = signal_regions[signal_regions["bin_index"] == bin_index].iloc[0]
    # k-fold models: data events (never trained on) are scored by event number (kfold.data_fold).
    model = kfold.KFoldModels(q.BDT_ROOT, log=log_message) if q.KFOLD else q._load_model()
    counts = {v: np.zeros(int(cfg["variables"][v]["bins"])) for v in plots}
    n_read = n_pre = 0
    with uproot.open(path) as f:
        for chunk in f[tree].iterate(load_branches, library="pd", step_size=2_000_000):
            n_read += len(chunk)
            chunk = chunk[q._threshold_mask(chunk, prefilter).to_numpy(dtype=bool)].reset_index(drop=True)
            n_pre += len(chunk)
            if chunk.empty:
                continue
            if q.KFOLD:
                proba = model.predict(model.model_inputs(chunk), model.data_fold(chunk["event"].to_numpy()))
            else:
                proba = _predict(q, chunk, model_branches, clip_ranges, log_transform, decorrelate, model)
            cut_masks, _ = _build_cut_masks(q, chunk, proba, thresholds, row, axis_names)
            for var, plot_cfg in plots.items():
                mask = _data_plot_mask(cut_masks, chunk, var, plot_cfg, overlay["blind_window"], is_data=True)
                edges = _hist_edges(cfg["variables"][var])
                counts[var] += np.histogram(_fold(chunk[var].to_numpy(dtype=float)[mask], edges), bins=edges)[0]
    out_dir = _data_hist_dir(output_root, tree, bin_index)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{file_index}.npz")
    tmp = out + ".tmp.npz"
    np.savez(tmp, source_file=path, n_read=n_read, n_prefilter=n_pre, **counts)
    os.replace(tmp, out)
    log_message(f"  read={n_read}, after prefilter={n_pre}, "
                + ", ".join(f"{v}: {int(c.sum())}" for v, c in counts.items()))
    log_message(f"Wrote {out}")


def _load_data_hists(q, overlay, output_root, tree, bin_index, var) -> np.ndarray:
    files = _data_files(q, overlay["sample"])
    out_dir = _data_hist_dir(output_root, tree, bin_index)
    total = None
    for k, path in enumerate(files):
        p = os.path.join(out_dir, f"{k}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing data histogram {p} (run --data-file {k})")
        with np.load(p) as d:
            if str(d["source_file"]) != path:
                raise RuntimeError(f"{p} was filled from {d['source_file']}, expected {path}")
            total = d[var].copy() if total is None else total + d[var]
    return total


def plot_target(target: dict, var_cfg: dict, output_root: str, overlay: dict | None = None) -> None:
    cfg_path = _resolve(target["qcd_est_config"], _SCRIPT_DIR)
    bin_index = int(target["bin_index"])
    q = _load_qcd_est(cfg_path, f"target_{bin_index}_{os.path.basename(cfg_path)}")
    tree = q.TREE_NAME
    log_message(f"Running SR distributions: tree={tree}, SR{bin_index}, config={cfg_path}")

    variables = list(target["variables"])
    missing_cfg = [v for v in variables if v not in var_cfg]
    if missing_cfg:
        raise KeyError(f"No binning in config 'variables' for: {missing_cfg}")

    model_branches = [item["name"] for item in q.br_cfg[tree]]
    selection = q.sel_cfg[tree]
    clip_ranges = {k: tuple(v) for k, v in selection.get("clip_ranges", {}).items()}
    log_transform = list(selection.get("log_transform", []))
    thresholds = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in selection.get("thresholds", {}).items()
    }
    decorrelate = q.cfg.get(tree, {}).get("decorrelate", [])
    raw_vars = [v for v in variables if not v.startswith("score_")]
    load_branches = sorted(set(model_branches) | set(thresholds) | set(decorrelate) | set(raw_vars))

    signal_regions, axis_names = q._load_signal_regions()
    rows = signal_regions[signal_regions["bin_index"] == bin_index]
    if rows.empty:
        raise RuntimeError(f"SR{bin_index} not found in {q.SIGNAL_REGION_CSV_PATH}")
    row = rows.iloc[0]

    if q.KFOLD:
        # k-fold models: every MC event with its out-of-fold score, as in qcd_est.py. Only the cuts
        # on variables that are not plotted are applied while reading (the N-1 plots need the events
        # failing a plotted variable's cut).
        load_branches = sorted(set(thresholds) | set(decorrelate) | set(raw_vars))
        prefilter = {k: c for k, c in thresholds.items() if k not in variables}
        df, proba = _load_kfold_events(q, load_branches, prefilter)
    else:
        df = q.load_test_data(load_branches)
        proba = None
    # The samples qcd_est.py leaves out of every yield (exclude_samples).
    keep = ~df["sample_name"].isin(sorted(q.EXCLUDE_SAMPLES)).to_numpy()
    df = df[keep].reset_index(drop=True)
    if proba is not None:
        proba = proba[keep]
    group_labels = df["group_name"].astype(str).to_numpy()
    w = df["weight"].to_numpy(dtype=float)
    X_raw = df[load_branches]

    if not q.KFOLD:
        log_message("Running model prediction")
        proba = _predict(q, X_raw, model_branches, clip_ranges, log_transform, decorrelate, q._load_model())
    cut_masks, score_cuts = _build_cut_masks(q, X_raw, proba, thresholds, row, axis_names)

    def selection_mask(exclude: str | None = None) -> np.ndarray:
        mask = np.ones(len(w), dtype=bool)
        for name, m in cut_masks.items():
            if name != exclude:
                mask &= m
        return mask

    # Cross-check the SR yields against the qcd_est ROOT output used by combine.
    sr_mask = selection_mask()
    root_path = os.path.join(q.OUTPUT_DIR, q.ROOT_FILE_NAME)
    with uproot.open(root_path) as f:
        for c in q.CLASS_NAMES:
            mine = float(w[sr_mask & (group_labels == c)].sum())
            ref = float(f[f"groups/{q._slugify(c)}/sr{bin_index}/yield"].values()[0])
            status = "OK" if abs(mine - ref) <= 1e-6 + 1e-4 * abs(ref) else "MISMATCH"
            log_message(f"  SR{bin_index} {c}: yield={mine:.6g}, qcd_est={ref:.6g} [{status}]")

    signal_names = _signal_classes(q)
    bkg_names = [c for c in q.CLASS_NAMES if c not in signal_names]
    s_sr = float(w[sr_mask & np.isin(group_labels, signal_names)].sum())
    b_sr = float(w[sr_mask & np.isin(group_labels, bkg_names)].sum())
    z_sr = _asimov_z(s_sr, b_sr)
    log_message(f"  SR{bin_index}: S={s_sr:.4g}, B={b_sr:.4g}, Z={z_sr:.3f}")

    colors = q._group_color_map()
    bkg_order = sorted(bkg_names, key=lambda c: w[sr_mask & (group_labels == c)].sum(), reverse=True)
    stack_order = bkg_order + signal_names
    line_scale = float(target.get("signal_line_scale", 0))
    out_dir = os.path.join(output_root, f"{tree}_sr{bin_index}")
    os.makedirs(out_dir, exist_ok=True)

    for var in variables:
        vc = var_cfg[var]
        mask = selection_mask(exclude=var)
        if var.startswith("score_"):
            cls = var[len("score_"):]
            values = proba[:, q.CLASS_NAMES.index(cls)]
            cuts = score_cuts.get(var, [])
        else:
            values = X_raw[var].to_numpy(dtype=float)
            cuts = _cut_lines(thresholds[var]) if var in thresholds else []
        mask &= values > -990
        lo, hi = vc["range"]
        edges = np.linspace(lo, hi, int(vc["bins"]) + 1)
        # Fold under/overflow into the first/last bin.
        vals = np.clip(values[mask], lo, np.nextafter(hi, lo))
        wv, gv = w[mask], group_labels[mask]

        hists, variances = {}, {}
        for c in stack_order:
            sel = gv == c
            hists[c] = np.histogram(vals[sel], bins=edges, weights=wv[sel])[0]
            variances[c] = np.histogram(vals[sel], bins=edges, weights=wv[sel] ** 2)[0]
        bkg = sum(hists[c] for c in bkg_names)
        sig = sum(hists[c] for c in signal_names)
        total = bkg + sig
        total_err = np.sqrt(sum(variances[c] for c in stack_order))

        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        )
        hep.histplot(
            [hists[c] for c in stack_order], edges, stack=True, histtype="fill", ax=ax,
            label=stack_order, color=[colors[c] for c in stack_order],
        )
        band_lo, band_hi = total - total_err, total + total_err
        ax.fill_between(
            edges, np.append(band_lo, band_lo[-1]), np.append(band_hi, band_hi[-1]), step="post",
            facecolor="none", edgecolor="gray", hatch="///", linewidth=0, label="MC stat. unc.",
        )
        if line_scale > 0:
            hep.histplot(sig * line_scale, edges, ax=ax, color="black", linestyle="--",
                         label=f"Signal $\\times${line_scale:g}")
        for x in cuts:
            ax.axvline(x, color="black", linestyle=":", linewidth=2)
            rax.axvline(x, color="black", linestyle=":", linewidth=2)
        ymax = float(max(np.max(total + total_err), np.max(sig * line_scale) if line_scale > 0 else 0.0))
        if vc.get("logy", False):
            ax.set_yscale("log")
            ax.set_ylim(max(1e-2, 0.5 * float(np.min(total[total > 0])) if np.any(total > 0) else 1e-2),
                        max(ymax, 1.0) * 50)
        else:
            ax.set_ylim(0, max(ymax, 1e-3) * 1.6)
        ax.set_ylabel("Events / bin")
        n1 = " (N-1)" if cuts else ""
        ax.legend(ncol=3, fontsize=14, loc="upper right", title_fontsize=14,
                  title=f"{tree}, SR{bin_index}{n1}: S={s_sr:.3g}, B={b_sr:.3g}, Z={z_sr:.2f}")
        _cms_label(ax, q.LUMI)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bkg > 0, sig / np.sqrt(np.where(bkg > 0, bkg, 1.0)), np.nan)
        hep.histplot(ratio, edges, ax=rax, color="black", histtype="step")
        rax.set_ylabel(r"S/$\sqrt{B}$")
        finite = ratio[np.isfinite(ratio)]
        rax.set_ylim(0, 1.3 * float(finite.max()) if finite.size and finite.max() > 0 else 1.0)
        rax.set_xlabel(vc.get("label", var))
        rax.set_xlim(lo, hi)

        out = os.path.join(out_dir, f"{var}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        log_message(f"Wrote {out}")

    if not overlay:
        return
    blo, bhi = overlay["blind_window"]
    for var, plot_cfg in overlay["plots"].items():
        if var not in variables:
            continue
        vc = var_cfg[var]
        edges = _hist_edges(vc)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if not (np.isclose(edges, blo).any() and np.isclose(edges, bhi).any()):
            raise ValueError(f"{var}: bin edges must include the blind window edges {blo}, {bhi}")
        sideband = (centers < blo) | (centers > bhi)
        data = _load_data_hists(q, overlay, output_root, tree, bin_index, var)

        mask = _data_plot_mask(cut_masks, X_raw, var, plot_cfg, overlay["blind_window"], is_data=False)
        vals = _fold(X_raw[var].to_numpy(dtype=float)[mask], edges)
        wv, gv = w[mask], group_labels[mask]
        hists = {c: np.histogram(vals[gv == c], bins=edges, weights=wv[gv == c])[0] for c in stack_order}
        var_mc = sum(np.histogram(vals[gv == c], bins=edges, weights=wv[gv == c] ** 2)[0] for c in stack_order)
        mc_sb, data_sb = float(sum(h[sideband].sum() for h in hists.values())), float(data[sideband].sum())
        if mc_sb <= 0 or data_sb <= 0:
            raise RuntimeError(f"{var}: empty sideband (MC={mc_sb}, data={data_sb})")
        scale = data_sb / mc_sb
        log_message(f"  {var}: data sideband={data_sb:.0f}, MC sideband={mc_sb:.4g}, MC scale={scale:.4g}")
        hists = {c: h * scale for c, h in hists.items()}
        total = sum(hists.values())
        total_err = np.sqrt(var_mc) * scale

        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        )
        hep.histplot([hists[c] for c in stack_order], edges, stack=True, histtype="fill", ax=ax,
                     label=stack_order, color=[colors[c] for c in stack_order])
        band_lo, band_hi = total - total_err, total + total_err
        ax.fill_between(edges, np.append(band_lo, band_lo[-1]), np.append(band_hi, band_hi[-1]), step="post",
                        facecolor="none", edgecolor="gray", hatch="///", linewidth=0, label="MC stat. unc.")
        shown = np.where(sideband, data, np.nan)
        ax.errorbar(centers, shown, yerr=np.sqrt(np.where(sideband, data, 0)), fmt="o", color="black",
                    markersize=6, label="Data")
        ax.axvspan(blo, bhi, color="lightgray", alpha=0.5, zorder=0, label="Blinded (data)")
        rax.axvspan(blo, bhi, color="lightgray", alpha=0.5, zorder=0)
        ymax = float(max(np.max(band_hi), np.nanmax(shown)))
        if vc.get("logy", False):
            ax.set_yscale("log")
            pos = total[total > 0]
            ax.set_ylim(max(1e-2, 0.5 * float(pos.min())) if pos.size else 1e-2, max(ymax, 1.0) * 50)
        else:
            ax.set_ylim(0, max(ymax, 1e-3) * 1.6)
        ax.set_ylabel("Events / bin")
        note = plot_cfg.get("note", "")
        ax.legend(ncol=3, fontsize=14, loc="upper right", title_fontsize=14,
                  title=f"{tree}, SR{bin_index} (N-1){note}\nMC $\\times${scale:.3g} (norm. to data in sidebands)")
        hep.cms.label("Private Work", data=True, com=13.6, year="2024", ax=ax)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sideband & (total > 0), data / total, np.nan)
            ratio_err = np.where(sideband & (total > 0), np.sqrt(data) / total, np.nan)
            mc_rel = np.where(total > 0, total_err / total, 0.0)
        rax.fill_between(edges, np.append(1 - mc_rel, 1 - mc_rel[-1]), np.append(1 + mc_rel, 1 + mc_rel[-1]),
                         step="post", facecolor="none", edgecolor="gray", hatch="///", linewidth=0)
        rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o", color="black", markersize=6)
        rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        rax.set_ylim(0, 2)
        rax.set_ylabel("Data/MC")
        rax.set_xlabel(vc.get("label", var))
        rax.set_xlim(edges[0], edges[-1])

        out = os.path.join(out_dir, f"{var}_data.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        log_message(f"Wrote {out}")


def main() -> None:
    cfg_path = _resolve(
        os.environ.get("SR_SENSITIVITY_CONFIG_PATH", "config_sr_sensitivity.json"), _SCRIPT_DIR)
    with open(cfg_path) as f:
        cfg = json.load(f)
    output_root = _resolve(cfg.get("output_root", "./sr_sensitivity"), _SCRIPT_DIR)
    os.makedirs(output_root, exist_ok=True)
    log_message(f"Running sr_sensitivity.py: config={cfg_path}, output_root={output_root}")
    if len(sys.argv) == 3 and sys.argv[1] == "--data-file":
        for target in cfg.get("targets", []):
            fill_data_hists(target, cfg, output_root, int(sys.argv[2]))
        log_message("Finished sr_sensitivity.py")
        return
    if cfg.get("summary_categories"):
        plot_sr_summary(cfg["summary_categories"], output_root)
    for target in cfg.get("targets", []):
        plot_target(target, cfg["variables"], output_root, cfg.get("data_overlay"))
    log_message("Finished sr_sensitivity.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_message(f"Runtime error: {exc}")
        sys.exit(1)
