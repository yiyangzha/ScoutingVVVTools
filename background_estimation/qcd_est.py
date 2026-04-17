#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""QCD ABCD estimation on MC using signal-region definitions from signal_region.py."""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb


plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_SELECTIONS_DIR = os.path.join(_ROOT_DIR, "selections")
_BDT_DIR = os.path.join(_SELECTIONS_DIR, "BDT")


def log_message(message: str) -> None:
    print(message, flush=True)


def log_warning(message: str) -> None:
    log_message(f"Warning: {message}")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


_cfg_path = os.environ.get("QCD_EST_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
_cfg_path = _resolve(_cfg_path, _SCRIPT_DIR)
qcd_cfg = _load_json(_cfg_path)

LUMI = float(qcd_cfg["lumi"])
BDT_ROOT = _resolve(qcd_cfg["bdt_root"], _SCRIPT_DIR)
OUTPUT_DIR = _resolve(qcd_cfg.get("output_dir", "./output"), _SCRIPT_DIR)
ROOT_FILE_NAME = qcd_cfg.get("root_file_name", "qcd_abcd_yields.root")
SIGNAL_REGION_CSV_PATH = _resolve(qcd_cfg["signal_region_csv"], _SCRIPT_DIR)


cfg = _load_json(os.path.join(BDT_ROOT, "config.json"))
br_cfg = _load_json(os.path.join(BDT_ROOT, "branch.json"))
sel_cfg = _load_json(os.path.join(BDT_ROOT, "selection.json"))
test_meta = _load_json(os.path.join(BDT_ROOT, "test_ranges.json"))

_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_BDT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

TREE_NAME = test_meta["tree_name"]
MODEL_PATTERN = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
CLASS_GROUPS = cfg["class_groups"]
CLASS_NAMES = list(CLASS_GROUPS.keys())
NUM_CLASSES = len(CLASS_NAMES)
AXIS_NAMES = CLASS_NAMES[: max(1, NUM_CLASSES - 1)]

QCD_CLASS_NAME = None
for class_name in CLASS_NAMES:
    if class_name.lower() == "qcd":
        QCD_CLASS_NAME = class_name
        break
if QCD_CLASS_NAME is None:
    raise RuntimeError("BDT class_groups must contain a QCD class")


SAMPLE_INFO = {}
for rule in sample_cfg["sample"]:
    SAMPLE_INFO[rule["name"]] = {
        "xsection": float(rule["xsection"]),
        "raw_entries": int(rule.get("raw_entries", -1)),
        "is_MC": bool(rule["is_MC"]),
        "is_signal": bool(rule["is_signal"]),
        "sample_ID": int(rule["sample_ID"]),
    }

SAMPLE_TO_CLASS = {}
SAMPLE_TO_GROUP = {}
for class_idx, (class_name, members) in enumerate(CLASS_GROUPS.items()):
    for sample_name in members:
        SAMPLE_TO_CLASS[sample_name] = class_idx
        SAMPLE_TO_GROUP[sample_name] = class_name

QCD_SAMPLES = set(CLASS_GROUPS[QCD_CLASS_NAME])


def _mask_from_cond(col: pd.Series, cond) -> pd.Series:
    idx = col.index
    if cond is None:
        return pd.Series(True, index=idx)
    if isinstance(cond, (int, float, np.integer, np.floating)):
        return col > float(cond)
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
        mn, mx = cond
        mask = pd.Series(True, index=idx)
        if mn is not None:
            mask &= col > mn
        if mx is not None:
            mask &= col < mx
        return mask
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
    raise TypeError(f"Unsupported condition type: {type(cond)}")


def filter_X(
    X: pd.DataFrame,
    y,
    w,
    branch: list,
    thresholds: dict | None = None,
    apply_to_sentinel: bool = True,
    sample_labels=None,
):
    """Apply per-branch threshold cuts, matching train.py and signal_region.py."""
    if not thresholds:
        if sample_labels is None:
            return X.copy(), y.copy(), w.copy()
        return X.copy(), y.copy(), w.copy(), np.asarray(sample_labels).copy()

    mask = pd.Series(True, index=X.index)
    for name, cond in thresholds.items():
        if name not in X.columns:
            raise KeyError(f"Column {name!r} not found in X")
        col = X[name]
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


def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
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


def load_test_data(branches: list[str]) -> pd.DataFrame:
    """Load the full test split with the same weight definition as signal_region.py."""
    log_message(f"Loading test data from: {os.path.join(BDT_ROOT, 'test_ranges.json')}")
    dfs = []

    for sample_name, sample_meta in test_meta["samples"].items():
        info = SAMPLE_INFO.get(sample_name)
        if info is None:
            log_warning(f"Sample '{sample_name}' not found in sample config, skipping")
            continue
        if not info["is_MC"]:
            log_warning(f"Skipping non-MC sample '{sample_name}'")
            continue
        if sample_name not in SAMPLE_TO_CLASS:
            log_warning(f"Sample '{sample_name}' not in any class group, skipping")
            continue

        xsec = float(info["xsection"])
        raw_entries = int(info["raw_entries"])
        total_entries = int(sample_meta["total_entries"])

        parts = []
        for seg in sample_meta["test_segments"]:
            fpath = seg["file"]
            if not os.path.exists(fpath):
                log_warning(f"File not found: {fpath}, skipping segment")
                continue
            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        log_warning(f"Tree '{TREE_NAME}' not in {fpath}, skipping")
                        continue
                    tree = uf[TREE_NAME]
                    available = set(tree.keys())
                    missing = [branch for branch in branches if branch not in available]
                    if missing:
                        raise KeyError(
                            f"Missing branches in {fpath}:{TREE_NAME}: "
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
            except Exception as exc:
                log_warning(f"Failed to read {fpath}: {exc}, skipping segment")
                continue

        if not parts:
            log_warning(f"No data loaded for '{sample_name}', skipping")
            continue

        df = pd.concat(parts, ignore_index=True)
        n_loaded = len(df)

        if xsec <= 0.0 or raw_entries <= 0:
            target_total = 0.0
            df["weight"] = 0.0
            log_warning(
                f"  {sample_name}: non-positive xsec={xsec} or raw_entries={raw_entries}, zero weight"
            )
        else:
            target_total = LUMI * xsec * total_entries / raw_entries
            df["weight"] = target_total / n_loaded

        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        df["group_name"] = SAMPLE_TO_GROUP[sample_name]
        dfs.append(df)
        log_message(
            f"  {sample_name}: n_loaded={n_loaded}, total_entries={total_entries}, "
            f"raw_entries={raw_entries}, xsec={xsec:.6g}, target_total={target_total:.6g}, "
            f"class={SAMPLE_TO_GROUP[sample_name]}"
        )

    if not dfs:
        raise RuntimeError("No MC test data loaded")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df_all


def _load_model():
    model_base = MODEL_PATTERN.format(output_root=BDT_ROOT, tree_name=TREE_NAME)
    if os.path.exists(model_base + ".json"):
        model_path = model_base + ".json"
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        log_message(f"Loaded model: {model_path}")
        return clf
    if os.path.exists(model_base + ".pkl"):
        model_path = model_base + ".pkl"
        with open(model_path, "rb") as handle:
            clf = pickle.load(handle)
        log_message(f"Loaded model: {model_path}")
        return clf
    raise FileNotFoundError(f"No model found at {model_base}(.json/.pkl)")


def _resolve_mass_thresholds(thresholds: dict) -> Tuple[dict, dict]:
    mass_thresholds = {}
    other_thresholds = {}
    for name, cond in thresholds.items():
        if name.startswith("ScoutingFatPFJetRecluster_msoftdrop_"):
            mass_thresholds[name] = cond
        else:
            other_thresholds[name] = cond
    if not mass_thresholds:
        raise RuntimeError("No ScoutingFatPFJetRecluster_msoftdrop_* thresholds found in selection.json")
    return mass_thresholds, other_thresholds


def _mass_pass_fail_masks(df: pd.DataFrame, mass_thresholds: dict) -> Tuple[np.ndarray, np.ndarray]:
    pass_mask = np.ones(len(df), dtype=bool)
    fail_mask = np.ones(len(df), dtype=bool)
    valid_mask = np.ones(len(df), dtype=bool)

    for name, cond in mass_thresholds.items():
        if name not in df.columns:
            raise KeyError(f"Mass threshold branch {name!r} not found in DataFrame")
        col = df[name]
        values = col.to_numpy(dtype=float, copy=False)
        sentinel = values < -990
        finite = np.isfinite(values)
        branch_valid = (~sentinel) & finite
        cond_mask = _mask_from_cond(col, cond).to_numpy(dtype=bool)
        valid_mask &= branch_valid
        pass_mask &= branch_valid & cond_mask
        fail_mask &= branch_valid & (~cond_mask)

    pass_mask &= valid_mask
    fail_mask &= valid_mask
    return pass_mask, fail_mask


def _load_signal_regions() -> pd.DataFrame:
    if not os.path.exists(SIGNAL_REGION_CSV_PATH):
        raise FileNotFoundError(
            f"Signal region CSV not found: {SIGNAL_REGION_CSV_PATH}. Run signal_region.py first."
        )

    df = pd.read_csv(SIGNAL_REGION_CSV_PATH)
    if df.empty:
        raise RuntimeError(f"Signal region CSV is empty: {SIGNAL_REGION_CSV_PATH}")

    required = ["bin_index"]
    for axis_name in AXIS_NAMES:
        required.extend([f"{axis_name}_low", f"{axis_name}_high"])
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(
            f"Signal region CSV missing required columns: {', '.join(missing)}"
        )

    return df.sort_values("bin_index").reset_index(drop=True)


def _region_mask(scores: np.ndarray, region_row: pd.Series) -> np.ndarray:
    mask = np.ones(scores.shape[0], dtype=bool)
    for dim, axis_name in enumerate(AXIS_NAMES):
        low = float(region_row[f"{axis_name}_low"])
        high = float(region_row[f"{axis_name}_high"])
        axis_scores = scores[:, dim]
        if high < 1.0 - 1e-12:
            mask &= (axis_scores >= low) & (axis_scores < high)
        else:
            mask &= axis_scores >= low
    return mask


def _hist_with_var(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, _ = np.histogram(values, bins=edges, weights=weights)
    vars_, _ = np.histogram(values, bins=edges, weights=weights ** 2)
    return vals.astype(float), vars_.astype(float)


def _ratio_pred_over_true(pred_vals, pred_vars, true_vals, true_vars):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(true_vals > 0, pred_vals / true_vals, np.nan)
        term_pred = np.where(pred_vals > 0, pred_vars / np.maximum(pred_vals, 1e-300) ** 2, 0.0)
        term_true = np.where(true_vals > 0, true_vars / np.maximum(true_vals, 1e-300) ** 2, 0.0)
        sigma = np.abs(ratio) * np.sqrt(term_pred + term_true)
    return ratio, sigma


def _add_uncert_band(ax, edges: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    lower_step = np.r_[lower, lower[-1]]
    upper_step = np.r_[upper, upper[-1]]
    ax.fill_between(
        edges,
        lower_step,
        upper_step,
        step="post",
        facecolor="none",
        edgecolor="gray",
        hatch="///",
        linewidth=0,
    )


def plot_abcd_region_counts(
    region_labels: List[str],
    group_vals: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
    out_path: str,
    normalize_per_bin: bool = False,
) -> None:
    edges = np.arange(len(region_labels) + 1, dtype=float)
    centers = edges[:-1] + 0.5
    widths = np.full(len(region_labels), 1.0)

    vals_map = {name: group_vals[name].copy() for name in CLASS_NAMES}
    vars_map = {name: group_vars[name].copy() for name in CLASS_NAMES}

    totals = np.zeros(len(region_labels), dtype=float)
    total_vars = np.zeros(len(region_labels), dtype=float)
    for name in CLASS_NAMES:
        totals += vals_map[name]
        total_vars += vars_map[name]

    if normalize_per_bin:
        scale = np.where(totals > 0, 1.0 / totals, 0.0)
        for name in CLASS_NAMES:
            vals_map[name] *= scale
            vars_map[name] *= scale ** 2
        totals *= scale
        total_vars *= scale ** 2

    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(region_labels), dtype=float)
    order = np.argsort([float(np.sum(vals_map[name])) for name in CLASS_NAMES])
    ordered_groups = [CLASS_NAMES[idx] for idx in order]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(CLASS_NAMES)}

    for name in ordered_groups:
        ax.bar(
            edges[:-1],
            vals_map[name],
            width=widths,
            bottom=bottom,
            align="edge",
            color=color_map[name],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=name,
        )
        bottom += vals_map[name]

    if not normalize_per_bin:
        sigma = np.sqrt(np.maximum(total_vars, 0.0))
        lower = np.clip(totals - sigma, 1e-12, None)
        upper = np.clip(totals + sigma, 1e-12, None)
        _add_uncert_band(ax, edges, lower, upper)
        ax.set_yscale("log")
        ax.set_ylim(0.1, max(1.0, float(np.max(totals[totals > 0])) * 3.0 if np.any(totals > 0) else 1.0))
        ax.set_ylabel("Events", fontsize=22)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction", fontsize=22)

    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel("Region", fontsize=22)
    ax.set_xticks(centers)
    ax.set_xticklabels(region_labels, fontsize=14)
    ax.margins(x=0)
    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
    ax.legend(loc="best", fontsize=14, frameon=False, ncol=2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Wrote plot file: {out_path}")


def plot_signal_region_prediction(
    region_labels: List[str],
    pred_group_vals: Dict[str, np.ndarray],
    pred_total_vals: np.ndarray,
    pred_total_vars: np.ndarray,
    true_vals: np.ndarray,
    true_vars: np.ndarray,
    out_path: str,
    groups: List[str],
    ylabel: str,
) -> None:
    n = len(region_labels)
    edges = np.arange(n + 1, dtype=float)
    centers = edges[:-1] + 0.5
    widths = np.full(n, 1.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(CLASS_NAMES)}

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    order = np.argsort([float(np.sum(pred_group_vals[name])) for name in groups])
    ordered_groups = [groups[idx] for idx in order]

    bottom = np.zeros(n, dtype=float)
    for name in ordered_groups:
        vals = pred_group_vals[name]
        ax.bar(
            edges[:-1],
            vals,
            width=widths,
            bottom=bottom,
            align="edge",
            color=color_map.get(name, "#1f77b4"),
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=name,
        )
        bottom += vals

    pred_sigma = np.sqrt(np.maximum(pred_total_vars, 0.0))
    lower = np.clip(pred_total_vals - pred_sigma, 1e-12, None)
    upper = np.clip(pred_total_vals + pred_sigma, 1e-12, None)
    _add_uncert_band(ax, edges, lower, upper)

    true_sigma = np.sqrt(np.maximum(true_vars, 0.0))
    y_plot = np.where(true_vals > 0, true_vals, np.nan)
    ax.errorbar(
        centers,
        y_plot,
        yerr=true_sigma,
        fmt="o",
        ms=7.2,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
        label="True",
    )

    ax.set_yscale("log")
    ymax = max(
        float(np.nanmax(pred_total_vals)) if pred_total_vals.size else 1.0,
        float(np.nanmax(true_vals)) if true_vals.size else 1.0,
        1.0,
    )
    ax.set_ylim(0.1, max(1.0, ymax * 4.0))
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylabel(ylabel, fontsize=22)
    ax.margins(x=0)
    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if "True" in labels:
        idx = labels.index("True")
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))
    ax.legend(handles, labels, loc="best", fontsize=14, frameon=False, ncol=2)

    ratio, ratio_err = _ratio_pred_over_true(pred_total_vals, pred_total_vars, true_vals, true_vars)
    axr.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        fmt="o",
        ms=7.2,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
    )
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    finite = np.isfinite(ratio)
    if np.any(finite):
        rmax = float(np.nanmax(ratio[finite] + np.nan_to_num(ratio_err[finite], nan=0.0)))
        rmin = float(np.nanmin(ratio[finite] - np.nan_to_num(ratio_err[finite], nan=0.0)))
        if rmax < 5.0:
            axr.set_ylim(max(0.0, 0.8 * rmin), max(2.0, 1.2 * rmax))
        else:
            axr.set_ylim(0.0, 5.0)
    else:
        axr.set_ylim(0.0, 2.0)

    axr.set_ylabel(r"$\frac{Pred}{True}$", fontsize=24)
    axr.yaxis.set_label_coords(-0.05, 0.6)
    axr.set_xlabel("Signal Region", fontsize=22)
    axr.set_xticks(centers)
    axr.set_xticklabels(region_labels, fontsize=14)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Wrote plot file: {out_path}")


def write_root_output(
    root_path: str,
    edges: np.ndarray,
    sample_yields: Dict[str, np.ndarray],
    sample_vars: Dict[str, np.ndarray],
    group_yields: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
    pred_qcd_vals: np.ndarray,
    pred_qcd_vars: np.ndarray,
    true_qcd_vals: np.ndarray,
    true_qcd_vars: np.ndarray,
    pred_total_vals: np.ndarray,
    pred_total_vars: np.ndarray,
    true_total_vals: np.ndarray,
    true_total_vars: np.ndarray,
) -> None:
    with uproot.recreate(root_path) as root_file:
        for sample_name in sorted(sample_yields):
            root_file[f"samples/{sample_name}/yield"] = (sample_yields[sample_name], edges)
            root_file[f"samples/{sample_name}/stat_error"] = (
                np.sqrt(np.maximum(sample_vars[sample_name], 0.0)),
                edges,
            )

        for group_name in CLASS_NAMES:
            root_file[f"groups/{_slugify(group_name)}/yield"] = (group_yields[group_name], edges)
            root_file[f"groups/{_slugify(group_name)}/stat_error"] = (
                np.sqrt(np.maximum(group_vars[group_name], 0.0)),
                edges,
            )

        root_file["qcd_predict/yield"] = (pred_qcd_vals, edges)
        root_file["qcd_predict/stat_error"] = (np.sqrt(np.maximum(pred_qcd_vars, 0.0)), edges)
        root_file["qcd_true/yield"] = (true_qcd_vals, edges)
        root_file["qcd_true/stat_error"] = (np.sqrt(np.maximum(true_qcd_vars, 0.0)), edges)
        root_file["total_predict/yield"] = (pred_total_vals, edges)
        root_file["total_predict/stat_error"] = (np.sqrt(np.maximum(pred_total_vars, 0.0)), edges)
        root_file["total_true/yield"] = (true_total_vals, edges)
        root_file["total_true/stat_error"] = (np.sqrt(np.maximum(true_total_vars, 0.0)), edges)

    log_message(f"Wrote ROOT file: {root_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_message(
        f"Running qcd_est.py: tree={TREE_NAME}, lumi={LUMI} fb^-1, "
        f"bdt_root={BDT_ROOT}, signal_region_csv={SIGNAL_REGION_CSV_PATH}, output_dir={OUTPUT_DIR}"
    )

    model_branches = [item["name"] for item in br_cfg[TREE_NAME]]
    selection = sel_cfg[TREE_NAME]
    clip_ranges = {key: tuple(val) for key, val in selection.get("clip_ranges", {}).items()}
    log_transform = list(selection.get("log_transform", []))
    thresholds = {
        key: (tuple(val) if isinstance(val, list) else val)
        for key, val in selection.get("thresholds", {}).items()
    }
    mass_thresholds, bdt_thresholds = _resolve_mass_thresholds(thresholds)
    decorrelate = cfg.get(TREE_NAME, {}).get("decorrelate", [])

    load_branches = sorted(set(model_branches) | set(thresholds.keys()))
    signal_regions = _load_signal_regions()
    region_labels = [f"SR{int(idx)}" for idx in signal_regions["bin_index"].tolist()]
    edges = np.arange(len(region_labels) + 1, dtype=float)
    log_message(
        f"Resolved inputs: model_branches={len(model_branches)}, "
        f"load_branches={len(load_branches)}, signal_regions={len(region_labels)}"
    )

    df_all = load_test_data(load_branches)
    X_raw = df_all[load_branches].copy()
    y = df_all["class_idx"].values.astype(int)
    w = df_all["weight"].values.astype(float)
    sample_labels = df_all["sample_name"].astype(str).values
    group_labels = df_all["group_name"].astype(str).values
    del df_all
    gc.collect()

    log_message("Applying non-mass thresholds")
    X_raw, y, w, sample_labels = filter_X(
        X_raw,
        y,
        w,
        load_branches,
        bdt_thresholds,
        apply_to_sentinel=True,
        sample_labels=sample_labels,
    )
    group_labels = np.asarray([SAMPLE_TO_GROUP[name] for name in sample_labels], dtype=object)
    log_message(f"After non-mass filtering: {len(X_raw)} events")

    log_message("Evaluating mass pass/fail masks")
    mass_pass, mass_fail = _mass_pass_fail_masks(X_raw, mass_thresholds)
    mass_mixed = ~(mass_pass | mass_fail)
    log_message(
        f"Mass categories: pass={int(np.count_nonzero(mass_pass))}, "
        f"fail={int(np.count_nonzero(mass_fail))}, excluded_mixed={int(np.count_nonzero(mass_mixed))}"
    )

    log_message("Standardising model features")
    X_model = standardize_X(X_raw[model_branches].copy(), clip_ranges, log_transform)
    if decorrelate:
        name_to_idx = {name: idx for idx, name in enumerate(X_model.columns)}
        decor_idx = sorted(name_to_idx[name] for name in decorrelate if name in name_to_idx)
        keep_idx = [idx for idx in range(len(X_model.columns)) if idx not in decor_idx]
        X_model = X_model.iloc[:, keep_idx]
        log_message(f"Removed decorrelated features: {decorrelate}")

    clf = _load_model()
    log_message("Running BDT prediction")
    proba = clf.predict_proba(X_model)
    log_message(f"Predicted probabilities shape: {proba.shape}")

    region_score_masks = []
    union_score_mask = np.zeros(len(X_raw), dtype=bool)
    membership = np.zeros(len(X_raw), dtype=int)
    for _, row in signal_regions.iterrows():
        mask = _region_mask(proba[:, : len(AXIS_NAMES)], row)
        region_score_masks.append(mask)
        union_score_mask |= mask
        membership += mask.astype(int)

    if np.any(membership > 1):
        raise RuntimeError("Signal region definitions overlap on the current event set")

    region_a_masks = [mask & mass_pass for mask in region_score_masks]
    a_union_mask = union_score_mask & mass_pass
    b_mask = (~union_score_mask) & mass_pass
    c_mask = union_score_mask & mass_fail
    d_mask = (~union_score_mask) & mass_fail

    log_message(
        f"ABCD event counts: A_union={int(np.count_nonzero(a_union_mask))}, "
        f"B={int(np.count_nonzero(b_mask))}, C={int(np.count_nonzero(c_mask))}, "
        f"D={int(np.count_nonzero(d_mask))}"
    )

    qcd_mask = np.isin(sample_labels, list(QCD_SAMPLES))
    weights = w

    def _sum_weight(mask):
        vals = weights[mask]
        return float(np.sum(vals)), float(np.sum(vals ** 2))

    qcd_a_total, qcd_a_var = _sum_weight(a_union_mask & qcd_mask)
    qcd_b_total, qcd_b_var = _sum_weight(b_mask & qcd_mask)
    qcd_c_total, qcd_c_var = _sum_weight(c_mask & qcd_mask)
    qcd_d_total, qcd_d_var = _sum_weight(d_mask & qcd_mask)

    if qcd_b_total <= 0.0 or qcd_c_total <= 0.0 or qcd_d_total <= 0.0:
        raise RuntimeError("QCD B/C/D totals must be positive for ABCD scaling")
    if qcd_a_total <= 0.0:
        raise RuntimeError("QCD A-union total is zero; cannot derive global QCD scale")

    pred_qcd_union = qcd_b_total * qcd_c_total / qcd_d_total
    pred_qcd_union_var = (
        (qcd_c_total / qcd_d_total) ** 2 * qcd_b_var
        + (qcd_b_total / qcd_d_total) ** 2 * qcd_c_var
        + (qcd_b_total * qcd_c_total / (qcd_d_total ** 2)) ** 2 * qcd_d_var
    )
    pred_qcd_union_sigma = math.sqrt(max(pred_qcd_union_var, 0.0))
    qcd_scale = pred_qcd_union / qcd_a_total
    qcd_scale_var = pred_qcd_union_var / (qcd_a_total ** 2)
    qcd_scale_sigma = math.sqrt(max(qcd_scale_var, 0.0))

    log_message(
        f"ABCD QCD totals: A_union={qcd_a_total:.6g}, B={qcd_b_total:.6g}, "
        f"C={qcd_c_total:.6g}, D={qcd_d_total:.6g}, pred_union={pred_qcd_union:.6g} ± "
        f"{pred_qcd_union_sigma:.6g}, scale={qcd_scale:.6g} ± {qcd_scale_sigma:.6g}"
    )

    sample_names = sorted({sample for sample in sample_labels})
    sample_yields = {sample: np.zeros(len(region_labels), dtype=float) for sample in sample_names}
    sample_vars = {sample: np.zeros(len(region_labels), dtype=float) for sample in sample_names}
    group_yields = {group: np.zeros(len(region_labels), dtype=float) for group in CLASS_NAMES}
    group_vars = {group: np.zeros(len(region_labels), dtype=float) for group in CLASS_NAMES}

    true_qcd_vals = np.zeros(len(region_labels), dtype=float)
    true_qcd_vars = np.zeros(len(region_labels), dtype=float)

    for idx, mask in enumerate(region_a_masks):
        for sample_name in sample_names:
            sample_mask = mask & (sample_labels == sample_name)
            vals = weights[sample_mask]
            sample_yields[sample_name][idx] = float(np.sum(vals))
            sample_vars[sample_name][idx] = float(np.sum(vals ** 2))
        for group_name in CLASS_NAMES:
            group_mask = mask & (group_labels == group_name)
            vals = weights[group_mask]
            group_yields[group_name][idx] = float(np.sum(vals))
            group_vars[group_name][idx] = float(np.sum(vals ** 2))
        qcd_vals = weights[mask & qcd_mask]
        true_qcd_vals[idx] = float(np.sum(qcd_vals))
        true_qcd_vars[idx] = float(np.sum(qcd_vals ** 2))

    qcd_fraction_vals = true_qcd_vals / qcd_a_total
    qcd_fraction_vars = np.zeros(len(region_labels), dtype=float)
    for idx in range(len(region_labels)):
        region_val = true_qcd_vals[idx]
        region_var = true_qcd_vars[idx]
        rest_val = qcd_a_total - region_val
        rest_var = max(0.0, qcd_a_var - region_var)
        qcd_fraction_vars[idx] = (
            ((rest_val / (qcd_a_total ** 2)) ** 2) * region_var
            + ((region_val / (qcd_a_total ** 2)) ** 2) * rest_var
        )

    pred_qcd_vals = pred_qcd_union * qcd_fraction_vals
    pred_qcd_var_fraction = (pred_qcd_union ** 2) * qcd_fraction_vars
    pred_qcd_var_scale = (true_qcd_vals ** 2) * qcd_scale_var
    pred_qcd_vars = pred_qcd_var_fraction + pred_qcd_var_scale

    pred_group_yields = {group: group_yields[group].copy() for group in CLASS_NAMES}
    pred_group_vars = {group: group_vars[group].copy() for group in CLASS_NAMES}
    pred_group_yields[QCD_CLASS_NAME] = pred_qcd_vals.copy()
    pred_group_vars[QCD_CLASS_NAME] = pred_qcd_vars.copy()

    true_total_vals = np.zeros(len(region_labels), dtype=float)
    true_total_vars = np.zeros(len(region_labels), dtype=float)
    pred_total_vals = np.zeros(len(region_labels), dtype=float)
    pred_total_vars = np.zeros(len(region_labels), dtype=float)
    for group_name in CLASS_NAMES:
        true_total_vals += group_yields[group_name]
        true_total_vars += group_vars[group_name]
        pred_total_vals += pred_group_yields[group_name]
        pred_total_vars += pred_group_vars[group_name]

    abcd_group_vals = {group: np.zeros(4, dtype=float) for group in CLASS_NAMES}
    abcd_group_vars = {group: np.zeros(4, dtype=float) for group in CLASS_NAMES}
    abcd_masks = [a_union_mask, b_mask, c_mask, d_mask]
    for reg_idx, mask in enumerate(abcd_masks):
        for group_name in CLASS_NAMES:
            group_mask = mask & (group_labels == group_name)
            vals = weights[group_mask]
            abcd_group_vals[group_name][reg_idx] = float(np.sum(vals))
            abcd_group_vars[group_name][reg_idx] = float(np.sum(vals ** 2))

    root_path = os.path.join(OUTPUT_DIR, ROOT_FILE_NAME)
    write_root_output(
        root_path,
        edges,
        sample_yields,
        sample_vars,
        group_yields,
        group_vars,
        pred_qcd_vals,
        pred_qcd_vars,
        true_qcd_vals,
        true_qcd_vars,
        pred_total_vals,
        pred_total_vars,
        true_total_vals,
        true_total_vars,
    )

    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_group_vals,
        abcd_group_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_counts.pdf"),
        normalize_per_bin=False,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_group_vals,
        abcd_group_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_fractions.pdf"),
        normalize_per_bin=True,
    )
    plot_signal_region_prediction(
        region_labels,
        pred_group_yields,
        pred_total_vals,
        pred_total_vars,
        true_total_vals,
        true_total_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_signal_regions_total.pdf"),
        CLASS_NAMES,
        "Events",
    )
    plot_signal_region_prediction(
        region_labels,
        {QCD_CLASS_NAME: pred_qcd_vals.copy()},
        pred_qcd_vals,
        pred_qcd_vars,
        true_qcd_vals,
        true_qcd_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_signal_regions_qcd.pdf"),
        [QCD_CLASS_NAME],
        "QCD Events",
    )

    log_message("Finished qcd_est.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_message(f"Runtime error: {exc}")
        raise
