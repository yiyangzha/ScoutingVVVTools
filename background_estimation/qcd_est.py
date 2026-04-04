#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summary: Estimate QCD backgrounds with an ABCD method."""

from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import pickle
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import uproot
import xgboost as xgb

import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator, ScalarFormatter

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)

# --- Configuration defaults (match train.py) ---
BASE_DIR = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/train/dataset"
SIG_DIR = os.path.join(BASE_DIR, "signal")
BKG_DIR = os.path.join(BASE_DIR, "bkg")
DATA_DIR = os.path.join(BASE_DIR, "data")

ENTRIES_PER_SAMPLE = 1_000_000
LUMI = 171.0
MODEL_STEM = "bdt_fat2_nocorr_v4"

alt_thr_sets = [
    # TODO: replace with real thresholds
    (np.array([0.850, 0.000, 0.000, 0.000], dtype=float), np.array([1.000, 1.000, 1.000, 1.000], dtype=float)),
    (np.array([0.800, 0.000, 0.000, 0.000], dtype=float), np.array([0.880, 1.000, 1.000, 1.000], dtype=float)),
    (np.array([0.770, 0.000, 0.000, 0.000], dtype=float), np.array([0.850, 1.000, 1.000, 1.000], dtype=float)),
    (np.array([0.700, 0.000, 0.000, 0.000], dtype=float), np.array([0.770, 1.000, 1.000, 1.000], dtype=float)),
]
alt_labels = ["1", "2", "3", "4"]

branches2 = [
    "N_ak8", "N_ak4", "H_T",
    "pt8_1", "pt8_2", "eta8_1", "eta8_2", "msd8_1", "msd8_2", "mr8_1", "mr8_2",
    "WvsQCD_1", "WvsQCD_2",
    "pt4_1", "pt4_2", "pt4_3", "pt4_4", "eta4_1", "eta4_2", "eta4_3", "eta4_4",
    "mPF4_1", "mPF4_2", "mPF4_3", "mPF4_4", "nConst4_1", "nConst4_2",
    "nConst4_3", "nConst4_4", "nCh4_1", "nCh4_2", "nCh4_3", "nCh4_4",
    "nEle4_1", "nEle4_2", "nEle4_3", "nEle4_4",
    "area4_1", "area4_2", "area4_3", "area4_4",
    "chEmEF4_1", "chEmEF4_2", "chEmEF4_3", "chEmEF4_4",
    "chHEF4_1", "chHEF4_2", "chHEF4_3", "chHEF4_4",
    "neEmEF4_1", "neEmEF4_2", "neEmEF4_3", "neEmEF4_4",
    "neHEF4_1", "neHEF4_2", "neHEF4_3", "neHEF4_4",
    "PT", "dR8", "dPhi", "m1overM", "m2overM", "sphereM",
    "dR84_min", "dR44_min", "dR8L_min",
    "ptL_1", "ptL_2", "ptL_3",
    "etaL_1", "etaL_2", "etaL_3", "phiL_1", "phiL_2", "phiL_3",
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3", "isoHcalL_1", "isoHcalL_2", "isoHcalL_3",
]

branches3 = [
    "N_ak8", "N_ak4", "H_T",
    "pt8_1", "pt8_2", "pt8_3", "eta8_1", "eta8_2", "eta8_3",
    "msd8_1", "msd8_2", "msd8_3", "mr8_1", "mr8_2", "mr8_3",
    "WvsQCD_1", "WvsQCD_2", "WvsQCD_3",
    "sphereM", "M", "m1overM", "m2overM", "m3overM", "PT",
    "dR_min", "dR_max", "dPhi_min", "dPhi_max", "dRL_min",
    "ptL_1", "ptL_2", "ptL_3",
    "etaL_1", "etaL_2", "etaL_3", "phiL_1", "phiL_2", "phiL_3",
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3", "isoHcalL_1", "isoHcalL_2", "isoHcalL_3",
]

BRANCH_CLIP_RANGES = {
    "H_T": (0, 13600),
    "M": (0, 13600),
    "M8": (0, 13600),
    "M84": (0, 13600),
    "PT": (0, 13600),
    "pt4_1": (0, 13600),
    "pt4_2": (0, 13600),
    "pt4_3": (0, 13600),
    "pt4_4": (0, 13600),
    "pt8_1": (0, 13600),
    "pt8_2": (0, 13600),
    "pt8_3": (0, 13600),
    "ptL_1": (0, 13600),
    "ptL_2": (0, 13600),
    "ptL_3": (0, 13600),
}

# x-section [pb] / raw entries * 1e8
_RAW_XSEC = {
    "www": 4.852,
    "wwz": 1.024,
    "wzz": 0.344,
    "zzz": 0.0863,
    "ww": 118.9,
    "wz": 45.16,
    "zz": 253.5,
    "qcd_ht100to200": 1.973e7,
    "qcd_ht200to400": 1.703e6,
    "qcd_ht400to600": 8.970e4,
    "qcd_ht600to800": 1.056e4,
    "qcd_ht800to1000": 2413,
    "qcd_ht1000to1200": 772.4,
    "qcd_ht1200to1500": 356.2,
    "qcd_ht1500to2000": 110.7,
    "qcd_ht2000toinf": 29.56,
    "tt_had": 79.80,
    "tt_semilep": 75.68,
    "zh": 2.26,
    "wplush": 3.72,
    "wminush": 2.15,
    "data": 20821,
}

TYPE = {
    "www": 0,
    "wwz": 1,
    "wzz": 2,
    "zzz": 3,
    "zh": 4,
    "wplush": 5,
    "wminush": 5,
    "qcd_ht100to200": 10,
    "qcd_ht200to400": 10,
    "qcd_ht400to600": 10,
    "qcd_ht600to800": 10,
    "qcd_ht800to1000": 10,
    "qcd_ht1000to1200": 10,
    "qcd_ht1200to1500": 10,
    "qcd_ht1500to2000": 10,
    "qcd_ht2000toinf": 10,
    "tt_had": 11,
    "tt_semilep": 12,
    "ww": 13,
    "wz": 14,
    "zz": 15,
    "data": -2,
    "unknown": -1,
}

CLASS_NAMES = ["VVV", "VH", "TT", "VV", "QCD"]
PROCESS_GROUPS = ["VVV", "VH", "VV", "QCD", "TT"]
_default_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
    "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
)
COLOR_MAP = {proc: _default_colors[i % len(_default_colors)] for i, proc in enumerate(PROCESS_GROUPS)}


def log(msg: str) -> None:
    print(msg, flush=True)


def fatal(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def group_files(path: str) -> Dict[str, List[str]]:
    files = glob.glob(os.path.join(path, "*.root"))
    groups: Dict[str, List[str]] = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        key = re.sub(r"_[0-9]+$", "", name)
        if "202" in key:
            key = "data"
        groups.setdefault(key, []).append(f)
    return groups


def count_entries(path: str, tree_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    groups = group_files(path)
    for sample, files in groups.items():
        total = 0
        for fpath in files:
            try:
                with uproot.open(fpath) as uf:
                    if tree_name in uf:
                        total += uf[tree_name].num_entries
            except Exception as exc:
                fatal(f"Failed to read {fpath}: {exc}")
        counts[sample.lower()] = total
    return counts


def build_xsec_map(sig_dir: str, bkg_dir: str, data_dir: str, tree_name: str) -> Dict[str, float]:
    if tree_name != "fat2":
        fatal(f"Unsupported tree_name for this script: {tree_name}")
    sig_fat2 = count_entries(sig_dir, "fat2") if os.path.isdir(sig_dir) else {}
    bkg_fat2 = count_entries(bkg_dir, "fat2") if os.path.isdir(bkg_dir) else {}
    data_fat2 = count_entries(data_dir, "fat2") if os.path.isdir(data_dir) else {}

    xsec_map: Dict[str, float] = {}
    for sample, xsec in _RAW_XSEC.items():
        e2 = sig_fat2.get(sample, 0) + bkg_fat2.get(sample, 0) + data_fat2.get(sample, 0)
        xsec_map[sample] = xsec * e2
    return xsec_map


def filter_X(
    X: pd.DataFrame,
    y,
    w,
    branch: list,
    thresholds: dict | None = None,
    apply_to_sentinel: bool = True,
):
    if thresholds is None:
        return X.copy(), y.copy(), w.copy()

    mask = pd.Series(True, index=X.index)

    def _combine_masks(masks: list[pd.Series], op: str, idx) -> pd.Series:
        if not masks:
            return pd.Series(True, index=idx) if op == "&" else pd.Series(False, index=idx)
        out = masks[0]
        for m in masks[1:]:
            out = (out & m) if op == "&" else (out | m)
        return out

    def _mask_from_cond(col: pd.Series, cond) -> pd.Series:
        idx = col.index
        if cond is None:
            return pd.Series(True, index=idx)
        if isinstance(cond, (int, float, np.integer, np.floating)):
            return col > float(cond)
        if isinstance(cond, tuple) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
            mn, mx = cond
            m = pd.Series(True, index=idx)
            if mn is not None:
                m &= col > mn
            if mx is not None:
                m &= col < mx
            return m
        if isinstance(cond, (list, tuple)):
            masks = [_mask_from_cond(col, c) for c in cond]
            return _combine_masks(masks, "|", idx)
        if isinstance(cond, dict):
            if "|" in cond:
                masks = [_mask_from_cond(col, c) for c in cond["|"]]
                return _combine_masks(masks, "|", idx)
            if "&" in cond:
                masks = [_mask_from_cond(col, c) for c in cond["&"]]
                return _combine_masks(masks, "&", idx)
            if "or" in cond:
                masks = [_mask_from_cond(col, c) for c in cond["or"]]
                return _combine_masks(masks, "|", idx)
            if "and" in cond:
                masks = [_mask_from_cond(col, c) for c in cond["and"]]
                return _combine_masks(masks, "&", idx)
            raise ValueError(f"Unsupported dict condition keys for column: {cond}")
        raise TypeError(f"Unsupported condition type for column: {type(cond)}")

    for b in branch:
        if b not in X.columns:
            raise KeyError(f"Column {b!r} not found in X")

        col = X[b]
        cond = thresholds.get(b, None)
        sentinel = col < -990

        if apply_to_sentinel:
            mask &= ~sentinel
            if cond is not None:
                cond_mask = _mask_from_cond(col, cond)
                mask &= cond_mask
        else:
            if cond is not None:
                cond_mask = _mask_from_cond(col, cond)
                mask &= cond_mask | sentinel

    return X.loc[mask].copy(), y[mask.values].copy(), w[mask.values].copy()


def standardize_X(X: pd.DataFrame) -> pd.DataFrame:
    def _need_ln(col: str) -> bool:
        if col.startswith("N_"):
            return True
        if col == "H_T":
            return True
        if col.startswith("pt"):
            return True
        if col.startswith("sphereM"):
            return True
        if col.startswith("iso"):
            return True
        if col.startswith("M"):
            return True
        if col == "PT":
            return True
        if col.startswith("mPF"):
            return True
        if col.startswith(("nMu", "nNh", "nPho", "nCh", "nEle", "nConst")):
            return True
        if "overPT" in col:
            return True
        if col.startswith("mr"):
            return True
        return False

    for col in X.columns:
        arr = X[col].values
        mask = arr < -990
        valid = ~mask
        if not valid.any():
            continue

        clip_min, clip_max = BRANCH_CLIP_RANGES.get(col, (None, None))
        if clip_min is not None:
            arr[valid & (arr < clip_min)] = clip_min
        if clip_max is not None:
            arr[valid & (arr > clip_max)] = clip_max

        if _need_ln(col):
            pos = valid & (arr > 0)
            if pos.any():
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float, copy=True)
                arr[pos] = np.log(arr[pos])

        X[col] = arr
    return X


def prepare_mc_data(
    tree_name: str,
    branches: list,
    entries_per_sample: int,
    sig_dir: str,
    bkg_dir: str,
    xsec_map: Dict[str, float],
    lumi: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    def load_samples(path: str) -> pd.DataFrame:
        groups = group_files(path)
        dfs = []
        for sample_id, file_list in groups.items():
            sid = sample_id.lower()
            if sid == "data":
                continue
            if sid not in xsec_map:
                fatal(f"Sample '{sample_id}' not found in xsec map")
            if sid not in TYPE:
                fatal(f"Sample '{sample_id}' not found in TYPE map")
            xsec = xsec_map[sid]

            parts = []
            n_read = 0
            for fpath in file_list:
                remain = entries_per_sample - n_read
                if remain <= 0:
                    break
                try:
                    with uproot.open(fpath) as uf:
                        tree = uf[tree_name]
                        n_entries = tree.num_entries
                        n_to_read = min(remain, n_entries)
                        if n_to_read <= 0:
                            continue
                        df_part = tree.arrays(
                            branches + ["type"],
                            library="pd",
                            entry_start=0,
                            entry_stop=n_to_read,
                        )
                except Exception as exc:
                    fatal(f"Failed to read {fpath}: {exc}")

                parts.append(df_part)
                n_read += len(df_part)
                if n_read >= entries_per_sample:
                    break

            if n_read == 0:
                log(f"[WARN] Sample '{sample_id}': no entries read")
                continue

            df_sample = pd.concat(parts, ignore_index=True)
            del parts
            gc.collect()

            per_evt_w = xsec / float(n_read) if n_read > 0 else 0.0
            per_evt_w *= lumi / 1e5
            df_sample["weight"] = per_evt_w
            df_sample["type"] = int(TYPE[sid])

            dfs.append(df_sample)
            log(f"[INFO] Sample '{sample_id}': read {n_read} entries, weight {per_evt_w:.6g}")

        if not dfs:
            cols = branches + ["weight", "type"]
            return pd.DataFrame(columns=cols)
        return pd.concat(dfs, ignore_index=True)

    df_sig = load_samples(sig_dir)
    df_bkg = load_samples(bkg_dir)

    df_all = pd.concat([df_sig, df_bkg], ignore_index=True)
    del df_sig, df_bkg
    gc.collect()

    if df_all.empty:
        fatal("No MC entries loaded from signal/background")

    #df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_all[branches]
    y = df_all.pop("type").values
    w = df_all.pop("weight").values
    del df_all
    gc.collect()

    return X, y, w


def resolve_model_path(model_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = []

    if os.path.isabs(model_path):
        candidates.append(model_path)
    else:
        candidates.append(os.path.join(os.getcwd(), model_path))
        candidates.append(os.path.join(script_dir, model_path))

    base_candidates = candidates[:]
    for base in base_candidates:
        root, ext = os.path.splitext(base)
        if ext:
            continue
        candidates.append(root + ".json")
        candidates.append(root + ".pkl")
        candidates.append(root + ".pickle")

    for path in candidates:
        if os.path.isfile(path):
            return path

    fatal(f"Model file not found (tried: {candidates})")
    return ""


def load_xgb_model(model_path: str) -> xgb.Booster:
    if model_path.endswith(".json"):
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster

    try:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        fatal(f"Failed to load model {model_path}: {exc}")
        raise

    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        return obj.get_booster()

    fatal(f"Unsupported model object in {model_path}")
    raise RuntimeError("unreachable")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def predict_scores(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    expected = booster.num_features()
    if expected and X.shape[1] != expected:
        fatal(f"Model expects {expected} features, got {X.shape[1]}")

    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat, output_margin=True)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        n = X.shape[0]
        if n == 0 or raw.size % n != 0:
            fatal("Unexpected prediction shape from model")
        raw = raw.reshape(n, raw.size // n)

    probs = softmax(raw)
    return probs


def map_type_to_group(t: np.ndarray) -> np.ndarray:
    vvv = {TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]}
    vh = {TYPE["zh"], TYPE["wplush"], TYPE["wminush"]}
    vv = {TYPE["ww"], TYPE["wz"], TYPE["zz"]}
    tt = {TYPE["tt_had"], TYPE["tt_semilep"]}
    qcd = {
        TYPE["qcd_ht100to200"],
        TYPE["qcd_ht200to400"],
        TYPE["qcd_ht400to600"],
        TYPE["qcd_ht600to800"],
        TYPE["qcd_ht800to1000"],
        TYPE["qcd_ht1000to1200"],
        TYPE["qcd_ht1200to1500"],
        TYPE["qcd_ht1500to2000"],
        TYPE["qcd_ht2000toinf"],
    }

    out = np.full(t.shape[0], "", dtype=object)
    out[np.isin(t, list(vvv))] = "VVV"
    out[np.isin(t, list(vh))] = "VH"
    out[np.isin(t, list(vv))] = "VV"
    out[np.isin(t, list(tt))] = "TT"
    out[np.isin(t, list(qcd))] = "QCD"
    return out


def hist_with_var(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, _ = np.histogram(values, bins=bins, weights=weights)
    vars_, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return vals.astype(float), vars_.astype(float)


def first_last_true(mask: np.ndarray) -> Tuple[int | None, int | None]:
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def calc_ratio_mc_over_data(mc_vals, mc_vars, data_vals, data_vars):
    mc_vals = np.asarray(mc_vals, dtype=float)
    mc_vars = np.asarray(mc_vars, dtype=float)
    data_vals = np.asarray(data_vals, dtype=float)
    data_vars = np.asarray(data_vars, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(data_vals > 0, mc_vals / data_vals, np.nan)
        term_mc = np.where(mc_vals > 0, mc_vars / np.maximum(mc_vals, 1e-300) ** 2, 0.0)
        term_dt = np.where(data_vals > 0, data_vars / np.maximum(data_vals, 1e-300) ** 2, 0.0)
        sigma_r = np.abs(r) * np.sqrt(term_mc + term_dt)
    return r, sigma_r


def pretty_ylim_max(ymax: float) -> float:
    if not np.isfinite(ymax) or ymax <= 0:
        return 1.0
    return 10 ** (math.log10(ymax) + 0.15)

def apply_log_ticks(ax, axis: str) -> None:
    locator = LogLocator(base=10.0, subs=(1.0,))
    formatter = LogFormatterMathtext(base=10.0)
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_locator(NullLocator())

def apply_linear_ticks(ax, axis: str) -> None:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)

def compute_visible_xlim(edges: np.ndarray, values: np.ndarray, y_min: float) -> Tuple[float, float]:
    mask = values > y_min
    i0, i1 = first_last_true(mask)
    if i0 is None:
        return float(edges[0]), float(edges[-1])
    return float(edges[i0]), float(edges[i1 + 1])

def bdt_mask_from_thresholds(scores: np.ndarray, thr_low: np.ndarray, thr_high: np.ndarray) -> np.ndarray:
    mask = np.ones(scores.shape[0], dtype=bool)
    for i in range(4):
        mask &= (scores[:, i] >= thr_low[i]) & (scores[:, i] <= thr_high[i])
    return mask

def add_uncert_band(ax, edges: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
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

def compute_bins(values: np.ndarray, logx: bool, max_range: float) -> np.ndarray | None:
    finite = np.isfinite(values)
    if not finite.any():
        return None
    vals = values[finite]
    if logx:
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        vmin = max(1.0, vmin)
        vmax = min(max_range, vmax)
        if vmax <= vmin:
            vmin = max(1.0, max_range / 10.0)
            vmax = max_range
            if vmax <= vmin:
                vmax = vmin * 1.1
        return np.logspace(np.log10(vmin), np.log10(vmax), 11)
    vmax = float(np.max(vals))
    vmax = min(max_range, vmax)
    vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, 11)


def plot_stacked_hist(
    edges: np.ndarray,
    mc_group_vals: Dict[str, np.ndarray],
    mc_group_vars: Dict[str, np.ndarray],
    var_label: str,
    out_path: str,
    lumi: float,
    logx: bool = False,
):
    ensure_dir(os.path.dirname(out_path))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]

    mc_vals_total = np.zeros(len(edges) - 1, dtype=float)
    mc_vars_total = np.zeros(len(edges) - 1, dtype=float)
    yields = {}
    for g in PROCESS_GROUPS:
        vals_g = mc_group_vals.get(g, np.zeros_like(mc_vals_total))
        vars_g = mc_group_vars.get(g, np.zeros_like(mc_vars_total))
        mc_vals_total += vals_g
        mc_vars_total += vars_g
        yields[g] = float(np.sum(vals_g))

    y_min = 0.1
    mask_vis = mc_vals_total > y_min
    if not np.any(mask_vis):
        log(f"[WARN] Skip empty histogram for {out_path}")
        return
    xlo, xhi = compute_visible_xlim(edges, mc_vals_total, y_min)
    if xhi <= xlo:
        xhi = float(edges[-1])

    order = np.argsort([yields[g] for g in PROCESS_GROUPS])
    groups_ordered = [PROCESS_GROUPS[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros_like(mc_vals_total)
    for g in groups_ordered:
        vals_g = mc_group_vals[g]
        ax.bar(
            edges[:-1],
            vals_g,
            width=bin_widths,
            bottom=bottom,
            align="edge",
            color=COLOR_MAP[g],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=g,
        )
        bottom += vals_g

    mc_sigma = np.sqrt(np.maximum(mc_vars_total, 0.0))
    lower = np.clip(mc_vals_total - mc_sigma, 1e-12, None)
    upper = np.clip(mc_vals_total + mc_sigma, 1e-12, None)
    add_uncert_band(ax, edges, lower, upper)

    ax.margins(x=0)
    ax.set_yscale("log")
    apply_log_ticks(ax, "y")
    ax.set_xlim(xlo, xhi)
    ymax = np.nanmax(mc_vals_total[mask_vis]) if np.any(mask_vis) else 1.0
    ax.set_ylim(y_min, max(1.0, pretty_ylim_max(ymax)))
    ax.set_ylabel("Events", fontsize=24)
    ax.set_xlabel(var_label, fontsize=24)
    if logx:
        ax.set_xscale("log")
        apply_log_ticks(ax, "x")
    else:
        apply_linear_ticks(ax, "x")

    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
    ax.legend(loc="best", fontsize=17, frameon=False, ncol=2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVE] {out_path}")


def plot_region_counts(
    region_labels: List[str],
    group_vals: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
    out_path: str,
    lumi: float,
    logy: bool = True,
    normalize: bool = False,
    normalize_per_bin: bool = False,
    show_uncert: bool = True,
    ylim: Tuple[float, float] | None = None,
):
    ensure_dir(os.path.dirname(out_path))
    n = len(region_labels)
    edges = np.arange(n + 1, dtype=float)
    centers = edges[:-1] + 0.5
    widths = np.full(n, 1.0)

    totals = np.zeros(n, dtype=float)
    total_vars = np.zeros(n, dtype=float)
    yields = {}
    for g in PROCESS_GROUPS:
        vals = group_vals.get(g, np.zeros(n, dtype=float))
        vars_ = group_vars.get(g, np.zeros(n, dtype=float))
        totals += vals
        total_vars += vars_
        yields[g] = float(np.sum(vals))

    order = np.argsort([yields[g] for g in PROCESS_GROUPS])
    groups_ordered = [PROCESS_GROUPS[i] for i in order]

    if normalize_per_bin:
        scales = np.where(totals > 0, 1.0 / totals, 0.0)
        for g in PROCESS_GROUPS:
            group_vals[g] = group_vals[g] * scales
            group_vars[g] = group_vars[g] * (scales ** 2)
        totals = totals * scales
        total_vars = total_vars * (scales ** 2)
    elif normalize:
        total_sum = float(np.sum(totals))
        if total_sum > 0:
            for g in PROCESS_GROUPS:
                group_vals[g] = group_vals[g] / total_sum
                group_vars[g] = group_vars[g] / (total_sum ** 2)
            totals = totals / total_sum
            total_vars = total_vars / (total_sum ** 2)
        else:
            log(f"[WARN] Skip empty normalized plot for {out_path}")
            return

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(n, dtype=float)
    for g in groups_ordered:
        vals_g = group_vals[g]
        ax.bar(
            edges[:-1],
            vals_g,
            width=widths,
            bottom=bottom,
            align="edge",
            color=COLOR_MAP[g],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=g,
        )
        bottom += vals_g

    if show_uncert:
        sigma = np.sqrt(np.maximum(total_vars, 0.0))
        lower = np.clip(totals - sigma, 1e-12, None)
        upper = np.clip(totals + sigma, 1e-12, None)
        add_uncert_band(ax, edges, lower, upper)

    if logy:
        y_min = 0.1
        ax.set_yscale("log")
        apply_log_ticks(ax, "y")
        ymax = np.nanmax(totals[totals > y_min]) if np.any(totals > y_min) else 1.0
        ax.set_ylim(y_min, max(1.0, pretty_ylim_max(ymax)))
    else:
        y_lo = 0.0 if ylim is None else ylim[0]
        y_hi = 1.0 if ylim is None else ylim[1]
        ax.set_ylim(y_lo, y_hi)
        apply_linear_ticks(ax, "y")

    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylabel("Events", fontsize=24)
    ax.set_xlabel("Selection", fontsize=24)
    ax.set_xticks(centers)
    ax.set_xticklabels(region_labels, fontsize=14)
    ax.margins(x=0)

    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=lumi, ax=ax)
    ax.legend(loc="best", fontsize=17, frameon=False, ncol=2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVE] {out_path}")


def plot_region_a_with_abcd(
    edges: np.ndarray,
    pred_group_vals: Dict[str, np.ndarray],
    pred_total_vals: np.ndarray,
    pred_total_vars: np.ndarray,
    true_vals: np.ndarray,
    true_vars: np.ndarray,
    var_label: str,
    out_path: str,
    lumi: float,
    logx: bool = False,
    groups: List[str] | None = None,
    xtick_labels: List[str] | None = None,
    force_full_range: bool = False,
):
    ensure_dir(os.path.dirname(out_path))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]

    y_min = 0.1
    mask_vis = (pred_total_vals > y_min) | (true_vals > y_min)
    if not np.any(mask_vis):
        log(f"[WARN] Skip empty A-region plot for {out_path}")
        return
    xlo, xhi = compute_visible_xlim(edges, np.maximum(pred_total_vals, true_vals), y_min)
    if xhi <= xlo:
        xhi = float(edges[-1])
    if force_full_range:
        xlo = float(edges[0])
        xhi = float(edges[-1])

    groups_use = groups if groups is not None else PROCESS_GROUPS
    yields = {g: float(np.sum(pred_group_vals.get(g, 0.0))) for g in groups_use}
    order = np.argsort([yields[g] for g in groups_use])
    groups_ordered = [groups_use[i] for i in order]

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    bottom = np.zeros_like(pred_total_vals)
    for g in groups_ordered:
        vals_g = pred_group_vals.get(g, np.zeros_like(pred_total_vals))
        ax.bar(
            edges[:-1],
            vals_g,
            width=bin_widths,
            bottom=bottom,
            align="edge",
            color=COLOR_MAP[g],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=g,
        )
        bottom += vals_g

    pred_sigma = np.sqrt(np.maximum(pred_total_vars, 0.0))
    lower = np.clip(pred_total_vals - pred_sigma, 1e-12, None)
    upper = np.clip(pred_total_vals + pred_sigma, 1e-12, None)
    add_uncert_band(ax, edges, lower, upper)

    true_sigma = np.sqrt(np.maximum(true_vars, 0.0))
    y_plot = np.where(true_vals > 0, true_vals, np.nan)
    ax.errorbar(
        bin_centers,
        y_plot,
        yerr=true_sigma,
        xerr=None,
        fmt="o",
        ms=7.6,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
        label="True",
    )

    ax.margins(x=0)
    ax.set_yscale("log")
    apply_log_ticks(ax, "y")
    ax.set_xlim(xlo, xhi)
    ymax = max(
        np.nanmax(pred_total_vals[mask_vis]) if np.any(mask_vis) else 1.0,
        np.nanmax(true_vals[mask_vis]) if np.any(mask_vis) else 1.0,
        1.0,
    )
    ax.set_ylim(y_min, max(1.0, pretty_ylim_max(ymax)))
    ax.set_ylabel("Events", fontsize=24)
    if logx:
        ax.set_xscale("log")
        apply_log_ticks(ax, "x")
    else:
        apply_linear_ticks(ax, "x")
    if xtick_labels is not None:
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(xtick_labels, fontsize=14)

    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=lumi, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if "True" in labels:
        idx = labels.index("True")
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))
    ax.legend(handles, labels, loc="best", fontsize=17, frameon=False, ncol=2)

    ratio, ratio_err = calc_ratio_mc_over_data(
        pred_total_vals, pred_total_vars, true_vals, true_vars
    )
    sel = (bin_centers >= xlo) & (bin_centers <= xhi)
    r_centers = bin_centers[sel]
    r_vals = ratio[sel]
    r_errs = ratio_err[sel]

    axr.errorbar(
        r_centers,
        r_vals,
        yerr=r_errs,
        xerr=None,
        fmt="o",
        ms=7.6,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
    )
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)

    finite_r = np.isfinite(r_vals)
    if np.any(finite_r):
        rmax = float(np.nanmax(r_vals[finite_r]))
        rmin = float(np.nanmin(r_vals[finite_r]))
    else:
        rmax = 1.0
        rmin = 1.0
    if rmax <= 0:
        rmax = 1.0
        rmin = 1.0
    if rmax < 5.0:
        ylo = 0.8 * rmin
        yhi = 1.2 * rmax
    else:
        ylo = 0.0
        yhi = 5.0
    ylo = min(ylo, 1.0)
    yhi = max(yhi, 1.0)
    axr.set_ylim(ylo, yhi)

    axr.set_ylabel(r"$\frac{Predict}{True}$", fontsize=26)
    axr.yaxis.set_label_coords(-0.05, 0.6)
    axr.set_xlabel(var_label, fontsize=24)
    if logx:
        axr.set_xscale("log")
        apply_log_ticks(axr, "x")
    else:
        apply_linear_ticks(axr, "x")
    if xtick_labels is not None:
        axr.set_xticks(bin_centers)
        axr.set_xticklabels(xtick_labels, fontsize=14)
    apply_linear_ticks(axr, "y")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVE] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCD estimation with BDT scores")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base dataset directory")
    parser.add_argument("--sig-dir", default=None, help="Signal directory override")
    parser.add_argument("--bkg-dir", default=None, help="Background directory override")
    parser.add_argument("--data-dir", default=None, help="Data directory override (counts only)")
    parser.add_argument("--model", default=MODEL_STEM, help="Model path or stem")
    parser.add_argument("--lumi", type=float, default=LUMI, help="Luminosity")
    parser.add_argument(
        "--entries-per-sample",
        type=int,
        default=ENTRIES_PER_SAMPLE * 2,
        help="Max entries per sample",
    )
    parser.add_argument("--out-dir", default="qcd_est", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sig_dir = args.sig_dir or os.path.join(args.base_dir, "signal")
    bkg_dir = args.bkg_dir or os.path.join(args.base_dir, "bkg")
    data_dir = args.data_dir or os.path.join(args.base_dir, "data")

    if not os.path.isdir(sig_dir):
        fatal(f"Signal directory not found: {sig_dir}")
    if not os.path.isdir(bkg_dir):
        fatal(f"Background directory not found: {bkg_dir}")

    log("[STEP 1/6] Resolve and load model")
    model_path = resolve_model_path(args.model)
    booster = load_xgb_model(model_path)
    log(f"[INFO] Loaded model: {model_path}")

    log("[STEP 2/6] Build xsec map and load MC samples")
    xsec_map = build_xsec_map(sig_dir, bkg_dir, data_dir, "fat2")
    X2, y2, w2 = prepare_mc_data(
        "fat2",
        branches2,
        args.entries_per_sample,
        sig_dir,
        bkg_dir,
        xsec_map,
        args.lumi,
    )

    thresholds = {
        "pt8_1": (180, None),
        "pt8_2": (180, None),
        "eta8_1": (-2.4, 2.4),
        "eta8_2": (-2.4, 2.4),
    }
    log("[STEP 3/6] Apply preselection filters (msd8_1/msd8_2 not applied)")
    X2, y2, w2 = filter_X(X2, y2, w2, branches2, thresholds, apply_to_sentinel=True)
    if X2.empty:
        fatal("No entries left after filtering")

    log("[STEP 4/6] Standardize features and run BDT inference")
    X2_std = standardize_X(X2.copy())
    if "msd8_1" not in X2_std.columns:
        fatal("msd8_1 missing from features; cannot align with training")
    X_input = X2_std.drop(columns=["msd8_1"])
    probs = predict_scores(booster, X_input.to_numpy())
    if probs.shape[1] != 5:
        fatal(f"Expected 5-class scores, got shape {probs.shape}")

    score_cols = ["score_vvv", "score_vh", "score_tt", "score_vv", "score_qcd"]
    df = X2.copy()
    df["weight"] = w2
    df["type"] = y2
    df["group"] = map_type_to_group(df["type"].to_numpy())
    for i, col in enumerate(score_cols):
        df[col] = probs[:, i]

    if (df["group"] == "").any():
        fatal("Unknown group found after TYPE mapping; update TYPE or inputs")

    log("[STEP 5/6] Build region masks and ABCD normalization")
    group_arr = df["group"].to_numpy()
    score_arr = df[score_cols[:4]].to_numpy()
    thr_low = np.array([0.700, 0.000, 0.000, 0.000], dtype=float)
    thr_high = np.array([1.000, 1.000, 1.000, 1.000], dtype=float)
    mask_B = np.ones(score_arr.shape[0], dtype=bool)
    for i in range(4):
        mask_B &= (score_arr[:, i] >= thr_low[i]) & (score_arr[:, i] <= thr_high[i])

    msd1 = df["msd8_1"].to_numpy()
    msd2 = df["msd8_2"].to_numpy()
    mask_msd = (msd1 > 40) & (msd1 < 150) & (msd2 > 40) & (msd2 < 150)

    masks = {
        "A": mask_msd & mask_B,
        "B": mask_msd & ~mask_B,
        "C": ~mask_msd & mask_B,
        "D": ~mask_msd & ~mask_B,
    }

    weights = df["weight"].to_numpy()
    qcd_mask = group_arr == "QCD"
    totals_qcd = {}
    vars_qcd = {}
    for key, m in masks.items():
        w = weights[m & qcd_mask]
        totals_qcd[key] = float(np.sum(w))
        vars_qcd[key] = float(np.sum(w ** 2))

    if totals_qcd["B"] <= 0 or totals_qcd["C"] <= 0 or totals_qcd["D"] <= 0:
        fatal("QCD B/C/D totals must be positive for ABCD scaling")
    if totals_qcd["A"] <= 0:
        fatal("QCD A total is zero; cannot scale to ABCD prediction")

    pred_total_qcd = totals_qcd["B"] * totals_qcd["C"] / totals_qcd["D"]
    pred_var_qcd = (
        (totals_qcd["C"] / totals_qcd["D"]) ** 2 * vars_qcd["B"]
        + (totals_qcd["B"] / totals_qcd["D"]) ** 2 * vars_qcd["C"]
        + (totals_qcd["B"] * totals_qcd["C"] / (totals_qcd["D"] ** 2)) ** 2 * vars_qcd["D"]
    )
    pred_sigma_qcd = math.sqrt(max(pred_var_qcd, 0.0))
    scale_factor_qcd = pred_total_qcd / totals_qcd["A"]
    frac_pred_qcd = pred_sigma_qcd / pred_total_qcd if pred_total_qcd > 0 else 0.0

    log(
        "[INFO] ABCD QCD totals: "
        f"A={totals_qcd['A']:.6g}, B={totals_qcd['B']:.6g}, "
        f"C={totals_qcd['C']:.6g}, D={totals_qcd['D']:.6g}, "
        f"Pred={pred_total_qcd:.6g} +/- {pred_sigma_qcd:.6g}"
    )

    log("[STEP 6/6] Produce plots")
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # (a) Selection count plot
    region_labels = [
        r"BDT Selected" + "\n" + r"$m_{SD}\in[40,150]$",
        r"BDT Unselected" + "\n" + r"$m_{SD}\in[40,150]$",
        r"BDT Selected" + "\n" + r"$m_{SD}\notin[40,150]$",
        r"BDT Unselected" + "\n" + r"$m_{SD}\notin[40,150]$",
    ]
    group_vals = {g: np.zeros(4, dtype=float) for g in PROCESS_GROUPS}
    group_vars = {g: np.zeros(4, dtype=float) for g in PROCESS_GROUPS}
    for i, key in enumerate(["A", "B", "C", "D"]):
        m = masks[key]
        for g in PROCESS_GROUPS:
            sel = m & (group_arr == g)
            w = weights[sel]
            group_vals[g][i] = float(np.sum(w))
            group_vars[g][i] = float(np.sum(w ** 2))

    plot_region_counts(
        region_labels,
        group_vals,
        group_vars,
        os.path.join(out_dir, "selection_regions.pdf"),
        args.lumi,
    )
    plot_region_counts(
        region_labels,
        group_vals,
        group_vars,
        os.path.join(out_dir, "selection_regions_norm.pdf"),
        args.lumi,
        logy=False,
        normalize_per_bin=True,
        show_uncert=False,
        ylim=(0.0, 1.0),
    )

    # (b) Variable plots per region
    var_specs = {
        "H_T": {"logx": True, "max_range": 10000.0, "label": r"$H_{T}$"},
        "msd8_1": {"logx": False, "max_range": 500.0, "label": r"$m_{SD,1}$"},
        "pt8_1": {"logx": True, "max_range": 1000.0, "label": r"$p_{T,1}$"},
    }

    for var, spec in var_specs.items():
        values = df[var].to_numpy()
        finite = np.isfinite(values)
        if not finite.any():
            log(f"[WARN] Skip variable {var}: no finite values")
            continue

        bins = compute_bins(values, spec["logx"], spec["max_range"])
        if bins is None:
            log(f"[WARN] Skip variable {var}: cannot build bins")
            continue

        for region_key in ["B", "C", "D"]:
            m = masks[region_key]
            group_vals = {}
            group_vars = {}
            for g in PROCESS_GROUPS:
                sel = m & (group_arr == g)
                vals_g = values[sel]
                w_g = weights[sel]
                valid = np.isfinite(vals_g)
                vals_g = vals_g[valid]
                w_g = w_g[valid]
                if spec["logx"]:
                    pos = vals_g > 0
                    vals_g = vals_g[pos]
                    w_g = w_g[pos]
                vals_hist, vars_hist = hist_with_var(vals_g, w_g, bins)
                group_vals[g] = vals_hist
                group_vars[g] = vars_hist

            plot_stacked_hist(
                bins,
                group_vals,
                group_vars,
                spec["label"],
                os.path.join(out_dir, f"{var}_region{region_key}.pdf"),
                args.lumi,
                logx=spec["logx"],
            )

        # Region A with ABCD prediction
        group_vals_a = {}
        group_vars_a = {}
        for g in PROCESS_GROUPS:
            sel = masks["A"] & (group_arr == g)
            vals_g = values[sel]
            w_g = weights[sel]
            valid = np.isfinite(vals_g)
            vals_g = vals_g[valid]
            w_g = w_g[valid]
            if spec["logx"]:
                pos = vals_g > 0
                vals_g = vals_g[pos]
                w_g = w_g[pos]
            vals_hist, vars_hist = hist_with_var(vals_g, w_g, bins)
            group_vals_a[g] = vals_hist
            group_vars_a[g] = vars_hist

        true_vals = np.zeros(len(bins) - 1, dtype=float)
        true_vars = np.zeros(len(bins) - 1, dtype=float)
        for g in PROCESS_GROUPS:
            true_vals += group_vals_a[g]
            true_vars += group_vars_a[g]

        pred_qcd_vals = group_vals_a["QCD"] * scale_factor_qcd
        pred_qcd_vars = (pred_qcd_vals * frac_pred_qcd) ** 2

        pred_group_vals = {g: group_vals_a[g].copy() for g in PROCESS_GROUPS}
        pred_group_vals["QCD"] = pred_qcd_vals

        pred_total_vals = np.zeros_like(pred_qcd_vals)
        pred_total_vars = np.zeros_like(pred_qcd_vals)
        for g in PROCESS_GROUPS:
            pred_total_vals += pred_group_vals[g]
            if g == "QCD":
                pred_total_vars += pred_qcd_vars
            else:
                pred_total_vars += group_vars_a[g]

        plot_region_a_with_abcd(
            bins,
            pred_group_vals,
            pred_total_vals,
            pred_total_vars,
            true_vals,
            true_vars,
            spec["label"],
            os.path.join(out_dir, f"{var}_regionA.pdf"),
            args.lumi,
            logx=spec["logx"],
        )

        plot_region_a_with_abcd(
            bins,
            {"QCD": pred_qcd_vals},
            pred_qcd_vals,
            pred_qcd_vars,
            group_vals_a["QCD"],
            group_vars_a["QCD"],
            spec["label"],
            os.path.join(out_dir, f"{var}_regionA_qcd.pdf"),
            args.lumi,
            logx=spec["logx"],
            groups=["QCD"],
        )

    # (c) Alternative BDT threshold sets for region A/C (4 bins)
    n_alt = len(alt_thr_sets)
    edges_alt = np.arange(n_alt + 1, dtype=float)

    base_b_qcd = totals_qcd["B"]
    base_b_var_qcd = vars_qcd["B"]
    base_d_qcd = totals_qcd["D"]
    base_d_var_qcd = vars_qcd["D"]
    if base_b_qcd <= 0 or base_d_qcd <= 0:
        fatal("QCD B/D totals must be positive for alternative threshold plots")

    alt_group_vals = {g: np.zeros(n_alt, dtype=float) for g in PROCESS_GROUPS}
    alt_group_vars = {g: np.zeros(n_alt, dtype=float) for g in PROCESS_GROUPS}
    alt_group_counts = {g: np.zeros(n_alt, dtype=float) for g in PROCESS_GROUPS}
    alt_true_vals = np.zeros(n_alt, dtype=float)
    alt_true_vars = np.zeros(n_alt, dtype=float)
    alt_pred_qcd_vals = np.zeros(n_alt, dtype=float)
    alt_pred_qcd_vars = np.zeros(n_alt, dtype=float)
    alt_pred_qcd_vars_entry = np.zeros(n_alt, dtype=float)
    alt_pred_total_vals = np.zeros(n_alt, dtype=float)
    alt_pred_total_vars = np.zeros(n_alt, dtype=float)

    for i, (thr_low_alt, thr_high_alt) in enumerate(alt_thr_sets):
        mask_B_alt = bdt_mask_from_thresholds(score_arr, thr_low_alt, thr_high_alt)
        mask_A_alt = mask_msd & mask_B_alt
        mask_C_alt = ~mask_msd & mask_B_alt

        for g in PROCESS_GROUPS:
            sel = mask_A_alt & (group_arr == g)
            w_sel = weights[sel]
            alt_group_vals[g][i] = float(np.sum(w_sel))
            alt_group_vars[g][i] = float(np.sum(w_sel ** 2))
            alt_group_counts[g][i] = float(np.count_nonzero(sel))

        w_c_qcd = weights[mask_C_alt & qcd_mask]
        c_qcd = float(np.sum(w_c_qcd))
        c_var_qcd = float(np.sum(w_c_qcd ** 2))

        pred_qcd = base_b_qcd * c_qcd / base_d_qcd
        pred_var_qcd = (
            (c_qcd / base_d_qcd) ** 2 * base_b_var_qcd
            + (base_b_qcd / base_d_qcd) ** 2 * c_var_qcd
            + (base_b_qcd * c_qcd / (base_d_qcd ** 2)) ** 2 * base_d_var_qcd
        )
        alt_pred_qcd_vals[i] = pred_qcd
        alt_pred_qcd_vars[i] = pred_var_qcd
        alt_pred_qcd_vars_entry[i] = (
            (c_qcd / base_d_qcd) ** 2 * base_b_qcd
            + (base_b_qcd / base_d_qcd) ** 2 * c_qcd
            + (base_b_qcd * c_qcd / (base_d_qcd ** 2)) ** 2 * base_d_qcd
        )

        total_true = 0.0
        total_true_var = 0.0
        total_pred = pred_qcd
        total_pred_var = pred_var_qcd
        for g in PROCESS_GROUPS:
            total_true += alt_group_vals[g][i]
            total_true_var += alt_group_vars[g][i]
            if g != "QCD":
                total_pred += alt_group_vals[g][i]
                total_pred_var += alt_group_vars[g][i]

        alt_true_vals[i] = total_true
        alt_true_vars[i] = total_true_var
        alt_pred_total_vals[i] = total_pred
        alt_pred_total_vars[i] = total_pred_var

    hists_path = os.path.join(out_dir, "hists.root")
    with uproot.recreate(hists_path) as f:
        for g in PROCESS_GROUPS:
            g_dir = g.lower()
            values = alt_group_vals[g]
            counts = alt_group_counts[g]
            errs = np.zeros_like(values)
            nonzero = counts > 0
            errs[nonzero] = np.abs(values[nonzero]) / np.sqrt(counts[nonzero])
            f[f"{g_dir}/mc"] = (values, edges_alt)
            f[f"{g_dir}/stat"] = (errs, edges_alt)

        pred_errs = np.sqrt(np.maximum(alt_pred_qcd_vars_entry, 0.0))
        f["qcd_predict/mc"] = (alt_pred_qcd_vals, edges_alt)
        f["qcd_predict/stat"] = (pred_errs, edges_alt)
    log(f"[SAVE] {hists_path}")

    def _read_hist(fobj, key: str) -> Tuple[np.ndarray, np.ndarray]:
        h = fobj[key]
        vals, edges = h.to_numpy()
        return vals.astype(float), edges.astype(float)

    with uproot.open(hists_path) as f:
        pred_qcd_vals, edges_alt = _read_hist(f, "qcd_predict/mc")
        pred_qcd_stat, _ = _read_hist(f, "qcd_predict/stat")
        pred_qcd_vars = pred_qcd_stat ** 2

        pred_group_vals = {}
        pred_group_vars = {}
        true_group_vals = {}
        true_group_vars = {}
        for g in PROCESS_GROUPS:
            g_dir = g.lower()
            vals_g, _ = _read_hist(f, f"{g_dir}/mc")
            stat_g, _ = _read_hist(f, f"{g_dir}/stat")
            pred_group_vals[g] = vals_g.copy()
            pred_group_vars[g] = stat_g ** 2
            true_group_vals[g] = vals_g.copy()
            true_group_vars[g] = stat_g ** 2

        pred_group_vals["QCD"] = pred_qcd_vals
        pred_group_vars["QCD"] = pred_qcd_vars

        pred_total_vals = np.zeros_like(pred_qcd_vals)
        pred_total_vars = np.zeros_like(pred_qcd_vals)
        true_vals = np.zeros_like(pred_qcd_vals)
        true_vars = np.zeros_like(pred_qcd_vals)
        for g in PROCESS_GROUPS:
            pred_total_vals += pred_group_vals[g]
            pred_total_vars += pred_group_vars[g]
            true_vals += true_group_vals[g]
            true_vars += true_group_vars[g]

        plot_region_a_with_abcd(
            edges_alt,
            pred_group_vals,
            pred_total_vals,
            pred_total_vars,
            true_vals,
            true_vars,
            "Selection",
            os.path.join(out_dir, "regionA_bdt_scan.pdf"),
            args.lumi,
            logx=False,
            xtick_labels=alt_labels,
            force_full_range=True,
        )
        plot_region_a_with_abcd(
            edges_alt,
            {"QCD": pred_qcd_vals},
            pred_qcd_vals,
            pred_qcd_vars,
            true_group_vals["QCD"],
            true_group_vars["QCD"],
            "Selection",
            os.path.join(out_dir, "regionA_bdt_scan_qcd.pdf"),
            args.lumi,
            logx=False,
            groups=["QCD"],
            xtick_labels=alt_labels,
            force_full_range=True,
        )


if __name__ == "__main__":
    main()
