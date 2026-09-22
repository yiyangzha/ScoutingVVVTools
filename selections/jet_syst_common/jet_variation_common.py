"""Shared engine for jes_syst.py / jer_syst.py.

Both scripts compare a nominal converted-ntuple set against an up/down
JES- or JER-shifted set (separate directories produced by re-running mode 0
with jet_pt_correction.variation set in convert_branch.C -- see
selections/convert/config_{jes,jer}_{up,down}.json). Unlike pileup/theory
systematics (a same-file alternate weight column), a JES/JER shift changes
jet kinematics themselves, so the "up"/"down" ratio here compares two
INDEPENDENT converted trees for the same sample, not two weight columns of
one tree. This module provides the shared pieces: BDT-model/signal-region
context (mirrors selections/pileup_syst/pileup_syst.py's
_build_tree_context/_infer_proba/_region_mask almost verbatim so the SR
bucketing is identical), and a single-dataset-root weighted-yield
accumulator that each of nominal/up/down is run through independently.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BDT_TOOLS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "BDT"))
if _BDT_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _BDT_TOOLS_DIR)
import model_io  # noqa: E402


def log(msg):
    print(msg, flush=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def per_tree(value, tree):
    if isinstance(value, dict):
        return value.get(tree)
    return value


def resolve_path(base_dir, path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def sample_group(sample_info, name):
    info = sample_info[name]
    return "signal" if info.get("is_signal", False) else "bkg"


def input_files(input_root, input_pattern, sample_info, sample_name):
    sg = sample_group(sample_info, sample_name)
    base = input_pattern.format(input_root=input_root, sample_group=sg, sample=sample_name)
    stem = base[:-5] if base.endswith(".root") else base
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


# ---------------------------------------------------------------------------
# Feature standardization (mirrors signal_region.py / train.py / pileup_syst.py)
# ---------------------------------------------------------------------------

def standardize_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        mask = arr < -990
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


# ---------------------------------------------------------------------------
# Signal-region handling (mirrors qcd_est.py / theory_syst.py / pileup_syst.py)
# ---------------------------------------------------------------------------

def detect_signal_region_axes(df, class_names):
    axes = []
    columns = set(df.columns)
    for col in df.columns:
        if not col.endswith("_low"):
            continue
        axis_name = col[:-4]
        if f"{axis_name}_high" not in columns:
            continue
        if axis_name not in class_names:
            raise KeyError(
                f"Signal region axis {axis_name!r} is not in BDT class_groups: {class_names}."
            )
        axes.append(axis_name)
    return axes


def region_mask(proba, region_row, axis_names, class_names):
    mask = np.ones(proba.shape[0], dtype=bool)
    for axis_name in axis_names:
        low = float(region_row[f"{axis_name}_low"])
        high = float(region_row[f"{axis_name}_high"])
        axis_scores = proba[:, class_names.index(axis_name)]
        if high < 1.0 - 1e-12:
            mask &= (axis_scores >= low) & (axis_scores < high)
        else:
            mask &= axis_scores >= low
    return mask


def threshold_mask(data_np, thresholds):
    if not thresholds:
        n = len(next(iter(data_np.values()))) if data_np else 0
        return np.ones(n, dtype=bool)
    n = len(next(iter(data_np.values())))
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        if branch not in data_np:
            continue
        arr = np.asarray(data_np[branch], dtype=float)
        lo = hi = None
        if isinstance(cond, (list, tuple)) and len(cond) == 2:
            lo, hi = cond[0], cond[1]
        elif isinstance(cond, (int, float)):
            lo = cond
        if lo is not None:
            mask &= arr > float(lo)
        if hi is not None:
            mask &= arr < float(hi)
    return mask


def load_thresholds(bdt_dir, tree_name):
    if bdt_dir is None:
        return {}
    sel_path = os.path.join(bdt_dir, "selection.json")
    if not os.path.exists(sel_path):
        return {}
    sel = load_json(sel_path)
    return sel.get(tree_name, {}).get("thresholds", {})


def build_tree_context(script_dir, bdt_root_cfg, sr_csv_cfg, tree):
    """Return context dict; always has 'thresholds'; has model/region info if enabled."""
    bdt_dir = resolve_path(script_dir, per_tree(bdt_root_cfg, tree))
    ctx = {"thresholds": load_thresholds(bdt_dir, tree), "regions_enabled": False}

    sr_csv_raw = per_tree(sr_csv_cfg, tree)
    if bdt_dir is None or sr_csv_raw is None:
        log(f"  [{tree}] no signal_region_csv/bdt_root configured -> inclusive only")
        return ctx
    sr_csv = resolve_path(script_dir, sr_csv_raw.format(tree_name=tree))
    if not os.path.exists(sr_csv):
        log(f"  [{tree}] signal_region CSV not found ({sr_csv}) -> inclusive only")
        return ctx

    bcfg_path = os.path.join(bdt_dir, "config.json")
    brj_path = os.path.join(bdt_dir, "branch.json")
    sel_path = os.path.join(bdt_dir, "selection.json")
    for p in (bcfg_path, brj_path, sel_path):
        if not os.path.exists(p):
            log(f"  [{tree}] missing {os.path.basename(p)} in bdt_root -> inclusive only")
            return ctx

    bcfg = load_json(bcfg_path)
    brj = load_json(brj_path)
    selj = load_json(sel_path)
    class_names = list(bcfg["class_groups"].keys())
    feature_cols = [b["name"] for b in brj[tree]]
    sel = selj.get(tree, {})
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_transform = sel.get("log_transform", [])
    decorrelate = bcfg.get(tree, {}).get("decorrelate", [])

    model_pattern = bcfg.get("model_pattern", "{output_root}/{tree_name}_model")
    model_base = model_pattern.format(output_root=bdt_dir, tree_name=tree)
    model = model_io.load_model(model_base, bcfg, len(class_names), log_message=log)

    sr_df = pd.read_csv(sr_csv)
    if sr_df.empty:
        log(f"  [{tree}] signal_region CSV is empty -> inclusive only")
        return ctx
    axes = detect_signal_region_axes(sr_df, class_names)
    if not axes:
        log(f"  [{tree}] no per-class axes detected in CSV -> inclusive only")
        return ctx
    bin_ids = [int(round(float(v))) for v in sr_df["bin_index"].tolist()]

    ctx.update(
        regions_enabled=True, model=model, class_names=class_names,
        num_classes=len(class_names), feature_cols=feature_cols,
        clip_ranges=clip_ranges, log_transform=log_transform,
        decorrelate=decorrelate, sr_df=sr_df, axes=axes, bin_ids=bin_ids,
        sr_csv=sr_csv,
    )
    log(f"  [{tree}] regions enabled: {len(bin_ids)} SRs from {os.path.basename(sr_csv)}, "
        f"axes={axes}")
    return ctx


def infer_proba(chunk, mask, ctx):
    cols = {}
    for name in ctx["feature_cols"]:
        if name not in chunk:
            raise KeyError(
                f"BDT feature branch {name!r} missing from input file; "
                f"cannot evaluate signal regions."
            )
        cols[name] = np.asarray(chunk[name])[mask]
    X = pd.DataFrame(cols)
    X = standardize_X(X, ctx["clip_ranges"], ctx["log_transform"])
    if ctx["decorrelate"]:
        drop = [c for c in ctx["decorrelate"] if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
    return model_io.predict_model_proba(ctx["model"], X, ctx["num_classes"])


# ---------------------------------------------------------------------------
# Single-dataset-root weighted SR yield accumulation
# ---------------------------------------------------------------------------

def accumulate_weighted_sr_yields(files, tree_name, ctx, weight_branch, chunk_size):
    """Sum `weight_branch` inclusive (slot 0) and per-SR (slots 1..N) over one
    independent set of converted-ntuple files (nominal, or a JES/JER-shifted
    variant) -- unlike pileup_syst.py's _compute_ratios, up/down live in
    SEPARATE files here, not separate columns of the same file, so this only
    ever accumulates ONE dataset root per call; the caller computes ratios
    across three separate calls (nominal, up, down)."""
    if not files:
        return None

    thresholds = ctx["thresholds"]
    enabled = ctx["regions_enabled"]
    n_slots = 1 + (len(ctx["bin_ids"]) if enabled else 0)

    load_set = {weight_branch, *thresholds.keys()}
    if enabled:
        load_set.update(ctx["feature_cols"])
    load_list = list(load_set)

    sum_w = np.zeros(n_slots, dtype=np.float64)
    n_events = np.zeros(n_slots, dtype=np.int64)
    found_weight = False

    for fpath in files:
        try:
            with uproot.open(fpath) as uf:
                if tree_name not in uf:
                    continue
                tree = uf[tree_name]
                avail = set(tree.keys())
                if weight_branch not in avail:
                    log(f"    WARNING: {os.path.basename(fpath)} missing {weight_branch!r}")
                    continue
                found_weight = True
                actual = [b for b in load_list if b in avail]

                for chunk in tree.iterate(expressions=actual, step_size=chunk_size, library="np"):
                    mask = threshold_mask(chunk, thresholds)
                    if not mask.any():
                        continue

                    w = np.asarray(chunk[weight_branch], dtype=np.float64)[mask]
                    sum_w[0] += float(w.sum())
                    n_events[0] += int(mask.sum())

                    if enabled:
                        proba = infer_proba(chunk, mask, ctx)
                        for j in range(len(ctx["bin_ids"])):
                            rmask = region_mask(
                                proba, ctx["sr_df"].iloc[j], ctx["axes"], ctx["class_names"]
                            )
                            if rmask.any():
                                sum_w[j + 1] += float(w[rmask].sum())
                                n_events[j + 1] += int(rmask.sum())
        except Exception as exc:
            log(f"    error reading {fpath}: {exc}")
            continue

    if not found_weight:
        return None

    def slot(idx):
        w = sum_w[idx]
        n = int(n_events[idx])
        if n < 1:
            return None
        return {"sum": float(w), "n_events": n}

    inclusive = slot(0)
    regions = None
    if enabled:
        regions = {bid: slot(j + 1) for j, bid in enumerate(ctx["bin_ids"])}
    return {"inclusive": inclusive, "regions": regions}
