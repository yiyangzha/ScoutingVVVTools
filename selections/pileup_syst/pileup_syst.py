#!/usr/bin/env python3
"""pileup_syst.py  (mode 9)

Computes pileup reweighting variation yield ratios for all MC samples.
The converted ROOT files must include branches added by mode 1 / weight.C:
  weight_pu      -- nominal pileup weight
  weight_pu_up   -- +1sigma PU-profile variation
  weight_pu_down -- -1sigma PU-profile variation

For each (sample, tree) reports:
  - inclusive (post-preselection) ratios at the top level
  - per-signal-region ratios in a ``regions`` sub-dict (when bdt_root and
    signal_region_csv are configured for the tree)

  pu_up   = sum(weight_pu_up)   / sum(weight_pu)
  pu_down = sum(weight_pu_down) / sum(weight_pu)

Region assignment uses the same BDT model + per-class rectangle definition
as qcd_est.py (via selections/BDT/model_io.py), so the SR bins match the
ABCD method exactly.  Trees without a configured signal_region_csv fall back
to inclusive-only (graceful degradation — fat3 has no SR CSV yet).
Low-stat regions fall back to the inclusive ratio (min_region_events config).

Outputs (written to output_dir):
  pileup_syst_yields.json  -- {sample: {tree: {nom_sum, n_events, pu_up, pu_down,
                                               regions: {bin_id: {...}}}}}
  pileup_syst_yields.csv   -- flat CSV: sample,tree,region,pu_up,pu_down,n_events
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# model_io lives under selections/BDT; make it importable for inference.
_BDT_TOOLS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "BDT"))
if _BDT_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _BDT_TOOLS_DIR)
import model_io  # noqa: E402


def log(msg):
    print(msg, flush=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_path = os.environ.get(
    "PILEUP_SYST_CONFIG_PATH",
    os.path.join(_SCRIPT_DIR, "config.json"),
)
if not os.path.isabs(_cfg_path):
    _cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _cfg_path))

cfg = _load_json(_cfg_path)

SUBMIT_TREES  = cfg.get("submit_trees", ["fat2", "fat3"])
INPUT_ROOT    = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN = cfg["input_pattern"]
OUTPUT_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg.get("output_dir", ".")))
CHUNK_SIZE    = cfg.get("chunk_size", "200 MB")
BDT_ROOT_CFG  = cfg.get("bdt_root", None)            # str or {tree: str}
SR_CSV_CFG    = cfg.get("signal_region_csv", None)    # str/pattern or {tree: str}
MIN_REGION_EVENTS = int(cfg.get("min_region_events", 1))

_sample_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["sample_config"]))
sample_cfg   = _load_json(_sample_cfg_path)
SAMPLE_INFO  = {s["name"]: s for s in sample_cfg["sample"]}

MC_SAMPLES = [
    s["name"]
    for s in sample_cfg["sample"]
    if s.get("is_MC", False)
]


# ---------------------------------------------------------------------------
# Per-tree helpers
# ---------------------------------------------------------------------------

def _per_tree(value, tree):
    if isinstance(value, dict):
        return value.get(tree)
    return value


def _resolve_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_SCRIPT_DIR, path))


def _bdt_dir_for(tree):
    return _resolve_path(_per_tree(BDT_ROOT_CFG, tree))


def _sr_csv_for(tree):
    raw = _per_tree(SR_CSV_CFG, tree)
    if raw is None:
        return None
    raw = raw.format(tree_name=tree)
    return _resolve_path(raw)


# ---------------------------------------------------------------------------
# Sample I/O helpers
# ---------------------------------------------------------------------------

def _sample_group(name):
    info = SAMPLE_INFO[name]
    return "signal" if info.get("is_signal", False) else "bkg"


def _input_files(sample_name):
    sg   = _sample_group(sample_name)
    base = INPUT_PATTERN.format(
        input_root=INPUT_ROOT, sample_group=sg, sample=sample_name
    )
    stem = base[:-5] if base.endswith(".root") else base
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


# ---------------------------------------------------------------------------
# Feature standardization (mirrors signal_region.py / train.py)
# ---------------------------------------------------------------------------

def _standardize_X(X, clip_ranges, log_transform):
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
# Signal-region handling (mirrors qcd_est.py / theory_syst.py)
# ---------------------------------------------------------------------------

def _detect_signal_region_axes(df, class_names):
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
                f"Signal region axis {axis_name!r} is not in BDT class_groups: "
                f"{class_names}."
            )
        axes.append(axis_name)
    return axes


def _region_mask(proba, region_row, axis_names, class_names):
    mask = np.ones(proba.shape[0], dtype=bool)
    for axis_name in axis_names:
        low  = float(region_row[f"{axis_name}_low"])
        high = float(region_row[f"{axis_name}_high"])
        axis_scores = proba[:, class_names.index(axis_name)]
        if high < 1.0 - 1e-12:
            mask &= (axis_scores >= low) & (axis_scores < high)
        else:
            mask &= axis_scores >= low
    return mask


# ---------------------------------------------------------------------------
# Per-tree context (thresholds always; model + regions when configured)
# ---------------------------------------------------------------------------

def _load_thresholds(bdt_dir, tree_name):
    if bdt_dir is None:
        return {}
    sel_path = os.path.join(bdt_dir, "selection.json")
    if not os.path.exists(sel_path):
        return {}
    sel = _load_json(sel_path)
    return sel.get(tree_name, {}).get("thresholds", {})


def _build_tree_context(tree):
    """Return context dict; always has 'thresholds'; has model/region info if enabled."""
    bdt_dir = _bdt_dir_for(tree)
    ctx = {"thresholds": _load_thresholds(bdt_dir, tree), "regions_enabled": False}

    sr_csv = _sr_csv_for(tree)
    if bdt_dir is None or sr_csv is None:
        log(f"  [{tree}] no signal_region_csv/bdt_root configured -> inclusive only")
        return ctx
    if not os.path.exists(sr_csv):
        log(f"  [{tree}] signal_region CSV not found ({sr_csv}) -> inclusive only")
        return ctx

    bcfg_path = os.path.join(bdt_dir, "config.json")
    brj_path  = os.path.join(bdt_dir, "branch.json")
    sel_path  = os.path.join(bdt_dir, "selection.json")
    for p in (bcfg_path, brj_path, sel_path):
        if not os.path.exists(p):
            log(f"  [{tree}] missing {os.path.basename(p)} in bdt_root -> inclusive only")
            return ctx

    bcfg = _load_json(bcfg_path)
    brj  = _load_json(brj_path)
    selj = _load_json(sel_path)
    class_names  = list(bcfg["class_groups"].keys())
    feature_cols = [b["name"] for b in brj[tree]]
    sel          = selj.get(tree, {})
    clip_ranges  = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_transform = sel.get("log_transform", [])
    decorrelate   = bcfg.get(tree, {}).get("decorrelate", [])

    model_pattern = bcfg.get("model_pattern", "{output_root}/{tree_name}_model")
    model_base = model_pattern.format(output_root=bdt_dir, tree_name=tree)
    model = model_io.load_model(model_base, bcfg, len(class_names), log_message=log)

    sr_df = pd.read_csv(sr_csv)
    if sr_df.empty:
        log(f"  [{tree}] signal_region CSV is empty -> inclusive only")
        return ctx
    axes = _detect_signal_region_axes(sr_df, class_names)
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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _threshold_mask(data_np, thresholds):
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


def _infer_proba(chunk, mask, ctx):
    cols = {}
    for name in ctx["feature_cols"]:
        if name not in chunk:
            raise KeyError(
                f"BDT feature branch {name!r} missing from input file; "
                f"cannot evaluate signal regions."
            )
        cols[name] = np.asarray(chunk[name])[mask]
    X = pd.DataFrame(cols)
    X = _standardize_X(X, ctx["clip_ranges"], ctx["log_transform"])
    if ctx["decorrelate"]:
        drop = [c for c in ctx["decorrelate"] if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
    return model_io.predict_model_proba(ctx["model"], X, ctx["num_classes"])


# ---------------------------------------------------------------------------
# Per-sample computation
# ---------------------------------------------------------------------------

def _compute_ratios(sample_name, ctx):
    """Accumulate inclusive (slot 0) and per-region (slots 1..N) PU weight sums.

    Returns {"inclusive": dict_or_None, "regions": {bin_id: dict_or_None}}
    or None if no files found or PU branches missing.
    """
    files = _input_files(sample_name)
    if not files:
        log(f"    no input files for {sample_name}, skipping")
        return None

    thresholds = ctx["thresholds"]
    enabled    = ctx["regions_enabled"]
    n_slots    = 1 + (len(ctx["bin_ids"]) if enabled else 0)

    load_set = {"weight_pu", "weight_pu_up", "weight_pu_down", *thresholds.keys()}
    if enabled:
        load_set.update(ctx["feature_cols"])
    load_list = list(load_set)

    sum_w    = np.zeros(n_slots, dtype=np.float64)
    sum_w_up = np.zeros(n_slots, dtype=np.float64)
    sum_w_dn = np.zeros(n_slots, dtype=np.float64)
    n_events = np.zeros(n_slots, dtype=np.int64)
    found_pu = False

    for fpath in files:
        try:
            with uproot.open(fpath) as uf:
                if ctx["_tree"] not in uf:
                    continue
                tree = uf[ctx["_tree"]]
                avail = set(tree.keys())
                missing = [b for b in ("weight_pu", "weight_pu_up", "weight_pu_down")
                           if b not in avail]
                if missing:
                    log(f"    WARNING: {os.path.basename(fpath)} missing {missing} "
                        f"— re-run mode 1 (weight.C)")
                    continue
                found_pu = True
                actual = [b for b in load_list if b in avail]

                for chunk in tree.iterate(expressions=actual, step_size=CHUNK_SIZE,
                                          library="np"):
                    mask = _threshold_mask(chunk, thresholds)
                    if not mask.any():
                        continue

                    w    = np.asarray(chunk["weight_pu"],      dtype=np.float64)[mask]
                    w_up = np.asarray(chunk["weight_pu_up"],   dtype=np.float64)[mask]
                    w_dn = np.asarray(chunk["weight_pu_down"], dtype=np.float64)[mask]

                    sum_w[0]    += float(w.sum())
                    sum_w_up[0] += float(w_up.sum())
                    sum_w_dn[0] += float(w_dn.sum())
                    n_events[0] += int(mask.sum())

                    if enabled:
                        proba = _infer_proba(chunk, mask, ctx)
                        for j in range(len(ctx["bin_ids"])):
                            rmask = _region_mask(
                                proba, ctx["sr_df"].iloc[j], ctx["axes"],
                                ctx["class_names"]
                            )
                            if rmask.any():
                                sum_w[j + 1]    += float(w[rmask].sum())
                                sum_w_up[j + 1] += float(w_up[rmask].sum())
                                sum_w_dn[j + 1] += float(w_dn[rmask].sum())
                                n_events[j + 1] += int(rmask.sum())
        except Exception as exc:
            log(f"    error reading {fpath}: {exc}")
            continue

    if not found_pu:
        log(f"    [{sample_name}] PU branches missing in all files")
        return None

    def _slot_ratios(slot):
        w = sum_w[slot]
        n = int(n_events[slot])
        if w <= 0.0 or n < 1:
            return None
        return {
            "nom_sum":  float(w),
            "n_events": n,
            "pu_up":    float(sum_w_up[slot] / w),
            "pu_down":  float(sum_w_dn[slot] / w),
        }

    inclusive = _slot_ratios(0)
    regions   = None
    if enabled:
        regions = {bid: _slot_ratios(j + 1) for j, bid in enumerate(ctx["bin_ids"])}

    return {"inclusive": inclusive, "regions": regions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _pct(r):
    return f"{100. * (r - 1.):+.2f}%"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("pileup_syst.py — pileup weight variations (inclusive + per signal region)")
    log(f"  trees:   {SUBMIT_TREES}")
    log(f"  samples: {MC_SAMPLES}")
    log(f"  output:  {OUTPUT_DIR}\n")

    all_results = {}   # (sample, tree) -> entry dict

    for tree_name in SUBMIT_TREES:
        log(f"=== Tree: {tree_name} ===")
        ctx = _build_tree_context(tree_name)
        ctx["_tree"] = tree_name
        if ctx["thresholds"]:
            log(f"  pre-selection cuts: {list(ctx['thresholds'].keys())}")
        else:
            log("  no pre-selection cuts applied")

        for sample_name in MC_SAMPLES:
            log(f"  {sample_name} ...")
            res = _compute_ratios(sample_name, ctx)
            if res is None:
                continue
            incl = res["inclusive"]
            if incl is None:
                log(f"    [{sample_name}/{tree_name}] nominal weight sum is zero, skipping")
                continue

            entry = dict(incl)   # inclusive ratios at top level (back-compatible)
            log(f"    inclusive: n={incl['n_events']}"
                f"  pu_up={_pct(incl['pu_up'])}  pu_down={_pct(incl['pu_down'])}")

            if res["regions"] is not None:
                region_out = {}
                for bid, rr in res["regions"].items():
                    if rr is None or rr["n_events"] < MIN_REGION_EVENTS:
                        n_have = 0 if rr is None else rr["n_events"]
                        log(f"    SR{bid}: only {n_have} events "
                            f"(< {MIN_REGION_EVENTS}); falling back to inclusive ratio")
                        region_out[str(bid)] = {**incl, "fallback_to_inclusive": True}
                    else:
                        region_out[str(bid)] = rr
                        log(f"    SR{bid}: n={rr['n_events']}"
                            f"  pu_up={_pct(rr['pu_up'])}  pu_down={_pct(rr['pu_down'])}")
                entry["regions"] = region_out

            all_results[(sample_name, tree_name)] = entry

    # Write JSON
    json_path = os.path.join(OUTPUT_DIR, "pileup_syst_yields.json")
    nested = {}
    for (sample, tree), entry in all_results.items():
        nested.setdefault(sample, {})[tree] = entry
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2)
    log(f"\nWrote {json_path}")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "pileup_syst_yields.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "tree", "region", "pu_up", "pu_down", "n_events"])
        for (sample, tree) in sorted(all_results.keys()):
            entry = all_results[(sample, tree)]
            writer.writerow([sample, tree, "inclusive",
                             f"{entry['pu_up']:.6f}", f"{entry['pu_down']:.6f}",
                             entry["n_events"]])
            for bid, rr in entry.get("regions", {}).items():
                writer.writerow([sample, tree, bid,
                                 f"{rr['pu_up']:.6f}", f"{rr['pu_down']:.6f}",
                                 rr["n_events"]])
    log(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
