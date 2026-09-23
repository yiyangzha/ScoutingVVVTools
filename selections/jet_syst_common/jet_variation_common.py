"""Shared engine of the jet-variation systematics jes_syst.py, jer_syst.py,
jms_syst.py, and jmr_syst.py.

convert_branch.C writes, into the same files as the nominal trees <tree> of
every MC sample, one tree <tree>__<variation> per jet_pt_correction variation
(jes_up, jes_down, jer_up, jer_down, jms_up, jms_down, jmr_up, jmr_down; see
selections/convert/config.json). A jet variation changes the jet kinematics,
so it changes which events pass and every jet-derived BDT feature: each ratio
compares the weighted yield of a variation tree with that of the nominal tree
in the same region, both evaluated with the nominal BDT model and
signal-region definition.

  <syst>_up   = sum(weight_branch on <tree>__<syst>_up)   / sum(weight_branch on <tree>)
  <syst>_down = sum(weight_branch on <tree>__<syst>_down) / sum(weight_branch on <tree>)

A variation without events in a region has ratio 0. Regions: the thresholds of
the BDT selection.json (the ABCD window included) are applied with
qcd_est.py's semantics (sentinel values fail), so each signal region is the
ABCD A region whose yields combine uses, and the inclusive entry is the
post-threshold selection. A region with fewer than min_region_events nominal
MC events takes the inclusive ratios (flagged fallback_to_inclusive).

Outputs (written to output_dir):
  <syst>_syst_yields.json -- {sample: {tree: {nom_sum, n_events, <syst>_up,
                              <syst>_down, regions: {bin_id: {...}}}}}
  <syst>_syst_yields.csv  -- flat CSV: sample,tree,region,<syst>_up,<syst>_down,n_events
"""

import csv
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

# Keys of the removed one-dataset-per-variation scheme.
REMOVED_CONFIG_KEYS = ("input_root_up", "input_root_down")


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


def mask_from_cond(values, cond):
    """qcd_est.py's _mask_from_cond on a numpy array: a number is a lower cut,
    [low, high] a window (None = open), a list of conditions their OR, and a
    {"&"/"and"/"|"/"or": [...]} dict their AND/OR."""
    if cond is None:
        return np.ones(values.shape[0], dtype=bool)
    if isinstance(cond, (int, float)):
        return values > float(cond)
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
        low, high = cond
        mask = np.ones(values.shape[0], dtype=bool)
        if low is not None:
            mask &= values > low
        if high is not None:
            mask &= values < high
        return mask
    if isinstance(cond, (list, tuple)):
        mask = np.zeros(values.shape[0], dtype=bool)
        for item in cond:
            mask |= mask_from_cond(values, item)
        return mask
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key not in cond:
                continue
            mask = np.full(values.shape[0], is_and, dtype=bool)
            for item in cond[key]:
                sub = mask_from_cond(values, item)
                mask = (mask & sub) if is_and else (mask | sub)
            return mask
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported condition type: {type(cond)}")


def threshold_mask(data_np, thresholds):
    """Events passing every threshold, with qcd_est.py's semantics: a sentinel
    value (< -990) fails, otherwise the condition decides."""
    n = len(next(iter(data_np.values())))
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        if branch not in data_np:
            raise KeyError(f"Threshold branch {branch!r} was not loaded from the input tree")
        values = np.asarray(data_np[branch], dtype=float)
        mask &= ~(values < -990)
        mask &= mask_from_cond(values, cond)
    return mask


def load_thresholds(bdt_dir, tree_name):
    if bdt_dir is None:
        return {}
    sel_path = os.path.join(bdt_dir, "selection.json")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"bdt_root has no selection.json: {sel_path}")
    sel = load_json(sel_path)
    return sel.get(tree_name, {}).get("thresholds", {})


def build_tree_context(script_dir, bdt_root_cfg, sr_csv_cfg, tree):
    """Return the context of one tree: always 'thresholds'; the model and the
    signal regions when bdt_root and signal_region_csv are configured."""
    bdt_dir = resolve_path(script_dir, per_tree(bdt_root_cfg, tree))
    ctx = {"thresholds": load_thresholds(bdt_dir, tree), "regions_enabled": False}

    sr_csv_raw = per_tree(sr_csv_cfg, tree)
    if bdt_dir is None or sr_csv_raw is None:
        log(f"  [{tree}] no signal_region_csv/bdt_root configured -> inclusive only")
        return ctx
    sr_csv = resolve_path(script_dir, sr_csv_raw.format(tree_name=tree))
    bcfg_path = os.path.join(bdt_dir, "config.json")
    brj_path = os.path.join(bdt_dir, "branch.json")
    for p in (sr_csv, bcfg_path, brj_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{tree}] configured input not found: {p}")

    bcfg = load_json(bcfg_path)
    brj = load_json(brj_path)
    selj = load_json(os.path.join(bdt_dir, "selection.json"))
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
        raise ValueError(f"[{tree}] signal_region CSV is empty: {sr_csv}")
    axes = detect_signal_region_axes(sr_df, class_names)
    if not axes:
        raise ValueError(f"[{tree}] no per-class axes in signal_region CSV: {sr_csv}")
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
# Weighted SR yield accumulation of one tree
# ---------------------------------------------------------------------------

def accumulate_weighted_sr_yields(files, tree_name, ctx, weight_branch, chunk_size):
    """Sum `weight_branch` inclusive (slot 0) and per SR (slots 1..N) over one
    tree (nominal or variation) of the converted files. Returns
    {"inclusive": slot, "regions": {bin_id: slot} or None} with
    slot = {"sum": float, "n_events": int}."""
    thresholds = ctx["thresholds"]
    enabled = ctx["regions_enabled"]
    n_slots = 1 + (len(ctx["bin_ids"]) if enabled else 0)

    load_set = {weight_branch, *thresholds.keys()}
    if enabled:
        load_set.update(ctx["feature_cols"])
    load_list = sorted(load_set)

    sum_w = np.zeros(n_slots, dtype=np.float64)
    n_events = np.zeros(n_slots, dtype=np.int64)

    for fpath in files:
        with uproot.open(fpath) as uf:
            if tree_name not in uf:
                raise KeyError(
                    f"{fpath} has no tree {tree_name!r}; re-run mode 0 (convert) with the "
                    f"jet_pt_correction variations of selections/convert/config.json"
                )
            tree = uf[tree_name]
            missing = [b for b in load_list if b not in set(tree.keys())]
            if missing:
                raise KeyError(f"{fpath}:{tree_name} lacks the branches {missing}")
            for chunk in tree.iterate(expressions=load_list, step_size=chunk_size, library="np"):
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

    def slot(idx):
        return {"sum": float(sum_w[idx]), "n_events": int(n_events[idx])}

    regions = None
    if enabled:
        regions = {bid: slot(j + 1) for j, bid in enumerate(ctx["bin_ids"])}
    return {"inclusive": slot(0), "regions": regions}


def ratio_entry(nominal_slot, up_slot, down_slot, up_key, down_key):
    """Up/down ratios of one region with a positive nominal yield."""
    return {
        "nom_sum": nominal_slot["sum"],
        "n_events": nominal_slot["n_events"],
        up_key: up_slot["sum"] / nominal_slot["sum"],
        down_key: down_slot["sum"] / nominal_slot["sum"],
    }


# ---------------------------------------------------------------------------
# Main flow of <syst>_syst.py and of its merge_shards.py
# ---------------------------------------------------------------------------

def _pct(r):
    return f"{100. * (r - 1.):+.2f}%"


def _write_outputs(syst, output_dir, nested):
    up_key, down_key = f"{syst}_up", f"{syst}_down"
    json_path = os.path.join(output_dir, f"{syst}_syst_yields.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2)
    log(f"Wrote {json_path}")

    csv_path = os.path.join(output_dir, f"{syst}_syst_yields.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "tree", "region", up_key, down_key, "n_events"])
        for sample in sorted(nested):
            for tree in sorted(nested[sample]):
                entry = nested[sample][tree]
                writer.writerow([sample, tree, "inclusive",
                                 f"{entry[up_key]:.6f}", f"{entry[down_key]:.6f}", entry["n_events"]])
                for bid, rr in entry.get("regions", {}).items():
                    writer.writerow([sample, tree, bid,
                                     f"{rr[up_key]:.6f}", f"{rr[down_key]:.6f}", rr["n_events"]])
    log(f"Wrote {csv_path}")


def run(syst, script_dir, config_env):
    """Compute and write the <syst>_up/<syst>_down ratios of every configured
    MC sample and tree; config from $<config_env> or script_dir/config.json."""
    cfg_path = os.environ.get(config_env, os.path.join(script_dir, "config.json"))
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.normpath(os.path.join(script_dir, cfg_path))
    cfg = load_json(cfg_path)
    for key in REMOVED_CONFIG_KEYS:
        if key in cfg:
            raise KeyError(f"{cfg_path}: {key} was removed; the {syst} variation trees are in the "
                           f"nominal converted files under input_root")

    submit_trees = cfg["submit_trees"]
    input_root = resolve_path(script_dir, cfg["input_root"])
    input_pattern = cfg["input_pattern"]
    output_dir = resolve_path(script_dir, cfg.get("output_dir", "."))
    chunk_size = cfg.get("chunk_size", "200 MB")
    bdt_root_cfg = cfg.get("bdt_root", None)
    sr_csv_cfg = cfg.get("signal_region_csv", None)
    weight_branch = cfg.get("weight_branch", "weight_pu")
    min_region_events = int(cfg["min_region_events"])

    sample_cfg = load_json(resolve_path(script_dir, cfg["sample_config"]))
    sample_info = {s["name"]: s for s in sample_cfg["sample"]}
    mc_samples = [s["name"] for s in sample_cfg["sample"] if s.get("is_MC", False)]
    submit_samples = cfg.get("submit_samples") or []
    if submit_samples:
        submit_set = set(submit_samples)
        mc_samples = [s for s in mc_samples if s in submit_set]

    up_key, down_key = f"{syst}_up", f"{syst}_down"
    os.makedirs(output_dir, exist_ok=True)
    log(f"{syst}_syst.py — {syst.upper()} yield ratios from the <tree>__{up_key}/__{down_key} trees "
        f"(inclusive + per signal region)")
    log(f"  trees:   {submit_trees}")
    log(f"  samples: {mc_samples}")
    log(f"  output:  {output_dir}\n")

    nested = {}
    for tree_name in submit_trees:
        log(f"=== Tree: {tree_name} ===")
        ctx = build_tree_context(script_dir, bdt_root_cfg, sr_csv_cfg, tree_name)
        if ctx["thresholds"]:
            log(f"  pre-selection cuts: {list(ctx['thresholds'].keys())}")
        else:
            log("  no pre-selection cuts applied")

        for sample_name in mc_samples:
            log(f"  {sample_name} ...")
            files = input_files(input_root, input_pattern, sample_info, sample_name)
            if not files:
                raise FileNotFoundError(
                    f"No converted files for MC sample {sample_name} under {input_root} "
                    f"(pattern {input_pattern}); convert it or leave it out of submit_samples"
                )
            nominal = accumulate_weighted_sr_yields(files, tree_name, ctx, weight_branch, chunk_size)
            up = accumulate_weighted_sr_yields(files, f"{tree_name}__{up_key}", ctx, weight_branch,
                                               chunk_size)
            down = accumulate_weighted_sr_yields(files, f"{tree_name}__{down_key}", ctx, weight_branch,
                                                 chunk_size)

            if nominal["inclusive"]["n_events"] < 1 or nominal["inclusive"]["sum"] <= 0.0:
                log(f"    [{sample_name}/{tree_name}] no nominal events after the thresholds -> no entry")
                continue
            incl = ratio_entry(nominal["inclusive"], up["inclusive"], down["inclusive"], up_key, down_key)
            log(f"    inclusive: n={incl['n_events']}"
                f"  {up_key}={_pct(incl[up_key])}  {down_key}={_pct(incl[down_key])}")

            entry = dict(incl)
            if ctx["regions_enabled"]:
                region_out = {}
                for bid in ctx["bin_ids"]:
                    nom = nominal["regions"][bid]
                    if nom["n_events"] < min_region_events or nom["sum"] <= 0.0:
                        log(f"    SR{bid}: only {nom['n_events']} events "
                            f"(< {min_region_events}); taking the inclusive ratios")
                        region_out[str(bid)] = {
                            "nom_sum": nom["sum"], "n_events": nom["n_events"],
                            up_key: incl[up_key], down_key: incl[down_key],
                            "fallback_to_inclusive": True,
                        }
                        continue
                    rr = ratio_entry(nom, up["regions"][bid], down["regions"][bid], up_key, down_key)
                    region_out[str(bid)] = rr
                    log(f"    SR{bid}: n={rr['n_events']}"
                        f"  {up_key}={_pct(rr[up_key])}  {down_key}={_pct(rr[down_key])}")
                entry["regions"] = region_out

            nested.setdefault(sample_name, {})[tree_name] = entry

    log("")
    _write_outputs(syst, output_dir, nested)


def merge_shards(syst, script_dir):
    """Merge per-sample <syst>_syst.py shard outputs (output/shards/<sample>/,
    samples listed in shards/manifest.txt) into output/<syst>_syst_yields.json
    and .csv, matching run()'s own output (a plain union: each shard holds only
    its sample). Returns 1 when a shard is missing."""
    shards_dir = os.path.join(script_dir, "output", "shards")
    out_dir = os.path.join(script_dir, "output")
    with open(os.path.join(script_dir, "shards", "manifest.txt"), encoding="utf-8") as fh:
        manifest = [line.strip() for line in fh if line.strip()]

    nested = {}
    missing = []
    for sample in manifest:
        path = os.path.join(shards_dir, sample, f"{syst}_syst_yields.json")
        if not os.path.exists(path):
            missing.append(sample)
            continue
        for s, trees in load_json(path).items():
            nested.setdefault(s, {}).update(trees)

    if missing:
        print(f"WARNING: {len(missing)} shard(s) missing (job failed): {missing}")
    _write_outputs(syst, out_dir, nested)
    print(f"Merged {len(nested)} samples")
    return 1 if missing else 0
