#!/usr/bin/env python3
"""jer_syst.py

Computes AK4/AK8 jet-energy-resolution (JER) shape-systematic yield ratios for
all MC samples, per tree and per signal region.

Unlike pileup_syst.py/theory_syst.py (an alternate weight column in the SAME
converted tree), a JER shift changes jet kinematics -- and therefore every
pt/mass-derived BDT feature -- so "up"/"down" here means three INDEPENDENT
converted-ntuple sets for the same sample: the nominal (scoutingPUPPI
correction only) and two JER-shifted variants (jet_pt_correction.variation
= "jer_up"/"jer_down" in convert_branch.C, written to their own dataset
roots -- see selections/convert/config_jer_{up,down}.json). Each is read
independently (same event-weight definition, weight_branch, typically
"weight_pu") and bucketed into the same BDT signal regions as qcd_est.py
(via selections/jet_syst_common/jet_variation_common.py, which mirrors
pileup_syst.py's context/inference/region-masking code).

  jer_up   = sum(weight_branch on jer_up tree)   / sum(weight_branch on nominal tree)
  jer_down = sum(weight_branch on jer_down tree) / sum(weight_branch on nominal tree)

Outputs (written to output_dir):
  jer_syst_yields.json  -- {sample: {tree: {n_events, jer_up, jer_down,
                                            regions: {bin_id: {...}}}}}
  jer_syst_yields.csv   -- flat CSV: sample,tree,region,jer_up,jer_down,n_events
"""

import csv
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402

log = jvc.log

_cfg_path = os.environ.get("JER_SYST_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
if not os.path.isabs(_cfg_path):
    _cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _cfg_path))
cfg = jvc.load_json(_cfg_path)

SUBMIT_TREES = cfg.get("submit_trees", [])
INPUT_ROOT = jvc.resolve_path(_SCRIPT_DIR, cfg["input_root"])
INPUT_ROOT_UP = jvc.resolve_path(_SCRIPT_DIR, cfg["input_root_up"])
INPUT_ROOT_DOWN = jvc.resolve_path(_SCRIPT_DIR, cfg["input_root_down"])
INPUT_PATTERN = cfg["input_pattern"]
OUTPUT_DIR = jvc.resolve_path(_SCRIPT_DIR, cfg.get("output_dir", "."))
CHUNK_SIZE = cfg.get("chunk_size", "200 MB")
BDT_ROOT_CFG = cfg.get("bdt_root", None)
SR_CSV_CFG = cfg.get("signal_region_csv", None)
WEIGHT_BRANCH = cfg.get("weight_branch", "weight_pu")
MIN_REGION_EVENTS = int(cfg.get("min_region_events", 1))

_sample_cfg_path = jvc.resolve_path(_SCRIPT_DIR, cfg["sample_config"])
sample_cfg = jvc.load_json(_sample_cfg_path)
SAMPLE_INFO = {s["name"]: s for s in sample_cfg["sample"]}

MC_SAMPLES = [s["name"] for s in sample_cfg["sample"] if s.get("is_MC", False)]
_submit_samples = cfg.get("submit_samples") or []
if _submit_samples:
    _submit_set = set(_submit_samples)
    MC_SAMPLES = [s for s in MC_SAMPLES if s in _submit_set]


def _pct(r):
    return f"{100. * (r - 1.):+.2f}%"


def _sample_ratios(sample_name, tree_name, ctx):
    nominal_files = jvc.input_files(INPUT_ROOT, INPUT_PATTERN, SAMPLE_INFO, sample_name)
    up_files = jvc.input_files(INPUT_ROOT_UP, INPUT_PATTERN, SAMPLE_INFO, sample_name)
    down_files = jvc.input_files(INPUT_ROOT_DOWN, INPUT_PATTERN, SAMPLE_INFO, sample_name)
    if not nominal_files or not up_files or not down_files:
        log(f"    missing nominal/up/down files for {sample_name}, skipping")
        return None

    nominal = jvc.accumulate_weighted_sr_yields(nominal_files, tree_name, ctx, WEIGHT_BRANCH, CHUNK_SIZE)
    up = jvc.accumulate_weighted_sr_yields(up_files, tree_name, ctx, WEIGHT_BRANCH, CHUNK_SIZE)
    down = jvc.accumulate_weighted_sr_yields(down_files, tree_name, ctx, WEIGHT_BRANCH, CHUNK_SIZE)
    if nominal is None or up is None or down is None:
        log(f"    [{sample_name}/{tree_name}] weight branch missing in nominal/up/down, skipping")
        return None
    return {"nominal": nominal, "up": up, "down": down}


def _ratio_entry(nominal_slot, up_slot, down_slot):
    if nominal_slot is None or nominal_slot["sum"] <= 0.0:
        return None
    return {
        "nom_sum": nominal_slot["sum"],
        "n_events": nominal_slot["n_events"],
        "jer_up": (up_slot["sum"] / nominal_slot["sum"]) if up_slot is not None else 1.0,
        "jer_down": (down_slot["sum"] / nominal_slot["sum"]) if down_slot is not None else 1.0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("jer_syst.py — JER shape-systematic yield ratios (inclusive + per signal region)")
    log(f"  trees:   {SUBMIT_TREES}")
    log(f"  samples: {MC_SAMPLES}")
    log(f"  output:  {OUTPUT_DIR}\n")

    all_results = {}

    for tree_name in SUBMIT_TREES:
        log(f"=== Tree: {tree_name} ===")
        ctx = jvc.build_tree_context(_SCRIPT_DIR, BDT_ROOT_CFG, SR_CSV_CFG, tree_name)
        if ctx["thresholds"]:
            log(f"  pre-selection cuts: {list(ctx['thresholds'].keys())}")
        else:
            log("  no pre-selection cuts applied")

        for sample_name in MC_SAMPLES:
            log(f"  {sample_name} ...")
            res = _sample_ratios(sample_name, tree_name, ctx)
            if res is None:
                continue

            incl = _ratio_entry(res["nominal"]["inclusive"], res["up"]["inclusive"], res["down"]["inclusive"])
            if incl is None:
                log(f"    [{sample_name}/{tree_name}] nominal weight sum is zero, skipping")
                continue
            log(f"    inclusive: n={incl['n_events']}"
                f"  jer_up={_pct(incl['jer_up'])}  jer_down={_pct(incl['jer_down'])}")

            entry = dict(incl)
            if ctx["regions_enabled"]:
                region_out = {}
                nominal_regions = res["nominal"]["regions"] or {}
                up_regions = res["up"]["regions"] or {}
                down_regions = res["down"]["regions"] or {}
                for bid in ctx["bin_ids"]:
                    rr = _ratio_entry(nominal_regions.get(bid), up_regions.get(bid), down_regions.get(bid))
                    if rr is None or rr["n_events"] < MIN_REGION_EVENTS:
                        n_have = 0 if rr is None else rr["n_events"]
                        log(f"    SR{bid}: only {n_have} events "
                            f"(< {MIN_REGION_EVENTS}); falling back to inclusive ratio")
                        region_out[str(bid)] = {**incl, "fallback_to_inclusive": True}
                    else:
                        region_out[str(bid)] = rr
                        log(f"    SR{bid}: n={rr['n_events']}"
                            f"  jer_up={_pct(rr['jer_up'])}  jer_down={_pct(rr['jer_down'])}")
                entry["regions"] = region_out

            all_results[(sample_name, tree_name)] = entry

    json_path = os.path.join(OUTPUT_DIR, "jer_syst_yields.json")
    nested = {}
    for (sample, tree), entry in all_results.items():
        nested.setdefault(sample, {})[tree] = entry
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2)
    log(f"\nWrote {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, "jer_syst_yields.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "tree", "region", "jer_up", "jer_down", "n_events"])
        for (sample, tree) in sorted(all_results.keys()):
            entry = all_results[(sample, tree)]
            writer.writerow([sample, tree, "inclusive",
                             f"{entry['jer_up']:.6f}", f"{entry['jer_down']:.6f}", entry["n_events"]])
            for bid, rr in entry.get("regions", {}).items():
                writer.writerow([sample, tree, bid,
                                 f"{rr['jer_up']:.6f}", f"{rr['jer_down']:.6f}", rr["n_events"]])
    log(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
