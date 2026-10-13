#!/usr/bin/env python3
"""Write the BDT and SR-scan configs of the boosted3 overtraining / event-weight study.

Every variant starts from the production configs (selections/BDT/config.json and
selections/signal_region/config_hist_boosted3.json) and changes only the keys listed in
VARIANTS (per-tree keys) and GLOBAL_CHANGES (top-level keys). `python3 make_study.py [variant ...]`
writes only the named variants (default: all), so the configs of finished variants trained from an
older production config are not rewritten. Outputs go to study_boosted3/<variant>/ (model, test_ranges.json, references),
<variant>/sr_hist/ (SRs optimised on the eval split) and <variant>/train.log.
"""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BDT_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BDT_DIR))
TREE = "boosted3"
LOW_HT_QCD = ["qcd_ht40to70", "qcd_ht70to100", "qcd_ht100to200", "qcd_ht200to400"]

VARIANTS = {
    "baseline": {},
    "mcw200": {"min_child_weight": 200},
    "shallow": {"max_depth": 4, "min_child_weight": 50},
    "noLowHT": {"training_sample_weight_scales": {s: 0 for s in LOW_HT_QCD}},
    "cap20": {"training_weight_cap": 20},
    "combo": {"min_child_weight": 200, "training_weight_cap": 20,
              "training_sample_weight_scales": {s: 0 for s in LOW_HT_QCD}},
    "balance05": {"class_balance_power": 0.5},
    # 2026-10-06, on the production settings of that day (depth 4, mcw 50, per-sample eval/test caps):
    "shallow_cap20": {"training_weight_cap": 20},
    "shallow_cap20_effnorm": {"training_weight_cap": 20},
    # 2026-10-07: shallow_cap20 as 5 k-fold rotations (every MC event with an out-of-fold score)
    "shallow_cap20_kfold": {"training_weight_cap": 20},
    # 2026-10-08: shallow_cap20_kfold with a finer and deeper decorrelation (the per-SR ABCD did not
    # close: the 5 msoftdrop bins of 40 GeV averaged 0-10 and 20-40 GeV, the thresholds stopped at
    # p = 0.002, and stage 2 was still improving after its 1000 rounds)
    "kfold_finedecor": {"training_weight_cap": 20, "learning_rate_decorr": 0.05, "n_estimators_decorr": 1500},
    # 2026-10-09: the kfold_finedecor settings for the other four categories, with the production
    # training budget (Top only; QCD and V+jets by the per-sample cap)
    "kfold_finedecor_resolved": {"training_weight_cap": 20, "learning_rate_decorr": 0.05, "n_estimators_decorr": 1500},
    "kfold_finedecor_unresolved": {"training_weight_cap": 20, "learning_rate_decorr": 0.05, "n_estimators_decorr": 1500},
    "kfold_finedecor_1lep": {"training_weight_cap": 20, "learning_rate_decorr": 0.05, "n_estimators_decorr": 1500},
    "kfold_finedecor_2lep": {"training_weight_cap": 20, "learning_rate_decorr": 0.05, "n_estimators_decorr": 1500},
}
# Tree of a variant (default boosted3); its configs start from the production config of that tree.
VARIANT_TREES = {
    "kfold_finedecor_resolved": "boosted2_resolved",
    "kfold_finedecor_unresolved": "boosted2_unresolved",
    "kfold_finedecor_1lep": "boosted2_leptonic_1lep",
    "kfold_finedecor_2lep": "boosted2_leptonic_2lep",
}
FINEDECOR = {"kfold": 5, "decor_n_bins": 10, "decor_msoftdrop_training_max": 250, "decor_tail_min_prob": 2e-4,
             "decor_tail_n": 10, "decor_eff_normalize": True, "decor_eff_norm_floor": 1e-4}
GLOBAL_CHANGES = {
    "shallow_cap20_effnorm": {"decor_eff_normalize": True},
    "shallow_cap20_kfold": {"kfold": 5},
    "kfold_finedecor": {"kfold": 5, "train_events_per_class": 3000000, "decor_n_bins": 10,
                        "decor_msoftdrop_training_max": 250, "decor_tail_min_prob": 2e-4, "decor_tail_n": 10,
                        "decor_eff_normalize": True, "decor_eff_norm_floor": 1e-4},
}
GLOBAL_CHANGES.update({name: FINEDECOR for name in VARIANT_TREES})
# Samples left out of a variant's class_groups: boosted2_leptonic_2lep holds only 2-3 entries of each
# HT < 400 GeV QCD sample, fewer than the 5 k-fold blocks need (train.py stops on an empty split);
# the per-SR ABCD studies leave these samples out anyway (exclude_samples). WZto2Nu2Q (5 entries) and
# WZto4Q (42 entries) have a block with a non-positive signed weight sum there (train.py stops on it)
# and give no real lepton pair; every other sample has positive sums in every block and training union.
DROP_SAMPLES = {"kfold_finedecor_2lep": LOW_HT_QCD + ["WZto2Nu2Q", "WZto4Q"]}

def main():
    base = json.load(open(os.path.join(BDT_DIR, "config.json")))
    names = sys.argv[1:] or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    for name in names:
        changes = VARIANTS[name]
        tree = VARIANT_TREES.get(name, TREE)
        sr_base = json.load(open(os.path.join(REPO, "selections", "signal_region", f"config_hist_{tree}.json")))
        out = os.path.join(HERE, name)
        os.makedirs(out, exist_ok=True)
        cfg = copy.deepcopy(base)
        cfg["submit_trees"] = [tree]
        cfg["output_root"] = os.path.join(out, "{tree_name}")
        cfg[tree].update(copy.deepcopy(changes))
        cfg.update(copy.deepcopy(GLOBAL_CHANGES.get(name, {})))
        drop = set(DROP_SAMPLES.get(name, []))
        cfg["class_groups"] = {c: [s for s in m if s not in drop] for c, m in cfg["class_groups"].items()}
        json.dump(cfg, open(os.path.join(out, "bdt_config.json"), "w"), indent=2)
        sr = copy.deepcopy(sr_base)
        sr.update({"bdt_root": os.path.join(out, tree), "output_dir": os.path.join(out, "sr_hist"),
                   "max_threads": 8, "validate_prediction_reference": True})
        json.dump(sr, open(os.path.join(out, "sr_config.json"), "w"), indent=2)
        print(f"{name:10s} {changes} {GLOBAL_CHANGES.get(name, {})}")

if __name__ == "__main__":
    main()
