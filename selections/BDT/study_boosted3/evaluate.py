#!/usr/bin/env python3
"""Score one variant of the boosted3 study, or print the table of all scored variants.

    python3 evaluate.py <variant>     # writes <variant>/summary.json
    python3 evaluate.py --table       # table of every <variant>/summary.json

Every variant is scored the same way, independent of its training weights:
  * test AUC per signal-vs-background class pair, score p_s / (p_s + p_b), weighted with the
    absolute physics weights of test_reference_signal_region.npz (full thresholds);
  * train/test AUC gap as logged by train.py (stage-2 model; weighted with that variant's
    training weights, so only indicative across weight variants);
  * the SRs that signal_region_hist.py chose on the eval split, evaluated on the eval and on the
    untouched test split from the prediction references: S, B (MC-true), sigma_B^2 = sum(w^2)
    and the sigma_B-aware Asimov Z (same formula as signal_region.asimov_z), Z_comb = sqrt(sum Z^2).
"""
import glob, json, math, os, re, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BDT_DIR = os.path.dirname(HERE)
TREE = "boosted3"


def asimov_z(s, b, v):
    """signal_region.asimov_z for scalars."""
    if s <= 0.0 or b <= 0.0:
        return 0.0
    v = max(v, 0.0)
    if v <= 1e-12 * b * b:
        f = (s + b) * math.log1p(s / b) - s
    else:
        f = ((s + b) * math.log((s + b) * (b + v) / (b * b + (s + b) * v))
             - (b * b / v) * math.log1p(v * s / (b * (b + v))))
    return math.sqrt(2.0 * max(f, 0.0))


def region_mask(proba, row, axes, class_names):
    """qcd_est._region_mask: low <= p < high on every axis, open-ended when high == 1."""
    mask = np.ones(len(proba), dtype=bool)
    for axis in axes:
        lo, hi = float(row[f"{axis}_low"]), float(row[f"{axis}_high"])
        p = proba[:, class_names.index(axis)]
        mask &= (p >= lo) & (p < hi) if hi < 1.0 - 1e-12 else (p >= lo)
    return mask


def score(variant):
    d = os.path.join(HERE, variant)
    root = os.path.join(d, TREE)
    cfg = json.load(open(os.path.join(root, "config.json")))
    sample_cfg_path = cfg["sample_config"]
    if not os.path.isabs(sample_cfg_path):
        sample_cfg_path = os.path.normpath(os.path.join(BDT_DIR, sample_cfg_path))
    is_signal = {e["name"]: e["is_signal"] for e in json.load(open(sample_cfg_path))["sample"]}
    class_names = list(cfg["class_groups"])
    signal = [i for i, c in enumerate(class_names) if all(is_signal[s] for s in cfg["class_groups"][c])]
    background = [i for i in range(len(class_names)) if i not in signal]
    lumi = float(json.load(open(os.path.join(d, "sr_config.json")))["lumi"])

    refs = {split: np.load(os.path.join(root, f"{split}_reference_signal_region.npz")) for split in ("eval", "test")}
    out = {"variant": variant, "test_auc": {}, "logged_auc": {}, "regions": []}

    # Test AUCs with physics weights.
    t = refs["test"]
    proba, cls, w = t["proba"], t["class_idx"], np.abs(t["weight"])
    for si in signal:
        for bi in background:
            m = (cls == si) | (cls == bi)
            sc = proba[m, si] / np.clip(proba[m, si] + proba[m, bi], 1e-12, None)
            out["test_auc"][f"{class_names[si]} vs {class_names[bi]}"] = float(
                roc_auc_score((cls[m] == si).astype(int), sc, sample_weight=w[m]))

    # Train/test AUCs logged by train.py for the final (stage-2) model.
    log = open(os.path.join(d, "train.log"), errors="replace").read()
    for split, pair, auc in re.findall(rf"{TREE} \[stage2\] (train|test) AUC \((\w+ vs \w+)\) = ([0-9.]+)", log):
        out["logged_auc"].setdefault(pair, {})[split] = float(auc)

    # SRs chosen on the eval split, evaluated on eval and test.
    srs = pd.read_csv(os.path.join(d, "sr_hist", "signal_region.csv"))
    axes = [c[:-4] for c in srs.columns if c.endswith("_low")]
    z2 = {"eval": 0.0, "test": 0.0}
    for _, row in srs.iterrows():
        entry = {"bin_index": int(row["bin_index"]), "csv_eval_Z": float(row["significance"])}
        for split, r in refs.items():
            m = region_mask(r["proba"], row, axes, class_names)
            ww = r["weight"][m] * lumi
            c = r["class_idx"][m]
            sig = np.isin(c, signal)
            S, B, V = float(ww[sig].sum()), float(ww[~sig].sum()), float((ww[~sig] ** 2).sum())
            Z = asimov_z(S, B, V)
            z2[split] += Z * Z
            entry[split] = {"S": S, "B": B, "sigma_B": math.sqrt(V), "n_bkg_events": int((~sig).sum()), "Z": Z}
        out["regions"].append(entry)
    out["Z_comb"] = {split: math.sqrt(v) for split, v in z2.items()}
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    return out


def table():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "*", "summary.json"))):
        s = json.load(open(path))
        gaps = [v["train"] - v["test"] for v in s["logged_auc"].values() if "train" in v and "test" in v]
        rows.append({
            "variant": s["variant"],
            "Z_comb eval": round(s["Z_comb"]["eval"], 3),
            "Z_comb test": round(s["Z_comb"]["test"], 3),
            "test AUC VVV vs VV": round(s["test_auc"].get("VVV vs VV", float("nan")), 4),
            "test AUC VH vs VV": round(s["test_auc"].get("VH vs VV", float("nan")), 4),
            "test AUC VVV vs QCD": round(s["test_auc"].get("VVV vs QCD", float("nan")), 4),
            "mean test AUC": round(float(np.mean(list(s["test_auc"].values()))), 4),
            "max train-test AUC gap": round(max(gaps), 4) if gaps else float("nan"),
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("no scored variants yet")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--table":
        table()
    else:
        s = score(sys.argv[1])
        print(json.dumps({"variant": s["variant"], "Z_comb": s["Z_comb"],
                          "test_auc": {k: round(v, 4) for k, v in s["test_auc"].items()}}, indent=1))
