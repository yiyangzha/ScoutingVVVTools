#!/usr/bin/env python3
"""Where does the per-SR ABCD non-closure of a k-fold variant come from? (MC only)

    python3 abcd_closure.py build <variant>                 # QCD event view from the k-fold score cache
    python3 abcd_closure.py analyze <variant> <signal_region.csv> [control_low,control_high]

build: the QCD events of <variant>/boosted3/kfold_scores passing every threshold except the ABCD
branch msoftdrop_1 (qcd_est.py's view), with lumi-scaled weights, class probabilities and the mass /
tagger branches of the cache, written to <variant>/closure/qcd_view.npz.

analyze: per-SR ABCD of qcd_est.py (abcd_mode per_sr, complement sideband: A_i = window in box,
C_i = control window 0 < msoftdrop_1 < 50 in box, B / D = window / control window outside the SR
union) and its breakdown by QCD sample, fold, msoftdrop_1 shape and score axis.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BDT_DIR = os.path.dirname(HERE)
sys.path.insert(0, BDT_DIR)
import kfold  # noqa: E402

LUMI = 255.8
MSD1 = "ScoutingFatPFJetRecluster_msoftdrop_1"
WINDOW = (50.0, 150.0)
CONTROL = (0.0, 50.0)
EXTRA = ["ScoutingFatPFJetRecluster_msoftdrop_2", "ScoutingFatPFJetRecluster_msoftdrop_3",
         "ScoutingFatPFJetRecluster_mass_fraction_1", "ScoutingFatPFJetRecluster_sphere_mass",
         "ScoutingFatPFJetRecluster_pt_1", "ScoutingFatPFJetRecluster_pt_2", "ScoutingFatPFJetRecluster_pt_3",
         "ScoutingFatPFJetRecluster_WvsQCD_1", "ScoutingFatPFJetRecluster_WvsQCD_2",
         "ScoutingFatPFJetRecluster_WvsQCD_3", "ScoutingFatPFJetRecluster_VvsQCD_1",
         "ScoutingFatPFJetRecluster_VvsQCD_2", "ScoutingFatPFJetRecluster_VvsQCD_3",
         "ScoutingPFJetRecluster2_scoutUParT_probb_max"]


def build(variant):
    tree = json.load(open(os.path.join(HERE, variant, "bdt_config.json")))["submit_trees"][0]
    root = os.path.join(HERE, variant, tree)
    models = kfold.KFoldModels(root)
    qcd_samples = [s for c, m in models.config["class_groups"].items() if "qcd" in c.lower() for s in m]
    thresholds = {k: v for k, v in models.thresholds.items() if k != MSD1}
    d = kfold.load_scores(models, qcd_samples, [MSD1] + EXTRA, thresholds)
    out = os.path.join(HERE, variant, "closure", "qcd_view.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sample_idx = np.array([qcd_samples.index(s) for s in d["sample_name"]], dtype=np.int8)
    np.savez(out, proba=d["proba"].astype(np.float32), weight=d["weight"] * LUMI, fold=d["fold"],
             sample=sample_idx, samples=np.array(qcd_samples), class_names=np.array(models.class_names),
             **{f"b_{b}": d["columns"][b].astype(np.float32) for b in [MSD1] + EXTRA})
    print(f"{len(sample_idx)} QCD events -> {out}")


def stat(w):
    return float(w.sum()), math.sqrt(float((w * w).sum()))


def ratio(a, sa, b, sb):
    if a <= 0 or b <= 0:
        return float("nan"), float("nan")
    r = a / b
    return r, r * math.hypot(sa / a, sb / b)


def box_mask(proba, row, axes, names, only=None, skip=None):
    """qcd_est._region_mask (low <= p < high, open when high == 1), optionally on a subset of axes."""
    mask = np.ones(len(proba), dtype=bool)
    for axis in axes:
        if (only is not None and axis not in only) or (skip is not None and axis in skip):
            continue
        lo, hi = float(row[f"{axis}_low"]), float(row[f"{axis}_high"])
        p = proba[:, names.index(axis)]
        mask &= (p >= lo) & (p < hi) if hi < 1.0 - 1e-12 else (p >= lo)
    return mask


def fmt(r, e):
    return f"{r:6.3f}±{e:5.3f}" if np.isfinite(r) else f"{'-':>12}"


def analyze(variant, csv, control=CONTROL):
    z = np.load(os.path.join(HERE, variant, "closure", "qcd_view.npz"), allow_pickle=False)
    proba = z["proba"].astype(np.float64)
    w = z["weight"]
    names = [str(x) for x in z["class_names"]]
    samples = [str(x) for x in z["samples"]]
    sidx, fold = z["sample"], z["fold"]
    msd = z[f"b_{MSD1}"].astype(np.float64)
    win = (msd > WINDOW[0]) & (msd < WINDOW[1])
    ctl = (msd > control[0]) & (msd < control[1])
    print(f"control window {control[0]:g} < msoftdrop_1 < {control[1]:g}")
    srs = pd.read_csv(csv)
    axes = [c[:-4] for c in srs.columns if c.endswith("_low")]
    boxes = [box_mask(proba, row, axes, names) for _, row in srs.iterrows()]
    union = np.logical_or.reduce(boxes)
    labels = [f"SR{int(r['bin_index'])}" for _, r in srs.iterrows()]
    print(f"{len(w)} QCD events, weight {w.sum():.4g}; window {int(win.sum())}, control {int(ctl.sum())}")

    def closure(sel, title):
        B, sB = stat(w[sel & win & ~union])
        D, sD = stat(w[sel & ctl & ~union])
        tf, tfe = ratio(B, sB, D, sD)
        print(f"\n{title}: TF = B/D = {fmt(tf, tfe)}  (B = {B:.4g} ± {sB:.2g}, D = {D:.4g} ± {sD:.2g})")
        print(f"  {'':5} {'A_true':>9} {'±':>6} {'nA':>5} | {'C':>9} {'±':>6} {'nC':>5} | {'A/C':>12} | {'(A/C)/TF':>12}")
        for lab, b in list(zip(labels, boxes)) + [("union", union)]:
            A, sA = stat(w[sel & win & b])
            C, sC = stat(w[sel & ctl & b])
            r, re = ratio(A, sA, C, sC)
            dr = (r / tf, math.hypot(re / r, tfe / tf) * r / tf) if np.isfinite(r) and tf > 0 else (float("nan"),) * 2
            print(f"  {lab:5} {A:9.2f} {sA:6.2f} {int((sel & win & b).sum()):5d} | {C:9.2f} {sC:6.2f} "
                  f"{int((sel & ctl & b).sum()):5d} | {fmt(r, re)} | {fmt(*dr)}")

    # 0) reproduce qcd_est
    allq = np.ones(len(w), dtype=bool)
    closure(allq, "All QCD (qcd_est.py)")

    # 1) by sample
    print("\nBy QCD sample: weight (events) in A_union, C_union, B, D")
    for i, s in enumerate(samples):
        m = sidx == i
        cells = []
        for reg in (win & union, ctl & union, win & ~union, ctl & ~union):
            cells.append(f"{w[m & reg].sum():10.3g} ({int((m & reg).sum()):7d})")
        print(f"  {s:18} " + " ".join(cells))
    low_ht = np.isin(sidx, [samples.index(s) for s in samples if s in ("qcd_ht40to70", "qcd_ht70to100",
                                                                       "qcd_ht100to200", "qcd_ht200to400")])
    closure(~low_ht, "QCD without HT < 400 GeV")

    # 2) by fold
    for f in np.unique(fold):
        closure(fold == f, f"fold {int(f)} only")

    # 3) msoftdrop_1 dependence of the SR selection efficiency: QCD weight fraction in each box
    # per msoftdrop_1 bin (flat for a decorrelated selection)
    q = ~low_ht
    edges = [-1e9, 0, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 1e9]
    print("\nQCD (HT >= 400) efficiency of the SR boxes per msoftdrop_1 bin, relative to the window "
          "(eff(bin) / eff(50-150)), with the events in the box")
    eff_win = {lab: w[q & win & b].sum() / w[q & win].sum() for lab, b in list(zip(labels, boxes)) + [("union", union)]}
    print(f"  {'msd1 bin':>12} {'w_bin':>10} " + " ".join(f"{lab:>18}" for lab in labels + ["union"]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = q & (msd > lo) & (msd <= hi)
        wb = w[m].sum()
        cells = []
        for lab, b in list(zip(labels, boxes)) + [("union", union)]:
            s, se = stat(w[m & b])
            r = s / wb / eff_win[lab] if wb > 0 else float("nan")
            cells.append(f"{r:7.2f}±{se / wb / eff_win[lab]:5.2f} ({int((m & b).sum()):4d})")
        print(f"  {f'({lo:g}, {hi:g}]':>12} {wb:10.4g} " + " ".join(f"{c:>18}" for c in cells))

    # 4) score-axis attribution: window / control efficiency ratio of each box with one axis only
    # and with all axes but one (QCD with HT >= 400)
    def eff_ratio(m):
        a, sa = stat(w[q & win & m])
        c, sc = stat(w[q & ctl & m])
        r, re = ratio(a / w[q & win].sum(), sa / w[q & win].sum(), c / w[q & ctl].sum(), sc / w[q & ctl].sum())
        return r, re, int((q & win & m).sum()), int((q & ctl & m).sum())

    print("\nScore-axis attribution (QCD, HT >= 400): eps_window / eps_control of the box cut, "
          "with (nA, nC) events; 1 = no msoftdrop_1 dependence")
    for (_, row), lab in zip(srs.iterrows(), labels):
        r, re, na, nc = eff_ratio(box_mask(proba, row, axes, names))
        print(f"  {lab}: full box {fmt(r, re)} ({na}, {nc})")
        for axis in axes:
            lo, hi = float(row[f"{axis}_low"]), float(row[f"{axis}_high"])
            if lo <= 0.0 and hi >= 1.0:
                continue
            r1, re1, na1, nc1 = eff_ratio(box_mask(proba, row, axes, names, only=[axis]))
            r2, re2, na2, nc2 = eff_ratio(box_mask(proba, row, axes, names, skip=[axis]))
            print(f"    {axis:6} [{lo:.3f}, {hi:.3f}): only this axis {fmt(r1, re1)} ({na1:7d}, {nc1:7d})   "
                  f"all but this axis {fmt(r2, re2)} ({na2:6d}, {nc2:6d})")

    # 5) tail of each class probability (QCD with HT >= 400): eps_window / eps_control of p_k >= t
    # (p_QCD < t for QCD) at fixed QCD efficiencies in the window
    print("\nPer-axis tail test: eps_window / eps_control at a cut keeping eps of the window QCD weight")
    effs = (1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5)
    print(f"  {'axis':6} " + " ".join(f"{e:>20.0e}" for e in effs))
    for k, name in enumerate(names):
        s = proba[:, k] if "qcd" not in name.lower() else -proba[:, k]
        sw = np.sort(s[q & win])[::-1]
        ww = w[q & win][np.argsort(s[q & win])[::-1]]
        cum = np.cumsum(ww) / ww.sum()
        cells = []
        for e in effs:
            t = sw[min(np.searchsorted(cum, e), len(sw) - 1)]
            r, re, na, nc = eff_ratio(s >= t)
            cells.append(f"{fmt(r, re)} ({nc:5d})")
        print(f"  {name:6} " + " ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3], *([tuple(float(x) for x in sys.argv[4].split(","))] if len(sys.argv) > 4 else []))
    else:
        raise SystemExit(__doc__)
