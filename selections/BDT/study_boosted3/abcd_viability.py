#!/usr/bin/env python3
"""Viability check of a per-SR data-driven ABCD for one study variant (MC only).

    python3 abcd_viability.py <variant> <qcd_est_config> [sr_tag ...]

<qcd_est_config> is a qcd_est.py config with bdt_root = <variant>/boosted3 (its SR csv is not
used). The first run rebuilds qcd_est.py's event view of the test split (model scores, all
non-ABCD thresholds, msoftdrop_1 pass/fail) and caches it in <variant>/viability/qcd_view.npz.

QCD means the QCD class without the HT < 400 GeV samples (LOW_HT_QCD): their few dozen events carry
about half of the QCD weight (boosted3 test split: 35 events, 47%) and would dominate every ratio.
The control region is the low-mass sideband 0 < msoftdrop_1 < 50 GeV of the fail region (the region
above the window has a much harder QCD score tail, and msoftdrop_1 <= 0 is outside the
decorrelation training window).

(a) Tail independence: for cuts on the signal-like score p(VVV)+p(VH) keeping 10% ... 0.03% of
    the QCD weight, the QCD tail fraction in the msoftdrop_1 window (pass region) and in the
    sideband above it, each over the one in the low sideband. A per-SR ABCD with C_i from the
    low sideband needs window / low = 1.
(b) Per-SR closure in MC for the SR sets <variant>/sr_hist[_<sr_tag>]/signal_region.csv: QCD in
    A_i (window, in box) against C_i x B/D (C_i: low sideband, in box; B, D: window / low
    sideband outside the SR union), with MC uncertainties from sum(w^2), and the signal and
    non-QCD content of C_i. Writes <variant>/viability/summary.json.
"""
import importlib.util, json, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import region_mask  # noqa: E402
from make_study import LOW_HT_QCD  # noqa: E402

QCD_EST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                            "background_estimation", "qcd_est.py")


def build_cache(config_path, cache_path):
    os.environ["QCD_EST_CONFIG_PATH"] = config_path
    spec = importlib.util.spec_from_file_location("qcd_est_viability", QCD_EST_PATH)
    qe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qe)
    qe.INFERENCE_THREADS = max(1, len(os.sched_getaffinity(0)))
    tree = qe.TREE_NAME
    model_branches = [item["name"] for item in qe.br_cfg[tree]]
    selection = qe.sel_cfg[tree]
    clip_ranges = {k: tuple(v) for k, v in selection.get("clip_ranges", {}).items()}
    log_transform = list(selection.get("log_transform", []))
    thresholds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in selection.get("thresholds", {}).items()}
    abcd_thresholds, bdt_thresholds = qe._resolve_abcd_thresholds(thresholds, qe.ABCD_BRANCH_NAMES)
    decorrelate = qe.cfg.get(tree, {}).get("decorrelate", [])
    load_branches = sorted(set(model_branches) | set(thresholds) | set(decorrelate))
    df = qe.load_test_data(load_branches)
    keep = qe._threshold_mask(df, bdt_thresholds, apply_to_sentinel=True).to_numpy(dtype=bool)
    df = df[keep].reset_index(drop=True)
    X = qe._drop_decorrelated_features(qe.standardize_X(df[model_branches].copy(), clip_ranges, log_transform), decorrelate)
    proba = qe._predict_model_proba(qe._load_model(), X, context="viability")
    abcd_pass, abcd_fail = qe._abcd_pass_fail_masks(df, abcd_thresholds)
    abcd_branch = qe.ABCD_BRANCH_NAMES[0]
    signal = [i for i, c in enumerate(qe.CLASS_NAMES)
              if all(qe.SAMPLE_INFO[s]["is_signal"] for s in qe.CLASS_GROUPS[c])]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, proba=proba.astype(np.float32), weight=df["weight"].to_numpy(dtype=float),
             class_idx=df["class_idx"].to_numpy(dtype=int), sample=df["sample_name"].astype(str).to_numpy(),
             msd=df[abcd_branch].to_numpy(dtype=float), abcd_pass=abcd_pass, abcd_fail=abcd_fail,
             class_names=np.array(qe.CLASS_NAMES), signal=np.array(signal),
             abcd_window=np.array(abcd_thresholds[abcd_branch], dtype=float))
    print(f"cached {len(df)} events -> {cache_path}")


def stat(w):
    """Weighted sum, its MC uncertainty and the Kish effective number of MC events."""
    s, s2 = float(w.sum()), float((w * w).sum())
    return s, math.sqrt(s2), (s * s / s2 if s2 > 0 else 0.0)


def ratio_err(a, sa, b, sb):
    if a <= 0 or b <= 0:
        return float("nan"), float("nan")
    r = a / b
    return r, r * math.sqrt((sa / a) ** 2 + (sb / b) ** 2)


def main():
    variant, config_path = sys.argv[1], os.path.abspath(sys.argv[2])
    sr_tags = sys.argv[3:] or [""]
    vdir = os.path.join(HERE, variant)
    cache = os.path.join(vdir, "viability", "qcd_view.npz")
    if not os.path.exists(cache):
        build_cache(config_path, cache)
    c = np.load(cache, allow_pickle=True)
    proba, w, cls, msd = c["proba"].astype(float), c["weight"], c["class_idx"], c["msd"]
    p_pass, p_fail = c["abcd_pass"], c["abcd_fail"]
    names = [str(x) for x in c["class_names"]]
    sig = list(c["signal"])
    lo_win, hi_win = c["abcd_window"]
    qcd_cls = cls == names.index("QCD")
    low_ht = qcd_cls & np.isin(c["sample"], LOW_HT_QCD)
    qcd = qcd_cls & ~low_ht
    is_sig = np.isin(cls, sig)
    d = proba[:, sig].sum(axis=1)
    out = {"variant": variant, "tail": [], "closure": {}}
    print(f"Excluded {int(low_ht.sum())} QCD events of {', '.join(LOW_HT_QCD)}: "
          f"{w[low_ht].sum() / w[qcd_cls].sum():.1%} of the QCD weight")
    ctrl = p_fail & (msd > 0) & (msd < lo_win)  # low-mass sideband: the ABCD control region

    # (a) tail independence
    regions = {"window": qcd & p_pass, "low": qcd & ctrl, "high": qcd & p_fail & (msd > hi_win)}
    tot = {k: float(w[m].sum()) for k, m in regions.items()}
    sel = regions["window"] | regions["low"] | regions["high"]
    order = np.argsort(-d[sel])
    dq, wq = d[sel][order], w[sel][order]
    cum = np.cumsum(wq) / wq.sum()
    print(f"QCD events: window {int(regions['window'].sum())}, low sideband (0, {lo_win:g}) {int(regions['low'].sum())}, "
          f"high sideband (> {hi_win:g}) {int(regions['high'].sum())}; QCD weight window/low = {tot['window'] / tot['low']:.3f}")
    print("\n(a) QCD tail fraction above a cut on p(VVV)+p(VH), over the one in the low sideband")
    print(f"{'QCD eff':>8} {'cut':>8} | {'eps_win':>10} {'N_win':>7} {'Neff':>7} | {'eps_low':>10} {'N_low':>7} {'Neff':>7} | "
          f"{'window/low':>13} | {'high/low':>13}")
    for eff in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        t = float(dq[min(np.searchsorted(cum, eff), len(dq) - 1)])
        row = {"target_eff": eff, "cut": t}
        vals = {}
        for k, m in regions.items():
            s, se, neff = stat(w[m & (d >= t)])
            vals[k] = (s / tot[k], se / tot[k], int((m & (d >= t)).sum()), neff)
            row[k] = {"eff": vals[k][0], "err": vals[k][1], "n": vals[k][2], "neff": vals[k][3]}
        r, re = ratio_err(vals["window"][0], vals["window"][1], vals["low"][0], vals["low"][1])
        rh, rhe = ratio_err(vals["high"][0], vals["high"][1], vals["low"][0], vals["low"][1])
        row["window_over_low"] = [r, re]
        row["high_over_low"] = [rh, rhe]
        out["tail"].append(row)
        print(f"{eff:8.0e} {t:8.4f} | {vals['window'][0]:10.3e} {vals['window'][2]:7d} {vals['window'][3]:7.0f} | "
              f"{vals['low'][0]:10.3e} {vals['low'][2]:7d} {vals['low'][3]:7.0f} | {r:6.3f}±{re:5.3f} | {rh:6.3f}±{rhe:5.3f}")

    # (b) per-SR closure
    for tag in sr_tags:
        csv = os.path.join(vdir, f"sr_hist_{tag}" if tag else "sr_hist", "signal_region.csv")
        if not os.path.exists(csv):
            print(f"\n(b) {tag or 'sr_hist'}: no SR csv")
            continue
        df = pd.read_csv(csv)
        axes = [col[:-4] for col in df.columns if col.endswith("_low")]
        boxes = [region_mask(proba, row, axes, names) for _, row in df.iterrows()]
        union = np.zeros(len(w), dtype=bool)
        for b in boxes:
            union |= b
        B, sB, _ = stat(w[qcd & p_pass & ~union])
        D, sD, _ = stat(w[qcd & ctrl & ~union])
        tf, tfe = ratio_err(B, sB, D, sD)
        print(f"\n(b) {tag or 'sr_hist'}: per-SR closure, TF = B/D = {tf:.4f} ± {tfe:.4f} "
              f"(QCD outside the SR union, window / low sideband)")
        print(f"{'SR':>4} | {'A_true':>9} {'±':>7} {'nA':>4} | {'C':>9} {'±':>7} {'nC':>4} | {'A_pred':>9} | {'pred/true':>14} | "
              f"{'C: signal':>9} {'nonQCD':>8}")
        rows = []
        for label, b in [(f"SR{int(r['bin_index'])}", bx) for (_, r), bx in zip(df.iterrows(), boxes)] + [("all", union)]:
            A, sA, _ = stat(w[qcd & p_pass & b])
            C, sC, _ = stat(w[qcd & ctrl & b])
            nA, nC = int((qcd & p_pass & b).sum()), int((qcd & ctrl & b).sum())
            pred = C * tf
            r, re = ratio_err(pred, C * tf * math.hypot(sC / C if C > 0 else 0, tfe / tf), A, sA)
            sC_sig = float(w[is_sig & ctrl & b].sum())
            sC_bkg = float(w[~is_sig & ~qcd_cls & ctrl & b].sum())
            rows.append({"region": label, "A_true": A, "A_err": sA, "nA": nA, "C": C, "C_err": sC, "nC": nC,
                         "A_pred": pred, "pred_over_true": [r, re], "C_signal": sC_sig, "C_nonqcd": sC_bkg})
            print(f"{label:>4} | {A:9.2f} {sA:7.2f} {nA:4d} | {C:9.2f} {sC:7.2f} {nC:4d} | {pred:9.2f} | {r:6.2f}±{re:6.2f} | "
                  f"{sC_sig:9.2f} {sC_bkg:8.2f}")
        out["closure"][tag or "sr_hist"] = {"tf": [tf, tfe], "regions": rows}
    with open(os.path.join(vdir, "viability", "summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
