#!/usr/bin/env python3
"""Offline ABCD non-closure / 2-D (BDT score vs soft-drop mass) diagnostic.

Reads the per-event dump written by qcd_est.py (QCD_EST_DUMP_2D=1) and, for QCD
MC, quantifies and visualizes why the ABCD factorization (msoftdrop independent
of the BDT score region) holds or fails.

ABCD recap (from qcd_est.py):
  A = (in SR score-box) & (msoftdrop PASS)        <- signal region
  B = (NOT in box)      & (msoftdrop PASS)
  C = (in box)          & (msoftdrop FAIL = sideband)
  D = (NOT in box)      & (msoftdrop FAIL)
  predict A = B*C/D ; closure = predict/true
Factorization assumption: msoftdrop PASS-fraction is the same in-box and out-of-box,
i.e. A/(A+C) == B/(B+D). Equivalently the (normalized) msoftdrop shape is identical
in-box vs out-of-box.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wsum(w, m):
    return float(np.sum(w[m]))


def analyze(tree, dump_path, outdir):
    d = np.load(dump_path, allow_pickle=True)
    axis_names = [str(a) for a in d["axis_names"]]
    qcd = d["qcd_mask"]
    msd = d["msoftdrop"][qcd]
    proba = d["proba"][qcd].astype(float)
    w = d["weight"][qcd].astype(float)
    inbox = d["union_score_mask"][qcd]
    passm = d["abcd_pass"][qcd]
    failm = d["abcd_fail"][qcd]
    region_masks = d["region_masks"][:, qcd]  # (nSR, nQCD)
    vvv = proba[:, axis_names.index("VVV")]

    # ---- A/B/C/D (weighted) and closure ----
    A = wsum(w, inbox & passm)
    C = wsum(w, inbox & failm)
    B = wsum(w, (~inbox) & passm)
    D = wsum(w, (~inbox) & failm)
    pred = B * C / D
    closure = pred / A
    pf_in = A / (A + C)
    pf_out = B / (B + D)
    # factorization-breaking ratio: (A/C)/(B/D) = A*D/(B*C) = 1/closure
    fac_ratio = (A * D) / (B * C)

    print(f"\n===== {tree} QCD ABCD diagnostic =====")
    print(f"  A(in,pass)={A:.6g}  C(in,fail)={C:.6g}  B(out,pass)={B:.6g}  D(out,fail)={D:.6g}")
    print(f"  predict A = B*C/D = {pred:.6g}   true A = {A:.6g}   closure = {closure:.4f}")
    print(f"  msoftdrop PASS-fraction:  in-box A/(A+C) = {pf_in:.4f}   out-box B/(B+D) = {pf_out:.4f}")
    print(f"  factorization-breaking ratio (A*D)/(B*C) = {fac_ratio:.3f}  (=1 if perfectly factorized; =1/closure)")

    # mass window inferred from pass mask (finite values)
    finite_pass = passm & np.isfinite(msd) & (msd > -990)
    win_lo, win_hi = float(np.min(msd[finite_pass])), float(np.max(msd[finite_pass]))
    print(f"  inferred msoftdrop PASS window ~ [{win_lo:.1f}, {win_hi:.1f}] GeV")

    # ---- pass-fraction vs VVV score (the correlation that breaks factorization) ----
    valid = np.isfinite(msd) & (msd > -990) & (passm | failm)
    vv = vvv[valid]
    pv = passm[valid]
    wv = w[valid]
    # quantile-based score bins so each bin has comparable QCD weight
    qedges = np.quantile(vv, np.linspace(0, 1, 21))
    qedges = np.unique(qedges)
    centers, pf, pf_err = [], [], []
    for i in range(len(qedges) - 1):
        lo, hi = qedges[i], qedges[i + 1]
        m = (vv >= lo) & (vv < hi if i < len(qedges) - 2 else vv <= hi)
        tot = wsum(wv, m)
        if tot <= 0:
            continue
        p = wsum(wv, m & pv)
        frac = p / tot
        # binned stat error on weighted fraction (approx)
        neff = tot ** 2 / max(np.sum(wv[m] ** 2), 1e-30)
        centers.append(0.5 * (lo + hi))
        pf.append(frac)
        pf_err.append(np.sqrt(max(frac * (1 - frac) / max(neff, 1.0), 0.0)))
    centers = np.array(centers); pf = np.array(pf); pf_err = np.array(pf_err)

    # ---- normalized msoftdrop shape: in-box vs out-of-box (the assumption test) ----
    mrange = (0.0, 250.0)
    mm = valid & (msd >= mrange[0]) & (msd <= mrange[1])
    bins = np.linspace(mrange[0], mrange[1], 51)
    h_in, _ = np.histogram(msd[mm & inbox], bins=bins, weights=w[mm & inbox])
    h_out, _ = np.histogram(msd[mm & (~inbox)], bins=bins, weights=w[mm & (~inbox)])
    bc = 0.5 * (bins[:-1] + bins[1:])
    h_in_n = h_in / max(h_in.sum(), 1e-30)
    h_out_n = h_out / max(h_out.sum(), 1e-30)

    # ---- 2-D weighted density: VVV score vs msoftdrop (QCD) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    sb = mm
    H, xe, ye = np.histogram2d(
        vvv[sb], msd[sb], bins=[np.linspace(0, 1, 60), bins], weights=w[sb]
    )
    # column-normalize (per VVV-score slice) to see the conditional msoftdrop shape
    col = H.sum(axis=1, keepdims=True)
    Hn = np.where(col > 0, H / col, 0.0)
    pcm = ax.pcolormesh(xe, ye, Hn.T, cmap="viridis", shading="auto")
    ax.axhline(win_lo, color="r", ls="--", lw=1)
    ax.axhline(win_hi, color="r", ls="--", lw=1, label="msoftdrop PASS window")
    ax.set_xlabel("VVV BDT score")
    ax.set_ylabel("msoftdrop_1 [GeV]")
    ax.set_title(f"{tree} QCD: msoftdrop shape vs VVV score (column-normalized)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(pcm, ax=ax, label="P(msoftdrop | score-slice)")

    ax = axes[0, 1]
    ax.errorbar(centers, pf, yerr=pf_err, fmt="o-", ms=3, label="QCD pass-fraction")
    ax.axhline(pf_out, color="g", ls=":", label=f"out-box B/(B+D)={pf_out:.3f}")
    ax.axhline(pf_in, color="r", ls=":", label=f"in-box A/(A+C)={pf_in:.3f}")
    ax.set_xlabel("VVV BDT score (quantile bins)")
    ax.set_ylabel("msoftdrop PASS-fraction")
    ax.set_title(f"{tree} QCD: PASS-fraction vs score (flat => factorizes)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.step(bc, h_out_n, where="mid", label="out-of-box (B+D)", color="g")
    ax.step(bc, h_in_n, where="mid", label="in-box (A+C)", color="r")
    ax.axvline(win_lo, color="k", ls="--", lw=0.8)
    ax.axvline(win_hi, color="k", ls="--", lw=0.8)
    ax.set_xlabel("msoftdrop_1 [GeV]")
    ax.set_ylabel("a.u. (unit norm)")
    ax.set_title(f"{tree} QCD msoftdrop shape: in-box vs out-of-box")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"{tree}  QCD ABCD non-closure diagnostic\n"
        f"--------------------------------------\n"
        f"A (in-box, mass PASS)  = {A:.4g}\n"
        f"C (in-box, mass FAIL)  = {C:.4g}\n"
        f"B (out-box, mass PASS) = {B:.4g}\n"
        f"D (out-box, mass FAIL) = {D:.4g}\n\n"
        f"predict A = B*C/D = {pred:.4g}\n"
        f"true A            = {A:.4g}\n"
        f"closure           = {closure:.3f}\n\n"
        f"mass PASS-fraction in-box  = {pf_in:.3f}\n"
        f"mass PASS-fraction out-box = {pf_out:.3f}\n"
        f"ratio (=1/closure)         = {fac_ratio:.2f}\n\n"
        f"PASS window ~ [{win_lo:.0f}, {win_hi:.0f}] GeV\n\n"
        f"Interpretation: closure fails when the in-box mass\n"
        f"PASS-fraction differs from out-of-box, i.e. high-score\n"
        f"QCD is mass-sculpted toward the signal window."
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout()
    out = os.path.join(outdir, "abcd_2d_nonclosure.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- per-SR local pass-fraction (which region drives non-closure) ----
    print("  per-SR  in-box passfrac  vs  out-box passfrac (B/(B+D)):")
    for i in range(region_masks.shape[0]):
        rm = region_masks[i] & valid
        a_i = wsum(w, rm & passm)
        c_i = wsum(w, rm & failm)
        if a_i + c_i <= 0:
            print(f"    SR{i+1}: (empty)")
            continue
        print(f"    SR{i+1}: in-box passfrac={a_i/(a_i+c_i):.3f}  (A={a_i:.4g}, C={c_i:.4g})  vs out-box {pf_out:.3f}")

    return dict(tree=tree, A=A, B=B, C=C, D=D, closure=closure, pf_in=pf_in, pf_out=pf_out)


if __name__ == "__main__":
    base = "/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/background_estimation/output"
    results = []
    for tree in ("fat2", "fat3"):
        outdir = os.path.join(base, f"{tree}_vjets_full_minbkg50")
        dump = os.path.join(outdir, "abcd_2d_dump.npz")
        if not os.path.exists(dump):
            print(f"MISSING dump: {dump}")
            continue
        results.append(analyze(tree, dump, outdir))
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {r['tree']}: closure={r['closure']:.3f}  passfrac in/out = {r['pf_in']:.3f}/{r['pf_out']:.3f}")
