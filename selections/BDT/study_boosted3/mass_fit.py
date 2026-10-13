#!/usr/bin/env python3
"""Prototype of a jet-mass fit for the per-SR QCD estimate of a k-fold study variant (MC only).

    python3 mass_fit.py build <variant>
    python3 mass_fit.py ratio <variant> <signal_region.csv> [--range 20,250] [--width 10]
    python3 mass_fit.py models <variant> <signal_region.csv> <tag> [--range ...] [--width ...] [--models a,b]
    python3 mass_fit.py bias-submit <variant> --tag <tag> --truth a,b --fit c,d [--toys 300] [--rinj 1]
    python3 mass_fit.py bias-summary <variant> --tag <tag>

build: every MC event of the score cache passing every threshold but msoftdrop_1 (the HT < 400 GeV
QCD samples LOW_HT_QCD left out, as in the per-SR ABCD studies), with class, weight x lumi, class
probabilities and msoftdrop_1, to <variant>/massfit/view.npz.

ratio: per SR (score box) the QCD msoftdrop_1 spectrum in the box over the QCD spectrum outside every
box (the fail template F), with fits of R(m) = p0 exp(p1 x + ... + pn x^n) (x = m mapped to [-1, 1])
of order 0-3 to all bins (chi2 / ndf) and, per order, the window (50-150 GeV) prediction of a fit to
the sidebands only, against the MC-true QCD in the window.

models: combine cards, one bin per SR and mass bin (F > 0), with the signal, the non-QCD MC
backgrounds (autoMCStats, lumi) and QCD from MC ('mc') or from a QCD model: 'persr_o<n>' = F x
p0_i exp(p1_i x + ... + pn_i x^n) with free parameters per SR, 'shared_o<n>' = F x n_i exp(s1 x +
... + sn x^n) with one shape for every SR and a free normalisation per SR (extArg + formula rateParam,
started at the fit to the MC ratios). Runs Significance and AsymptoticLimits on the model's Asimov
data (-t -1) and the significance on pseudo-data equal to the MC sum with the MC-true QCD (data_obs;
not a sensitivity: with the noisy MC QCD it measures the model's residuals). Outputs in
<variant>/massfit/<tag>/ (results.json).

bias / bias-submit / bias-summary: functional-form bias test: toys from the workspace of a truth
model (parameters at their MC fit, systematics not varied, signal strength --rinj), fitted with
another model (FitDiagnostics, s+b), pull of r = (r - rinj) / (upper error if r < rinj else lower
error); bias-submit runs one SLURM job per truth x fit pair, bias-summary tabulates the mean pulls
with the models' Asimov Z and limits.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import kfold  # noqa: E402
from evaluate import region_mask  # noqa: E402
from make_study import LOW_HT_QCD  # noqa: E402

LUMI = 255.8
MSD1 = "ScoutingFatPFJetRecluster_msoftdrop_1"
WINDOW = (50.0, 150.0)
LUMI_UNC = 0.016


def _tree(variant):
    return json.load(open(os.path.join(HERE, variant, "bdt_config.json")))["submit_trees"][0]


def build(variant):
    models = kfold.KFoldModels(os.path.join(HERE, variant, _tree(variant)))
    samples = [s for m in models.config["class_groups"].values() for s in m if s not in LOW_HT_QCD]
    thresholds = {k: v for k, v in models.thresholds.items() if k != MSD1}
    d = kfold.load_scores(models, samples, [MSD1], thresholds)
    out = os.path.join(HERE, variant, "massfit", "view.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, proba=d["proba"].astype(np.float32), weight=d["weight"] * LUMI,
             class_idx=d["class_idx"].astype(np.int8), msd=d["columns"][MSD1].astype(np.float32),
             class_names=np.array(models.class_names))
    print(f"{len(d['weight'])} events -> {out}")


def _load(variant, csv, lo, hi, width):
    z = np.load(os.path.join(HERE, variant, "massfit", "view.npz"))
    names = [str(x) for x in z["class_names"]]
    proba, w, cls, msd = z["proba"].astype(np.float64), z["weight"], z["class_idx"], z["msd"].astype(np.float64)
    srs = pd.read_csv(csv)
    axes = [c[:-4] for c in srs.columns if c.endswith("_low")]
    boxes = [region_mask(proba, row, axes, names) for _, row in srs.iterrows()]
    labels = [f"sr{int(r['bin_index'])}" for _, r in srs.iterrows()]
    edges = np.arange(lo, hi + 0.5 * width, width)
    return names, w, cls, msd, boxes, labels, edges


def _hist(msd, w, m, edges):
    s, _ = np.histogram(msd[m], bins=edges, weights=w[m])
    s2, _ = np.histogram(msd[m], bins=edges, weights=w[m] ** 2)
    return s, s2


def _xvar(edges):
    c = 0.5 * (edges[:-1] + edges[1:])
    return 2.0 * (c - edges[0]) / (edges[-1] - edges[0]) - 1.0


def _fit_ratio(x, r, var, order, use):
    """Weighted least squares of log R = log p0 + p1 x + ... on the bins `use` (R > 0), then a
    Newton refinement of the chi2 of R itself; returns (params, chi2, ndf)."""
    ok = use & (r > 0) & (var > 0)
    if ok.sum() <= order + 1:
        return None, float("nan"), 0
    X = np.vander(x[ok], order + 1, increasing=True)
    wl = r[ok] ** 2 / var[ok]
    beta = np.linalg.lstsq(X * np.sqrt(wl)[:, None], np.log(r[ok]) * np.sqrt(wl), rcond=None)[0]
    for _ in range(50):
        model = np.exp(X @ beta)
        res = (r[ok] - model) / np.sqrt(var[ok])
        J = (model / np.sqrt(var[ok]))[:, None] * X
        step = np.linalg.lstsq(J, res, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    model = np.exp(X @ beta)
    chi2 = float(np.sum((r[ok] - model) ** 2 / var[ok]))
    return beta, chi2, int(ok.sum() - order - 1)


def _ratios(variant, csv, lo, hi, width):
    names, w, cls, msd, boxes, labels, edges = _load(variant, csv, lo, hi, width)
    qcd = cls == names.index("QCD")
    union = np.logical_or.reduce(boxes)
    F, F2 = _hist(msd, w, qcd & ~union, edges)
    out = []
    for lab, b in zip(labels, boxes):
        A, A2 = _hist(msd, w, qcd & b, edges)
        r = np.where(F > 0, A / np.where(F > 0, F, 1), 0.0)
        var = np.where(F > 0, A2 / np.where(F > 0, F, 1) ** 2 + r ** 2 * F2 / np.where(F > 0, F, 1) ** 2, 0.0)
        out.append((lab, A, A2, r, var))
    return names, w, cls, msd, boxes, labels, edges, F, F2, out


def ratio(variant, csv, lo, hi, width):
    names, w, cls, msd, boxes, labels, edges, F, F2, out = _ratios(variant, csv, lo, hi, width)
    x = _xvar(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    in_win = (centres > WINDOW[0]) & (centres < WINDOW[1])
    print(f"QCD fail template outside every SR box: {F.sum():.4g} in {lo:g}-{hi:g} GeV, {len(centres)} bins of {width:g} GeV")
    for lab, A, A2, r, var in out:
        print(f"\n{lab}: QCD {A.sum():.2f} (window {A[in_win].sum():.2f} +- {math.sqrt(A2[in_win].sum()):.2f}), "
              f"ratio to F per bin x 1e6:")
        print("   " + " ".join(f"{m:5.0f}" for m in centres))
        print("   " + " ".join(f"{1e6 * v:5.2f}" for v in r))
        print("   " + " ".join(f"{1e6 * math.sqrt(v):5.2f}" for v in var))
        for order in range(4):
            beta, chi2, ndf = _fit_ratio(x, r, var, order, np.ones_like(x, dtype=bool))
            sb, _, _ = _fit_ratio(x, r, var, order, ~in_win)
            if beta is None:
                print(f"   order {order}: too few bins")
                continue
            pred = float(np.sum(F[in_win] * np.exp(np.vander(x[in_win], order + 1, increasing=True) @ sb))) if sb is not None else float("nan")
            print(f"   order {order}: chi2/ndf (all bins) = {chi2:6.1f}/{ndf:2d};  sidebands-only fit predicts "
                  f"{pred:8.2f} in the window (MC-true {A[in_win].sum():.2f}, ratio {pred / max(A[in_win].sum(), 1e-9):.2f})")


def _th1(name, value, err):
    import ROOT
    h = ROOT.TH1F(name, name, 1, 0.0, 1.0)
    h.SetBinContent(1, value)
    h.SetBinError(1, err)
    return h


def _fit_shared(x, rats, order):
    """Joint fit of R_i(m) = exp(a_i + b_1 x + ... + b_n x^n) to every SR's ratio (shared shape,
    one normalisation per SR; bins with R > 0): weighted least squares of log R, then Gauss-Newton
    on the chi2 of R. Returns ([exp(a_i)], [b_k], chi2, ndf)."""
    rows, rs, vs = [], [], []
    n = len(rats)
    for i, (_lab, _A, _A2, r, var) in enumerate(rats):
        for j in np.flatnonzero((r > 0) & (var > 0)):
            row = np.zeros(n + order)
            row[i] = 1.0
            row[n:] = [x[j] ** k for k in range(1, order + 1)]
            rows.append(row); rs.append(r[j]); vs.append(var[j])
    D, rs, vs = np.array(rows), np.array(rs), np.array(vs)
    wl = rs ** 2 / vs
    theta = np.linalg.lstsq(D * np.sqrt(wl)[:, None], np.log(rs) * np.sqrt(wl), rcond=None)[0]
    for _ in range(50):
        model = np.exp(D @ theta)
        J = (model / np.sqrt(vs))[:, None] * D
        step = np.linalg.lstsq(J, (rs - model) / np.sqrt(vs), rcond=None)[0]
        theta = theta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    chi2 = float(np.sum((rs - np.exp(D @ theta)) ** 2 / vs))
    return np.exp(theta[:n]), theta[n:], chi2, int(len(rs) - n - order)


def _model_params(model, x, rats):
    """{sr label: (formula parameter names, {parameter: (initial value, range)})} of a QCD model
    'persr_o<n>' (R_i = p0_i exp(p1_i x + ...)) or 'shared_o<n>' (R_i = n_i exp(s1 x + ...)), started
    at its fit to the MC ratios."""
    kind, order = model.split("_o")[0], int(model.split("_o")[1])
    out = {}
    if kind == "persr":
        for lab, _A, _A2, r, var in rats:
            beta, _, _ = _fit_ratio(x, r, var, order, np.ones_like(x, dtype=bool))
            pars = {f"p0_{lab}": (float(np.exp(beta[0])), f"[0,{10 * float(np.exp(beta[0])):.6g}]")}
            pars.update({f"p{k}_{lab}": (float(beta[k]), "[-20,20]") for k in range(1, order + 1)})
            out[lab] = (list(pars), pars)
    else:
        norms, shape, _, _ = _fit_shared(x, rats, order)
        for (lab, *_), nrm in zip(rats, norms):
            pars = {f"n_{lab}": (float(nrm), f"[0,{10 * float(nrm):.6g}]")}
            pars.update({f"s{k}": (float(shape[k - 1]), "[-20,20]") for k in range(1, order + 1)})
            out[lab] = (list(pars), pars)
    return out, order


def _write_card(outdir, name, model, x, rats, boxes, S_all, B_all, F):
    """Card <outdir>/card_<name>.txt: one bin per SR and mass bin with F > 0, the signal and the
    non-QCD MC backgrounds (autoMCStats, lumi), and QCD from MC (model 'mc') or as F x R_i(m)
    (extArg parameters, formula rateParam). data_obs = the MC sum with the MC-true QCD."""
    import ROOT
    extra, declared = [], set()
    if model != "mc":
        params, order = _model_params(model, x, rats)
    bins, rows = [], []
    shapes = ROOT.TFile.Open(os.path.join(outdir, f"shapes_{name}.root"), "RECREATE")
    for (lab, A, A2, _r, _var), S, B in zip(rats, S_all, B_all):
        if model != "mc":
            pnames, pars = params[lab]
            for pn in pnames:
                if pn not in declared:
                    v, rng = pars[pn]
                    extra.append(f"{pn} extArg {v:.10g} {rng}")
                    declared.add(pn)
        for j in np.flatnonzero(F > 0):
            ch = f"{lab}_m{j:02d}"
            procs = {"signal": (S[0][j], S[1][j])}
            procs.update({p: (h[0][j], h[1][j]) for p, h in B.items()})
            if model == "mc":
                procs["bkg_qcd"] = (A[j], A2[j])
            else:
                procs["bkg_qcd"] = (F[j], 0.0)
                terms = "+".join(f"@{k}*{x[j] ** k:.10g}" for k in range(1, order + 1))
                formula = f"(@0*exp({terms}))" if order > 0 else "(@0)"
                extra.append(f"qcd_{ch} rateParam {ch} bkg_qcd {formula} " + ",".join(pnames))
            procs = {p: v for p, v in procs.items() if v[0] > 0}
            if not any(p.startswith("bkg_") for p in procs):
                continue
            d = shapes.mkdir(ch)
            d.cd()
            for p, (val, v2) in procs.items():
                _th1(p, val, math.sqrt(max(v2, 0.0))).Write()
            truth = sum(v[0] for p, v in procs.items() if p != "bkg_qcd") + A[j]
            _th1("data_obs", truth, 0.0).Write()
            bins.append(ch)
            rows.append(list(procs))
    shapes.Close()
    with open(os.path.join(outdir, f"card_{name}.txt"), "w") as fh:
        fh.write(f"imax {len(bins)}\njmax *\nkmax *\n")
        fh.write(f"shapes * * {outdir}/shapes_{name}.root $CHANNEL/$PROCESS\n")
        fh.write("bin " + " ".join(bins) + "\nobservation " + " ".join("-1" for _ in bins) + "\n")
        bl, pl, il = [], [], []
        for ch, procs in zip(bins, rows):
            for p in procs:
                bl.append(ch); pl.append(p)
                il.append("0" if p == "signal" else str(1 + ["bkg_top", "bkg_vv", "bkg_vjets", "bkg_qcd"].index(p)))
        fh.write("bin " + " ".join(bl) + "\nprocess " + " ".join(pl) + "\nprocess " + " ".join(il)
                 + "\nrate " + " ".join("-1" for _ in pl) + "\n")
        fh.write("lumi lnN " + " ".join(("-" if (p == "bkg_qcd" and model != "mc") else f"{1 + LUMI_UNC}") for p in pl) + "\n")
        fh.write("* autoMCStats 10\n")
        for line in extra:
            fh.write(line + "\n")


def models(variant, csv, tag, lo, hi, width, model_list):
    """Cards of the MC-QCD reference and of every QCD model in model_list, with Asimov
    significance / median expected limit and the significance on the MC-true pseudo-data."""
    names, w, cls, msd, boxes, labels, edges, F, F2, rats = _ratios(variant, csv, lo, hi, width)
    x = _xvar(edges)
    sig = np.isin(cls, [names.index("VVV"), names.index("VH")])
    bkgs = [c for c in names if c not in ("VVV", "VH", "QCD")]
    S_all = [_hist(msd, w, sig & b, edges) for b in boxes]
    B_all = [{f"bkg_{c.lower()}": _hist(msd, w, (cls == names.index(c)) & b, edges) for c in bkgs} for b in boxes]
    outdir = os.path.join(HERE, variant, "massfit", tag)
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for model in ["mc"] + list(model_list):
        _write_card(outdir, model, model, x, rats, boxes, S_all, B_all, F)
        results[model] = _run_combine(outdir, model)
        print(f"{model:12s} {results[model]}", flush=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


def _run_combine(outdir, model):
    def run(args, name):
        out = subprocess.run(args, cwd=outdir, capture_output=True, text=True)
        open(os.path.join(outdir, f"{name}.log"), "w").write(out.stdout + out.stderr)
        return out.stdout
    run(["text2workspace.py", f"card_{model}.txt", "-o", f"ws_{model}.root"], f"t2w_{model}")
    res = {}
    out = run(["combine", "-M", "Significance", "-t", "-1", "--expectSignal", "1", "-n", f"_{model}", f"ws_{model}.root"], f"sig_{model}")
    res["significance"] = next((float(l.split()[1]) for l in out.splitlines() if l.startswith("Significance:")), None)
    out = run(["combine", "-M", "AsymptoticLimits", "-t", "-1", "--run", "expected", "-n", f"_{model}", f"ws_{model}.root"], f"lim_{model}")
    res["limit_50"] = next((float(l.split("<")[1]) for l in out.splitlines() if l.startswith("Expected 50.0%")), None)
    out = run(["combine", "-M", "Significance", "-n", f"_{model}_truth", f"ws_{model}.root"], f"sig_{model}_truth")
    res["significance_mc_truth"] = next((float(l.split()[1]) for l in out.splitlines() if l.startswith("Significance:")), None)
    return res


def bias(variant, tag, truth, fit, toys, rinj, seed):
    """Toys from the workspace of QCD model `truth` (parameters at their MC fit, no systematics
    varied) with signal strength rinj, fitted with model `fit` (FitDiagnostics, s+b): pull of r,
    (r - rinj) / (upper error if r < rinj else lower error)."""
    import ROOT
    outdir = os.path.join(HERE, variant, "massfit", tag)
    name = f"{truth}__{fit}_r{rinj:g}"
    def run(args, logname):
        out = subprocess.run(args, cwd=outdir, capture_output=True, text=True)
        open(os.path.join(outdir, f"{logname}.log"), "w").write(out.stdout + out.stderr)
    run(["combine", "-M", "GenerateOnly", "-t", str(toys), "-s", str(seed), "--expectSignal", str(rinj), "--saveToys",
         "--toysNoSystematics", "--bypassFrequentistFit", "-m", "120", "-n", f"_gen_{name}", f"ws_{truth}.root"], f"gen_{name}")
    toyfile = f"higgsCombine_gen_{name}.GenerateOnly.mH120.{seed}.root"
    run(["combine", "-M", "FitDiagnostics", "-t", str(toys), "--toysFile", toyfile, "--skipBOnlyFit", "--rMin", "-20",
         "--rMax", "20", "--cminDefaultMinimizerStrategy", "0", "-m", "120", "-n", f"_bias_{name}", f"ws_{fit}.root"], f"fit_{name}")
    f = ROOT.TFile.Open(os.path.join(outdir, f"fitDiagnostics_bias_{name}.root"))
    t = f.Get("tree_fit_sb")
    pulls, rs, failed = [], [], 0
    for e in t:
        if e.fit_status != 0:
            failed += 1
            continue
        err = e.rHiErr if e.r < rinj else e.rLoErr
        if err > 0:
            pulls.append((e.r - rinj) / err)
            rs.append(e.r)
    pulls = np.array(pulls)
    res = {"truth": truth, "fit": fit, "r_injected": rinj, "toys": toys, "fits_ok": int(len(pulls)), "fits_failed": failed,
           "pull_mean": float(pulls.mean()) if len(pulls) else None,
           "pull_mean_err": float(pulls.std(ddof=1) / math.sqrt(len(pulls))) if len(pulls) > 1 else None,
           "pull_median": float(np.median(pulls)) if len(pulls) else None,
           "pull_std": float(pulls.std(ddof=1)) if len(pulls) > 1 else None,
           "r_median": float(np.median(rs)) if rs else None}
    with open(os.path.join(outdir, f"bias_{name}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


def bias_submit(variant, tag, truths, fits, toys, rinj, account, exclude):
    for truth in truths:
        for fit in fits:
            cmd = (f"cd {HERE} && python3 mass_fit.py bias {variant} --tag {tag} --truth {truth} --fit {fit} "
                   f"--toys {toys} --rinj {rinj}")
            log = os.path.join(HERE, variant, "massfit", tag, f"bias_{truth}__{fit}_r{rinj:g}.out")
            args = ["sbatch", "--parsable", f"--job-name=massfit_bias", f"--account={account}", "--cpus-per-task=1",
                    "--mem=8G", "--time=06:00:00", f"--output={log}", f"--wrap={cmd}"]
            if exclude:
                args.insert(-1, f"--exclude={exclude}")
            print(truth, fit, subprocess.check_output(args, text=True).strip())


def bias_summary(variant, tag):
    outdir = os.path.join(HERE, variant, "massfit", tag)
    res = json.load(open(os.path.join(outdir, "results.json")))
    rows = [json.load(open(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir)) if f.startswith("bias_") and f.endswith(".json")]
    fits = sorted({r["fit"] for r in rows})
    truths = sorted({r["truth"] for r in rows})
    print(f"{'fit model':12s} {'Z':>6s} {'r<':>6s} | mean pull of r per truth model (+- stat; failed fits)")
    print(f"{'':12s} {'':6s} {'':6s} | " + " ".join(f"{t:>18s}" for t in truths))
    for fit in fits:
        cells = []
        for t in truths:
            r = next((r for r in rows if r["fit"] == fit and r["truth"] == t), None)
            cells.append(f"{r['pull_mean']:+.2f}+-{r['pull_mean_err']:.2f} ({r['fits_failed']})" if r and r["pull_mean"] is not None else "-")
        z = res.get(fit, {})
        print(f"{fit:12s} {z.get('significance') or 0:6.3f} {z.get('limit_50') or 0:6.2f} | " + " ".join(f"{c:>18s}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["build", "ratio", "models", "bias", "bias-submit", "bias-summary"])
    p.add_argument("variant")
    p.add_argument("csv", nargs="?")
    p.add_argument("tag_pos", nargs="?")
    p.add_argument("--tag", default="")
    p.add_argument("--range", default="20,250")
    p.add_argument("--width", type=float, default=10.0)
    p.add_argument("--models", default="persr_o0,persr_o1,persr_o2,persr_o3,shared_o1,shared_o2,shared_o3")
    p.add_argument("--truth", default="")
    p.add_argument("--fit", default="")
    p.add_argument("--toys", type=int, default=300)
    p.add_argument("--rinj", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--account", default="cms-express")
    p.add_argument("--exclude", default="hammer-f015")
    a = p.parse_args()
    lo, hi = (float(v) for v in a.range.split(","))
    tag = a.tag or a.tag_pos or ""
    if a.action == "build":
        build(a.variant)
    elif a.action == "ratio":
        ratio(a.variant, a.csv, lo, hi, a.width)
    elif a.action == "models":
        models(a.variant, a.csv, tag, lo, hi, a.width, [m for m in a.models.split(",") if m])
    elif a.action == "bias":
        bias(a.variant, tag, a.truth, a.fit, a.toys, a.rinj, a.seed)
    elif a.action == "bias-submit":
        bias_submit(a.variant, tag, a.truth.split(","), a.fit.split(","), a.toys, a.rinj, a.account, a.exclude)
    else:
        bias_summary(a.variant, tag)


if __name__ == "__main__":
    main()
