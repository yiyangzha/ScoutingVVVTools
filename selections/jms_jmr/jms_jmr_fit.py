#!/usr/bin/env python3
"""
Tag-and-probe JMS / JMR fit of the hadronic-W peak in the scouting AK8 soft-drop mass
(DP-2025-052 style, reco-level).

The probe jet is the leading AK8 OPPOSITE the muon (built in the convert as `probe`), in
semileptonic-ttbar events selected with a tight muon, a b-tag, and pT(leptonic W) > 150 GeV.
Events are split by the ParticleNet-style W-tagger on the probe:
    Pass = Probe_WvsQCD > tag_wp   (W-enriched)
    Fail = the rest                (background-enriched)
The W-peak position and width are fit in the Pass region for DATA and MC; the corrections are
    JMS = mu_data / mu_MC          JMR = sigma_data / sigma_MC

NOTE: gen-matched templates (Top-/W-/Non-merged, as in DP-2025-052) are NOT reproducible here
because the scouting MC has no GenPart collection — the W enrichment comes from the tagger.

Run:  pixi run python selections/jms_jmr/jms_jmr_fit.py
Requires the convert to have been re-skimmed with the probe branches.
"""
import os, json, argparse
import numpy as np
import uproot
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(p, base):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


# -------------------- fit model: Gaussian signal + exponential background --------------------
def _bkg(m, B, C, m0):
    return B * np.exp(-C * (m - m0))


def _model(m, A, mu, sigma, B, C, m0):
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + _bkg(m, B, C, m0)


def _fit_peak(centers, counts, errors, window, label):
    m0 = window[0]
    peak = max(counts.max(), 1.0)
    p0 = [peak, 80.0, 9.0, max(counts[0], 1.0), 0.01, m0]
    lo = [0.0, 65.0, 3.0, 0.0, -0.05, m0 - 1e-6]
    hi = [10 * peak + 10, 100.0, 25.0, 1e9, 0.2, m0 + 1e-6]
    sig = np.where(errors > 0, errors, 1.0)
    popt, pcov = curve_fit(_model, centers, counts, p0=p0, sigma=sig,
                           absolute_sigma=True, bounds=(lo, hi), maxfev=300000)
    perr = np.sqrt(np.diag(pcov))
    fit = _model(centers, *popt)
    chi2 = float(np.sum(((counts - fit) / sig) ** 2))
    return {"mu": float(popt[1]), "mu_err": float(perr[1]),
            "sigma": float(abs(popt[2])), "sigma_err": float(perr[2]),
            "B": float(popt[3]), "C": float(popt[4]), "m0": m0,
            "chi2": chi2, "ndf": max(len(centers) - 5, 1),
            "popt": [float(x) for x in popt], "label": label}


def _mask(arrs, thresholds):
    n = len(next(iter(arrs.values())))
    m = np.ones(n, dtype=bool)
    for br, cond in thresholds.items():
        if br not in arrs:
            continue
        col = np.asarray(arrs[br], dtype=float)
        m &= ~(col < -990)
        lo, hi = (cond if isinstance(cond, (list, tuple)) else (cond, None))
        if lo is not None:
            m &= col > lo
        if hi is not None:
            m &= col < hi
    return m


def main():
    ap = argparse.ArgumentParser(description="Tag-and-probe JMS/JMR W-peak fit")
    ap.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    plot_cfg = json.load(open(_resolve(cfg["samples_from"], _HERE)))
    info = {s["name"]: s for s in json.load(open(_resolve(cfg["sample_config"], _HERE)))["sample"]}
    tree = cfg["tree"]
    mass_br, tag_br, w_br = cfg["mass_branch"], cfg["tag_branch"], cfg["weight_branch"]
    wp = float(cfg["tag_wp"])
    window = tuple(cfg["fit_window"]); nb = int(cfg["n_bins"])
    sel = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["selection"].items()}
    input_root = _resolve(cfg["input_root"], _HERE)
    out_dir = _resolve(cfg["output_dir"], _HERE); os.makedirs(out_dir, exist_ok=True)

    mc_samples = [s for v in plot_cfg["class_groups"].values() for s in v]
    data_samples = list(plot_cfg["data_samples"])
    lumi = sum(float(info[s].get("lumi", 0.0)) for s in data_samples)

    edges = np.linspace(window[0], window[1], nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    read = sorted(set(list(sel.keys()) + [mass_br, tag_br]))

    def _file(s):
        return os.path.join(input_root, "data" if not info[s].get("is_MC") else "bkg", f"{s}.root")

    def _accumulate(samples, is_mc):
        # returns pass/fail histograms (sum-w, sum-w2)
        hp = np.zeros(nb); hp2 = np.zeros(nb); hf = np.zeros(nb); hf2 = np.zeros(nb)
        for s in samples:
            f = _file(s)
            if not os.path.exists(f):
                print(f"[WARN] missing {f}"); continue
            cols = read + ([w_br] if is_mc else [])
            a = uproot.open(f)[tree].arrays(cols, library="np")
            m = _mask(a, sel)
            if is_mc:
                xs = float(info[s].get("xsection", 0)); raw = float(info[s].get("raw_entries", 0))
                if xs <= 0 or raw <= 0:
                    continue
                wgt = (np.asarray(a[w_br], float) if w_br in a else np.ones(len(m)))[m] * (lumi * xs / raw)
            else:
                wgt = np.ones(int(m.sum()))
            mass = np.asarray(a[mass_br])[m]; tag = np.asarray(a[tag_br])[m]
            passm = tag > wp
            hp += np.histogram(mass[passm], bins=edges, weights=wgt[passm])[0]
            hp2 += np.histogram(mass[passm], bins=edges, weights=wgt[passm] ** 2)[0]
            hf += np.histogram(mass[~passm], bins=edges, weights=wgt[~passm])[0]
            hf2 += np.histogram(mass[~passm], bins=edges, weights=wgt[~passm] ** 2)[0]
        return hp, np.sqrt(hp2), hf, np.sqrt(hf2)

    d_pass, d_pass_e, d_fail, d_fail_e = _accumulate(data_samples, False)
    m_pass, m_pass_e, m_fail, m_fail_e = _accumulate(mc_samples, True)

    print(f"Lumi={lumi:.2f}/fb  tag WP(Probe_WvsQCD)>{wp}  window={window} bins={nb}")
    print(f"  PASS: data={d_pass.sum():.0f}  MC={m_pass.sum():.1f}")
    print(f"  FAIL: data={d_fail.sum():.0f}  MC={m_fail.sum():.1f}")

    # fit the W peak in the PASS region
    data_fit = _fit_peak(centers, d_pass, d_pass_e, window, "Data (Pass)")
    mc_fit = _fit_peak(centers, m_pass, m_pass_e, window, "MC (Pass)")

    md, sd, md_e, sd_e = data_fit["mu"], data_fit["sigma"], data_fit["mu_err"], data_fit["sigma_err"]
    mm, sm, mm_e, sm_e = mc_fit["mu"], mc_fit["sigma"], mc_fit["mu_err"], mc_fit["sigma_err"]
    jms = md / mm; jms_e = jms * np.hypot(md_e / md, mm_e / mm)
    jmr = sd / sm; jmr_e = jmr * np.hypot(sd_e / sd, sm_e / sm)

    results = {"lumi_fb": lumi, "tag_branch": tag_br, "tag_wp": wp,
               "fit_window": list(window), "n_bins": nb, "w_mass_ref": cfg["w_mass_ref"],
               "pass": {"data": data_fit, "mc": mc_fit},
               "yields": {"data_pass": float(d_pass.sum()), "mc_pass": float(m_pass.sum()),
                          "data_fail": float(d_fail.sum()), "mc_fail": float(m_fail.sum())},
               "JMS": jms, "JMS_err": jms_e, "JMR": jmr, "JMR_err": jmr_e,
               "selection": {k: list(v) if isinstance(v, tuple) else v for k, v in sel.items()}}
    json.dump(results, open(os.path.join(out_dir, "jms_jmr_results.json"), "w"), indent=2)

    print("\n=========  JMS / JMR  (tag-and-probe, Pass region)  =========")
    print(f"  Data:  mu={md:6.2f}+/-{md_e:4.2f}  sigma={sd:5.2f}+/-{sd_e:4.2f}  chi2/ndf={data_fit['chi2']:.1f}/{data_fit['ndf']}")
    print(f"  MC:    mu={mm:6.2f}+/-{mm_e:4.2f}  sigma={sm:5.2f}+/-{sm_e:4.2f}  chi2/ndf={mc_fit['chi2']:.1f}/{mc_fit['ndf']}")
    print(f"  JMS = {jms:.4f} +/- {jms_e:.4f}")
    print(f"  JMR = {jmr:.4f} +/- {jmr_e:.4f}")
    print("=============================================================")

    # ---- plot: 2x2 (data/MC x pass/fail), with the Pass fits ----
    xf = np.linspace(window[0], window[1], 400)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    panels = [
        (ax[0, 0], d_pass, d_pass_e, data_fit, "black", "Data — Pass (W-tagged)"),
        (ax[0, 1], m_pass, m_pass_e, mc_fit, "#5790fc", "MC — Pass (W-tagged)"),
        (ax[1, 0], d_fail, d_fail_e, None, "black", "Data — Fail"),
        (ax[1, 1], m_fail, m_fail_e, None, "#5790fc", "MC — Fail"),
    ]
    for a, h, e, fit, col, lab in panels:
        a.errorbar(centers, h, yerr=e, fmt="o", color=col, ms=4)
        if fit is not None:
            a.plot(xf, _model(xf, *fit["popt"]), "-", color="#e42536", lw=2, label="S+B fit")
            a.plot(xf, _bkg(xf, fit["B"], fit["C"], fit["m0"]), "--", color="gray", lw=1.3, label="bkg")
            a.axvline(fit["mu"], color="#e42536", ls=":", lw=1)
            a.text(0.04, 0.95, f"$\\mu$={fit['mu']:.2f}$\\pm${fit['mu_err']:.2f}\n"
                              f"$\\sigma$={fit['sigma']:.2f}$\\pm${fit['sigma_err']:.2f}",
                   transform=a.transAxes, va="top", fontsize=12,
                   bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
            a.legend(loc="upper right", fontsize=11)
        a.set_title(lab, fontsize=13); a.set_ylim(bottom=0)
    ax[1, 0].set_xlabel(r"Probe $m_{\mathrm{SD}}$ [GeV]")
    ax[1, 1].set_xlabel(r"Probe $m_{\mathrm{SD}}$ [GeV]")
    fig.suptitle(f"JMS = {jms:.3f} $\\pm$ {jms_e:.3f}      JMR = {jmr:.3f} $\\pm$ {jmr_e:.3f}",
                 fontsize=15, y=0.99)
    hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=round(lumi, 1), ax=ax[0, 0])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "jms_jmr_W_tagprobe.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=120)
    print(f"\nSaved: {out}")
    print(f"Saved: {os.path.join(out_dir, 'jms_jmr_results.json')}")


if __name__ == "__main__":
    main()
