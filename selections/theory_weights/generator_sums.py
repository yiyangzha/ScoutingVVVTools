#!/usr/bin/env python3
"""generator_sums.py -- theory-weight sums over all generated events, for theory_syst.py.

    python3 generator_sums.py [sample ...]     # one JSON per sample (default: every theory sample)
    python3 generator_sums.py --merge          # collect the existing ones into generator_sums_json

theory_syst.py normalises every theory variation member to its value over all generated events,
so the variations change only the selection efficiency (the samples are normalised to their
sample.json cross-sections; the inclusive cross-section uncertainty enters combine separately as
xsec_unc). The NanoAOD Runs tree of every input file stores, per member i,
LHEScaleSumw[i] / LHEPdfSumw[i] / PSSumw[i] = sum(genWeight * weight_i) / genEventSumw; this script
averages them over the sample's input files weighted by genEventSumw. The input files are the ones
of the conversion: the snapshot {input_root}/{sample_group}_tmp/{sample}.files written by
convert_branch (remote ROOT files; only their Runs trees are read, serially, with retries). The
arrays are mapped to the layout convert_branch.C stores per event (copyTheoryWeights): 101 PDF
members, the alpha_s members (source indices 101/102) separately, 9 scale entries (an 8-entry set
gets 1.0 at the nominal index 4), 4 PS entries, missing entries 1.0. A file that cannot be read
after the retries is skipped and listed (the sums are means).

Config: THEORY_CONFIG_PATH (selections/theory_weights/config.json): input_root, sample_config,
submit_samples, generator_sums_json.

Outputs:
  <dirname(generator_sums_json)>/generator_sums/<sample>.json   per sample
  generator_sums_json   {sample: {genEventSumw, n_files, n_files_used, skipped_files, n_lhe_scale,
                         n_lhe_pdf, n_ps, lhe_scale_sumw[9], lhe_pdf_sumw[101],
                         lhe_alphas_sumw[2], ps_sumw[4]}}
"""
import json
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_PDF, N_ALPHAS, N_SCALE, N_PS = 101, 2, 9, 4
OPEN_RETRIES = 5


def log(msg):
    print(msg, flush=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_cfg_path = os.environ.get("THEORY_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
if not os.path.isabs(_cfg_path):
    _cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _cfg_path))
cfg = _load_json(_cfg_path)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_SCRIPT_DIR, path))


INPUT_ROOT = _resolve(cfg["input_root"])
SAMPLE_INFO = {s["name"]: s for s in _load_json(_resolve(cfg["sample_config"]))["sample"]}
OUT_JSON = _resolve(cfg["generator_sums_json"])
SHARD_DIR = os.path.join(os.path.dirname(OUT_JSON), "generator_sums")
THEORY_SAMPLES = [n for n, s in SAMPLE_INFO.items() if s.get("has_theory_weights", False) and s.get("is_MC", True)]
if cfg.get("submit_samples"):
    THEORY_SAMPLES = [s for s in THEORY_SAMPLES if s in set(cfg["submit_samples"])]


def _mapped(values, n_target, scale=False):
    """Map one Runs array to convert_branch.C's stored per-event layout (missing entries 1.0)."""
    v = list(values)
    if scale and len(v) == N_SCALE - 1:
        v = v[:4] + [1.0] + v[4:]
    v = v[:n_target]
    return np.array(v + [1.0] * (n_target - len(v)), dtype=np.float64)


def _open(url):
    import ROOT
    for attempt in range(1, OPEN_RETRIES + 1):
        try:
            f = ROOT.TFile.Open(url)  # PyROOT raises OSError when the open fails
        except OSError:
            f = None
        if f and not f.IsZombie():
            return f
        log(f"    open failed ({attempt}/{OPEN_RETRIES}): {url}")
        time.sleep(5 * attempt)
    return None


def sample_sums(sample):
    info = SAMPLE_INFO[sample]
    group = "signal" if info.get("is_signal", False) else "bkg"
    listing = os.path.join(INPUT_ROOT, f"{group}_tmp", f"{sample}.files")
    if not os.path.exists(listing):
        raise FileNotFoundError(f"{listing} missing: the input-file snapshot of the conversion")
    files = [l.strip() for l in open(listing) if l.strip()]
    sumw = 0.0
    acc = {"scale": np.zeros(N_SCALE), "pdf": np.zeros(N_PDF), "alphas": np.zeros(N_ALPHAS), "ps": np.zeros(N_PS)}
    sizes = {"scale": set(), "pdf": set(), "ps": set()}
    skipped = []
    t0 = time.time()
    for k, url in enumerate(files):
        f = _open(url)
        runs = f.Get("Runs") if f else None
        if not runs:
            skipped.append(url)
            log(f"    WARNING skipped (unreadable or no Runs tree): {url}")
            if f:
                f.Close()
            continue
        for e in runs:
            w = float(e.genEventSumw)
            pdf = list(e.LHEPdfSumw)
            scale = list(e.LHEScaleSumw)
            ps = list(e.PSSumw)
            sizes["pdf"].add(len(pdf)); sizes["scale"].add(len(scale)); sizes["ps"].add(len(ps))
            sumw += w
            acc["scale"] += w * _mapped(scale, N_SCALE, scale=True)
            acc["pdf"] += w * _mapped(pdf, N_PDF)
            acc["alphas"] += w * np.array([pdf[N_PDF + i] if len(pdf) > N_PDF + i else 1.0 for i in range(N_ALPHAS)])
            acc["ps"] += w * _mapped(ps, N_PS)
        f.Close()
        if (k + 1) % 50 == 0:
            log(f"    {k + 1}/{len(files)} files, {time.time() - t0:.0f} s")
    for key, s in sizes.items():
        if len(s) > 1:
            raise RuntimeError(f"{sample}: the input files store different numbers of {key} weights: {sorted(s)}")
    if not (sumw != 0.0) or len(skipped) == len(files):
        raise RuntimeError(f"{sample}: no readable Runs tree with a nonzero genEventSumw")
    out = {"sample": sample, "genEventSumw": sumw, "n_files": len(files), "n_files_used": len(files) - len(skipped),
           "skipped_files": skipped, "n_lhe_scale": min(sizes["scale"]), "n_lhe_pdf": min(sizes["pdf"]),
           "n_ps": min(sizes["ps"]),
           "lhe_scale_sumw": list(acc["scale"] / sumw), "lhe_pdf_sumw": list(acc["pdf"] / sumw),
           "lhe_alphas_sumw": list(acc["alphas"] / sumw), "ps_sumw": list(acc["ps"] / sumw)}
    log(f"  {sample}: {out['n_files_used']}/{len(files)} files, n(scale/pdf/ps) = {out['n_lhe_scale']}/"
        f"{out['n_lhe_pdf']}/{out['n_ps']}, scale sums {np.round(acc['scale'] / sumw, 4).tolist()}, "
        f"ps sums {np.round(acc['ps'] / sumw, 4).tolist()}, {time.time() - t0:.0f} s")
    return out


def main():
    args = sys.argv[1:]
    if args == ["--merge"]:
        merged, missing = {}, []
        for s in THEORY_SAMPLES:
            p = os.path.join(SHARD_DIR, f"{s}.json")
            if os.path.exists(p):
                merged[s] = _load_json(p)
            else:
                missing.append(s)
        if missing:
            # theory_syst.py stops on any sample it processes without generator sums
            log(f"WARNING: no generator sums for {missing} (not converted, or their job failed)")
        with open(OUT_JSON, "w") as fh:
            json.dump(merged, fh, indent=2)
        log(f"Wrote {OUT_JSON} ({len(merged)} samples)")
        return
    os.makedirs(SHARD_DIR, exist_ok=True)
    for s in args or THEORY_SAMPLES:
        if s not in THEORY_SAMPLES:
            raise KeyError(f"{s} is not a theory-weight sample of the config")
        out = sample_sums(s)
        path = os.path.join(SHARD_DIR, f"{s}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        log(f"Wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log(f"Runtime error: {ex}")
        raise
