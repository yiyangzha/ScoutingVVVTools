#!/usr/bin/env python3
"""SR granularity study of one study variant: number of SRs x minimum background MC events.

    python3 sr_granularity.py submit [--variant shallow_cap20]   # write configs, submit one job per point
    python3 sr_granularity.py summary [--variant shallow_cap20]  # table + sr_granularity/<variant>/summary.json

Every grid point (N_SRS x MIN_BKG_ENTRIES) gets sr_granularity/<variant>/n<N>_minb<M>/ and one SLURM
job that runs, in sequence:
  1. signal_region_hist.py on the eval split (the variant's sr_config.json with n_signal_regions and
     min_bkg_entries changed),
  2. qcd_est.py on the test split in abcd_mode per_sr (the variant's qcd_est_per_sr.json with the
     B/D sideband as the complement of the SR union, robust for small SRs),
  3. combine.C for the combined scenario only (scenario_scopes), MC-true QCD and per-SR ABCD, with
     autoMCStats, the zero-count floors, the ABCD non-closure and lumi. The theory/pileup JSONs are left
     out: they belong to older SRs.
The combine binary is sr_granularity/combine_run (built from combine/combine.C).

    python3 sr_granularity.py abcd --tag <tag> [--variant ...] [--control low,high] [--nonclosure f] [--exclude-low-ht]
    python3 sr_granularity.py summary --abcd-tag <tag> [--variant ...]

abcd reruns only steps 2 and 3 on the SRs of every grid point, with the per-SR ABCD settings changed
(control window abcd_control_window, abcd_per_sr_nonclosure, exclude_samples = the HT < 400 GeV QCD
samples LOW_HT_QCD), into n<N>_minb<M>/abcd_<tag>/; summary --abcd-tag reads those outputs.

    python3 sr_granularity.py submit --grid-tag <tag> [--min-bkg 0,...] [--control ...] [--nonclosure f]
                                     [--exclude-low-ht] [--abcd-aware-scan]

submit --grid-tag writes the grid to sr_granularity/<variant>_<tag>/ with the same qcd_est overrides
as abcd (also given to the scan as exclude_samples), and --abcd-aware-scan adds the scan's
abcd_uncertainty (signal_region_hist.py: the SRs are chosen with the per-SR ABCD uncertainty of
QCD, control window and non-closure as for qcd_est.py); summary --grid-tag reads that grid.
"""
import argparse, csv, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_study import LOW_HT_QCD  # noqa: E402
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SR_DIR = os.path.join(REPO, "selections", "signal_region")
STUDY = os.path.join(HERE, "sr_granularity")
COMBINE_BIN = os.path.join(STUDY, "combine_run")
N_SRS = [1, 2, 3, 4]
# combine nuisances of the study (combine.C requires the input of every enabled nuisance, and the
# theory/pileup/jet-systematics JSONs belong to older SRs).
STUDY_NUISANCES = ["lumi", "autoMCStats", "zero_count_floor", "qcd_abcd_stat", "qcd_abcd_nonclosure"]
MIN_BKG_ENTRIES = [0, 30, 100, 300, 1000]


def variant_tree(variant):
    """The analysis tree of a study variant (its BDT config's submit_trees)."""
    return json.load(open(os.path.join(HERE, variant, "bdt_config.json")))["submit_trees"][0]


def grid_dir(variant, tag=""):
    return os.path.join(STUDY, f"{variant}_{tag}" if tag else variant)


def point_dir(variant, n, m, tag=""):
    return os.path.join(grid_dir(variant, tag), f"n{n}_minb{m}")


def write_configs(variant, n, m, tag="", sr_overrides=None, qe_overrides=None):
    vdir = os.path.join(HERE, variant)
    d = point_dir(variant, n, m, tag)
    os.makedirs(d, exist_ok=True)
    sr = json.load(open(os.path.join(vdir, "sr_config.json")))
    sr.update({"n_signal_regions": n, "min_bkg_entries": m, "output_dir": os.path.join(d, "sr")})
    sr.update(sr_overrides or {})
    json.dump(sr, open(os.path.join(d, "sr_config.json"), "w"), indent=2)
    write_abcd_configs(variant, d, d, qe_overrides or {})
    return d


def write_abcd_configs(variant, d, out, qe_overrides):
    """qcd_est.py and combine.C configs on the SRs of grid point d, with their outputs in out (d for
    the grid itself, d/abcd_<tag> for an ABCD variation) and qe_overrides applied to qcd_est.py's."""
    vdir = os.path.join(HERE, variant)
    tree = variant_tree(variant)
    os.makedirs(out, exist_ok=True)
    qe = json.load(open(os.path.join(vdir, "qcd_est_per_sr.json")))
    qe.update({"bdt_root": os.path.join(vdir, tree), "signal_region_csv": os.path.join(d, "sr", "signal_region.csv"),
               "output_dir": os.path.join(out, "qcd_est"), "abcd_mode": "per_sr", "abcd_sideband_mode": "complement"})
    qe.update(qe_overrides)
    json.dump(qe, open(os.path.join(out, "qcd_est.json"), "w"), indent=2)
    cb = json.load(open(os.path.join(REPO, "combine", "config.json")))
    for key in ("theory_syst_json", "pileup_syst_json", "jes_syst_json", "jer_syst_json"):
        cb.pop(key, None)
    cb.update({"channels": [{"name": tree, "root_file": os.path.join(out, "qcd_est", "qcd_abcd_yields.root"),
                             "bdt_root": os.path.join(vdir, tree)}],
               "output_dir": os.path.join(out, "combine"), "scenario_scopes": ["combined"], "enabled_nuisances": STUDY_NUISANCES})
    json.dump(cb, open(os.path.join(out, "combine.json"), "w"), indent=2)


def submit(variant, account, exclude, after="", tag="", min_bkg=None, sr_overrides=None, qe_overrides=None,
           mem="48G", time_limit="12:00:00"):
    if not os.path.exists(COMBINE_BIN):
        sys.exit(f"missing {COMBINE_BIN}: build it from combine/combine.C first")
    for n in N_SRS:
        for m in (min_bkg or MIN_BKG_ENTRIES):
            d = write_configs(variant, n, m, tag, sr_overrides, qe_overrides)
            cmd = (f"cd {SR_DIR} && python3 signal_region_hist.py {d}/sr_config.json"
                   f" && QCD_EST_CONFIG_PATH={d}/qcd_est.json python3 {REPO}/background_estimation/qcd_est.py"
                   f" && cd {d} && COMBINE_CONFIG_PATH={d}/combine.json {COMBINE_BIN}")
            args = ["sbatch", "--parsable", f"--job-name=srgran{'_' + tag if tag else ''}_n{n}_m{m}", f"--account={account}",
                    "--cpus-per-task=8", f"--mem={mem}", f"--time={time_limit}", f"--output={d}/job.log", f"--wrap={cmd}"]
            if exclude:
                args.insert(-1, f"--exclude={exclude}")
            if after:
                args.insert(-1, "--dependency=afterok:" + after.replace(",", ":"))
            job = subprocess.check_output(args, text=True).strip()
            print(f"n={n} min_bkg_entries={m}: job {job} -> {d}")


def submit_abcd(variant, tag, overrides, account, exclude):
    """Rerun qcd_est.py and combine.C on the SRs of every grid point with qcd_est overrides."""
    if not os.path.exists(COMBINE_BIN):
        sys.exit(f"missing {COMBINE_BIN}: build it from combine/combine.C first")
    for n in N_SRS:
        for m in MIN_BKG_ENTRIES:
            d = point_dir(variant, n, m)
            if not os.path.exists(os.path.join(d, "sr", "signal_region.csv")):
                print(f"n={n} min_bkg_entries={m}: no signal_region.csv, skipped")
                continue
            out = os.path.join(d, f"abcd_{tag}")
            write_abcd_configs(variant, d, out, overrides)
            cmd = (f"QCD_EST_CONFIG_PATH={out}/qcd_est.json python3 {REPO}/background_estimation/qcd_est.py"
                   f" && cd {out} && COMBINE_CONFIG_PATH={out}/combine.json {COMBINE_BIN}")
            args = ["sbatch", "--parsable", f"--job-name=srabcd_{tag}_n{n}_m{m}", f"--account={account}",
                    "--cpus-per-task=4", "--mem=48G", "--time=02:00:00", f"--output={out}/job.log", f"--wrap={cmd}"]
            if exclude:
                args.insert(-1, f"--exclude={exclude}")
            job = subprocess.check_output(args, text=True).strip()
            print(f"n={n} min_bkg_entries={m}: job {job} -> {out}")


def z_simple(s, b):
    return math.sqrt(2 * ((s + b) * math.log(1 + s / b) - s)) if b > 0 and s > 0 else 0.0


def z_sigma(s, b, sig2):
    """Asimov Z with background uncertainty (as signal_region.asimov_z)."""
    if s <= 0 or b <= 0:
        return 0.0
    if sig2 <= 0:
        return z_simple(s, b)
    t1 = (s + b) * math.log((s + b) * (b + sig2) / (b * b + (s + b) * sig2))
    t2 = (b * b / sig2) * math.log(1 + sig2 * s / (b * (b + sig2)))
    return math.sqrt(max(2 * (t1 - t2), 0.0))


def read_csv_value(path, column):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["scope"] == "combined":
                return float(row[column])
    return None


def summary(variant, tag="", grid_tag=""):
    import uproot
    rows = []
    for n in N_SRS:
        for m in MIN_BKG_ENTRIES:
            d = point_dir(variant, n, m, grid_tag)
            if not os.path.isdir(d):
                continue
            out = os.path.join(d, f"abcd_{tag}") if tag else d
            row = {"n_srs": n, "min_bkg_entries": m}
            sr_csv = os.path.join(d, "sr", "signal_region.csv")
            if os.path.exists(sr_csv):
                with open(sr_csv) as fh:
                    srs = list(csv.DictReader(fh))
                row["eval_z_sigma"] = math.sqrt(sum(z_sigma(float(r["S"]), float(r["B"]), float(r["B_err"]) ** 2) ** 2 for r in srs))
                row["eval_min_bkg_mc"] = min(int(float(r["B_entries"])) for r in srs)
            root = os.path.join(out, "qcd_est", "qcd_abcd_yields.root")
            if os.path.exists(root):
                f = uproot.open(root)
                ids = [int(x) for x in f["metadata/signal_regions"]["bin_index"].array(library="np")]
                groups = sorted({k.split("/")[1].split(";")[0] for k in f.keys() if k.startswith("groups/")})
                sig = [g for g in groups if g in ("vvv", "vh")]
                z2_simple = z2_sigma = 0.0
                min_mc = None
                for i in ids:
                    get = lambda g, h: float(f[f"groups/{g}/sr{i}/{h}"].values()[0])
                    s = sum(get(g, "yield") for g in sig)
                    b = sum(get(g, "yield") for g in groups if g not in sig)
                    s2 = sum(get(g, "stat_error") ** 2 for g in groups if g not in sig)
                    nmc = sum(get(g, "n_entries") for g in groups if g not in sig)
                    z2_simple += z_simple(s, b) ** 2
                    z2_sigma += z_sigma(s, b, s2) ** 2
                    min_mc = nmc if min_mc is None else min(min_mc, nmc)
                row.update(test_z_simple=math.sqrt(z2_simple), test_z_sigma=math.sqrt(z2_sigma),
                           test_min_bkg_mc=int(min_mc), abcd_tf=float(f["qcd_predict/abcd_tf"].values()[0]))
            cdir = os.path.join(out, "combine")
            row["combine_z_mc"] = read_csv_value(os.path.join(cdir, "significance.csv"), "significance")
            row["combine_z_abcd"] = read_csv_value(os.path.join(cdir, "significance_abcd_mc.csv"), "significance")
            row["combine_limit_mc"] = read_csv_value(os.path.join(cdir, "limits.csv"), "exp_50")
            row["combine_limit_abcd"] = read_csv_value(os.path.join(cdir, "limits_abcd_mc.csv"), "exp_50")
            rows.append(row)
    os.makedirs(grid_dir(variant, grid_tag), exist_ok=True)
    json.dump(rows, open(os.path.join(grid_dir(variant, grid_tag), f"summary_abcd_{tag}.json" if tag else "summary.json"), "w"), indent=1)
    fmt = lambda v, p=2: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    print(f"{'N':>2} {'minMC':>5} | {'eval Z':>6} {'MC':>4} | {'test Z':>6} {'Z(sB)':>6} {'MC':>4} | "
          f"{'comb Z(MC QCD)':>14} {'r<':>6} | {'comb Z(ABCD)':>12} {'r<':>6}")
    for r in rows:
        print(f"{r['n_srs']:>2} {r['min_bkg_entries']:>5} | {fmt(r.get('eval_z_sigma')):>6} {fmt(r.get('eval_min_bkg_mc')):>4} | "
              f"{fmt(r.get('test_z_simple')):>6} {fmt(r.get('test_z_sigma')):>6} {fmt(r.get('test_min_bkg_mc')):>4} | "
              f"{fmt(r.get('combine_z_mc')):>14} {fmt(r.get('combine_limit_mc'), 1):>6} | "
              f"{fmt(r.get('combine_z_abcd')):>12} {fmt(r.get('combine_limit_abcd'), 1):>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["submit", "summary", "abcd"])
    parser.add_argument("--variant", default="shallow_cap20")
    parser.add_argument("--account", default="cms-express")
    parser.add_argument("--exclude", default="hammer-f015")
    parser.add_argument("--after", default="", help="job ids (comma-separated) the grid jobs wait for")
    parser.add_argument("--tag", default="", help="abcd: name of the ABCD variation (output abcd_<tag>/)")
    parser.add_argument("--abcd-tag", default="", help="summary: read the outputs of this ABCD variation")
    parser.add_argument("--control", default="", help="abcd: control window low,high of msoftdrop_1")
    parser.add_argument("--nonclosure", type=float, default=None, help="abcd: abcd_per_sr_nonclosure")
    parser.add_argument("--exclude-low-ht", action="store_true", help="abcd/submit: exclude_samples = LOW_HT_QCD")
    parser.add_argument("--grid-tag", default="", help="submit/summary: grid in sr_granularity/<variant>_<tag>/")
    parser.add_argument("--min-bkg", default="", help="submit: min_bkg_entries values (comma-separated)")
    parser.add_argument("--mem", default="48G", help="submit: memory of the grid jobs")
    parser.add_argument("--time", default="12:00:00", help="submit: time limit of the grid jobs")
    parser.add_argument("--abcd-aware-scan", action="store_true",
                        help="submit: SR scan with abcd_uncertainty (needs --control and --nonclosure)")
    a = parser.parse_args()
    overrides = {}
    tree = variant_tree(a.variant)
    if a.control:
        overrides["abcd_control_window"] = {tree: [float(x) for x in a.control.split(",")]}
    if a.nonclosure is not None:
        overrides["abcd_per_sr_nonclosure"] = {tree: a.nonclosure}
    if a.exclude_low_ht:
        overrides["exclude_samples"] = LOW_HT_QCD
    if a.action == "submit":
        sr_overrides = {"exclude_samples": LOW_HT_QCD} if a.exclude_low_ht else {}
        if a.abcd_aware_scan:
            if not a.control or a.nonclosure is None:
                sys.exit("--abcd-aware-scan needs --control and --nonclosure")
            sr_overrides["abcd_uncertainty"] = {"branch": "ScoutingFatPFJetRecluster_msoftdrop_1",
                                                "control_window": [float(x) for x in a.control.split(",")],
                                                "nonclosure": a.nonclosure}
        if (overrides or sr_overrides) and not a.grid_tag:
            sys.exit("submit with overrides needs --grid-tag (the default grid keeps the variant's configs)")
        min_bkg = [int(x) for x in a.min_bkg.split(",")] if a.min_bkg else None
        submit(a.variant, a.account, a.exclude, a.after, a.grid_tag, min_bkg, sr_overrides, overrides, a.mem, a.time)
    elif a.action == "summary":
        summary(a.variant, a.abcd_tag, a.grid_tag)
    else:
        if not a.tag:
            sys.exit("abcd needs --tag")
        submit_abcd(a.variant, a.tag, overrides, a.account, a.exclude)
