#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversion QA + recovery tool.

Inspects the output of the conversion step (run.py mode 0 / convert_branch) and
decides, per sample, whether everything finished correctly by cross-checking three
sources of truth:

  1. SLURM merge logs   (selections/convert/convert_branch_<sample>_merge_*.out)
  2. temp batch files   (dataset/<group>_tmp/<sample>_<i>.root  + .raw_entries sidecar)
  3. merged ROOT files  (dataset/<group>/<sample>.root  [+ split siblings <sample>_<i>.root])

For anything broken it prepares the cheapest sufficient recovery, written to a recovery
shell script:

  * re-merge          all temps present & valid, only the merged output is bad
  * resubmit batches  some temp batches missing/corrupt but most are fine — re-run ONLY
                      the failed batch indices (one SLURM job each) + a dependent re-merge,
                      mirroring run.py's mode-0 SLURM layout. Avoids reprocessing a whole
                      sample for a handful of missing files.
  * reprocess         temps unusable / count unknown / schema stale — full run.py mode 0

Generate-only by default; pass --submit to actually launch them.

The partial-resubmit path needs the batch index -> file mapping to match the original
conversion, which depends on files-per-job (CONVERT_FILES_PER_BATCH).  Use --files-per-job
to match whatever run.py used (default 50, same as run.py --slurm-files-per-job).  By
default the chosen value is verified against the batch count the merge log recorded; on a
mismatch that sample falls back to a full reprocess.

Run inside the CMSSW environment (needs `uproot`).  Read-only unless --submit is given.

Examples:
  python3 check_conversion.py --data-only
  python3 check_conversion.py                 # all samples
  python3 check_conversion.py --samples data_2024 wwz
  python3 check_conversion.py --submit        # also launch the recovery jobs
  python3 check_conversion.py --files-per-job 50 --submit   # match run.py's batch size
"""

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time

import uproot

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))           # selections/convert
_ROOT_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))  # repo root (has run.py)


# ----------------------------------------------------------------------------- config
def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def load_configs():
    cfg = _load_json(os.path.join(_SCRIPT_DIR, "config.json"))
    branch_cfg = _load_json(os.path.join(_SCRIPT_DIR, "branch.json"))
    sample_path = _resolve(cfg.get("sample_config", "../../src/sample.json"), _SCRIPT_DIR)
    sample_cfg = _load_json(sample_path)
    cfg["_output_root_abs"] = _resolve(cfg["output_root"], _SCRIPT_DIR)
    cfg["_sample_path"] = sample_path
    return cfg, branch_cfg, sample_cfg


def sample_group(info):
    if not info.get("is_MC", True):
        return "data"
    return "signal" if info.get("is_signal", False) else "bkg"


def expected_trees(branch_cfg):
    return [t["name"] for t in branch_cfg.get("output", {}).get("trees", [])]


def expected_branches(branch_cfg, tree_name, is_mc):
    """Branch names convert_branch writes for one tree (slot-expanded, onlyMC-aware)."""
    out, seen = [], set()
    for tree in branch_cfg.get("output", {}).get("trees", []):
        if tree["name"] != tree_name:
            continue
        scalars = tree.get("scalars", {})
        for entry in list(scalars.get("regular", [])) + list(scalars.get("extrema", [])):
            if entry.get("onlyMC", False) and not is_mc:
                continue
            name = entry["name"]
            slots = entry.get("slots")
            names = [f"{name}_{i + 1}" for i in range(int(slots))] if slots else [name]
            for n in names:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
    return out


# ------------------------------------------------------------------- ROOT inspection
def root_health(path, trees):
    """Return (ok, info) for a merged/temp ROOT file. info has per-tree entries + branch set."""
    info = {"entries": {}, "branches": set(), "error": None}
    try:
        with uproot.open(path) as uf:
            keys = set(k.split(";")[0] for k in uf.keys())
            for tn in trees:
                if tn not in keys:
                    info["error"] = f"missing tree {tn}"
                    return False, info
                t = uf[tn]
                info["entries"][tn] = int(t.num_entries)
                info["branches"].update(t.keys())
        return True, info
    except Exception as exc:  # zombie / unreadable / no keys
        info["error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        return False, info


def temp_batch_valid(tmp_dir, sample, idx, trees):
    """Replicate convert_branch validateBatchTempOutput for one batch index."""
    root_path = os.path.join(tmp_dir, f"{sample}_{idx}.root")
    if not os.path.exists(root_path):
        return False, "missing ROOT output"
    if not os.path.exists(root_path + ".raw_entries"):
        return False, "missing raw_entries"
    try:
        with open(root_path + ".raw_entries") as fh:
            int(fh.read().strip())
    except Exception:
        return False, "bad raw_entries"
    ok, hinfo = root_health(root_path, trees)
    return (ok, None if ok else hinfo["error"])


# -------------------------------------------------------------------------- log parse
_CRASH_RE = re.compile(r"Bus error|segmentation|\*\*\* Break \*\*\*|core dumped|There was a crash",
                       re.IGNORECASE)


def newest(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def parse_merge_log(sample):
    """Parse the newest merge .out for this sample.

    Returns dict: path, n_batches, n_merged, skipped(set of 0-based idx), wrote_output,
    no_successful, crash.
    """
    logs = glob.glob(os.path.join(_SCRIPT_DIR, f"convert_branch_{sample}_merge_*.out"))
    log = newest(logs)
    res = dict(path=log, n_batches=None, n_merged=None, skipped=set(),
               wrote_output=False, no_successful=False, crash=False)
    if not log:
        return res
    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if _CRASH_RE.search(text):
        res["crash"] = True
    if "No successful temporary batch outputs found" in text:
        res["no_successful"] = True
    if "Wrote output file" in text:
        res["wrote_output"] = True
    m = re.search(r"Merging\s+(\d+)\s+successful temporary batch file[s]?\s+out of\s+(\d+)", text)
    if m:
        res["n_merged"], res["n_batches"] = int(m.group(1)), int(m.group(2))
    for mm in re.finditer(r"skipping incomplete batch\s+(\d+)/(\d+)", text):
        res["skipped"].add(int(mm.group(1)) - 1)  # log is 1-based
        res["n_batches"] = int(mm.group(2))
    return res


def query_batch_count(sample, cfg, files_per_job=None):
    """convert_branch <sample> --batch-count (needs proxy/DAS).

    files_per_job sets CONVERT_FILES_PER_BATCH so the reported count matches the
    SLURM layout produced by run.py (whose batch size is --slurm-files-per-job).
    Without it convert_branch falls back to its threads*32 default, which does NOT
    match the SLURM batch indices.
    """
    binp = os.path.join(_SCRIPT_DIR, "convert_branch")
    if not os.path.exists(binp):
        return None
    env = {**os.environ, "CONVERT_CONFIG_PATH": os.path.join(_SCRIPT_DIR, "config.json")}
    if files_per_job:
        env["CONVERT_FILES_PER_BATCH"] = str(files_per_job)
    try:
        r = subprocess.run([binp, sample, "--batch-count"], env=env, cwd=_SCRIPT_DIR,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        out = r.stdout.decode().strip().splitlines()
        return int(out[-1]) if out and out[-1].strip().isdigit() else None
    except Exception:
        return None


def find_golden_json():
    """Path to Cert_*_Golden.json in the repo root, or None (data quality selection)."""
    cands = sorted(glob.glob(os.path.join(_ROOT_DIR, "Cert_*_Golden.json")))
    return cands[0] if cands else None


def root_libdir():
    """`root-config --libdir`, prepended to LD_LIBRARY_PATH in SLURM wraps (mirrors run.py)."""
    try:
        r = subprocess.run(["root-config", "--libdir"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return r.stdout.decode().strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------- per sample
def inspect_sample(name, cfg, branch_cfg, sample_cfg_map, trees, batch_count_fallback,
                   files_per_job=None, allow_partial=True):
    info = sample_cfg_map[name]
    is_mc = bool(info.get("is_MC", True))
    group = sample_group(info)
    out_dir = os.path.join(cfg["_output_root_abs"], group)
    tmp_dir = os.path.join(cfg["_output_root_abs"], f"{group}_tmp")

    r = dict(name=name, group=group, is_mc=is_mc, status=None, detail=[],
             n_batches=None, valid_temps=0, missing_batches=[], merged_files=[],
             merged_entries={}, missing_branches=[], recovery=None)

    merge = parse_merge_log(name)
    n = merge["n_batches"]
    if n is None and batch_count_fallback:
        n = query_batch_count(name, cfg, files_per_job)
    r["n_batches"] = n

    # --- temp batches ---
    if n is not None:
        valid_idx, missing_idx = [], []
        for i in range(n):
            ok, _reason = temp_batch_valid(tmp_dir, name, i, trees)
            (valid_idx if ok else missing_idx).append(i)
        r["valid_temps"] = len(valid_idx)
        r["missing_batches"] = missing_idx
    temps_complete = (n is not None and r["valid_temps"] == n and n > 0)

    # --- merged output(s) ---
    base = os.path.join(out_dir, f"{name}.root")
    split = sorted(glob.glob(os.path.join(out_dir, f"{name}_[0-9]*.root")))
    merged_paths = ([base] if os.path.exists(base) else []) + split
    r["merged_files"] = merged_paths
    exp_branches = {tn: set(expected_branches(branch_cfg, tn, is_mc)) for tn in trees}
    merged_ok = bool(merged_paths)
    schema_ok = True
    if not merged_paths:
        r["detail"].append("no merged output file")
    for p in merged_paths:
        ok, hinfo = root_health(p, trees)
        if not ok:
            merged_ok = False
            r["detail"].append(f"{os.path.basename(p)}: {hinfo['error']}")
            continue
        for tn, ent in hinfo["entries"].items():
            r["merged_entries"][tn] = r["merged_entries"].get(tn, 0) + ent
        miss = sorted(set().union(*exp_branches.values()) - hinfo["branches"])
        if miss:
            schema_ok = False
            r["missing_branches"] = miss[:12]

    # --- classify ---
    if merge["no_successful"]:
        r["status"] = "TOTAL_FAILURE"
    elif not merged_paths:
        r["status"] = "OUTPUT_MISSING"
    elif not merged_ok:
        r["status"] = "MERGE_CORRUPT"
        if merge["crash"]:
            r["detail"].append("merge log shows a crash (Bus error / core dumped)")
    elif not schema_ok:
        r["status"] = "STALE_SCHEMA"
    elif merge["n_merged"] is not None and merge["n_batches"] is not None \
            and merge["n_merged"] < merge["n_batches"]:
        r["status"] = "INCOMPLETE"
        r["detail"].append(f"merge used {merge['n_merged']}/{merge['n_batches']} batches "
                           f"({merge['n_batches'] - merge['n_merged']} dropped)")
    else:
        r["status"] = "OK"

    # --- recovery action ---
    # Pick the cheapest sufficient action:
    #   remerge   - every temp is valid, only the merged file is bad
    #   resubmit  - a strict, non-empty subset of batches is missing/corrupt; re-run just
    #               those indices + re-merge (the merge re-validates all temps and keeps
    #               the good ones). STALE_SCHEMA is excluded: its *present* temps are stale
    #               too, so re-running only the missing ones would not refresh them.
    #   reprocess - temps unusable, count unknown, or schema stale -> full run.py mode 0
    if r["status"] != "OK":
        missing = r["missing_batches"]
        partial_ok = (allow_partial and n is not None and n > 0
                      and 0 < len(missing) < n and r["status"] != "STALE_SCHEMA")
        if temps_complete and r["status"] in ("MERGE_CORRUPT", "STALE_SCHEMA"):
            r["recovery"] = ("remerge", name)
        elif partial_ok:
            r["recovery"] = ("resubmit", name, list(missing))
        else:
            r["recovery"] = ("reprocess", name)
    return r


# --------------------------------------------------------------- SLURM resubmit helpers
# These mirror run.py's launch_mode0_slurm so resubmitted batches are produced identically
# to the original conversion: same convert_branch binary, config, batch size, golden JSON,
# X509 proxy and LD_LIBRARY_PATH.

def _wrap_prefix(ctx, want_proxy):
    """Shell prefix exported inside the sbatch --wrap (LD_LIBRARY_PATH + optional proxy)."""
    p = ""
    if ctx["libdir"]:
        p += f"export LD_LIBRARY_PATH={ctx['libdir']}:${{LD_LIBRARY_PATH:-}}; "
    if want_proxy and ctx["x509_dst"]:
        p += f"export X509_USER_PROXY={ctx['x509_dst']}; "
    return p


def _env_str(ctx, golden):
    s = f"env CONVERT_CONFIG_PATH={ctx['config']} CONVERT_FILES_PER_BATCH={ctx['files_per_job']}"
    if golden:
        s += f" CONVERT_GOLDEN_JSON={golden}"
    return s


def _batch_wrap(ctx, name, idx, golden):
    return (f"{_wrap_prefix(ctx, want_proxy=bool(golden))}{_env_str(ctx, golden)} "
            f"CONVERT_DEFER_FINAL_MERGE=1 {ctx['bin']} {name} {idx}")


def _merge_wrap(ctx, name, golden):
    return (f"{_wrap_prefix(ctx, want_proxy=bool(golden))}{_env_str(ctx, golden)} "
            f"{ctx['bin']} {name} --merge-successful-batches")


def _sbatch_argv(ctx, job_name, wrap, depends_csv=None):
    a = ["sbatch", "--parsable",
         f"--job-name={job_name}",
         f"--output={ctx['workdir']}/{job_name}_%j.out",
         f"--error={ctx['workdir']}/{job_name}_%j.out",
         f"--cpus-per-task={ctx['cpus']}",
         f"--mem={ctx['mem']}",
         f"--time={ctx['time']}"]
    if ctx["partition"]:
        a.append(f"--account={ctx['partition']}")
    if depends_csv:
        a.append(f"--dependency=afterany:{depends_csv}")
    if ctx["extra"]:
        a.extend(ctx["extra"].split())
    a.append(f"--wrap={wrap}")
    return a


def _render_sbatch_line(ctx, job_name, wrap, capture_var=None, depends_expr=None):
    """Render an sbatch invocation as a bash line for the recovery script.

    The --wrap value is single-quoted so its `${LD_LIBRARY_PATH:-}` survives the
    submitting shell and is expanded at job runtime; --dependency uses depends_expr
    verbatim (e.g. "$deps") so it DOES expand at submit time.
    """
    parts = ["sbatch", "--parsable",
             shlex.quote(f"--job-name={job_name}"),
             shlex.quote(f"--output={ctx['workdir']}/{job_name}_%j.out"),
             shlex.quote(f"--error={ctx['workdir']}/{job_name}_%j.out"),
             shlex.quote(f"--cpus-per-task={ctx['cpus']}"),
             shlex.quote(f"--mem={ctx['mem']}"),
             shlex.quote(f"--time={ctx['time']}")]
    if ctx["partition"]:
        parts.append(shlex.quote(f"--account={ctx['partition']}"))
    if depends_expr:
        parts.append(f"--dependency=afterany:{depends_expr}")
    if ctx["extra"]:
        parts.extend(shlex.quote(t) for t in ctx["extra"].split())
    parts.append(shlex.quote(f"--wrap={wrap}"))
    cmd = " ".join(parts)
    return f"{capture_var}=$({cmd})" if capture_var else cmd


def resubmit_script_lines(ctx, r):
    """Bash lines that re-run the failed batch indices of one sample, then re-merge."""
    name = r["recovery"][1]
    idxs = r["recovery"][2]
    golden = ctx["golden"] if not r["is_mc"] else None
    lines = [f"# {name}: {r['status']} — resubmit {len(idxs)}/{r['n_batches']} failed "
             f"batch(es): {', '.join(map(str, idxs))}; then re-merge",
             "deps=\"\""]
    for idx in idxs:
        jn = f"convert_branch_{name}_{idx}"
        lines.append(_render_sbatch_line(ctx, jn, _batch_wrap(ctx, name, idx, golden),
                                         capture_var="jid"))
        lines.append('deps="${deps:+$deps:}$jid"')
    lines.append(_render_sbatch_line(ctx, f"convert_branch_{name}_merge",
                                     _merge_wrap(ctx, name, golden),
                                     depends_expr='"$deps"'))
    lines.append("")
    return lines


def resubmit_live(ctx, r):
    """Submit the failed batches + dependent merge directly via sbatch."""
    name = r["recovery"][1]
    idxs = r["recovery"][2]
    golden = ctx["golden"] if not r["is_mc"] else None
    batch_ids = []
    for idx in idxs:
        jn = f"convert_branch_{name}_{idx}"
        res = subprocess.run(_sbatch_argv(ctx, jn, _batch_wrap(ctx, name, idx, golden)),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=_SCRIPT_DIR)
        if res.returncode != 0:
            print(f"    sbatch failed for {jn}: {res.stderr.decode().strip()}")
            return False
        jid = res.stdout.decode().strip().split()[-1]
        batch_ids.append(jid)
        print(f"    batch {idx} -> slurm_job_id={jid}")
    res = subprocess.run(_sbatch_argv(ctx, f"convert_branch_{name}_merge",
                                      _merge_wrap(ctx, name, golden),
                                      depends_csv=":".join(batch_ids)),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=_SCRIPT_DIR)
    if res.returncode != 0:
        print(f"    sbatch failed for {name} merge: {res.stderr.decode().strip()}")
        return False
    print(f"    merge -> slurm_job_id={res.stdout.decode().strip().split()[-1]} "
          f"(afterany:{':'.join(batch_ids)})")
    return True


def prepare_x509():
    """Copy the grid proxy where SLURM jobs can read it; return dst path or None.

    Mirrors run.py: /tmp/x509up_u<uid> -> /depot/cms/users/<user>/x509up_u<uid>.
    """
    uid = os.getuid()
    src = f"/tmp/x509up_u{uid}"
    dst = f"/depot/cms/users/{os.environ.get('USER', '')}/x509up_u{uid}"
    if not os.path.exists(src):
        return None, src, dst
    try:
        import shutil
        shutil.copy2(src, dst)
        return dst, src, dst
    except OSError:
        return None, src, dst


# ---------------------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description="Conversion QA + recovery preparation")
    p.add_argument("--samples", nargs="*", default=None, help="restrict to these sample names")
    p.add_argument("--group", choices=["data", "signal", "bkg"], default=None)
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--batch-count-fallback", action="store_true",
                   help="query convert_branch --batch-count when no merge log (needs proxy/DAS)")
    p.add_argument("--submit", action="store_true", help="launch the prepared recovery jobs")
    p.add_argument("--json", default=None, help="write the full report as JSON to this path")
    # partial-resubmit controls
    p.add_argument("--files-per-job", type=int, default=50, metavar="N",
                   help="batch size for resubmitted batches; must match run.py "
                        "--slurm-files-per-job used originally (default 50)")
    p.add_argument("--no-partial", action="store_true",
                   help="disable per-batch resubmit; always full-reprocess broken samples")
    p.add_argument("--skip-batch-verify", action="store_true",
                   help="skip the batch-count cross-check for resubmit candidates "
                        "(verification needs proxy/DAS; on mismatch a sample is reprocessed)")
    # SLURM resources for resubmitted batches (mirror run.py defaults)
    p.add_argument("--slurm-partition", default="cms-express", metavar="NAME")
    p.add_argument("--slurm-time", default="24:00:00", metavar="HH:MM:SS")
    p.add_argument("--slurm-mem", default="4G", metavar="MEM")
    p.add_argument("--slurm-cpus", type=int, default=1, metavar="N")
    p.add_argument("--slurm-extra", default="", metavar="ARGS",
                   help="extra sbatch arguments (space-separated)")
    args = p.parse_args()

    cfg, branch_cfg, sample_cfg = load_configs()
    sample_map = {s["name"]: s for s in sample_cfg["sample"]}
    trees = expected_trees(branch_cfg)

    samples = [s["name"] for s in sample_cfg["sample"]]
    if args.data_only:
        samples = [s for s in samples if not sample_map[s].get("is_MC", True)]
    if args.group:
        samples = [s for s in samples if sample_group(sample_map[s]) == args.group]
    if args.samples:
        unknown = [s for s in args.samples if s not in sample_map]
        if unknown:
            sys.exit(f"Unknown sample(s): {', '.join(unknown)}")
        samples = list(args.samples)

    if cfg.get("max_output_file_size_gb", 0) in (0, None):
        print("[WARN] max_output_file_size_gb=0 in config.json: large-sample merges may crash "
              "(single huge file). Recommend a split size (e.g. 5) before recovering big samples.\n")

    results = [inspect_sample(s, cfg, branch_cfg, sample_map, trees, args.batch_count_fallback,
                              files_per_job=args.files_per_job,
                              allow_partial=not args.no_partial)
               for s in samples]

    # ---- report ----
    print(f"{'sample':28s} {'group':6s} {'status':14s} {'batches':>9s} {'merged entries':>22s}  notes")
    print("-" * 110)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        ent = ", ".join(f"{tn}={r['merged_entries'][tn]}" for tn in r["merged_entries"]) or "-"
        nb = "-" if r["n_batches"] is None else f"{r['valid_temps']}/{r['n_batches']}"
        notes = "; ".join(r["detail"])
        if r["missing_branches"]:
            notes += f"  missing branches: {', '.join(r['missing_branches'])}"
        print(f"{r['name']:28s} {r['group']:6s} {r['status']:14s} {nb:>9s} {ent:>22s}  {notes}")
    print("-" * 110)
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # ---- recovery preparation ----
    broken = [r for r in results if r["recovery"]]
    if not broken:
        print("\nAll inspected samples are OK — no recovery needed.")
        if args.json:
            _write_json(args.json, results)
        return

    # ---- verify batch-index alignment for resubmit candidates ----
    # A resubmit re-runs specific batch indices; those indices only line up with the
    # existing temps if files-per-job matches the original conversion. Cross-check against
    # the count the merge log recorded; on a mismatch fall back to a full reprocess.
    resubmits = [r for r in broken if r["recovery"][0] == "resubmit"]
    if resubmits and not args.skip_batch_verify:
        print(f"\nVerifying batch-count alignment for {len(resubmits)} resubmit candidate(s) "
              f"(files-per-job={args.files_per_job}; needs proxy/DAS)...")
        for r in resubmits:
            q = query_batch_count(r["name"], cfg, args.files_per_job)
            if q is None:
                r["detail"].append(f"batch-count unverifiable; trusting files-per-job="
                                   f"{args.files_per_job}")
                print(f"  {r['name']:28s} could not query — proceeding UNVERIFIED")
            elif q != r["n_batches"]:
                r["detail"].append(f"batch-count {q} != log {r['n_batches']} at files-per-job="
                                   f"{args.files_per_job}; reprocessing instead")
                r["recovery"] = ("reprocess", r["name"])
                print(f"  {r['name']:28s} MISMATCH {q} != {r['n_batches']} -> reprocess")
            else:
                print(f"  {r['name']:28s} OK ({q} batches)")

    needs_resubmit = any(r["recovery"][0] == "resubmit" for r in broken)
    data_resubmit = any(r["recovery"][0] == "resubmit" and not r["is_mc"] for r in broken)

    # ---- shared SLURM context for resubmits (mirrors run.py mode-0 layout) ----
    uid = os.getuid()
    ctx = dict(
        bin=os.path.join(_SCRIPT_DIR, "convert_branch"),
        config=os.path.join(_SCRIPT_DIR, "config.json"),
        workdir=_SCRIPT_DIR,
        files_per_job=args.files_per_job,
        libdir=root_libdir() if needs_resubmit else "",
        golden=find_golden_json() if data_resubmit else None,
        x509_dst=(f"/depot/cms/users/{os.environ.get('USER', '')}/x509up_u{uid}"
                  if data_resubmit else None),
        partition=args.slurm_partition, time=args.slurm_time, mem=args.slurm_mem,
        cpus=args.slurm_cpus, extra=args.slurm_extra,
    )
    # resubmit needs the compiled binary; without it, fall back to reprocess (run.py compiles)
    if needs_resubmit and not os.path.exists(ctx["bin"]):
        print(f"\n[WARN] {ctx['bin']} not found — resubmit candidates will be reprocessed.")
        for r in broken:
            if r["recovery"][0] == "resubmit":
                r["recovery"] = ("reprocess", r["name"])
        needs_resubmit = data_resubmit = False
        ctx["golden"] = ctx["x509_dst"] = None

    ts = time.strftime("%Y%m%d_%H%M%S")
    rec_path = os.path.join(_SCRIPT_DIR, f"recovery_{ts}.sh")
    lines = ["#!/usr/bin/env bash", "set -e",
             f"# Recovery for {len(broken)} sample(s) generated {ts}",
             f"cd {_ROOT_DIR}", ""]
    if data_resubmit:
        lines += ["# data batches re-discover input files via DAS — give SLURM jobs the proxy",
                  f"cp /tmp/x509up_u{uid} {ctx['x509_dst']}", ""]
    for r in broken:
        action = r["recovery"][0]
        name = r["recovery"][1]
        if action == "remerge":
            lines.append(f"# {name}: {r['status']} — all {r['n_batches']} temps valid, just re-merge")
            lines.append(f"CONVERT_CONFIG_PATH={_SCRIPT_DIR}/config.json "
                         f"{_SCRIPT_DIR}/convert_branch {name} --merge-successful-batches")
            lines.append("")
        elif action == "resubmit":
            lines += resubmit_script_lines(ctx, r)
        else:
            lines.append(f"# {name}: {r['status']} — reprocess + merge (temps incomplete/removed)")
            lines.append(f"python3 run.py 0 {name} --slurm")
            lines.append("")
    with open(rec_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(rec_path, 0o755)

    print(f"\nPrepared recovery for {len(broken)} sample(s):")
    for r in broken:
        action = r["recovery"][0]
        extra = ""
        if action == "resubmit":
            extra = f" ({len(r['recovery'][2])}/{r['n_batches']} batches: " \
                    f"{', '.join(map(str, r['recovery'][2]))})"
        print(f"  {r['name']:28s} {r['status']:14s} -> {action}{extra}")
    print(f"Recovery script written: {rec_path}")

    if args.json:
        _write_json(args.json, results)

    if args.submit:
        print("\n--submit: launching recovery jobs ...")
        if data_resubmit:
            dst, src, _ = prepare_x509()
            if dst:
                print(f"  copied proxy {src} -> {dst}")
            else:
                print(f"  [WARN] proxy {src} not found; data batch jobs may fail DAS access")
        for r in broken:
            action = r["recovery"][0]
            name = r["recovery"][1]
            if action == "remerge":
                print(f"  -> {name}: remerge")
                cmd = [ctx["bin"], name, "--merge-successful-batches"]
                env = {**os.environ, "CONVERT_CONFIG_PATH": ctx["config"]}
                subprocess.run(cmd, env=env, cwd=_SCRIPT_DIR, check=False)
            elif action == "resubmit":
                print(f"  -> {name}: resubmit {len(r['recovery'][2])} batch(es) + merge")
                resubmit_live(ctx, r)
            else:
                print(f"  -> {name}: reprocess (run.py 0 {name} --slurm)")
                subprocess.run(["python3", "run.py", "0", name, "--slurm"],
                               env=os.environ, cwd=_ROOT_DIR, check=False)
    else:
        print(f"\nGenerate-only (no jobs submitted). Review and run:  bash {rec_path}\n"
              f"or re-run this script with --submit.")


def _write_json(path, results):
    serial = []
    for r in results:
        d = dict(r)
        d["recovery"] = list(r["recovery"]) if r["recovery"] else None
        serial.append(d)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serial, fh, indent=2)
    print(f"JSON report written: {path}")


if __name__ == "__main__":
    main()
