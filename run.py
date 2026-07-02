#!/usr/bin/env python3
"""run.py — Python replacement for run.sh"""

import argparse
import atexit
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Mode table
# ---------------------------------------------------------------------------

MODES = {
    0: dict(label="convert_branch", subdir="selections/convert",
            source="convert_branch.C", bin_name="convert_branch",
            config_env="CONVERT_CONFIG_PATH"),
    1: dict(label="pileup", subdir="selections/weight",
            source="weight.C", bin_name="weight",
            config_env="WEIGHT_CONFIG_PATH"),
    2: dict(label="bdt_train", subdir="selections/BDT",
            script="train.py", config_env="BDT_CONFIG_PATH"),
    3: dict(label="signal_region", subdir="selections/signal_region",
            script="signal_region.py", config_env="SCAN_CONFIG_PATH"),
    4: dict(label="data_mc", subdir="plotting",
            script="data_mc.py", config_env="PLOT_CONFIG_PATH"),
    5: dict(label="qcd_est", subdir="background_estimation",
            script="qcd_est.py", config_env="QCD_EST_CONFIG_PATH"),
    6: dict(label="mix", subdir="selections/mix",
            source="mix.C", bin_name="mix",
            config_env="MIX_CONFIG_PATH"),
    7: dict(label="combine", subdir="combine",
            source="combine.C", bin_name="combine_run",
            config_env="COMBINE_CONFIG_PATH"),
    8: dict(label="theory_syst", subdir="selections/theory_weights",
            script="theory_syst.py", config_env="THEORY_CONFIG_PATH"),
    9: dict(label="pileup_syst", subdir="selections/pileup_syst",
            script="pileup_syst.py", config_env="PILEUP_SYST_CONFIG_PATH"),
    10: dict(label="class_shapes", subdir="plotting",
             script="class_shapes.py", config_env="PLOT_CONFIG_PATH"),
}

PYTHON_MODES = frozenset({2, 3, 4, 5, 8, 9, 10})
SAMPLE_MODES = frozenset({0, 1, 6})

ROOT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{timestamp()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Job dispatcher for ScoutingVVVTools",
        epilog="""
Modes:
  0  selections/convert/convert_branch.C  (per-sample, batch-parallel)
  1  selections/weight/weight.C           (per-sample)
  2  selections/BDT/train.py             (no samples)
  3  selections/signal_region/signal_region.py  (no samples)
  4  plotting/data_mc.py                 (no samples)
  5  background_estimation/qcd_est.py   (no samples)
  6  selections/mix/mix.C               (per-sample)
  7  combine/combine.C                  (no samples)
  8  selections/theory_weights/theory_syst.py  (no samples)
  9  selections/pileup_syst/pileup_syst.py  (no samples)
  10 plotting/class_shapes.py            (no samples)

Sample selection for modes 0, 1, 6:
  1. CLI sample names (highest priority)
  2. submit_samples from config.json
  3. All MC samples (fallback)
""",
    )
    p.add_argument("mode", type=int, choices=MODES, metavar="MODE",
                   help="Execution mode 0-10")
    p.add_argument("rest", nargs="*", metavar="ARG",
                   help="Optional: [config.json] [sample1 sample2 ...]")
    p.add_argument("--slurm", action="store_true",
                   help="Submit jobs to SLURM instead of running locally")
    p.add_argument("--slurm-partition", default="cms-express", metavar="NAME")
    p.add_argument("--slurm-time", default="24:00:00", metavar="HH:MM:SS")
    p.add_argument("--slurm-mem", default="4G", metavar="MEM")
    p.add_argument("--slurm-cpus", type=int, default=1, metavar="N")
    p.add_argument("--slurm-extra", default="", metavar="ARGS",
                   help="Extra sbatch arguments (space-separated)")
    p.add_argument("--slurm-files-per-job", type=int, default=50, metavar="N",
                   help="Target input files per SLURM job for mode 0; default 50 pairs with --slurm-cpus=1 (serial) to keep per-job wall time reasonable")
    p.add_argument("--max-jobs", type=int, default=1, metavar="N",
                   help="Max concurrent local jobs (default: 1)")

    args, passthrough = p.parse_known_args()

    rest = list(args.rest)
    # argparse closes the nargs="*" positional as soon as an optional (e.g. --slurm)
    # is seen, so positionals given AFTER an optional land in parse_known_args'
    # leftovers instead of args.rest. For the sample-taking modes, recover any
    # non-flag leftovers as positionals so argument order does not matter
    # (`0 --slurm www` == `0 www --slurm`). Real flags stay in passthrough, which
    # is forwarded verbatim to python-script modes (e.g. --no-selection).
    if args.mode in SAMPLE_MODES:
        rest += [tok for tok in passthrough if not tok.startswith("-")]
        passthrough = [tok for tok in passthrough if tok.startswith("-")]
    args.passthrough = passthrough

    # config.json may appear in any position; pull it out, the rest are sample names.
    args.config_input = None
    json_tokens = [tok for tok in rest if tok.endswith(".json")]
    if json_tokens:
        args.config_input = json_tokens[0]
        rest = [tok for tok in rest if tok != args.config_input]
    args.samples = rest
    return args


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_config(config_input, work_dir):
    path = Path(config_input) if config_input else work_dir / "config.json"
    if not path.exists():
        sys.exit(f"config file not found: {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# Golden JSON helpers (mode 0, collision data only)
# ---------------------------------------------------------------------------

def find_golden_json():
    """Return the path to Cert_*_Golden.json in ROOT_DIR, or None."""
    candidates = sorted(ROOT_DIR.glob("Cert_*_Golden.json"))
    return str(candidates[0]) if candidates else None


def build_sample_mc_map(config_path):
    """Return {sample_name: is_MC} dict from the sample config."""
    with open(config_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_sc = payload.get("sample_config", "../../src/sample.json")
    sc_path = (
        Path(raw_sc) if os.path.isabs(raw_sc)
        else (config_path.parent / raw_sc).resolve()
    )
    with open(sc_path, encoding="utf-8") as fh:
        sp = json.load(fh)
    return {
        r["name"]: r.get("is_MC", True)
        for r in sp.get("sample", [])
        if isinstance(r, dict) and "name" in r
    }


# ---------------------------------------------------------------------------
# Sample resolution
# ---------------------------------------------------------------------------

def resolve_samples(config_path, requested_samples):
    """Three-tier selection: CLI > submit_samples > all MC."""
    with open(config_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    raw_sc = payload.get("sample_config", "../../src/sample.json")
    if not isinstance(raw_sc, str) or not raw_sc:
        sys.exit("sample_config must be a non-empty string")
    sc_path = (
        Path(raw_sc) if os.path.isabs(raw_sc)
        else (config_path.parent / raw_sc).resolve()
    )

    with open(sc_path, encoding="utf-8") as fh:
        sample_payload = json.load(fh)

    rules = sample_payload.get("sample", [])
    if not isinstance(rules, list):
        sys.exit("sample must be a JSON array")

    seen, all_samples, mc_samples = set(), [], []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = rule.get("name", "")
        if not isinstance(name, str) or not name:
            sys.exit("sample.name must be a non-empty string")
        is_mc = rule.get("is_MC")
        if not isinstance(is_mc, bool):
            sys.exit("sample.is_MC must be a boolean")
        if name in seen:
            continue
        seen.add(name)
        all_samples.append(name)
        if is_mc:
            mc_samples.append(name)

    configured = payload.get("submit_samples") or []
    if not isinstance(configured, list):
        sys.exit("submit_samples must be a JSON array")
    for s in configured:
        if not isinstance(s, str):
            sys.exit("submit_samples must contain only strings")

    selected = list(requested_samples) if requested_samples else (configured or mc_samples)

    available = set(all_samples)
    emitted, result = set(), []
    for s in selected:
        if s not in available:
            sys.exit(f"Unknown sample requested: {s}")
        if s not in emitted:
            emitted.add(s)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# OpenMP detection
# ---------------------------------------------------------------------------

def detect_openmp():
    """Return (cflags, ldflags) for OpenMP, or ('', '') if unavailable."""
    src_fd, src = tempfile.mkstemp(suffix=".cpp", prefix="omp_test_")
    bin_fd, out = tempfile.mkstemp(suffix=".bin", prefix="omp_test_")
    os.close(src_fd)
    os.close(bin_fd)
    try:
        with open(src, "w") as fh:
            fh.write("#include <omp.h>\nint main() { return 0; }\n")

        brew_cf = "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        brew_lf = "-L/opt/homebrew/opt/libomp/lib -lomp"
        r = subprocess.run(
            ["c++"] + brew_cf.split() + [src, "-o", out] + brew_lf.split(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if r.returncode == 0:
            return brew_cf, brew_lf

        r = subprocess.run(["c++", "-fopenmp", src, "-o", out],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return "-fopenmp", ""

        return "", ""
    finally:
        for path in (src, out):
            try:
                os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_binary(work_dir, source, bin_path, omp_cflags, omp_ldflags):
    for tool in ("c++", "root-config"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is required but not found in PATH")

    root_cflags  = subprocess.check_output(["root-config", "--cflags"],  text=True).strip()
    root_libs    = subprocess.check_output(["root-config", "--libs"],    text=True).strip()
    root_libdir  = subprocess.check_output(["root-config", "--libdir"],  text=True).strip()

    cmd = (
        ["c++", "-O3", "-DNDEBUG", "-std=c++17"]
        + root_cflags.split()
        + (omp_cflags.split() if omp_cflags else [])
        + [f"./{source}", "-o", str(bin_path)]
        + root_libs.split()
        + (omp_ldflags.split() if omp_ldflags else [])
    )
    log(f"compile: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=work_dir)
    if r.returncode != 0:
        sys.exit(f"compilation failed (status {r.returncode})")
    log("compile finished")
    return root_libdir


# ---------------------------------------------------------------------------
# Cleanup & log copy
# ---------------------------------------------------------------------------

def cleanup_build_artifacts(bin_path):
    for p in (bin_path, Path(str(bin_path) + ".dSYM")):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except OSError:
            pass


def copy_log_to_output_dirs(mode, config_path, work_dir, log_path):
    if not log_path.exists():
        return
    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return

    def resolve(rel):
        if not isinstance(rel, str) or not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (work_dir / p).resolve()

    dirs = []
    if mode == 1:
        d = resolve(cfg.get("output_root"))
        if d:
            dirs.append(d)
    elif mode in (2, 4):
        patt = cfg.get("output_root") or ""
        for tree in cfg.get("submit_trees") or []:
            if isinstance(tree, str) and tree:
                d = resolve(patt.replace("{tree_name}", tree))
                if d:
                    dirs.append(d)
    elif mode in (3, 5, 7):
        d = resolve(cfg.get("output_dir") or cfg.get("output_root"))
        if d:
            dirs.append(d)
    # modes 0, 6: intentionally skipped

    seen = set()
    for raw_d in dirs:
        d = raw_d.resolve()
        if d in seen:
            continue
        seen.add(d)
        if not d.is_dir():
            log(f"log copy: skipping missing output dir {d}")
            continue
        dest = d / "log.txt"
        try:
            shutil.copy2(log_path, dest)
            log(f"copied log to {dest}")
        except OSError as exc:
            log(f"warning: failed to copy log to {dest}: {exc}")


# ---------------------------------------------------------------------------
# Python-script modes (2-5)
# ---------------------------------------------------------------------------

def run_python_mode(mode_cfg, config_path, work_dir, passthrough=None):
    env = {**os.environ, mode_cfg["config_env"]: str(config_path)}
    script = mode_cfg["script"]
    cmd = ["python3", f"./{script}", *(passthrough or [])]
    log(f"run: env {mode_cfg['config_env']}={config_path} {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env, cwd=work_dir)
    return r.returncode


# ---------------------------------------------------------------------------
# C++ single-run mode (mode 7)
# ---------------------------------------------------------------------------

def run_combine_mode(mode_cfg, config_path, bin_path, work_dir):
    env = {**os.environ, mode_cfg["config_env"]: str(config_path)}
    log(f"run: env {mode_cfg['config_env']}={config_path} {bin_path}")
    r = subprocess.run([str(bin_path)], env=env, cwd=work_dir)
    return r.returncode


# ---------------------------------------------------------------------------
# Mode-0 batch loop (target for multiprocessing.Process)
# ---------------------------------------------------------------------------

def _run_convert_batches(sample, config_env, config_path_str, bin_path_str, work_dir_str,
                         golden_json_path=None):
    """Runs the convert batch loop in a subprocess — forked by launch_job_local."""
    config_path = Path(config_path_str)
    bin_path    = Path(bin_path_str)
    work_dir    = Path(work_dir_str)
    env_base    = {**os.environ, config_env: str(config_path)}
    if golden_json_path:
        env_base["CONVERT_GOLDEN_JSON"] = golden_json_path


    
    r = subprocess.run(
        [str(bin_path), sample, "--batch-count"],
        env=env_base, cwd=work_dir,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    raw = r.stdout.decode().strip()
    print (raw)
    if not raw.isdigit() or int(raw) <= 0:
        print(f"Invalid convert batch count for sample={sample}: {raw!r}", flush=True)
        sys.exit(1)
    batch_count = int(raw)

    batch_failures = 0
    successful_batches = []

    for batch_index in range(batch_count):
        log(f"running sample={sample} batch={batch_index + 1}/{batch_count}")
        env = {
            **env_base,
            "CONVERT_SUCCESSFUL_BATCHES": ",".join(map(str, successful_batches)),
            "CONVERT_DEFER_FINAL_MERGE": "1",
        }
        r = subprocess.run(
            [str(bin_path), sample, str(batch_index)],
            env=env, cwd=work_dir,
        )
        if r.returncode != 0:
            batch_failures += 1
            log(
                f"warning: sample={sample} batch={batch_index + 1}/{batch_count} "
                f"failed status={r.returncode}; continuing"
            )
            continue
        successful_batches.append(batch_index)

    if batch_failures:
        log(
            f"warning: sample={sample} completed with "
            f"{batch_failures}/{batch_count} failed batch(es); "
            f"final output uses successful batches only"
        )

    log(f"running sample={sample} final merge from successful batches")
    r = subprocess.run(
        [str(bin_path), sample, "--merge-successful-batches"],
        env={**env_base,
             "CONVERT_SUCCESSFUL_BATCHES": ",".join(map(str, successful_batches))},
        cwd=work_dir,
    )
    sys.exit(r.returncode)


# ---------------------------------------------------------------------------
# Uniform process handle (multiprocessing.Process → poll/wait API)
# ---------------------------------------------------------------------------

class _ProcHandle:
    def __init__(self, proc):
        self._proc = proc

    @property
    def pid(self):
        return self._proc.pid

    def poll(self):
        if self._proc.is_alive():
            return None
        code = self._proc.exitcode
        return code if code is not None else 1

    def wait(self):
        self._proc.join()
        code = self._proc.exitcode
        return code if code is not None else 1


# ---------------------------------------------------------------------------
# Local job launching
# ---------------------------------------------------------------------------

def launch_job_local(sample, mode, mode_cfg, config_path, bin_path, work_dir,
                     golden_json_path=None):
    if mode == 0:
        proc = multiprocessing.Process(
            target=_run_convert_batches,
            args=(sample, mode_cfg["config_env"],
                  str(config_path), str(bin_path), str(work_dir),
                  golden_json_path),
            daemon=False,
        )
        proc.start()
        return _ProcHandle(proc)

    # Modes 1 and 6: subprocess inherits dup2'd fd 1/2 → output goes to log
    env = {**os.environ, mode_cfg["config_env"]: str(config_path)}
    return subprocess.Popen([str(bin_path), sample], env=env, cwd=work_dir)


# ---------------------------------------------------------------------------
# SLURM job launching
# ---------------------------------------------------------------------------

def launch_job_slurm(sample, mode_cfg, config_path, bin_path, work_dir, args, x509_dst,
                     root_libdir=""):
    config_env = mode_cfg["config_env"]
    label      = mode_cfg["label"]

    sbatch_args = [
        "sbatch",
        f"--job-name={label}_{sample}",
        f"--output={work_dir}/{sample}_%j.out",
        f"--error={work_dir}/{sample}_%j.out",
        f"--cpus-per-task={args.slurm_cpus}",
        f"--mem={args.slurm_mem}",
        f"--time={args.slurm_time}",
        f"--exclude=hammer-f004,hammer-f007",
    ]
    if args.slurm_partition:
        sbatch_args.append(f"--account={args.slurm_partition}")
    if args.slurm_extra:
        sbatch_args.extend(args.slurm_extra.split())
    ldpath_prefix = f"export LD_LIBRARY_PATH={root_libdir}:${{LD_LIBRARY_PATH:-}}; " if root_libdir else ""
    sbatch_args.append(
        f"--wrap={ldpath_prefix}export X509_USER_PROXY={x509_dst}; "
        f"env {config_env}={config_path} {bin_path} {sample}"
    )

    r = subprocess.run(sbatch_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
    if r.returncode != 0:
        sys.exit(f"sbatch failed for sample={sample}: {r.stderr.decode().strip()}")
    job_id = r.stdout.decode().strip().split()[-1]
    log(f"submitted sample={sample} slurm_job_id={job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Mode-0 SLURM multi-job launch
# ---------------------------------------------------------------------------

def launch_mode0_slurm(sample, mode_cfg, config_path, bin_path, work_dir, args, x509_dst,
                       golden_json_path=None, root_libdir=""):
    """Submit one SLURM job per ~files-per-job-sized batch, plus a dependency merge job.

    Each batch job runs: convert_branch {sample} {idx} with CONVERT_DEFER_FINAL_MERGE=1
    so it writes a temp file and exits without merging.  The merge job, submitted
    with --dependency=afterany on all batch jobs, runs --merge-successful-batches
    which tolerates missing temp files from any failed batch jobs.
    """
    config_env    = mode_cfg["config_env"]
    label         = mode_cfg["label"]
    files_per_job = args.slurm_files_per_job

    # Query batch count locally so we know how many SLURM jobs to submit.
    r = subprocess.run(
        [str(bin_path), sample, "--batch-count"],
        env={**os.environ, config_env: str(config_path),
             "CONVERT_FILES_PER_BATCH": str(files_per_job)},
        cwd=work_dir,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    raw = r.stdout.decode().strip()
    if not raw.isdigit() or int(raw) <= 0:
        sys.exit(
            f"batch count query failed for sample={sample}: "
            f"exit={r.returncode} stderr={r.stderr.decode().strip()!r}"
        )
    n_batches = int(raw)
    log(f"sample={sample} n_batches={n_batches} files_per_job={files_per_job}")

    x509_prefix   = f"export X509_USER_PROXY={x509_dst}; " if x509_dst else ""
    ldpath_prefix = f"export LD_LIBRARY_PATH={root_libdir}:${{LD_LIBRARY_PATH:-}}; " if root_libdir else ""
    # Base env string shared by all jobs: sets config path and batch size override.
    base_env_str = (
        f"env {config_env}={config_path} "
        f"CONVERT_FILES_PER_BATCH={files_per_job}"
    )
    if golden_json_path:
        base_env_str += f" CONVERT_GOLDEN_JSON={golden_json_path}"

    def _sbatch(job_name, wrap_body, depends_on=None):
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output={work_dir}/{job_name}_%j.out",
            f"--error={work_dir}/{job_name}_%j.out",
            f"--cpus-per-task={args.slurm_cpus}",
            f"--mem={args.slurm_mem}",
            f"--time={args.slurm_time}",
        ]
        if args.slurm_partition:
            cmd.append(f"--account={args.slurm_partition}")
        if depends_on:
            cmd.append(f"--dependency=afterany:{':'.join(depends_on)}")
        if args.slurm_extra:
            cmd.extend(args.slurm_extra.split())
        # wrap_body is appended after base_env_str; any NAME=VALUE tokens before
        # the executable are absorbed by `env` as additional environment variables.
        cmd.append(f"--wrap={ldpath_prefix}{x509_prefix}{base_env_str} {wrap_body}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
        if result.returncode != 0:
            sys.exit(f"sbatch failed for {job_name}: {result.stderr.decode().strip()}")
        return result.stdout.decode().strip().split()[-1]

    batch_ids = []
    for idx in range(n_batches):
        job_name = f"{label}_{sample}_{idx}"
        job_id = _sbatch(
            job_name,
            f"CONVERT_DEFER_FINAL_MERGE=1 {bin_path} {sample} {idx}",
        )
        batch_ids.append(job_id)
        log(f"submitted sample={sample} batch={idx + 1}/{n_batches} slurm_job_id={job_id}")

    merge_id = _sbatch(
        f"{label}_{sample}_merge",
        f"{bin_path} {sample} --merge-successful-batches",
        depends_on=batch_ids,
    )
    log(f"submitted sample={sample} merge slurm_job_id={merge_id}")


# ---------------------------------------------------------------------------
# Job reaping
# ---------------------------------------------------------------------------

def reap_local(running, failed_jobs):
    still_running = []
    any_finished  = False
    for entry in running:
        proc, sample = entry["proc"], entry["sample"]
        status = proc.poll()
        if status is None:
            still_running.append(entry)
            continue
        any_finished = True
        proc.wait()  # ensure multiprocessing.Process is fully reaped
        if status != 0:
            failed_jobs += 1
        log(f"finished sample={sample} pid={proc.pid} status={status}")
    return still_running, any_finished, failed_jobs


def reap_slurm(running, failed_jobs):
    still_running = []
    any_finished  = False
    for entry in running:
        job_id, sample = entry["job_id"], entry["sample"]
        r = subprocess.run(
            ["squeue", f"--jobs={job_id}", "--noheader", "--format=%T"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if r.stdout.decode().strip():
            still_running.append(entry)
            continue
        any_finished = True
        r2 = subprocess.run(
            ["sacct", "-j", job_id, "--noheader",
             "--format=ExitCode", "--parsable2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            exit_code = int(r2.stdout.decode().strip().splitlines()[0].split(":")[0])
        except (IndexError, ValueError):
            exit_code = 1
        if exit_code != 0:
            failed_jobs += 1
        log(f"finished sample={sample} slurm_job_id={job_id} status={exit_code}")
    return still_running, any_finished, failed_jobs


# ---------------------------------------------------------------------------
# Dispatch loop
# ---------------------------------------------------------------------------

def dispatch_jobs(samples, mode, mode_cfg, config_path, bin_path, work_dir, args,
                  x509_dst=None, golden_json_path=None, sample_mc_map=None, root_libdir=""):
    failed_jobs = 0

    def _sample_golden(sample):
        """Return golden JSON path if sample is data and one is available, else None."""
        if not golden_json_path:
            return None
        if sample_mc_map and not sample_mc_map.get(sample, True):
            return golden_json_path
        return None

    if args.slurm:
        for sample in samples:
            if mode == 0:
                launch_mode0_slurm(sample, mode_cfg, config_path, bin_path,
                                   work_dir, args, x509_dst, _sample_golden(sample),
                                   root_libdir=root_libdir)
            else:
                launch_job_slurm(sample, mode_cfg, config_path, bin_path,
                                 work_dir, args, x509_dst, root_libdir=root_libdir)
        log("all jobs submitted to SLURM")
        return 0

    running = []
    for sample in samples:
        while len(running) >= args.max_jobs:
            running, any_finished, failed_jobs = reap_local(running, failed_jobs)
            if not any_finished:
                time.sleep(2)
        handle = launch_job_local(sample, mode, mode_cfg, config_path, bin_path, work_dir,
                                  _sample_golden(sample))
        running.append({"proc": handle, "sample": sample})
        log(f"started sample={sample} pid={handle.pid}")

    while running:
        running, any_finished, failed_jobs = reap_local(running, failed_jobs)
        if not any_finished:
            time.sleep(2)

    log(f"all jobs finished, failed_jobs={failed_jobs}")
    return failed_jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    mode     = args.mode
    mode_cfg = MODES[mode]
    label    = mode_cfg["label"]

    if args.samples and mode not in SAMPLE_MODES:
        sys.exit(
            f"mode={mode} does not accept sample arguments: " + " ".join(args.samples)
        )

    work_dir = ROOT_DIR / mode_cfg["subdir"]
    log_path = work_dir / "log.txt"

    config_path = resolve_config(args.config_input, work_dir)

    # X509 cert copy happens before log redirect so errors go to terminal
    x509_dst = None
    if args.slurm:
        uid      = os.getuid()
        x509_src = Path(f"/tmp/x509up_u{uid}")
        x509_dst = Path(f"/depot/cms/users/{os.environ['USER']}/x509up_u{uid}")
        if not x509_src.exists():
            sys.exit(f"Certificate not found: {x509_src}")
        shutil.copy2(x509_src, x509_dst)
        print(f"[{timestamp()}] copied certificate {x509_src} -> {x509_dst}", flush=True)
    print (log_path)
    # Redirect stdout/stderr to log file (truncate existing); mirrors bash `exec >> log 2>&1`
    log_fh     = open(log_path, "w", buffering=1)
    sys.stdout = log_fh
    sys.stderr = log_fh
    os.dup2(log_fh.fileno(), 1)
    os.dup2(log_fh.fileno(), 2)

    if not args.slurm:
        atexit.register(copy_log_to_output_dirs, mode, config_path, work_dir, log_path)

    # Python-only modes — no compilation, no sample loop
    if mode in PYTHON_MODES:
        log(f"mode={mode} ({label})")
        log(f"work_dir={work_dir}")
        log(f"config={config_path}")
        log(f"started job={label} pid={os.getpid()}")
        status = run_python_mode(mode_cfg, config_path, work_dir, args.passthrough)
        log(f"finished job={label} pid={os.getpid()} status={status}")
        sys.exit(status)

    # C++ modes — detect OpenMP, compile
    omp_cflags, omp_ldflags = detect_openmp()
    bin_path = work_dir / mode_cfg["bin_name"]

    log(f"mode={mode} ({label})")
    log(f"work_dir={work_dir}")
    log(f"config={config_path}")
    log(f"max_concurrent_jobs={args.max_jobs}")
    if args.samples:
        log(f"cli_samples={' '.join(args.samples)}")

    if not args.slurm:
        atexit.register(cleanup_build_artifacts, bin_path)

    root_libdir = compile_binary(work_dir, mode_cfg["source"], bin_path, omp_cflags, omp_ldflags)

    # Mode 7 — single run, no sample loop
    if mode == 7:
        log(f"started job={label} pid={os.getpid()}")
        status = run_combine_mode(mode_cfg, config_path, bin_path, work_dir)
        log(f"finished job={label} pid={os.getpid()} status={status}")
        sys.exit(status)

    # Modes 0, 1, 6 — per-sample parallel dispatch
    samples = resolve_samples(config_path, args.samples)
    if not samples:
        sys.exit(f"No samples selected from {config_path}")
    log(f"selected_samples={' '.join(samples)}")

    golden_json_path = None
    sample_mc_map = None
    if mode == 0:
        golden_json_path = find_golden_json()
        sample_mc_map = build_sample_mc_map(config_path)
        if golden_json_path:
            log(f"golden_json={golden_json_path}")
        else:
            log("no golden JSON found in project root — data quality selection disabled")

    failed = dispatch_jobs(
        samples, mode, mode_cfg, config_path, bin_path, work_dir, args, x509_dst,
        golden_json_path=golden_json_path, sample_mc_map=sample_mc_map,
        root_libdir=root_libdir,
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
