#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCALE_FACTOR_DIR="${REPO_ROOT}/systematics/scale_factor"
NANO_DIR="${SCALE_FACTOR_DIR}/nano.cpp"
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt"
LCG_SETUP="${LCG_VIEW}/setup.sh"

log() {
  printf '[mode11-preflight] %s\n' "$*"
}

fail() {
  printf '[mode11-preflight] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing required file: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is not available after site setup: $1"
}

source_setup_file() {
  local setup=$1
  local saved_flags=$-
  local rc=0

  set +e
  set +u
  # shellcheck disable=SC1090
  source "${setup}"
  rc=$?

  case "${saved_flags}" in
    *e*) set -e ;;
    *) set +e ;;
  esac
  case "${saved_flags}" in
    *u*) set -u ;;
    *) set +u ;;
  esac

  return "${rc}"
}

log "repository: ${REPO_ROOT}"
require_file "${SCALE_FACTOR_DIR}/scale_factor.py"
require_file "${SCALE_FACTOR_DIR}/ntuple_config.json"
require_file "${NANO_DIR}/templates/condor/process.sh.in"
require_file "${NANO_DIR}/templates/condor/submit.sh.in"
require_file "${NANO_DIR}/tools/package_worker_runtime.py"
require_file "${LCG_SETUP}"

log "checking shell templates"
bash -n "${NANO_DIR}/templates/condor/process.sh.in"
bash -n "${NANO_DIR}/templates/condor/submit.sh.in"

log "checking fixed LCG setup under set -u"
if ! source_setup_file "${LCG_SETUP}" >/dev/null; then
  fail "failed to source fixed LCG setup safely: ${LCG_SETUP}"
fi

log "checking required site tools"
for tool in python3 cmake gcc g++ root-config hadd xrdcp xrdfs tar awk sed flock sha256sum; do
  require_command "${tool}"
done

log "checking exact scale_factor.py LCG package resolution"
python3 - "${REPO_ROOT}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
scale_dir = repo_root / "systematics" / "scale_factor"
scale_path = scale_dir / "scale_factor.py"

spec = importlib.util.spec_from_file_location("scale_factor_preflight", scale_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

exports = module.lcg_env_exports()
if not exports:
    raise SystemExit("fixed LCG view was not selected")

required_paths = [
    "SCALE_FACTOR_CC_COMPILER",
    "SCALE_FACTOR_CXX_COMPILER",
    "SCALE_FACTOR_PYTHON3_EXECUTABLE",
    "SCALE_FACTOR_ROOT_DIR",
    "SCALE_FACTOR_CORRECTIONLIB_DIR",
    "SCALE_FACTOR_YAML_CPP_DIR",
    "SCALE_FACTOR_YAML_CPP_INCLUDE_DIR",
    "SCALE_FACTOR_YAML_CPP_LIBRARY",
]
for key in required_paths:
    value = exports.get(key)
    if not value:
        raise SystemExit(f"{key} was not resolved")
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"{key} does not exist: {path}")
    if "/cvmfs/sft.cern.ch/lcg/" not in str(path.resolve()):
        raise SystemExit(f"{key} is not from the fixed LCG/CVMFS runtime: {path}")

cfg_path = scale_dir / "ntuple_config.json"
with cfg_path.open(encoding="utf-8") as handle:
    cfg = json.load(handle)

ntuple = cfg["ntuple"]
targets = module.run_targets(cfg)
sample_sets = []
if ntuple.get("samples", {}).get("mc_groups"):
    sample_sets.append("mc")
if ntuple.get("samples", {}).get("data"):
    sample_sets.append("data")

job_pattern = ntuple.get("job_dir", "jobs/ntuples/{jet_type}_{jet_category}_{year}_{sample_set}")
generated = []
stale = []
missing = []
for target in targets:
    tokens = {
        "jet_type": target["jet_type"],
        "jet_category": target["jet_category"],
        "year": cfg["year"],
    }
    for sample_set in sample_sets:
        job_dir = module.resolve_path(job_pattern.format(**tokens, sample_set=sample_set))
        if not job_dir.exists():
            continue
        generated.append(job_dir)
        for name in ("process.sh", "submit.sh", "submit_lxplus.jdl", "repo.tar.gz", "worker_runtime.tar.gz", "job_manifest.tsv", "config_snapshot.yaml"):
            if not (job_dir / name).exists():
                missing.append(job_dir / name)
        process = job_dir / "process.sh"
        if process.exists():
            text = process.read_text(encoding="utf-8", errors="replace")
            if "source_setup_file" not in text or "LCG_109/x86_64-el9-gcc13-opt" not in text:
                stale.append(process)

if missing:
    raise SystemExit("generated job directory is incomplete:\n  " + "\n  ".join(str(path) for path in missing))
if stale:
    raise SystemExit("generated job directory has stale process.sh; regenerate it with python3 run.py 11:\n  " + "\n  ".join(str(path) for path in stale))

print("LCG runtime:", exports["SCALE_FACTOR_RUNTIME_PREFIX"] or "fixed CVMFS LCG")
print("ROOT_DIR:", exports["SCALE_FACTOR_ROOT_DIR"])
print("correctionlib_DIR:", exports["SCALE_FACTOR_CORRECTIONLIB_DIR"])
print("yaml-cpp_DIR:", exports["SCALE_FACTOR_YAML_CPP_DIR"])
print("yaml-cpp library:", exports["SCALE_FACTOR_YAML_CPP_LIBRARY"])
if generated:
    print("generated job dirs checked:", len(generated))
else:
    print("generated job dirs checked: 0 (run python3 run.py 11 before the final pre-submit check)")
PY

log "preflight passed"
