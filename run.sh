#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_CONCURRENT_JOBS=1
USE_SLURM=false
SLURM_PARTITION="cms-express"
SLURM_TIME="24:00:00"
SLURM_MEM="4G"
SLURM_CPUS=12
SLURM_EXTRA=""

X509_SRC="/tmp/x509up_u$(id -u)"
X509_DST="/depot/cms/users/$(whoami)/x509up_u$(id -u)"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}



usage() {
  cat >&2 <<'EOF'
Usage:
  ./run.sh <mode> [config.json] [sample1 sample2 ...]

Modes:
  mode=0  Run convert/convert_branch.C jobs.
  mode=1  Run weight/weight.C pileup jobs.
  mode=2  Run BDT/train.py.
  mode=3  Run signal_region/signal_region.py.
  mode=4  Run plotting/data_mc.py.
  mode=5  Run background_estimation/qcd_est.py.
  mode=6  Run selections/mix/mix.C (shuffle sample entries across chunks).
  mode=7  Run combine/combine.C (CMS combine wrapper for MC-true and ABCD QCD).

Sample selection:
  1. If sample names are given on the command line, they are used.
  2. Otherwise the script reads submit_samples from the chosen config.json.
  3. If submit_samples is empty or missing, all MC samples are submitted.
  4. Sample arguments are only supported for mode=0, mode=1, and mode=6.
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

MODE="$1"
shift

case "${MODE}" in
  0)
    WORK_DIR="${ROOT_DIR}/selections/convert"
    SOURCE_FILE="convert_branch.C"
    BIN_NAME="convert_branch"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="CONVERT_CONFIG_PATH"
    MODE_LABEL="convert_branch"
    ;;
  1)
    WORK_DIR="${ROOT_DIR}/selections/weight"
    SOURCE_FILE="weight.C"
    BIN_NAME="weight"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="WEIGHT_CONFIG_PATH"
    MODE_LABEL="pileup"
    ;;
  2)
    WORK_DIR="${ROOT_DIR}/selections/BDT"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="BDT_CONFIG_PATH"
    MODE_LABEL="bdt_train"
    PYTHON_SCRIPT="train.py"
    ;;
  3)
    WORK_DIR="${ROOT_DIR}/selections/signal_region"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="SCAN_CONFIG_PATH"
    MODE_LABEL="signal_region"
    PYTHON_SCRIPT="signal_region.py"
    ;;
  4)
    WORK_DIR="${ROOT_DIR}/plotting"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="PLOT_CONFIG_PATH"
    MODE_LABEL="data_mc"
    PYTHON_SCRIPT="data_mc.py"
    ;;
  5)
    WORK_DIR="${ROOT_DIR}/background_estimation"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="QCD_EST_CONFIG_PATH"
    MODE_LABEL="qcd_est"
    PYTHON_SCRIPT="qcd_est.py"
    ;;
  6)
    WORK_DIR="${ROOT_DIR}/selections/mix"
    SOURCE_FILE="mix.C"
    BIN_NAME="mix"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="MIX_CONFIG_PATH"
    MODE_LABEL="mix"
    ;;
  7)
    WORK_DIR="${ROOT_DIR}/combine"
    SOURCE_FILE="combine.C"
    BIN_NAME="combine_run"
    DEFAULT_CONFIG="${WORK_DIR}/config.json"
    CONFIG_ENV_VAR="COMBINE_CONFIG_PATH"
    MODE_LABEL="combine"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --slurm)            USE_SLURM=true;           shift ;;
    --slurm-partition)  SLURM_PARTITION="$2";     shift 2 ;;
    --slurm-time)       SLURM_TIME="$2";          shift 2 ;;
    --slurm-mem)        SLURM_MEM="$2";           shift 2 ;;
    --slurm-cpus)       SLURM_CPUS="$2";          shift 2 ;;
    --slurm-extra)      SLURM_EXTRA="$2";         shift 2 ;;
    *)                  break ;;
  esac
done

if $USE_SLURM; then
  if [ ! -f "${X509_SRC}" ]; then
    echo "Certificate not found: ${X509_SRC}" >&2
    exit 1
  fi
  cp "${X509_SRC}" "${X509_DST}"
  echo "[$(timestamp)] copied certificate ${X509_SRC} -> ${X509_DST}"
fi


CONFIG_INPUT="${DEFAULT_CONFIG}"
if [ "$#" -gt 0 ]; then
  case "$1" in
    *.json)
      CONFIG_INPUT="$1"
      shift
      ;;
  esac
fi

REQUESTED_SAMPLES=("$@")
LOG_PATH="${WORK_DIR}/log.txt"
BIN_PATH="${WORK_DIR}/${BIN_NAME:-}"

if [ ! -f "${CONFIG_INPUT}" ]; then
  echo "config file not found: ${CONFIG_INPUT}" >&2
  exit 1
fi

case "${MAX_CONCURRENT_JOBS}" in
  ''|*[!0-9]*|0)
    echo "MAX_CONCURRENT_JOBS must be a positive integer, got: ${MAX_CONCURRENT_JOBS}" >&2
    exit 1
    ;;
esac

CONFIG_PATH="$(cd "$(dirname "${CONFIG_INPUT}")" && pwd)/$(basename "${CONFIG_INPUT}")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to read JSON config files." >&2
  exit 1
fi


detect_openmp_flags() {
  local test_src test_bin
  test_src="$(mktemp "${TMPDIR:-/tmp}/omp_test.XXXXXX.cpp")"
  test_bin="$(mktemp "${TMPDIR:-/tmp}/omp_test.XXXXXX.bin")"
  cat > "${test_src}" <<'EOF'
#include <omp.h>
int main() { return 0; }
EOF

  if c++ -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include "${test_src}" -L/opt/homebrew/opt/libomp/lib -lomp -o "${test_bin}" >/dev/null 2>&1; then
    rm -f "${test_src}" "${test_bin}"
    printf '%s\n' "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include|-L/opt/homebrew/opt/libomp/lib -lomp"
    return 0
  fi

  if c++ -fopenmp "${test_src}" -o "${test_bin}" >/dev/null 2>&1; then
    rm -f "${test_src}" "${test_bin}"
    printf '%s\n' "-fopenmp|"
    return 0
  fi

  rm -f "${test_src}" "${test_bin}"
  return 1
}

OMP_INFO="$(detect_openmp_flags || true)"
OMP_CFLAGS=""
OMP_LDFLAGS=""
if [ -n "${OMP_INFO}" ]; then
  OMP_CFLAGS="${OMP_INFO%%|*}"
  OMP_LDFLAGS="${OMP_INFO#*|}"
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

# At end-of-run, copy the run-log into the program's configured output dir(s).
# Modes 0 (convert_branch) and 6 (mix) are intentionally skipped per design.
# Modes 2 and 4 use a `{tree_name}` template against `submit_trees` so the log
# is copied once per tree output directory; the other modes have a single dir.
copy_log_to_output_dirs() {
  if [ -z "${LOG_PATH:-}" ] || [ ! -f "${LOG_PATH}" ]; then
    return 0
  fi
  if [ -z "${MODE:-}" ] || [ -z "${CONFIG_PATH:-}" ] || [ -z "${WORK_DIR:-}" ]; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi

  local dirs_output
  dirs_output="$(python3 - "${MODE}" "${CONFIG_PATH}" "${WORK_DIR}" 2>/dev/null <<'PY' || true
import json, os, sys

mode, cfg_path, work_dir = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
except Exception:
    sys.exit(0)

def _resolve(rel):
    if not isinstance(rel, str) or not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(work_dir, rel))

dirs = []
if mode == "1":
    d = _resolve(cfg.get("output_root"))
    if d:
        dirs.append(d)
elif mode in ("2", "4"):
    patt = cfg.get("output_root", "") or ""
    trees = cfg.get("submit_trees", []) or []
    for tree in trees:
        if isinstance(tree, str) and tree:
            d = _resolve(patt.replace("{tree_name}", tree))
            if d:
                dirs.append(d)
elif mode in ("3", "5", "7"):
    rel = cfg.get("output_dir") or cfg.get("output_root")
    d = _resolve(rel)
    if d:
        dirs.append(d)

seen = set()
for d in dirs:
    norm = os.path.normpath(d)
    if norm in seen:
        continue
    seen.add(norm)
    print(norm)
PY
  )"

  if [ -z "${dirs_output}" ]; then
    return 0
  fi

  while IFS= read -r out_dir; do
    [ -z "${out_dir}" ] && continue
    if [ ! -d "${out_dir}" ]; then
      echo "[$(timestamp)] log copy: skipping missing output dir ${out_dir}"
      continue
    fi
    if cp "${LOG_PATH}" "${out_dir}/log.txt" 2>/dev/null; then
      echo "[$(timestamp)] copied log to ${out_dir}/log.txt"
    else
      echo "[$(timestamp)] warning: failed to copy log to ${out_dir}/log.txt"
    fi
  done <<<"${dirs_output}"

  return 0
}

cd "${WORK_DIR}"
: > "${LOG_PATH}"
exec >> "${LOG_PATH}" 2>&1
trap copy_log_to_output_dirs EXIT


if [ "${MODE}" = "2" ] || [ "${MODE}" = "3" ] || [ "${MODE}" = "4" ] || [ "${MODE}" = "5" ]; then
  if [ "$#" -gt 0 ]; then
    echo "mode=${MODE} does not accept sample arguments: $*" >&2
    exit 1
  fi

  echo "[$(timestamp)] mode=${MODE} (${MODE_LABEL})"
  echo "[$(timestamp)] work_dir=${WORK_DIR}"
  echo "[$(timestamp)] config=${CONFIG_PATH}"
  echo "[$(timestamp)] started job=${MODE_LABEL} pid=$$"
  echo "[$(timestamp)] run: env ${CONFIG_ENV_VAR}=${CONFIG_PATH} python3 ./${PYTHON_SCRIPT}"
  set +e
  env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" python3 "./${PYTHON_SCRIPT}"
  status=$?
  set -e
  echo "[$(timestamp)] finished job=${MODE_LABEL} pid=$$ status=${status}"
  exit "${status}"
fi


if ! command -v c++ >/dev/null 2>&1; then
  echo "c++ is required to compile ${MODE_LABEL}." >&2
  exit 1
fi

if ! command -v root-config >/dev/null 2>&1; then
  echo "root-config is required to compile ${MODE_LABEL}." >&2
  exit 1
fi

cleanup_build_artifacts() {
  if [ -f "${BIN_PATH}" ]; then
    rm -f "${BIN_PATH}"
  fi
  if [ -d "${BIN_PATH}.dSYM" ]; then
    rm -rf "${BIN_PATH}.dSYM"
  fi
}



if $USE_SLURM; then
  trap - EXIT
else
  trap 'cleanup_build_artifacts; copy_log_to_output_dirs' EXIT
fi


timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] mode=${MODE} (${MODE_LABEL})"
echo "[$(timestamp)] work_dir=${WORK_DIR}"
echo "[$(timestamp)] config=${CONFIG_PATH}"
echo "[$(timestamp)] max_concurrent_jobs=${MAX_CONCURRENT_JOBS}"
if [ "${#REQUESTED_SAMPLES[@]}" -gt 0 ]; then
  echo "[$(timestamp)] cli_samples=${REQUESTED_SAMPLES[*]}"
fi

ROOT_CFLAGS="$(root-config --cflags)"
ROOT_LIBS="$(root-config --libs)"
ROOT_LIBDIR="$(root-config --libdir)"
COMPILE_CMD="c++ -O3 -DNDEBUG -std=c++17 ${ROOT_CFLAGS} ${OMP_CFLAGS} ./${SOURCE_FILE} -o ${BIN_PATH} ${ROOT_LIBS} ${OMP_LDFLAGS}"
echo "[$(timestamp)] compile: ${COMPILE_CMD}"
eval "${COMPILE_CMD}"
echo "[$(timestamp)] compile finished"

if [ "${MODE}" = "7" ]; then
  if [ "$#" -gt 0 ]; then
    echo "mode=${MODE} does not accept sample arguments: $*" >&2
    exit 1
  fi
  echo "[$(timestamp)] started job=${MODE_LABEL} pid=$$"
  echo "[$(timestamp)] run: env ${CONFIG_ENV_VAR}=${CONFIG_PATH} ${BIN_PATH}"
  set +e
  env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" "${BIN_PATH}"
  status=$?
  set -e
  echo "[$(timestamp)] finished job=${MODE_LABEL} pid=$$ status=${status}"
  exit "${status}"
fi

samples=()
while IFS= read -r sample; do
  if [ -n "${sample}" ]; then
    samples+=("${sample}")
  fi
done < <(
  python3 - "${CONFIG_PATH}" "${REQUESTED_SAMPLES[@]}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

sample_config = payload.get("sample_config", "../../src/sample.json")
if not isinstance(sample_config, str) or not sample_config:
    raise SystemExit("sample_config must be a non-empty string")

config_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
sample_config_path = sample_config if os.path.isabs(sample_config) else os.path.normpath(os.path.join(config_dir, sample_config))

with open(sample_config_path, "r", encoding="utf-8") as handle:
    sample_payload = json.load(handle)

rules = sample_payload.get("sample", [])
if not isinstance(rules, list):
    raise SystemExit("sample must be a JSON array")

seen = set()
all_samples = []
mc_samples = []
for rule in rules:
    if not isinstance(rule, dict):
        continue

    name = rule.get("name", "")
    if not isinstance(name, str) or not name:
        raise SystemExit("sample.name must be a non-empty string")

    is_mc = rule.get("is_MC")
    if not isinstance(is_mc, bool):
        raise SystemExit("sample.is_MC must be a boolean")

    if name in seen:
        continue
    seen.add(name)
    all_samples.append(name)
    if is_mc:
        mc_samples.append(name)

configured = payload.get("submit_samples", [])
if configured is None:
    configured = []
if not isinstance(configured, list):
    raise SystemExit("submit_samples must be a JSON array")
for sample in configured:
    if not isinstance(sample, str):
        raise SystemExit("submit_samples must contain only strings")

requested = [sample for sample in sys.argv[2:] if sample]
selected = requested if requested else configured
if not selected:
    selected = mc_samples

available = set(all_samples)
emitted = set()
for sample in selected:
    if sample not in available:
        raise SystemExit(f"Unknown sample requested: {sample}")
    if sample in emitted:
        continue
    emitted.add(sample)
    print(sample)
PY
)


if [ "${#samples[@]}" -eq 0 ]; then
  echo "No samples selected from ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[$(timestamp)] selected_samples=${samples[*]}"

if [ "${MODE}" = "0" ]; then
  RESUME_SUCCESSFUL_BATCHES="$(
    python3 - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

resume = payload.get("resume_successful_batches", True)
if not isinstance(resume, bool):
    raise SystemExit("resume_successful_batches must be a boolean")
print("1" if resume else "0")
PY
  )"
  if [ "${RESUME_SUCCESSFUL_BATCHES}" = "1" ]; then
    RESUME_SUCCESSFUL_BATCHES_LABEL=true
  else
    RESUME_SUCCESSFUL_BATCHES_LABEL=false
  fi
  echo "[$(timestamp)] resume_successful_batches=${RESUME_SUCCESSFUL_BATCHES_LABEL}"
  if [ "${RESUME_SUCCESSFUL_BATCHES}" = "1" ] && [ "${#samples[@]}" -ne 1 ]; then
    echo "resume_successful_batches requires exactly one selected sample, got ${#samples[@]}: ${samples[*]}" >&2
    exit 1
  fi
fi

declare -a RUNNING_PIDS=()
declare -a RUNNING_SAMPLES=()
FAILED_JOBS=0

reap_finished_jobs() {
  if $USE_SLURM; then
    reap_finished_jobs_slurm
  else
    reap_finished_jobs_local
  fi
}

reap_finished_jobs_slurm() {
  local new_pids=() new_samples=()
  local idx job_id sample state finished_any=0

  for idx in "${!RUNNING_PIDS[@]}"; do
    job_id="${RUNNING_PIDS[$idx]}"
    sample="${RUNNING_SAMPLES[$idx]}"

    state=$(squeue --jobs="${job_id}" --noheader --format="%T" 2>/dev/null || true)

    if [ -n "${state}" ]; then
      # Job still in queue (PENDING, RUNNING, etc.)
      new_pids+=("${job_id}")
      new_samples+=("${sample}")
      continue
    fi

    # Job gone from squeue — check final state via sacct
    finished_any=1
    local exit_code
    exit_code=$(sacct -j "${job_id}" --noheader --format=ExitCode --parsable2 \
                  2>/dev/null | head -1 | cut -d: -f1 || echo "1")

    if [ "${exit_code}" = "0" ]; then
      echo "[$(timestamp)] finished sample=${sample} slurm_job_id=${job_id} status=0"
    else
      echo "[$(timestamp)] finished sample=${sample} slurm_job_id=${job_id} status=${exit_code}"
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
  done

  RUNNING_PIDS=("${new_pids[@]}")
  RUNNING_SAMPLES=("${new_samples[@]}")
  return "${finished_any}"
}

reap_finished_jobs_local() {
  local new_pids=()
  local new_samples=()
  local idx pid sample status finished_any=0 state

  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    sample="${RUNNING_SAMPLES[$idx]}"
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"

    if [ -n "${state}" ] && [[ "${state}" != *Z* ]]; then
      new_pids+=("${pid}")
      new_samples+=("${sample}")
      continue
    fi

    finished_any=1
    if wait "${pid}"; then
      status=0
    else
      status=$?
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
    echo "[$(timestamp)] finished sample=${sample} pid=${pid} status=${status}"
  done

  RUNNING_PIDS=("${new_pids[@]}")
  RUNNING_SAMPLES=("${new_samples[@]}")
  return "${finished_any}"
}

launch_job() {
  local sample="$1"

  if $USE_SLURM; then
    local job_name="${MODE_LABEL}_${sample}"
    local slurm_log="${WORK_DIR}/${sample}_%j.out"

    local sbatch_args=(
      --job-name="${job_name}"
      --output="${slurm_log}"
      --error="${slurm_log}"
      --cpus-per-task="${SLURM_CPUS}"
      --mem="${SLURM_MEM}"
      --time="${SLURM_TIME}"
    )
    [ -n "${SLURM_PARTITION}" ] && sbatch_args+=(--account="${SLURM_PARTITION}")
    [ -n "${SLURM_EXTRA}" ]     && sbatch_args+=($SLURM_EXTRA)

    local job_id
    job_id=$(sbatch "${sbatch_args[@]}" \
      --wrap="export X509_USER_PROXY=${X509_DST}; export LD_LIBRARY_PATH=${ROOT_LIBDIR}:\${LD_LIBRARY_PATH:-}; env ${CONFIG_ENV_VAR}=${CONFIG_PATH} ${BIN_PATH} ${sample}" \
      | awk '{print $NF}')

    # Reuse the same pid-tracking arrays — store job_id in place of pid
    RUNNING_PIDS+=("${job_id}")
    RUNNING_SAMPLES+=("${sample}")
    echo "[$(timestamp)] submitted sample=${sample} slurm_job_id=${job_id}"
  else
    if [ "${MODE}" = "0" ]; then
      (
        set -e
        batch_count="$(env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" "${BIN_PATH}" "${sample}" --batch-count)"
        if [[ ! "${batch_count}" =~ ^[0-9]+$ ]] || [ "${batch_count}" -le 0 ]; then
          echo "Invalid convert batch count for sample=${sample}: ${batch_count}" >&2
          exit 1
        fi
        batch_failures=0
        successful_batches=""
        for ((batch_index = 0; batch_index < batch_count; batch_index++)); do
          echo "[$(timestamp)] running sample=${sample} batch=$((batch_index + 1))/${batch_count}"
          set +e
          env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" \
            "CONVERT_SUCCESSFUL_BATCHES=${successful_batches}" \
            "CONVERT_DEFER_FINAL_MERGE=1" \
            "${BIN_PATH}" "${sample}" "${batch_index}"
          batch_status=$?
          set -e
          if [ "${batch_status}" -ne 0 ]; then
            batch_failures=$((batch_failures + 1))
            echo "[$(timestamp)] warning: sample=${sample} batch=$((batch_index + 1))/${batch_count} failed status=${batch_status}; continuing"
            continue
          fi
          if [ -z "${successful_batches}" ]; then
            successful_batches="${batch_index}"
          else
            successful_batches="${successful_batches},${batch_index}"
          fi
        done
        if [ "${batch_failures}" -gt 0 ]; then
          echo "[$(timestamp)] warning: sample=${sample} completed with ${batch_failures}/${batch_count} failed batch(es); final output uses successful batches only"
        fi
        echo "[$(timestamp)] running sample=${sample} final merge from successful batches"
        env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" \
          "CONVERT_SUCCESSFUL_BATCHES=${successful_batches}" \
          "${BIN_PATH}" "${sample}" --merge-successful-batches
      ) &
    else
      nohup env "${CONFIG_ENV_VAR}=${CONFIG_PATH}" "${BIN_PATH}" "${sample}" >> "${LOG_PATH}" 2>&1 &
    fi
    local pid=$!
    RUNNING_PIDS+=("${pid}")
    RUNNING_SAMPLES+=("${sample}")
    echo "[$(timestamp)] started sample=${sample} pid=${pid}"
  fi
}

if $USE_SLURM; then
  # Submit all jobs at once — SLURM handles scheduling
  for sample in "${samples[@]}"; do
    launch_job "${sample}"
  done
  echo "[$(timestamp)] all jobs submitted to SLURM"
else
  for sample in "${samples[@]}"; do
    while [ "${#RUNNING_PIDS[@]}" -ge "${MAX_CONCURRENT_JOBS}" ]; do
      if ! reap_finished_jobs; then
        sleep 2
      fi
    done
    launch_job "${sample}"
  done

  while [ "${#RUNNING_PIDS[@]}" -gt 0 ]; do
    if ! reap_finished_jobs; then
      sleep 2
    fi
  done

  echo "[$(timestamp)] all jobs finished, failed_jobs=${FAILED_JOBS}"
  if [ "${FAILED_JOBS}" -ne 0 ]; then
    exit 1
  fi
fi
