#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_INPUT="${1:-${SCRIPT_DIR}/config.json}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
LOG_PATH="${SCRIPT_DIR}/log.txt"
BIN_PATH="${SCRIPT_DIR}/convert_branch"

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
  echo "python3 is required to read and rewrite JSON config files." >&2
  exit 1
fi

if ! command -v c++ >/dev/null 2>&1; then
  echo "c++ is required to compile convert_branch." >&2
  exit 1
fi

if ! command -v root-config >/dev/null 2>&1; then
  echo "root-config is required to compile convert_branch." >&2
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

cd "${SCRIPT_DIR}"
: > "${LOG_PATH}"
exec >> "${LOG_PATH}" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] start run_all_mc_nohup.sh"
echo "[$(timestamp)] config=${CONFIG_PATH}"
echo "[$(timestamp)] max_concurrent_jobs=${MAX_CONCURRENT_JOBS}"

ROOT_CFLAGS="$(root-config --cflags)"
ROOT_LIBS="$(root-config --libs)"
COMPILE_CMD="c++ -O3 -DNDEBUG -std=c++17 ${ROOT_CFLAGS} ${OMP_CFLAGS} ./convert_branch.C -o ${BIN_PATH} ${ROOT_LIBS} ${OMP_LDFLAGS}"
echo "[$(timestamp)] compile: ${COMPILE_CMD}"
eval "${COMPILE_CMD}"
echo "[$(timestamp)] compile finished"

samples=()
while IFS= read -r sample; do
  if [ -n "${sample}" ]; then
    samples+=("${sample}")
  fi
done < <(
  python3 - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

rules = payload.get("sample_rules", [])
if not isinstance(rules, list):
    raise SystemExit("sample_rules must be a JSON array")

seen = set()
for rule in rules:
    if not isinstance(rule, dict):
        continue

    category = rule.get("category", "")
    if category == "data":
        continue

    contains_any = rule.get("contains_any", [])
    if not isinstance(contains_any, list):
        raise SystemExit("sample_rules.contains_any must be a JSON array")

    for sample in contains_any:
        if not isinstance(sample, str) or not sample or sample in seen:
            continue
        seen.add(sample)
        print(sample)
PY
)

if [ "${#samples[@]}" -eq 0 ]; then
  echo "No non-data samples found in sample_rules of ${CONFIG_PATH}" >&2
  exit 1
fi

declare -a RUNNING_PIDS=()
declare -a RUNNING_SAMPLES=()
FAILED_JOBS=0

reap_finished_jobs() {
  local new_pids=()
  local new_samples=()
  local idx pid sample status finished_any=0

  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    sample="${RUNNING_SAMPLES[$idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
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
  nohup env CONVERT_CONFIG_PATH="${CONFIG_PATH}" "${BIN_PATH}" "${sample}" >> "${LOG_PATH}" 2>&1 &
  local pid=$!
  RUNNING_PIDS+=("${pid}")
  RUNNING_SAMPLES+=("${sample}")
  echo "[$(timestamp)] started sample=${sample} pid=${pid}"
}

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
