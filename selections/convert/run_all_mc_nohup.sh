#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${SCRIPT_DIR}/config.json}"
JOB_CONFIG_DIR="${2:-${SCRIPT_DIR}/job_configs}"
LOG_DIR="${3:-${SCRIPT_DIR}/logs}"
MACRO_ARG="${REPO_ROOT}/convert/convert_branch.C()"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to read and rewrite JSON config files." >&2
  exit 1
fi

if ! command -v root >/dev/null 2>&1; then
  echo "ROOT is required to launch convert jobs." >&2
  exit 1
fi

mkdir -p "${JOB_CONFIG_DIR}" "${LOG_DIR}"

mapfile -t samples < <(
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
    input_root_key = rule.get("input_root_key", "")
    if category == "data" or input_root_key == "data":
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

for sample in "${samples[@]}"; do
  job_config="${JOB_CONFIG_DIR}/${sample}.json"
  log_path="${LOG_DIR}/convert_${sample}.log"
  pid_path="${LOG_DIR}/convert_${sample}.pid"

  python3 - "${CONFIG_PATH}" "${job_config}" "${sample}" <<'PY'
import json
import sys

src_path, dst_path, sample_name = sys.argv[1:]
with open(src_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

payload["run_sample"] = sample_name

with open(dst_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  nohup env CONVERT_CONFIG_PATH="${job_config}" root -l -b -q "${MACRO_ARG}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" > "${pid_path}"
  printf 'started sample=%s pid=%s log=%s config=%s\n' "${sample}" "${pid}" "${log_path}" "${job_config}"
done
