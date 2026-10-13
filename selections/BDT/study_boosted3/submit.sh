#!/usr/bin/env bash
# Submit the boosted3 study: per variant one training job and, after it, the SR scan on the eval
# split plus the scoring (evaluate.py). Usage: bash submit.sh [variant ...] (default: all).
# STUDY_AFTER=<jobid:jobid...> holds the training jobs until those jobs succeeded (e.g. merges);
# STUDY_HOLD=1 submits the training jobs held (release with scontrol release).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BDT_DIR="$(dirname "$HERE")"
SR_DIR="$(dirname "$BDT_DIR")/signal_region"
ACCOUNT="${STUDY_ACCOUNT:-cms-express}"
EXCLUDE=()
[ -n "${STUDY_EXCLUDE:-}" ] && EXCLUDE=(--exclude="$STUDY_EXCLUDE")
AFTER=()
[ -n "${STUDY_AFTER:-}" ] && AFTER=(--dependency=afterok:"$STUDY_AFTER")
HOLD=()
[ -n "${STUDY_HOLD:-}" ] && HOLD=(--hold)
python3 "$HERE/make_study.py" "$@" > /dev/null
variants=("$@")
[ "${#variants[@]}" -gt 0 ] || variants=($(python3 -c "import sys; sys.path.insert(0, '$HERE'); import make_study; print(' '.join(make_study.VARIANTS))"))
for v in "${variants[@]}"; do
  d="$HERE/$v"
  train_id=$(sbatch --parsable --job-name="study_b3_${v}" --account="$ACCOUNT" "${EXCLUDE[@]}" "${AFTER[@]}" "${HOLD[@]}" \
    --cpus-per-task=32 --mem=48G --time=24:00:00 --output="$d/train.log" \
    --wrap="cd '$BDT_DIR' && BDT_CONFIG_PATH='$d/bdt_config.json' python3 train.py")
  sr_id=$(sbatch --parsable --job-name="study_b3_${v}_sr" --account="$ACCOUNT" "${EXCLUDE[@]}" \
    --dependency=afterok:"$train_id" --cpus-per-task=8 --mem=16G --time=06:00:00 --output="$d/sr_eval.log" \
    --wrap="cd '$SR_DIR' && python3 signal_region_hist.py '$d/sr_config.json' && python3 '$HERE/evaluate.py' '$v'")
  echo "$v: train=$train_id sr+evaluate=$sr_id"
done
