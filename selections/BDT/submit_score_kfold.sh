#!/bin/bash
# Submit score_kfold.py for every sample of a k-fold tree root (train.py kfold), one SLURM job per
# sample; prints the job ids (comma-separated) on the last line.
#   submit_score_kfold.sh <tree_root> [--after <jobids>] [--branches a,b] [--account A] [--exclude nodes]
#                         [--mem 16G] [--time 04:00:00]
# Logs: <tree_root>/kfold_scores/logs/<sample>.log. The environment of the calling shell (cmsenv) is
# passed to the jobs.
set -euo pipefail
ROOT=$(readlink -f "$1"); shift
AFTER=""; BRANCHES=""; ACCOUNT="cms-express"; EXCLUDE=""; MEM="16G"; TIME="04:00:00"
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER="$2"; shift 2 ;;
        --branches) BRANCHES="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done
HERE=$(dirname "$(readlink -f "$0")")
SAMPLES=$(python3 -c "import json; c = json.load(open('$ROOT/fold0/config.json')); print(' '.join(s for m in c['class_groups'].values() for s in m))")
mkdir -p "$ROOT/kfold_scores/logs"
IDS=()
for s in $SAMPLES; do
    args=(sbatch --parsable --job-name="kfold_score_$s" --account="$ACCOUNT" --cpus-per-task=4 --mem="$MEM"
          --time="$TIME" --output="$ROOT/kfold_scores/logs/$s.log")
    [ -n "$AFTER" ] && args+=(--dependency="afterok:${AFTER//,/:}")
    [ -n "$EXCLUDE" ] && args+=(--exclude="$EXCLUDE")
    IDS+=("$("${args[@]}" --wrap="cd $HERE && python3 score_kfold.py $ROOT $s ${BRANCHES:+--branches $BRANCHES}")")
done
echo "submitted ${#IDS[@]} scoring jobs" >&2
(IFS=,; echo "${IDS[*]}")
