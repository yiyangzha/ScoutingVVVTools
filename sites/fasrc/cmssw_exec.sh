#!/bin/bash
# Run a command in the CMSSW_16_1_0_pre4 (el9_amd64_gcc13) environment of this repository's
# project area. On FASRC Cannon (Rocky 8 hosts) run.py calls it through the CVMFS container
# wrapper: /cvmfs/cms.cern.ch/common/cmssw-el9 --command-to-run sites/fasrc/cmssw_exec.sh <cmd> ...
# The working directory and the caller's environment (config paths, X509_USER_PROXY, TMPDIR)
# are kept.
set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmssw_area="${VVV_CMSSW_AREA:-$repo_dir/CMSSW_16_1_0_pre4}"
if [ ! -d "$cmssw_area/src" ]; then
    echo "cmssw_exec.sh: CMSSW area $cmssw_area not found; create it once inside the container with" \
         "'cd $repo_dir && SCRAM_ARCH=el9_amd64_gcc13 scram project CMSSW CMSSW_16_1_0_pre4'" >&2
    exit 1
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
work_dir="$PWD"
cd "$cmssw_area/src"
eval "$(scram runtime -sh)"
cd "$work_dir"
# correctionlib and the Python packages come from CMSSW, never from a user-site copy.
export PYTHONNOUSERSITE=1

exec "$@"
