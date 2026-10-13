#!/bin/bash
# Run a command in the CMSSW_16_1_0_pre4 (el9_amd64_gcc13) environment of this repository's
# project area. On IHEP (AlmaLinux 9 lxlogin and HTCondor workers) run.py calls it directly:
# sites/ihep/cmssw_exec.sh <cmd> ... The working directory and the caller's environment
# (config paths, X509_USER_PROXY) are kept.
set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmssw_area="${VVV_CMSSW_AREA:-$repo_dir/CMSSW_16_1_0_pre4}"
if [ ! -d "$cmssw_area/src" ]; then
    echo "cmssw_exec.sh: CMSSW area $cmssw_area not found; create it once with" \
         "'source /cvmfs/cms.cern.ch/cmsset_default.sh && cd $repo_dir &&" \
         "SCRAM_ARCH=el9_amd64_gcc13 scram project CMSSW CMSSW_16_1_0_pre4'" >&2
    exit 1
fi

# The AFS home is unreadable without an AFS token (SSH key logins, batch jobs); dasgoclient
# needs a writable home for its DAS maps.
if [ ! -w "${HOME:-/nonexistent}" ]; then
    export HOME="$repo_dir/build/ihep/home"
    mkdir -p "$HOME"
fi

# The HTCondor workers have no /etc/grid-security: without a CA directory the XRootD client
# of ROOT is refused by the IHEP EOS door ([3010] Operation not permitted). Use the CVMFS
# grid trust store there.
if [ -z "${X509_CERT_DIR:-}" ] && [ ! -d /etc/grid-security/certificates ]; then
    export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
fi
if [ -z "${X509_VOMS_DIR:-}" ] && [ ! -d /etc/grid-security/vomsdir/cms ]; then
    export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
work_dir="$PWD"
cd "$cmssw_area/src"
eval "$(scram runtime -sh)"
cd "$work_dir"
# correctionlib and the Python packages come from CMSSW, never from a user-site copy.
export PYTHONNOUSERSITE=1

exec "$@"
