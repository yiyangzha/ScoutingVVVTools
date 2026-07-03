#!/bin/bash

WORKDIR=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
# load CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}
export RELEASE_DIR=${BOOHFT_CMSSW_RELEASE_DIR:-$ROUTINE_DIR/$RELEASE}
if [ -r "$RELEASE_DIR/src" ] ; then
  echo found "$RELEASE_DIR"
else
  echo please setup "$RELEASE_DIR" env first
  exit 1
fi
cd "$RELEASE_DIR/src"
eval `scram runtime -sh`

# launch the fit
cd $WORKDIR
python3 "$SCRIPT_DIR/fit.py" "$@"
