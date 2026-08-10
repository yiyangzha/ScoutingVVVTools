#!/usr/bin/env bash
set -e
# Recovery for 4 sample(s) generated 20260804_210815
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools


# wjets_h100to400: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h100to400 --slurm


