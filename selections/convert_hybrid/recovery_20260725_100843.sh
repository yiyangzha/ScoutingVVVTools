#!/usr/bin/env bash
set -e
# Recovery for 5 sample(s) generated 20260725_100843
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools

# qcd_ht1000to1200: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht1000to1200 --slurm

# wlnu_h100to400_m0to120: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h100to400_m0to120 --slurm


