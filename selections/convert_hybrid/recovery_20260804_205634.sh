#!/usr/bin/env bash
set -e
# Recovery for 26 sample(s) generated 20260804_205634
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools

# 2024B: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json 2024B --slurm

# 2024I: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json 2024I --slurm

# qcd_ht600to800: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht600to800 --slurm

# qcd_ht800to1000: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht800to1000 --slurm

# qcd_ht1200to1500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht1200to1500 --slurm

# qcd_ht1500to2000: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht1500to2000 --slurm

# qcd_ht2000: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json qcd_ht2000 --slurm

# ttbar_had: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json ttbar_had --slurm

# ttbar_semilep: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json ttbar_semilep --slurm

# wjets_h100to400: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h100to400 --slurm

# wjets_h400to800: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h400to800 --slurm

# wjets_h800to1500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h800to1500 --slurm

# wjets_h1500to2500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h1500to2500 --slurm

# wjets_h2500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wjets_h2500 --slurm

# zjets_h400to800: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json zjets_h400to800 --slurm

# zjets_h800to1500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json zjets_h800to1500 --slurm

# zjets_h1500to2500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json zjets_h1500to2500 --slurm

# zjets_h2500: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json zjets_h2500 --slurm

# wlnu_h1500to2500_m0to120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h1500to2500_m0to120 --slurm

# wlnu_h1500to2500_m120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h1500to2500_m120 --slurm

# wlnu_h2500_m0to120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h2500_m0to120 --slurm

# wlnu_h2500_m120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h2500_m120 --slurm

# wlnu_h400to800_m0to120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h400to800_m0to120 --slurm

# wlnu_h800to1500_m0to120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h800to1500_m0to120 --slurm

# wlnu_h800to1500_m120: STALE_SCHEMA — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json wlnu_h800to1500_m120 --slurm

# data_2024: OUTPUT_MISSING — reprocess + merge (temps incomplete/removed)
python3 run.py 0 /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/selections/convert_hybrid/config.json data_2024 --slurm

