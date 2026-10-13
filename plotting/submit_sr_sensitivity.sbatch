#!/bin/bash
#SBATCH --job-name=sr_sensitivity
#SBATCH --account=cms-express
#SBATCH --partition=hammer-nodes
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/plotting/sr_sensitivity_%j.log
#SBATCH --error=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/plotting/sr_sensitivity_%j.log
# SR sensitivity plots (plotting/sr_sensitivity.py, config plotting/config_sr_sensitivity.json).
set -u
echo "[sbatch] start $(date) host $(hostname) job=$SLURM_JOB_ID"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src
eval "$(scramv1 runtime -sh)"
cd ScoutingVVVTools/plotting
export MPLBACKEND=Agg
python3 ./sr_sensitivity.py
st=$?
echo "[sbatch] done $(date) exit=$st"
exit $st
