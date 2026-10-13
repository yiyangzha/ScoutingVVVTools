#!/bin/bash
#SBATCH --job-name=sr_sensitivity_data
#SBATCH --account=cms-express
#SBATCH --partition=hammer-nodes
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-311%40
#SBATCH --output=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/plotting/sr_sensitivity_data_logs/sr_sensitivity_data_%a_%A.log
#SBATCH --error=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/plotting/sr_sensitivity_data_logs/sr_sensitivity_data_%a_%A.log
# Blinded data histograms for the SR sensitivity plots: array task K runs
# `sr_sensitivity.py --data-file K` on data file K (sorted data_2024_*.root) for every target.
# The --array range must cover all data files; plotting fails if any file's histogram is missing.
set -u
echo "[sbatch] start $(date) host $(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src
eval "$(scramv1 runtime -sh)"
cd ScoutingVVVTools/plotting
export MPLBACKEND=Agg
python3 ./sr_sensitivity.py --data-file $SLURM_ARRAY_TASK_ID
st=$?
echo "[sbatch] done $(date) exit=$st"
exit $st
