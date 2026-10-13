#!/bin/bash
#SBATCH --job-name=sr_qcd_est
#SBATCH --account=cms-express
#SBATCH --partition=hammer-nodes
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/background_estimation/sr_qcd_est_%x_%j.log
#SBATCH --error=/depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src/ScoutingVVVTools/background_estimation/sr_qcd_est_%x_%j.log
# Signal-region optimisation (signal_region_hist.py) followed by the QCD ABCD
# estimate (qcd_est.py) for one analysis category.
# Usage: sbatch --job-name=sr_qcd_<cat> submit_sr_qcd_est.sbatch <category>
set -u
CAT=$1
echo "[sbatch] start $(date) host $(hostname) job=$SLURM_JOB_ID category=$CAT"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /depot/cms/private/users/schul105/VVV/analysis/CMSSW_16_1_0_pre4/src
eval "$(scramv1 runtime -sh)"
export MPLBACKEND=Agg
TOOLS=$PWD/ScoutingVVVTools

cd $TOOLS/selections/signal_region
echo "[sbatch] running signal_region_hist.py config_hist_$CAT.json ..."
python3 ./signal_region_hist.py config_hist_$CAT.json
st=$?
echo "[sbatch] signal_region_hist exit=$st"
[ $st -ne 0 ] && exit $st

cd $TOOLS/background_estimation
echo "[sbatch] running qcd_est.py config_$CAT.json ..."
QCD_EST_CONFIG_PATH=$PWD/config_$CAT.json python3 ./qcd_est.py
st=$?
echo "[sbatch] qcd_est exit=$st"
echo "[sbatch] done $(date)"
exit $st
