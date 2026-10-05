#!/usr/bin/env bash
# Build the data pileup profiles (nominal / low / high minimum-bias cross section) for exactly
# the luminosity the analysis processed, from the processed-lumi JSON that convert_branch
# writes next to the merged data output ({output_root}/data/{sample}_processed_lumis.json).
#
# Run where the BRIL pileup JSON is readable (lxplus / EOS) inside a CMSSW environment:
#   bash make_data_pileup.sh <processed_lumis.json> <pileup_JSON.txt> [tag]
# e.g. the 2024 pileup JSON published by the DQM/certification team under
#   /eos/user/c/cmsdqm/www/CAF/certification/Collisions24/PileUp/   (check the current file)
#
# Writes dataPileupHistogram-<tag>-{69200,66000,72400}ub.root (histogram "pileup", 100 bins in
# [0, 100], the binning of the existing profiles). Copy them to selections/weight/pileup/, point
# pileup_data_files in selections/weight/config.json at them, rerun mode 1 (weight.C) and then
# reconvert the MC so weight_pu and the stored weight_pu*_mean follow the analysed data.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <processed_lumis.json> <pileup_JSON.txt> [tag]" >&2
  exit 1
fi
LUMI_JSON="$1"
PILEUP_JSON="$2"
TAG="${3:-2024GHI_processed}"

for xsec in 69200 66000 72400; do
  out="dataPileupHistogram-${TAG}-${xsec}ub.root"
  echo "pileupCalc: minBiasXsec=${xsec} ub -> ${out}"
  pileupCalc.py -i "${LUMI_JSON}" --inputLumiJSON "${PILEUP_JSON}" --calcMode true \
    --minBiasXsec "${xsec}" --maxPileupBin 100 --numPileupBins 100 "${out}"
done
