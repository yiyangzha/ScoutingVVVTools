# FASRC fresh-campaign plan

Created: 2026-10-08T10:14:23-04:00 (America/New_York).

## Objective and acceptance

Complete the fresh Part B campaign at source commit fbe3bc40214ede6366d999605d5b06b361679179: PU, corrected ttbar CR, new JMS/JMR fit, five nominal analysis trees plus eight MC variations per tree, confirmed mixing/training, models/diagnostics, SRs, QCD ABCD, every evaluable systematic, Data/MC/class-shape plots, and full-systematics expected combine significance/limits for the agreed combined/class/sample/channel scenarios. A submission or zero exit code alone is not completion.

## Confirmed operating decisions

- Use only 2024G_UParT_v6 data; preserve the historical 255.8 fb^-1 sensitivity target, retained C-I PU inputs, six BDT classes, and current physics until evidence and confirmation justify a change.
- Use private proxy backup build/fasrc/credentials/cms-20261008T101031-0400/x509up.
- Write only in the repository, approved project scratch, and the new singular external ntuple directory. Preserve every file. Step outputs use default step directories; ask about non-default locations and any fresh-namespace conflict.
- No local tests, arbitrary installs/downloads, or mutable git actions. Ask visibly in Chinese and record each answer.

## Stages

1. Orientation and initial safety inventory (in progress). Verify complete prompt/current docs/history/site-reference reading, working visible card, exact checkout, credential backup, current storage/account/partition/software evidence. Complete worker inventory and accurately record source-reading coverage.
2. Complete current pipeline audit and identify decisions (pending). Read every relevant source/config fully. Verify cleanup/replacement/write behavior, path-resolution bases, model/split/variation correspondence, physical weighting and required calibration/source scope. Ask before physical fixes, remote source access, extra installations, or unresolved output layout choices.
3. Modular server/environment integration (pending). State affected files/functions/config/output/downstream/docs changes. Fix delegated CPU/GPU pixi design and Slurm/proxy/build/resource integration, retain other sites, preserve tracked convert binary, and retain build/intermediate artifacts. Verify on suitable workers without local tests.
4. Source metadata and normalization (pending). Inspect registered input schemas and Events/Runs/LuminosityBlocks/theory weights on workers within confirmed remote access scope. Establish fresh source manifests and processing counters; confirm any physics correction before editing.
5. Fresh PU, ttbar CR, and JMS/JMR (pending). Rebuild MC PU CSVs from the retained references, convert agreed CR samples/data, fit at actual CR precision, and verify complete input/chunk coverage and fit diagnostics.
6. Fresh analysis conversion and durable ntuples (pending). Produce nominal plus all eight MC jet variations with fixed source lists; validate every batch/merge/schema/sidecar/hash and store reusable ntuples/provenance in the exempt Lab directory. Preserve all intermediates.
7. Confirmed mixing and GPU training (pending). Produce five models and diagnostics with exact sample/split/reference correspondence and verified actual GPU execution. Account explicitly for zero-entry channels/samples.
8. SR optimization and QCD (pending). Compare existing optimizers fairly and reevaluate promising results with full systematics within a stated feasible budget. Produce final disjoint SRs and matching QCD outputs, verify metadata format and reference checks.
9. Systematics and plots (pending). Complete modes 8/9/11/12/13/14, coverage and ratio validation, all enabled/evaluable nuisances, and confirmed-normalization Data/MC/class-shape plots with fresh inputs.
10. Combine, standalone studies, and final verification (pending). Complete agreed scenarios and ancillary scope on workers, inspect fit/quantile/card evidence, reconcile jobs and all outputs/provenance. Reread complete records; list new-file git/ignore state. Present any necessary deletion proposals only at the final decision.

## Source-reading coverage

Completely read in this Cannon session: tmp/prompt.md (659 lines), current AGENTS.md (430 lines before this update), README.md (172 lines), all three lxplus historical records, historical backup prompt (241 lines), copied global AGENTS.md, IAIFI PDF via full pdftotext, lep_calib_FASRC_HANDOFF.md (863 lines), .gitignore, pixi.toml, run.py (942 lines), and existing build/fasrc/session.env. Truncated outputs were reread. Full current pipeline source/config reading remains incomplete; previous-agent coverage is not current-session coverage.

## Immediate independent work

Submitted read-only inventory jobs 51318524 (test CPU) and 51318526 (iaifi_gpu GPU), each one task/CPU, 4G, 10 minutes, with retained submission text/stdout/stderr under build/fasrc/fasrc_20261008/inventory/. CPU worker holy8a24102 is Rocky Linux 8.10, can read CMS CVMFS and the EL9 CMSSW_16_1_0_pre4 release, and validates the backed-up CMS proxy; this OS/release pairing requires a compatible runtime before execution. GPU inventory and complete source/config audit remain in progress. Do not invoke the existing runner before its cleanup and write behavior is adapted. Actual source ROOT/DAS accesses and physics edits remain unperformed.

## Claude session plan (2026-10-08T11:50-04:00)

Paths: repo R=/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools; scratch S=/n/netscratch/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting (VVV_SCRATCH; 50 TB/lab, 90-day purge per https://docs.rc.fas.harvard.edu/kb/policy-scratch/); durable ntuples copied to /n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ntuples/fasrc_20261008/ only after completion (ask if larger than the free Lab quota). Step results in default directories. Campaign converter configs are new untracked JSON files next to the checked-in ones (same directory, so the converter finds the sibling branch.json/selection.json) whose only differences are output_root on scratch and campaign paths; downstream configs: checked-in defaults where they already point to default locations, otherwise campaign copies (to be confirmed per stage: non-default *input* paths such as the scratch ntuple root are unavoidable).

Stage status:
1. Inventory/decisions: done for storage/environment/access (card 1). Round 2 pending: program-internal temp deletions, generator-weight/PU normalization, AK4 cleaning, mixing. Round 3: SR optimizer and downstream SR paths, disabled validations, Data/MC 255.8 projection, PS index convention, standalone studies.
2. Server integration: run.py site profiles + sites/fasrc/cmssw_exec.sh + format/thread/GPU-log fixes + weight.C mutex implemented; pixi fixed and verified (CPU+GPU). Next: worker smoke test of run.py --site fasrc (driver submission, compile in container, one small sample).
3. Source metadata: job 51327134 (one file per sample, 87 samples). Then fresh genweight_mean (all Runs trees) via convert_branch --update-genweight-mean pilot/production once weighting is decided.
4. PU: copy the three verified reference ROOT files into selections/weight/pileup/ (cp -p --no-clobber); validate histogram name/binning on a worker; pilot mode 1 on one small sample; then all MC in scope.
5. ttbar CR: pilot conversion of 1-2 batches of ttbar_semilep and 2024G_UParT_v6 with apply_jms_jmr false; then full CR (39 samples); fit JMS/JMR (actual 27.51 fb^-1 statistics); review fit.
6. Analysis conversion: pilot one batch each of a signal, QCD, ttbar and data sample; measure time/size per file -> resource plan and storage estimate; full production with nominal + 8 variations; validate batches/merges; record sizes.
7-10. Mix/train (GPU), SR + QCD, systematics + plots, combine, standalone studies, final verification and copy to ntuples/.
