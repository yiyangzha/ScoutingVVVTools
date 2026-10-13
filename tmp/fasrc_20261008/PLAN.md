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

## IHEP integration plan (proposed 2026-10-08T21:00-04:00, awaiting approval)

Facts (read-only checks over SSH, key login as yiyangzhao): lxlogin AlmaLinux 9.8 (128 cores, CMSSW_16_1_0_pre4 el9_amd64_gcc13 native from CVMFS), HTCondor through /afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin/hep_sub (-g cms, -os AlmaLinux9, -cpu, -m, -wt), IHEP Slurm only on the GPU cluster (account cmsgpu); /ihepfs Lustre 75 TB free (user has 3.75 TB, no quota); AFS home unreadable without a token (scripts run with bash --noprofile --norc and HOME-independent settings); GitHub reachable (branch cloned); proxy at /ihepfs/cms_thu/yiyangzhao/Research/claude/proxy/x509up (VO cms, ~8 days). Reads: root://cceos.ihep.ac.cn//eos/ihep/cms/<LFN> 0.69 s per 110 MB file (~160 MB/s); global redirector from IHEP 146 s per file.

Split: IHEP converts the IHEP-hosted samples (2024G_UParT_v6 + the 32 yiyangz MC) for the ttbar CR and the main analysis; Cannon converts the 52 Purdue-hosted MC; merged outputs from IHEP are rsynced to the Cannon scratch dataset trees, and their sample.json fields (raw_entries, weight_pu*_mean, sum_genweight*) are copied into Cannon's src/sample.json; everything downstream runs on Cannon.

Code changes:
1. convert_branch.C and weight.C: optional config key `remote_input_rules` = list of {"lfn_prefix", "url_prefix"}; a discovered DAS LFN starting with lfn_prefix becomes url_prefix + LFN, every other LFN keeps root://cms-xrd-global.cern.ch/ (default unchanged when the key is absent). IHEP configs map /store/user/yiyangz/ to root://cceos.ihep.ac.cn//eos/ihep/cms. The .files snapshot and the .meta slice hash use the resulting URLs as now.
2. weight.C: remote opens retried 5 times with 5 s pauses (converter constants); a file that still cannot be opened, or lacks the tree/branch, is skipped with a warning (MC only, as in convert_branch); the run fails if every file is skipped; skipped files are listed in the log and in pileup_<sample>_skipped.txt next to the CSV.
3. run.py: site profile `ihep` (explicit proxy, cxx_wrapper sites/ihep/cmssw_exec.sh = cmsset_default + cmsenv in the repo CMSSW_16_1_0_pre4 el9 area, HOME-independent, no Slurm) with an HTCondor scheduler through hep_sub: mode 0 --condor submits one hep_sub job per batch (job script under build/ihep/jobs/, -g cms -os AlmaLinux9 -cpu N -m MEM -wt), mode 0 --merge-only submits the per-sample merge job after the batches; mode 1 per-sample jobs likewise. Compilation and the --batch-count DAS query run on lxlogin (one short compile per source).
4. New untracked campaign configs on IHEP: selections/convert/config_ihep.json and selections/convert_ttbar_cr/config_ihep.json (output_root /ihepfs/cms_thu/yiyangzhao/Research/claude/ntuples/fasrc_20261008/dataset{,_cr}, remote_input_rules), weight config_ihep.json if PU runs there (not needed: PU CSVs come from Cannon).
5. README/AGENTS: site ihep, remote_input_rules, weight.C skip behavior.
Verification: pilot one data batch and one yiyangz MC batch on IHEP (time, size, meta), merge, rsync, compare with the Cannon CR data pilot output (same events for the same 5 files).

## Merge, optimizer extension and Purdue handoff (approved 2026-10-13, requirements 48-66)

0. Commit the uncommitted IHEP/records work (done: 2bf5ceb).
1. Merge refs/remotes/jan/optimizations (e972efa) with the card decisions (weights mc_weights.py everywhere; theory acceptance-only + xsec lnN; Jan's class_groups, HT < 400 GeV QCD excluded from physics yields; production configs of the review; min_bkg_entries 0; our branch-and-bound limits).
2. signal_region_hist.py: combine-consistent background model (predicted QCD in ABCD mode, per-class theory_scale and xsec rows, no MC nuisances on the data-driven QCD), control-region-aware deduplication and boundary placement with a sorted point index, joint correlated stage (joint_z, joint_refine_selection), fine refinement guarded by Z_joint.
3. Documentation: AGENTS.md (condensed, section 6 updated) and README.md.
4. Handoff prompt tmp/prompt_purdue_20261013.md (Part A for Yiyang at Purdue, Part B for the new agent; ABCDEF, VV MC statistics, data access, validation items).
5. Independent static review, ast syntax check (approved), merge commit, push to origin fasrc-campaign-20261008, git status / check-ignore report.
