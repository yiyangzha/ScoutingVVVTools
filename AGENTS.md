# AGENTS.md

Whenever you modify the code, you must also update the contents of AGENTS.md accordingly. Do not run test locally at any circumstance.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Guardrails

- Never overwrite user edits between reads.
- Never restore deleted code without confirmation.
- Make the smallest fix that solves the problem.
- No scope drift: no refactors, restyles, or extras unless asked.
- Fix root causes, not symptoms.
- Use web search for unstable or version-specific behavior; cite sources.
- State assumptions; ask only when blocked.
- Briefly narrate multi-step tool usage.
- Finish the full plan once started.
- If anything in the user's request is unclear, surprising, contradictory, or looks possibly wrong, stop and ask for clarification before modifying the code. Only proceed once the user has confirmed.

## 1. Non-Destructive File Handling

**Preserve user data. Prefer additive changes over destructive ones.**

- Never delete files or directories.
- Never run destructive commands or cleanup operations that remove user data.
- Do not use `rm`, `del`, `erase`, `rmdir`, `Remove-Item`, or equivalents.
- Do not remove files just because they look temporary, generated, redundant, cached, old, or replaceable.
- Do not delete files before recreating or renaming them.
- If replacement is needed, write a new file and leave the original intact.
- If renaming fails, keep the old file and create a new one with a different name.
- If deletion seems necessary, stop and propose a non-destructive alternative first.

Default rule: when in doubt, keep all existing files.

## 2. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 4. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


# Repository Information

This repository is a CMS Run 3 Scouting VVV analysis toolkit. It processes ScoutingNano-style ROOT files, builds `fat2`/`fat3` analysis trees, optionally shuffles converted MC chunks, trains BDT/NN classifiers, defines signal regions, validates QCD with ABCD, plots data/MC, runs CMS combine, and derives tagger scale factors.

## Operating Rules

- Do not run local tests, builds, or sample processing unless the user explicitly reverses that instruction.
- Prefer static inspection (`rg`, `sed`, `git diff`) for verification in this repository.
- Treat `src/sample.json` as the single source for sample paths, luminosity, and xsections.
- Do not use offline reco `Muon`, `Jet`, `FatJet`, or offline MET branches for Scouting workflows when Scouting branches exist.
- Keep edits scoped to the requested workflow; do not refactor unrelated analysis code.

## Main Modes

- `./run.sh 1`: pileup weights from `selections/weight/config.json`.
- `./run.sh 0`: convert ScoutingNano into `fat2`/`fat3` trees using `selections/convert/` configs.
- `./run.sh 6`: deterministic MC chunk mixing using `selections/mix/config.json`.
- `./run.sh 2`: BDT/NN training from `selections/BDT/config.json`.
- `./run.sh 3`: signal-region optimization.
- `./run.sh 5`: QCD ABCD validation and combine-facing ROOT output.
- `./run.sh 4`: data/MC plotting.
- `./run.sh 7`: CMS combine wrapper.
- `python3 run.py 11`: ScoutingNano to topwsf ntuple jobs; default config `systematics/scale_factor/ntuple_config.json`.
- `python3 run.py 12`: topwsf scale-factor cards/fits; default config `systematics/scale_factor/sf_config.json`.

For sample-parallel C++ modes (`0`, `1`, `6`), `run.py --slurm` submits SLURM jobs. The scale-factor ntuple step uses the vendored `nano.cpp` Condor workflow instead.

## Scale-Factor Workflow

`systematics/scale_factor/` vendors `nano.cpp` for ntuple production and `boohft-calib` for the top/W SF fit. The controller in `scale_factor.py` only generates sample YAMLs and topwsf cards and launches those existing tools. `nano.cpp/external/CMSJMECalculators` is vendored source from the upstream `nano.cpp` submodule at commit `f31313d87bc917b0e6a2114b08a83a408bf8608c`; keep nested git metadata out of that directory.

Current configured target:

- AK8 W jets only.
- Taggers: `WvsQCD`, `ZvsQCD`, `VvsQCD`.
- Data sample: `data_2024` requiring `DST_PFScouting_SingleMuon`.
- Scouting objects: `ScoutingMuonVtx`, `ScoutingFatPFJetRecluster`, `ScoutingPFJetRecluster2`, and `ScoutingMET`.

Remote ScoutingNano inputs may be DAS datasets, `/store/...` files, `root://...` files, local directories, local ROOT files, or text/list/YAML file lists. Remote ROOT files are streamed by default; do not change to full local download unless requested. Step-1 output and all step-2 paths are local.

For `/.../USER` dataset resolution, scale-factor `nano.cpp` first tries quoted DAS QL values with the legacy `instance="prod/phys03"` query and then falls back to quoted `system="rucio"`, quoted `system="dbs3"`, quoted plain DAS, and the original unquoted legacy forms. The controller prefers `/cvmfs/cms.cern.ch/common/dasgoclient` when available, with `SCALE_FACTOR_DASGOCLIENT` as an explicit override. DAS queries run with pixi/conda identity variables removed, active `CONDA_PREFIX` paths stripped from path-like variables, and `/cvmfs/cms.cern.ch/cmsset_default.sh` sourced immediately before executing the CVMFS DAS client, while preserving proxy environment.

The scale-factor ntuple config uses `ntuple.build_jobs` for the local `nano.cpp` build; `"auto"` uses the visible CPU count. Condor ntuple jobs still request one CPU, but worker builds use all CPUs visible inside the slot unless `BUILD_JOBS` is set.

The Scouting ntuple card uses strict branch binding. Missing configured ScoutingNano branches fail immediately, except `GenPart_*`: MC jobs warn in their per-job Condor logs and fall back to default W/Z matches with `GenJet` flavour hints. `GenJet_*` remains required for MC jobs. The topwsf step also fails on missing local ntuple files, missing Events branches, missing xsections, or missing `Runs/genEventSumw`.

## Logs And Outputs

Each mode writes a controller `log.txt` in its work directory. Scale-factor modes write `systematics/scale_factor/log.txt`; Condor ntuple jobs keep stdout/stderr/scheduler logs under each generated `ntuple.job_dir` `logs/` directory, matching the convert SLURM pattern of separate per-job logs.

Key outputs:

- Convert: `{output_root}/{signal|bkg|data}/...root`.
- Mix: mixed MC under `{signal_mixed|bkg_mixed}`.
- Training: per-tree model output directories with copied configs and prediction references.
- Signal region: `signal_region.csv` plus score plots.
- QCD ABCD: PDFs and `qcd_abcd_yields.root`.
- Combine: significance and limit CSVs under `combine/config.json` `output_dir`.
- Scale-factor ntuples: `systematics/scale_factor/output/ntuples/..._{year}_{nano_version}`.
- Scale-factor fits: generated topwsf cards under `systematics/scale_factor/generated/topwsf/`, with `boohft-calib/output/` and `boohft-calib/web/` fit products.

## Dependencies

The repository expects ROOT, C++17, XRootD tools, Python 3, and the packages listed in `pixi.toml`, including the conda-forge C/C++ compiler wrappers, ROOT, XRootD, `yaml-cpp`, `correctionlib`, and Boost C++ headers. Mode 11 resolves `ROOTConfig.cmake` from the active pixi/conda prefix and forwards `CC`, `CXX`, `ROOT_DIR`, `correctionlib_DIR`, `yaml-cpp_DIR`, and pixi runtime library paths to CMake so vendored C++ dependencies resolve inside one pixi environment; it rejects a stale `nano.cpp/build` cache configured with a compiler, ROOT, `correctionlib`, or `yaml-cpp` outside the active pixi/conda prefix. Condor ntuple jobs unpack into a tarball-hash-specific work directory so a regenerated job directory cannot silently reuse an old extracted `nano.cpp` tree. `boohft-calib/topwsf` uses coffea/uproot/hist/correctionlib-style Python packages and requires CMSSW/combine only for the fit step that invokes combine.
