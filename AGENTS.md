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
- Ntuple production is tagger-independent; the Scouting card stores all configured Scouting GloParT W-jet tagger scores so changing the calibrated tagger does not require remaking ntuples.
- Taggers: `XcsVsQCD`, `XudVsQCD`, `WvsQCD`, `XbbVsQCD`, `XccVsQCD`, `XssVsQCD`, `XqqVsQCD`, `ZvsQCD`, `VvsQCD`.
- Data samples: official ScoutingNano 2024C-I and 2025B-D samples, one sample per era named like `2024C_official`, requiring `DST_PFScouting_SingleMuon`.
- Scouting objects: `ScoutingMuonVtx`, `ScoutingFatPFJetRecluster`, `ScoutingPFJetRecluster2`, and `ScoutingMET`.

Remote ScoutingNano inputs may be DAS datasets, `/store/...` files, `root://...` files, local directories, local ROOT files, or text/list/YAML file lists. The scale-factor default config stages remote ROOT files with `xrdcp` into worker scratch when available before processing on batch workers because direct ROOT streaming from IHEP workers to the CMS global redirector can time out; do not otherwise change remote staging behavior unless requested. POSIX-mounted `/eos/...` paths, such as lxplus `/eos/home-*` workspaces, are allowed as local paths; Tier output paths must use `root://...`. When `ntuple.use_tier_storage` is true, step-1 Condor/HepJob workers copy each per-job merged piece to the configured Tier path, and `nano_merge` can later read those Tier pieces with `--pieces-dir` while writing merged ntuples locally. With `split_by_era` enabled, ntuple job directories, Tier pieces, local merged outputs, and topwsf cards are generated separately per data era under year subdirectories; each era output contains that era's data sample and the corresponding-year MC samples. Step-2 topwsf input remains local through `calibration.input_sample_base`.

For `/.../USER` dataset resolution, scale-factor `nano.cpp` first tries the legacy DAS query `instance=prod/phys03` and then falls back to `system=rucio`, `system=dbs3`, and plain DAS queries. Official `/.../NANOAOD` datasets are queried without `instance=prod/phys03`. The controller prefers `/cvmfs/cms.cern.ch/common/dasgoclient` when available, with `SCALE_FACTOR_DASGOCLIENT` as an explicit override. DAS queries run with pixi/conda identity variables removed, active `CONDA_PREFIX` paths stripped from path-like variables, `/cvmfs/cms.cern.ch/cmsset_default.sh` sourced immediately before executing the CVMFS DAS client, and `HOME` set to `SCALE_FACTOR_DAS_HOME` so `dasgoclient` can load DAS key definitions. On IHEP, `SCALE_FACTOR_DAS_HOME` is inferred from `/afs/ihep.ac.cn/users/<initial>/<user>` when `HOME` points at `/publicfs/cms/user/...`; users may override it explicitly. Generated Condor `process.sh` exports the same DAS home.

The scale-factor ntuple config uses `ntuple.build_jobs` for the local `nano.cpp` build; `"auto"` uses the visible CPU count. Generated ntuple job directories include `submit.sh`; every submission first stages the current X509 proxy as `x509up_proxy` in the job directory, and workers use that staged file as `X509_USER_PROXY` instead of relying on Condor's default `/tmp/x509up_*` lookup. On IHEP `lxlogin*` hosts it submits with `hep_sub ./process.sh -g cms -wt mid ...`, on CERN `lxplus*` hosts it submits `submit_lxplus.jdl` with the staged proxy in `transfer_input_files`, CERN-style `+JobFlavour = "tomorrow"` and `MY.WantOS = "el9"` instead of IHEP accounting/walltime ClassAds, and loads `lxbatch/eossubmit` first when the job directory is under `/eos/...`, and other hosts fall back to `condor_submit submit.jdl`, which also transfers `x509up_proxy` as a normal input file. Condor JDLs set `transfer_output_files = ""` so no placeholder output file can put jobs on hold if it is missing; `on_exit_hold` still holds nonzero worker exits. The IHEP `hep_sub` path passes `download_remote_inputs` as the eighth worker argument so remote input staging works there too. Condor ntuple jobs request one CPU and `ntuple.request_disk_mb` disk, defaulting to 50000 MB. Mode 11 can use ordinary Python for the controller when the fixed CVMFS LCG view is available; otherwise it falls back to pixi/conda. The C++ build and Condor workers use `/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt` directly when it exists, with no local LCG environment variable needed. In LCG mode, ROOT, correctionlib, yaml-cpp, and Python3 CMake inputs are resolved from the sourced LCG setup environment, including its `CMAKE_PREFIX_PATH` and Python package paths; yaml-cpp is linked through an explicit imported target using the resolved LCG include directory and library path, and each executable links that target directly; generated workers source the same setup script with `set -u` temporarily disabled and then restored because LCG setup scripts are not nounset-safe, and `worker_runtime.tar.gz` contains only `nano_run` plus local build products. In fallback mode it also carries the private pixi/conda shared-library closure needed by `nano_run`. `hadd`, `xrdcp`, and `xrdfs` are always resolved from the worker node or fixed CVMFS LCG environment, not copied from pixi, and `process.sh` keeps their site runtime separate from the `nano_run` runtime. Workers unpack the bundle in scratch and run the bundled `nano_run` instead of compiling on the worker or reading `/eos/.../.pixi/envs/default`. The generated worker script prints proxy, site tool, XRootD, host, and input diagnostics, sets XRootD to IPv4 by default, tries multiple CMS XRootD redirectors for remote input staging via `SCALE_FACTOR_XRD_REDIRECTORS`, and guards shared tarball/runtime extraction with `flock` because all ProcId jobs in one generated work directory reuse the same tarball-hash-specific repository and runtime directories. Before lxplus EOS submission, `systematics/scale_factor/check_lxplus_mode11.sh` can be run from the repository root to preflight the fixed LCG setup, site tools, exact ROOT/correctionlib/yaml-cpp CMake resolution, shell template syntax, and already generated job directories without submitting jobs or processing samples.

The Scouting ntuple card uses strict branch binding. Its schema must match ROOT physical branch types, including compact `vec_uint8`/`vec_int16` encodings for Scouting and `GenJet` flavour, multiplicity, constituent, and index branches; analysis producers still access these values through integer helpers where needed. Missing configured ScoutingNano branches fail immediately, except `GenPart_*`: MC jobs warn in their per-job Condor logs and fall back to default W/Z matches with `GenJet` flavour hints. Scouting `Flag_*` MET filter branches are optional because current Scouting data productions can omit them; when absent, `passmetfilters` treats the missing filter as not vetoing the event. `GenJet_*` remains required for MC jobs. The topwsf step also fails on missing local ntuple files, missing Events branches, missing xsections, or missing `Runs/genEventSumw`. Mode 12 launches `boohft-calib/launcher.py` with the same Python executable that runs `scale_factor.py`, removes `PYTHONPATH`/`PYTHONHOME`, sets `PYTHONNOUSERSITE=1`, and defaults BLAS/OpenMP thread variables to one so `pixi run python3 run.py 12` does not inherit incompatible LCG/CVMFS Python packages from a previously sourced site environment. The default `sf_config.json` uses conservative topwsf workers `[4, 2]` and `fit_impact_parallel=2` for lxlogin-style process limits; raise them only when running on a host with enough process and memory headroom. The `boohft-calib` CMSSW setup scripts clone Combine and CombineHarvester with GitHub SSH URLs, not HTTPS.

## Logs And Outputs

Each mode writes a controller `log.txt` in its work directory. Scale-factor modes write `systematics/scale_factor/log.txt`; Condor ntuple jobs keep stdout/stderr/scheduler logs under each generated `ntuple.job_dir` `logs/` directory, matching the convert SLURM pattern of separate per-job logs.

Key outputs:

- Convert: `{output_root}/{signal|bkg|data}/...root`.
- Mix: mixed MC under `{signal_mixed|bkg_mixed}`.
- Training: per-tree model output directories with copied configs and prediction references.
- Signal region: `signal_region.csv` plus score plots.
- QCD ABCD: PDFs and `qcd_abcd_yields.root`.
- Combine: significance and limit CSVs under `combine/config.json` `output_dir`.
- Scale-factor ntuples: `systematics/scale_factor/output/ntuples/.../{year}/{era}_{nano_version}` when `split_by_era` is enabled.
- Scale-factor fits: generated topwsf cards under `systematics/scale_factor/generated/topwsf/{year}/`, with `boohft-calib/output/` and `boohft-calib/web/` fit products separated by era in the routine name.

## Dependencies

The repository expects ROOT, C++17, XRootD tools, Python 3, and the packages listed in `pixi.toml`, including the conda-forge C/C++ compiler wrappers, ROOT, XRootD, `yaml-cpp`, `correctionlib`, and Boost C++ headers. Mode 11 resolves `ROOTConfig.cmake` from the selected C++ dependency environment. On lxplus-style hosts with CVMFS it uses `/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt` directly so ROOT, XRootD, compiler, Python3, `yaml-cpp`, and `correctionlib` are ABI-compatible with worker nodes; otherwise it falls back to the active pixi/conda prefix and forwards `CC`, `CXX`, `Python3_EXECUTABLE`, `ROOT_DIR`, `correctionlib_DIR`, `yaml-cpp_DIR`, `NANO_YAML_CPP_INCLUDE_DIR`, `NANO_YAML_CPP_LIBRARY`, and pixi runtime library paths to CMake. It rejects a stale `nano.cpp/build` cache configured with a compiler, Python3, ROOT, `correctionlib`, or `yaml-cpp` outside the selected C++ dependency environment. Condor ntuple jobs use the local build to package a worker runtime bundle, so worker nodes need shell/coreutils/tar plus normal batch access to CVMFS/grid certificates but do not need CMake, compilers, Python packages, or direct access to the submitting EOS pixi environment. Condor ntuple jobs unpack into tarball-hash-specific work and runtime directories so regenerated job directories cannot silently reuse old extracted inputs. `boohft-calib/topwsf` uses coffea/uproot/hist/correctionlib-style Python packages and requires CMSSW/combine only for the fit step that invokes combine.
