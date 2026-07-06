# ScoutingVVVTools

CMS Run 3 Scouting VVV analysis tools. The main workflow converts ScoutingNano ROOT files, trains BDT/NN models, defines signal regions, validates QCD with ABCD, plots data/MC, and runs CMS combine. A separate `systematics/scale_factor` workflow derives AK8/AK4 tagger scale factors from semileptonic ttbar muon tag-and-probe ntuples.

## Environment

Use the project's normal ROOT/CMSSW environment on the batch machine for non-pixi workflows. The Python environment is described by `pixi.toml`; it includes the Python packages needed by training, plotting, coffea/topwsf, the scale-factor controller, conda-forge C/C++ compiler wrappers, ROOT, XRootD, `yaml-cpp`, `correctionlib`, and Boost C++ headers. The `nano.cpp` JME helper dependency `external/CMSJMECalculators` is vendored in-tree with nested git metadata removed; mode 11 resolves `ROOTConfig.cmake` from the active pixi/conda prefix and passes `CC`, `CXX`, `ROOT_DIR`, `correctionlib_DIR`, `yaml-cpp_DIR`, and pixi runtime library paths to CMake so C++ dependencies resolve inside one environment. If an existing `nano.cpp/build` cache was configured with a system compiler, ROOT, `correctionlib`, or `yaml-cpp`, mode 11 exits with a clear message instead of mixing system and pixi libraries.

Do not use the scale-factor ntuple step as a replacement for `selections/convert`: it reads ScoutingNano directly and writes topwsf ntuples, not `fat2`/`fat3` BDT trees.

## Main Commands

```bash
# pileup weights
./run.sh 1 [selections/weight/config.json] [sample ...]

# convert ScoutingNano to fat2/fat3 analysis trees
./run.sh 0 [selections/convert/config.json] [sample ...]

# shuffle converted MC chunks for stable train/test splitting
./run.sh 6 [selections/mix/config.json] [sample ...]

# train BDT or NN models
./run.sh 2 [selections/BDT/config.json]

# optimize signal regions and run QCD ABCD
./run.sh 3 [selections/signal_region/config.json]
./run.sh 5 [background_estimation/config.json]

# data/MC plots and combine
./run.sh 4 [plotting/config.json]
./run.sh 7 [combine/config.json]
```

`run.py` supports SLURM for the sample-parallel C++ modes:

```bash
python3 run.py 0 selections/convert/config.json --slurm [sample ...]
python3 run.py 6 selections/mix/config.json --slurm [sample ...]
```

For local parallelism in sample modes, use `--max-jobs N`. Controller logs are written to the mode directory as `log.txt`; batch job stdout/stderr stays in per-job logs, as in the convert SLURM workflow.

## Scale Factors

The AK8 W-tagger calibration currently runs only:

- jet type/category: `ak8` / `W`
- taggers: `WvsQCD`, `ZvsQCD`, `VvsQCD`
- data: `data_2024` with `DST_PFScouting_SingleMuon`

Step 1 prepares or submits `nano.cpp` Condor jobs from ScoutingNano:

```bash
python3 run.py 11
# default config: systematics/scale_factor/ntuple_config.json
```

Important `ntuple_config.json` fields:

- `run_targets`: the jet type, jet category, and exact tagger list to produce.
- `ntuple.samples`: data and MC groups, by names from `src/sample.json`.
- `ntuple.sample_base`: local output prefix; the year/nano suffix is appended.
- `ntuple.job_dir`: local Condor work directory pattern.
- `ntuple.build_jobs`: local `nano.cpp` build parallelism; `"auto"` uses the visible CPU count.
- `ntuple.download_remote_inputs`: defaults to `false`, so remote `root://` ScoutingNano inputs are streamed rather than copied locally.
- `ntuple.variations`: JES/JER/MET switches are present but disabled for Scouting until implemented.

Submit Condor either by setting `ntuple.submit_condor: true` or manually from each generated job directory:

```bash
cd systematics/scale_factor/jobs/ntuples/ak8_W_2024_mc
condor_submit submit.jdl
```

After Condor finishes, merge pieces locally:

```bash
systematics/scale_factor/nano.cpp/build/nano_merge systematics/scale_factor/output/ntuples/topwsf_scouting_muon_2024_v15
```

Step 2 generates and runs `boohft-calib/topwsf` cards:

```bash
python3 run.py 12
# default config: systematics/scale_factor/sf_config.json
```

Important `sf_config.json` fields:

- `run_targets`: the exact taggers/categories to run; only these generate cards.
- `calibration.input_sample_base`: local merged ntuple prefix from step 1.
- `calibration.samples`: data and MC groups, again resolved from `src/sample.json`.
- `calibration.binning`: per AK type, jet category, tagger, pT bin, and score bin.
- `calibration.systematics.enabled`: currently `pu`, `jms`, and `jmr`; JES/JER/MET/LHE switches are listed but disabled.
- `calibration.run_launcher`: set `false` to only generate cards.

Scale-factor controller logs are written to `systematics/scale_factor/log.txt`. Ntuple Condor job stdout/stderr/scheduler logs are under each generated `ntuple.job_dir` `logs/` directory, which is printed by `python3 run.py 11`. Condor jobs still request one CPU in `submit.jdl`; their build step uses all CPUs visible inside the worker slot unless `BUILD_JOBS` is set. Workers unpack the vendored `nano.cpp` tarball into a tarball-hash-specific directory and build with the same pixi/conda compiler and CMake package paths resolved by mode 11.

Mode 11 resolves DAS datasets before submission. The controller prefers `/cvmfs/cms.cern.ch/common/dasgoclient` when available; set `SCALE_FACTOR_DASGOCLIENT=/path/to/dasgoclient` only to override it explicitly. DAS queries keep `instance=prod/phys03` inside the query for `/.../USER` datasets, run with pixi/conda variables removed, active `CONDA_PREFIX` paths stripped from path-like variables, `/cvmfs/cms.cern.ch/cmsset_default.sh` sourced immediately before executing the CVMFS DAS client, and `HOME` set to `SCALE_FACTOR_DAS_HOME` so `dasgoclient` can load DAS key definitions. On IHEP, the controller infers `SCALE_FACTOR_DAS_HOME=/afs/ihep.ac.cn/users/<initial>/<user>` when the login script sets `HOME=/publicfs/cms/user/...`; override it explicitly if needed. Generated Condor jobs export the same DAS home.

Missing configured samples, missing xsections/lumi, missing non-`GenPart_*` ScoutingNano branches, missing ntuple branches, missing local ntuple files, and missing `Runs/genEventSumw` fail explicitly. Missing `GenPart_*` branches warn in the affected ntuple job log and use default W/Z matching with `GenJet` flavour hints.

## Config Files

- `src/sample.json`: single source for sample names, paths, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`.
- `selections/weight/config.json`: pileup input histograms and output CSV paths.
- `selections/convert/config.json`, `branch.json`, `selection.json`: ScoutingNano conversion to `fat2`/`fat3` trees.
- `selections/mix/config.json`: deterministic MC chunk shuffling.
- `selections/BDT/config.json`: model type, class groups, input patterns, training branches, decorrelation, and hyperparameters.
- `selections/signal_region/config.json`: signal-region search settings.
- `background_estimation/config.json`: QCD ABCD setup and optional A-region shape plots.
- `plotting/config.json`, `plotting/branch.json`: data/MC plotting inputs and branch plot overrides.
- `combine/config.json`: combine channels, ROOT inputs, output directory, and covariance option.
- `systematics/scale_factor/ntuple_config.json`: ScoutingNano to topwsf ntuple step.
- `systematics/scale_factor/sf_config.json`: topwsf scale-factor card generation and fit step.
