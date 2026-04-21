# ScoutingVVVTools

CMS scouting VVV analysis workflow for pileup reweighting, branch conversion, BDT training, signal-region optimization, QCD ABCD estimation, and data/MC plotting.

## Pipeline order

1. `mode=1`: build pileup-weight CSV files.
2. `mode=0`: convert the selected NanoAOD-style inputs into `fat2` and `fat3` ROOT trees.
3. `mode=2`: train the BDT and save the model plus copied configs.
4. `mode=3`: scan the test split and write `signal_region.csv`.
5. `mode=5`: run the QCD ABCD validation on the MC test split.
6. `mode=4`: make data/MC comparison plots.

`selections/b_veto/` and `systematics/` are separate studies and are not launched through `run.sh`.

## `run.sh`

Usage:

```bash
./run.sh <mode> [config.json] [sample1 sample2 ...]
```

Examples:

```bash
./run.sh 1
./run.sh 0
./run.sh 0 selections/convert/config.json www qcd_ht2000
./run.sh 2
./run.sh 3 selections/signal_region/config.json
./run.sh 5 background_estimation/config.json
./run.sh 4 plotting/config.json
```

Modes:

- `mode=0`: compile and run `selections/convert/convert_branch.C`.
  Input: `selections/convert/config.json`, `selections/convert/branch.json`, `selections/convert/selection.json`, `src/sample.json`, and the pileup CSVs from `mode=1`.
  Output: converted ROOT files under `{output_root}/{signal|bkg|data}/`, plus `selections/convert/log.txt`.
- `mode=1`: compile and run `selections/weight/weight.C`.
  Input: `selections/weight/config.json` and `src/sample.json`.
  Output: pileup CSV files under the configured `output_root`, plus `selections/weight/log.txt`.
- `mode=2`: run `selections/BDT/train.py`.
  Input: `selections/BDT/config.json`, the converted ROOT files from `mode=0`, and `src/sample.json`.
  Output: one trained BDT directory per tree under `output_root`, containing the model, plots, copied `config.json` / `branch.json` / `selection.json`, `test_ranges.json`, and the saved test-set prediction references `test_reference_signal_region.npz` / `test_reference_qcd_est.npz`, plus `selections/BDT/log.txt`. During training, `train.py` logs six loss values per boosting round: train/test `classification_loss`, `decorrelation_loss`, and `total_loss`. For the custom-objective decorrelation modes (`smooth_cvm` and `cvm`), the classification term uses native weighted-sum XGBoost softprob scale on class-balanced unit-scale training weights, while the decorrelation term is multiplied by one fixed train-split scale so its loss/gradient/hessian stay on a comparable scale. The saved reference files keep the downstream physics-normalised test weights.
- `mode=3`: run `selections/signal_region/signal_region.py`.
  Input: `selections/signal_region/config.json` and one trained BDT directory from `mode=2`.
  Output: `sr_score_*.pdf`, `sr_regions_2d.pdf`, and `signal_region.csv` inside the configured `output_dir`, plus `selections/signal_region/log.txt`. Before scanning, the script reloads the saved model, reproduces the test-split prediction used in training, and aborts if it does not match `test_reference_signal_region.npz` within the stored tolerances.
- `mode=4`: run `plotting/data_mc.py`.
  Input: `plotting/config.json`, `plotting/branch.json`, the converted ROOT files from `mode=0`, and one trained BDT directory from `mode=2`.
  Output: one PDF per plotted branch under the configured `output_root`, plus `plotting/log.txt`.
- `mode=5`: run `background_estimation/qcd_est.py`.
  Input: `background_estimation/config.json`, one trained BDT directory from `mode=2`, and the `signal_region.csv` written by `mode=3`.
  Output: ABCD summary PDFs and one ROOT file under the configured `output_dir`, plus `background_estimation/log.txt`. The ROOT file stores `yield`, `stat_error`, `scale_error`, and `covariance_total` for each saved category. Before building the ABCD regions, the script reloads the saved model, reproduces the test-split prediction used for the `qcd_est.py` preprocessing chain, and aborts if it does not match `test_reference_qcd_est.npz` within the stored tolerances.

Sample arguments:

- Extra sample names are supported only for `mode=0` and `mode=1`.
- If no sample names are given, `run.sh` uses `submit_samples` from the chosen config.
- If `submit_samples` is empty or missing, all MC samples in `src/sample.json` are used.

## Main JSON files

- `src/sample.json`: master sample registry with `name`, `path`, `sample_ID`, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`.
- `selections/weight/config.json`: pileup histogram inputs and pileup-weight output paths.
- `selections/convert/config.json`: convert-step paths, threading, file-size splitting, and pileup CSV pattern.
- `selections/convert/selection.json`: event selection and tree split (`fat2` / `fat3`).
- `selections/convert/branch.json`: input branches to read and output branches to write.
- `selections/BDT/config.json`: BDT inputs, class groups, training settings, output directories, and per-tree `event_reweight_branches` (per-event raw-weight branches multiplied before sample normalisation; default `["weight_pu"]`).
- `selections/signal_region/config.json`: signal-region scan settings and `output_dir`.
- `background_estimation/config.json`: `qcd_est.py` settings, including `bdt_root`, `signal_region_csv`, `output_dir`, and `root_file_name`.
- `plotting/config.json`: `data_mc.py` settings, including `bdt_root`, `output_root`, `data_samples`, and per-tree `event_reweight_branches` (applied to MC events only; data weights stay 1.0).
- `plotting/branch.json`: per-tree plot overrides such as `skip_branches`, `bins`, `x_range`, `y_range`, `logx`, and `logy`.

## Step-by-step file flow

- `mode=1` writes the pileup CSVs used by `mode=0` for MC samples.
- `mode=0` writes the converted `fat2` and `fat3` ROOT trees used by `mode=2` and `mode=4`.
- `mode=2` writes the model, copied configs, and saved test-set prediction references used by `mode=3`, `mode=4`, and `mode=5`.
- `mode=3` writes `signal_region.csv`, which defines the A-region score bins for `mode=5`.
- `mode=5` writes the ABCD validation ROOT file and PDFs for the chosen tree.
- `mode=4` writes one data/MC comparison PDF per branch.
