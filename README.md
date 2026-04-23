# ScoutingVVVTools

CMS scouting VVV analysis workflow for pileup reweighting, branch conversion, optional sample-entry mixing, BDT training, signal-region optimization, QCD ABCD estimation, and data/MC plotting.

## Pipeline order

1. `mode=1`: build pileup-weight CSV files.
2. `mode=0`: convert the selected NanoAOD-style inputs into `fat2` and `fat3` ROOT trees.
3. `mode=6`: shuffle each selected MC sample across its ROOT chunks while preserving chunk sizes and filenames, using a deterministic block shuffle that keeps ROOT reads mostly sequential.
4. `mode=2`: train the BDT and save the model plus copied configs. The checked-in config reads from `dataset_mixed` by default.
5. `mode=3`: scan the test split and write `signal_region.csv`.
6. `mode=5`: run the QCD ABCD validation on the MC test split.
7. `mode=4`: make data/MC comparison plots.

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
./run.sh 6
./run.sh 6 selections/mix/config.json www
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
- `mode=6`: compile and run `selections/mix/mix.C`.
  Input: `selections/mix/config.json`, `src/sample.json`, and the converted MC ROOT files from `mode=0`.
  Output: shuffled ROOT files under the configured `output_root/{signal|bkg}/`, preserving the original chunk filenames and per-chunk entry counts while applying a deterministic per-tree block shuffle for each selected MC sample, plus `selections/mix/log.txt`. The implementation scans input chunk entry counts first (warming ROOT serially, then using OpenMP across the remaining files when available) and then rewrites each tree through a `TChain` using shuffled contiguous entry blocks plus an in-block cyclic rotation. The block size is chosen as `clamp(total_entries / 512, min_block_entries, max_block_entries)`; the checked-in defaults are `32` and `4096`. This keeps the output format unchanged while avoiding the old fully random entry-by-entry ROOT read pattern.
- `mode=2`: run `selections/BDT/train.py`.
  Input: `selections/BDT/config.json`, the ROOT files resolved from its `input_root` / `input_pattern` (the checked-in config points to the mixed `mode=6` output under `../../dataset_mixed`), and `src/sample.json`.
  Output: one trained BDT directory per tree under `output_root`, containing the main model `{tree}_model.json`, the stage-1 baseline `{tree}_model_stage1.json`, shared `feature_corr.pdf`, `loss_mlogloss.pdf`, and the other `loss_*.pdf`, stage-tagged comparison plots such as `importance_cls.pdf` / `importance_decorr.pdf`, `roc_*_cls.pdf` / `roc_*_decorr.pdf`, `score_*_cls.pdf` / `score_*_decorr.pdf`, and `decor_corr_{train,test}_cls.pdf` / `decor_corr_{train,test}_decorr.pdf`, a `branches/` subdirectory with one normalized per-class input-distribution PDF per training branch (train+test combined, values after thresholds and clip ranges but before log transform), plus copied `config.json` / `branch.json` / `selection.json`, `test_ranges.json`, and the saved test-set prediction references `test_reference_signal_region.npz` / `test_reference_qcd_est.npz`, plus `selections/BDT/log.txt`. When decorrelation is enabled, `train.py` first runs a native `multi:softprob` stage with `n_estimators`, `learning_rate`, and `early_stopping_rounds`, then resumes from the stage-1 best model for `n_estimators_decorr` additional rounds at `learning_rate_decorr` using the custom decorrelation objective. Stage 1 logs only train/test `mlogloss`; stage 2 logs train/test `mlogloss`, `classification_loss`, `decorrelation_loss`, and `total_loss`. In both stages, `classification_loss` is the same sum-scale weighted softprob cross-entropy (`Σ w·(-log p_true)`), `mlogloss` is the native weighted average of that same quantity (`Σ w·(-log p_true) / Σ w`), `regularization_loss` is reconstructed from the trained trees on the unshrunk leaf weights of each stage (`leaf / eta`, then applying `gamma` / `reg_lambda` / `reg_alpha`), and `total_loss = classification_loss + decorrelation_loss`. Per-sample raw weights are fixed before thresholding, then class totals are equalised only after thresholding so each filtered split keeps the post-cut sample mixture within each class while giving each class the same overall training weight scale. Stage 1 chooses early stopping / best round on test `classification_loss`, while stage 2 chooses them on test `total_loss`; when `lr_reduce_patience > 0` and `min_learning_rate > 0`, stage 2 suppresses early stopping while `eta` remains above the floor, halves `eta` after each stale block of `lr_reduce_patience` rounds, resets the stale counters after each reduction, and only starts enforcing the normal `early_stopping_rounds` condition once `eta` reaches `min_learning_rate`. The decorrelation term is scaled once from the saved stage-1 best model using `(cls_loss_stage1 + reg_loss_stage1) / max(raw_decor_loss_stage1, ε)`, so the stage-2 decorrelation term stays on the same overall scale as the stage-1 endpoint. The saved reference files keep the pre-lumi physics test weights, and downstream comparison multiplies them by the configured `lumi` before checking the reconstructed physical weights.
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

- Extra sample names are supported only for `mode=0`, `mode=1`, and `mode=6`.
- If no sample names are given, `run.sh` uses `submit_samples` from the chosen config.
- If `submit_samples` is empty or missing, all MC samples in `src/sample.json` are used.

## Main JSON files

- `src/sample.json`: master sample registry with `name`, `path`, `sample_ID`, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`.
- `selections/weight/config.json`: pileup histogram inputs and pileup-weight output paths.
- `selections/convert/config.json`: convert-step paths, threading, file-size splitting, and pileup CSV pattern.
- `selections/mix/config.json`: mix-step tree selection, input/output ROOT roots and patterns, sample config, threading for the input-chunk scan, deterministic `random_state`, and the block-size bounds `min_block_entries` / `max_block_entries` (default `32` / `4096`).
- `selections/convert/selection.json`: event selection and tree split (`fat2` / `fat3`).
- `selections/convert/branch.json`: input branches to read and output branches to write.
- `selections/BDT/config.json`: BDT inputs, `input_root` / `input_pattern`, class groups, training settings, output directories, top-level `decor_loss_mode`, `decor_lambda`, `decor_n_bins`, `decor_n_thresholds`, `decor_score_tau`, `decor_bin_tau_scale`, and per-tree hyperparameters (`n_estimators`, `n_estimators_decorr`, `max_depth`, `learning_rate`, `learning_rate_decorr`, optional `min_learning_rate` / `lr_reduce_patience`, `gamma`, `reg_lambda`, `reg_alpha`, `min_child_weight`, `subsample`, `colsample_bytree`, `early_stopping_rounds`), `decorrelate`, and `event_reweight_branches` (per-event raw-weight branches multiplied before sample normalisation; default `["weight_pu"]`).
- `selections/signal_region/config.json`: signal-region scan settings and `output_dir`.
- `background_estimation/config.json`: `qcd_est.py` settings, including `bdt_root`, `signal_region_csv`, `output_dir`, and `root_file_name`.
- `plotting/config.json`: `data_mc.py` settings, including `bdt_root`, `output_root`, `data_samples`, and per-tree `event_reweight_branches` (applied to MC events only; data weights stay 1.0).
- `plotting/branch.json`: per-tree plot overrides such as `skip_branches`, `bins`, `x_range`, `y_range`, `logx`, and `logy`.

## Step-by-step file flow

- `mode=1` writes the pileup CSVs used by `mode=0` for MC samples.
- `mode=0` writes the converted `fat2` and `fat3` ROOT trees used directly by `mode=4`, and as the input source for `mode=6` or any `mode=2` config that still points `input_root` at the unmixed dataset.
- `mode=6` can rewrite those converted MC trees into a shuffled mirror with the same chunk layout using a deterministic block shuffle, which the checked-in BDT config uses by default.
- `mode=2` writes the model, copied configs, and saved test-set prediction references used by `mode=3`, `mode=4`, and `mode=5`.
- `mode=3` writes `signal_region.csv`, which defines the A-region score bins for `mode=5`.
- `mode=5` writes the ABCD validation ROOT file and PDFs for the chosen tree.
- `mode=4` writes one data/MC comparison PDF per branch.
