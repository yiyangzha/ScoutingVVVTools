# AGENTS.md

Whenever you modify the code, you must also update the contents of AGENTS.md accordingly.

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

This file provides guidance to Claude Code (claude.ai/code) or Codex when working with code in this repository.

## Overview

This is a CMS physics analysis toolkit for a VVV (triple vector boson) scouting search using Run 3 data. The workflow processes CMS NanoAOD-style ROOT files from DAS, applies selections, can reshuffle converted sample entries into a mixed dataset for stable train/test splitting, trains BDT models, and estimates backgrounds.

**Dependencies:** ROOT (with `root-config`), C++17, Python 3, OpenMP (optional), XRootD for remote files.

## Pipeline Overview

The analysis runs in this order:

1. **Skim** — reduce raw NanoAOD files to manageable size (`legacy/skim/`)
2. **Pileup weights** — derive MC pileup reweighting CSVs (`selections/weight/`)
3. **Convert** — apply physics selections, build BDT training trees (`selections/convert/`)
4. **Mix** — shuffle per-sample tree entries across ROOT chunks while preserving chunk sizes (`selections/mix/`)
5. **B-veto** — derive AK4 b-jet veto working points (`selections/b_veto/`)
6. **BDT training** — train `fat2`/`fat3` classifiers (`selections/BDT/`)
7. **Background estimation** — QCD ABCD validation on the MC test split (`background_estimation/`)
8. **Data/MC plotting** — compare distributions of `fat2`/`fat3` variables in data vs MC (`plotting/`)
9. **Combine** — wrap CMS `combine` to compute expected significance and `AsymptoticLimits` from the `mode=5` ROOT output (`combine/`)
10. **Systematics** — trigger efficiency studies (`systematics/`)

`systematics/find_duplicate_entries.C` is a standalone ROOT diagnostic macro that scans either one ROOT file or a directory searched recursively for `.root` files, chains the matching trees with `TChain`, and reports duplicated event keys (`run`, `luminosityBlock`, `event`) plus their multiplicities and first/last entry indices.

## Convert_branch.C output layout (consumed downstream)

`convert_branch.C` writes per-sample ROOT files laid out as
`{output_root}/{sample_group}/{sample}.root`, where `sample_group ∈ {signal, bkg, data}` is derived from each sample's `is_MC`/`is_signal` flags. When the combined estimated output exceeds `max_output_file_size_gb`, the writer splits per-tree entries evenly across N chunks (sized so each chunk stays below the limit with ~10% margin) and writes them as `{sample}_0.root`, `{sample}_1.root`, …; when the estimate is within the limit a single `{sample}.root` is written instead. Each output chunk is built by opening a `TChain` over all per-thread temp ROOT files, cloning the output tree from that `TChain` itself so ROOT keeps branch addresses synced across temp-file boundaries, and streaming the chunk's entry range into that clone. The writer validates the planned entry count for every non-empty tree chunk before closing the file. Basket size is fixed at 32 KB and auto-flush/auto-save are disabled on the output trees so ROOT's `OptimizeBaskets` cannot inflate a basket past its 1 GB `TBufferFile` serialization limit on highly-compressible branches. Every ROOT file contains the `fat2` and `fat3` trees described by `selections/convert/branch.json`.

Downstream scripts — `selections/BDT/train.py`, `selections/BDT/signal_region/signal_region.py`, and `plotting/data_mc.py` — all consume these files via their config's `input_root` + `input_pattern` (`{input_root}/{sample_group}/{sample}.root`). They discover every file for a given sample by globbing the base pattern **and** the `{sample}_*.root` split variant, then concatenate their trees in sorted order.

`selections/mix/mix.C` can read that same layout and write a shuffled mirror under the same dataset root using `{sample_group}_mixed` directories (for example `dataset/signal_mixed/` and `dataset/bkg_mixed/`). For each selected MC sample and requested tree, it first scans every input chunk's tree entries once (warming ROOT serially, then using OpenMP across remaining files when available), then concatenates the chunk files through a `TChain`, applies a deterministic per-tree block shuffle seeded by `random_state` (with a random cyclic rotation inside each block), and writes the entries back into output files that preserve the original chunk filenames and per-chunk entry counts. Each non-empty output chunk is cloned from the `TChain` itself so ROOT keeps branch addresses synced as the chain advances across input-file boundaries, and the writer validates both per-chunk and total written entry counts. The block size is chosen as `clamp(total_entries / 512, min_block_entries, max_block_entries)` with defaults `32` and `4096`. The checked-in BDT config reads this mixed layout via `input_root="../../dataset"` and `input_pattern="{input_root}/{sample_group}_mixed/{sample}.root"`.

## Key Commands

### Convert branches (build training trees)
```bash
# Run all MC samples defined in selections/convert/config.json
./run.sh 0

# Run specific samples
./run.sh 0 selections/convert/config.json www qcd_ht2000

# Run with a custom config
./run.sh 0 /path/to/custom_config.json
```

### Pileup reweighting
```bash
./run.sh 1
./run.sh 1 selections/weight/config.json www
```

### BDT training via run.sh
```bash
./run.sh 2
./run.sh 2 selections/BDT/config.json
```

### Mix converted chunks (randomise per-sample entry order)
```bash
./run.sh 6
./run.sh 6 selections/mix/config.json www
./run.sh 6 /path/to/custom_config.json
```

### Compile and run C++ tools manually
```bash
# compile
c++ -O3 -std=c++17 $(root-config --cflags --libs) selections/convert/convert_branch.C -o convert_branch
# run one sample
CONVERT_CONFIG_PATH=selections/convert/config.json ./convert_branch www
```

When `convert_branch.C` runs a sample, it first sums the configured `tree_name` entries across all ROOT files resolved from that sample's `path` entries, then writes that total back to the active sample config's `raw_entries` field for the running sample before processing events.

### BDT training

BDT outputs are written under the per-tree `output_root` configured in `selections/BDT/config.json` (for example `selections/BDT/fat2/`). The checked-in BDT config currently points `input_root` to `../../dataset` and `input_pattern` to `{input_root}/{sample_group}_mixed/{sample}.root`, i.e. the shuffled mirror produced by `selections/mix/mix.C` under the same dataset root. `train.py` runs in two stages when decorrelation is enabled: stage 1 trains a native `multi:softprob` model with the tree's `n_estimators`, `learning_rate`, and `early_stopping_rounds`; stage 2 resumes from the stage-1 best model for `n_estimators_decorr` additional rounds at `learning_rate_decorr` and adds the configured decorrelation loss. The stage-1 baseline is saved as `{tree}_model_stage1.json`, and the final model used downstream is saved as `{tree}_model.json` (if decorrelation is disabled, the stage-1 model is also written to the main path). Each tree output directory also stores copies of `config.json`, `branch.json`, `selection.json`, the fixed per-sample test split definition `test_ranges.json`, and the saved test-set prediction references `test_reference_signal_region.npz` / `test_reference_qcd_est.npz`.

Before training starts, `train.py` also writes a `branches/` subdirectory under the tree output directory containing one PDF per training branch with normalized per-class distributions overlaid (train+test events combined, values after threshold filtering and clip ranges but **before** the log transform); these plots are intended as a data-leakage / input-sanity diagnostic and use step histograms colored by class with legend.

Model-dependent comparison plots are saved twice with `_cls` / `_decorr` suffixes where applicable: `importance_cls.pdf` / `importance_decorr.pdf`, `roc_*_cls.pdf` / `roc_*_decorr.pdf`, `score_*_cls.pdf` / `score_*_decorr.pdf`, `decor_corr_{train,test}_cls.pdf` / `decor_corr_{train,test}_decorr.pdf`, `decor_score_vs_branch_cls.pdf` / `decor_score_vs_branch_decorr.pdf`, and `decor_branch_shapes_by_signal_score_cls.pdf` / `decor_branch_shapes_by_signal_score_decorr.pdf`. The two new decorrelation diagnostics are multipage PDFs built from test events: score-vs-branch pages use the matching class's test sample with a black weighted-average score curve over fine branch-value bins, while signal-score-shape pages compare normalized decorrelate-branch distributions after weighted-efficiency cuts from each tree's `decor_efficiencies`. Both use decorrelate branch values after threshold filtering and clipping but before log transform, and normalized plots use `A.U.` on the y-axis. `feature_corr.pdf`, `loss_mlogloss.pdf`, `loss_classification.pdf`, `loss_decorrelation.pdf`, and `loss_total.pdf` are saved once, and the loss plots draw a vertical line at the stage boundary. During training, stage 1 prints only train/test `mlogloss`; stage 2 prints train/test `mlogloss`, `classification_loss`, `decorrelation_loss`, and `total_loss`. In both stages, `classification_loss` is the same sum-scale weighted softprob cross-entropy (`Σ w·(-log p_true)`), `mlogloss` is the native weighted average of that same quantity (`Σ w·(-log p_true) / Σ w`), `regularization_loss` is reconstructed from the trained trees on the unshrunk leaf weights (saved leaf values divided by that stage's `eta`, then applying `gamma` / `reg_lambda` / `reg_alpha`), and `total_loss = classification_loss + decorrelation_loss`. The decorrelation loss itself uses the current event weights in both smooth and hard CvM modes. Stage 1 chooses early stopping / best round on test `classification_loss`, while stage 2 chooses them on test `total_loss`; when `lr_reduce_patience > 0` and `min_learning_rate > 0`, stage 2 suppresses early stopping while `eta` is still above the floor, halves `eta` after each stale block of `lr_reduce_patience` rounds, resets the stale counters after each reduction, and only starts applying the normal `early_stopping_rounds` check once `eta` has reached `min_learning_rate`. The fixed decorrelation scale is calibrated once from the saved stage-1 best model as `(cls_loss_stage1 + reg_loss_stage1) / max(raw_decor_loss_stage1, ε)`, so the stage-2 logged decorrelation term stays on the same overall scale as the stage-1 endpoint. The saved reference files keep the pre-lumi physics test weights (`σ × N_tree / N_raw` with the same per-event raw-w shaping as the downstream scripts), and downstream comparison multiplies those stored weights by the local analysis `lumi` before checking them against the reconstructed physical weights.

### Signal region optimisation
```bash
# Edit selections/signal_region/config.json, then:
python3 selections/signal_region/signal_region.py
# Or with a custom scan config:
SCAN_CONFIG_PATH=/path/to/config.json python3 selections/signal_region/signal_region.py
```

`signal_region.py` reads all parameters from [selections/signal_region/config.json](selections/signal_region/config.json). It loads the saved model, branch, and selection configs from the `bdt_root` directory written by `train.py`, reloads the exact test split defined in `test_ranges.json`, applies physics-normalised weights (`lumi × σ × N_tree / N_raw`), converts the stored reference weights to the same scale by multiplying them with the local `lumi`, validates that its reconstructed test-split prediction matches `test_reference_signal_region.npz` within the stored tolerances, and then builds `N = n_signal_regions` non-overlapping general high-dimensional rectangles in the configured BDT score axes (`score_axes`, defaulting to the independent first `NUM_CLASSES − 1` axes; the checked-in config uses all class-score axes). It first builds one shared candidate pool using dense per-axis boundaries from total/signal/background weighted quantiles plus tail-heavy quantiles, deterministic beam coordinate search, compatibility expansion, and optional nearby real event-threshold refinement. Local refinement can choose seeds by distinct event masks so it does not spend all refinement passes on many geometric variants of the same selected events. Candidates are filtered by weighted signal/background minima, optional event-count minima, and geometric validity, but are not trimmed by single-region Z or by a fixed global candidate cap. Before global selection, every candidate is shrunk to its event-preserving minimal score box, and duplicate removal only merges candidates that have the same exact event mask and the same rounded canonical shrunk box. The final selection is global rather than sequential: it chooses K mutually non-overlapping candidates that maximize the existing combined objective `sqrt(sum Z_i^2)`, using the same per-bin Asimov significance calculation already used by the script. With `max_threads > 1` and a working OpenMP compiler/libomp setup, [openmp_region_select.cpp](selections/signal_region/openmp_region_select.cpp) uses a beam incumbent followed by branch-and-bound over the candidate list; if the configured time/node limits stop the search before exhaustion, the script reports the best found Z and a conservative upper-bound certificate. With `max_threads == 1` or if the helper cannot build, it falls back to the Python branch-and-bound implementation. After the K non-overlapping SRs are selected, an additional empty-bin expansion step is applied in order (SR1, SR2, …): each SR's signal-class score axes have their upper bound pushed toward `1.0` and its background-class score axes have their lower bound pushed toward `0.0`, but only into MC-empty space and only as far as the expanded box remains geometrically non-overlapping with every other selected SR. The expansion is constrained to keep each SR's exact selected event mask, so per-bin `S`/`B`/`Z` are unchanged; only the empty-side bounds in the per-SR report and in `signal_region.csv` are widened. Plots are saved as `sr_score_*.pdf`, `scores_no_regions.pdf`, and `scores.pdf` in the configured `output_dir` (defaulting to `bdt_root` for backward compatibility). The 2D region views project the full multiclass probability vector onto the vertices of a regular `NUM_CLASSES`-gon; the no-region version shows class-colored transparent test points and class-labeled vertices, while the region version overlays projected high-dimensional SR polygons instead of rectangles. The selected high-dimensional region definitions are written to `signal_region.csv` in the same directory.

### QCD ABCD estimation
```bash
python3 background_estimation/qcd_est.py
# Or with a custom config:
QCD_EST_CONFIG_PATH=/path/to/config.json python3 background_estimation/qcd_est.py
```

`background_estimation/qcd_est.py` reads [background_estimation/config.json](background_estimation/config.json), then reuses the trained `bdt_root` output in the same way as `signal_region.py`: it loads the saved `config.json`, `model`, `branch.json`, `selection.json`, and `test_ranges.json`, reads the full test split with the same per-event weight definition (loading `model_branches ∪ thresholds.keys() ∪ decorrelate` from each ROOT file so every branch needed for filtering, mass pass/fail, and decorrelation is available even when it is not in `branch.json`), removes only the non-mass thresholds before BDT inference, applies the same clip/log preprocessing, converts the stored reference weights to the local physical scale by multiplying them with the configured `lumi`, validates that the reconstructed `qcd_est.py` test-split prediction matches `test_reference_qcd_est.npz` within the stored tolerances, and uses the `signal_region.csv` produced by `signal_region.py` as the set of A-region score bins. The SR score axes are detected from the CSV `{axis}_low` / `{axis}_high` columns so qcd_est can consume either the legacy first `NUM_CLASSES − 1` axes or all class-score axes. It defines `A` as the union of those score bins with mass-pass, `B` as outside that union with mass-pass, `C` as inside that union with mass-fail, and `D` as outside that union with mass-fail, where mass-pass requires all `ScoutingFatPFJetRecluster_msoftdrop_*` thresholds to pass and mass-fail requires all of them to fail. QCD classes are identified by `class_groups` names containing `qcd` case-insensitively, and all samples from those classes are summed for the ABCD `A/B/C/D` totals, `qcd_true`, `qcd_predict`, and `total_predict`; the per-class `groups/*` outputs remain the MC-true class yields. A single QCD ABCD scale from the union-level `B/C/D` totals is then applied to all individual signal regions. The validation plots keep the finite-MC / ABCD-propagated uncertainties based on `sum(w^2)` and the B/C/D propagation. The ROOT output is combine-facing: every saved category (`samples/*`, `groups/*`, `qcd_predict`, `qcd_true`, `total_predict`, and `total_true`) stores each SR as one-bin histograms under `srN/yield`, `srN/stat_error`, and `srN/scale_error`, plus a category-level `covariance_total` TH2. The combine-facing convention is `stat_error = sqrt(yield)`, `scale_error = 0`, and diagonal `covariance_total = diag(yield)`, treating each weighted yield as the Poisson event count.

### CMS combine wrapper
```bash
./run.sh 7
./run.sh 7 combine/config.json
```

`combine/combine.C` wraps CMS `combine` to compute expected significance and expected `AsymptoticLimits` (CLs) from the ROOT output of `qcd_est.py`. It reads [combine/config.json](combine/config.json), which lists one or more channels, each pointing to a `qcd_abcd_yields.root` written by `mode=5` and the matching per-channel `bdt_root` directory written by `train.py`. `combine.C` loads `class_groups` from each channel's `bdt_root/config.json`, resolves that copied config's `sample_config` the same way as `qcd_est.py`, and requires all channel BDT configs to share the same class membership plus signal/QCD definitions before building one shared scenario list. Group names from `class_groups` are matched against `groups/*` in the ROOT file by the same slugified lowercase convention used by `qcd_est.py`, and QCD classes are the class names containing `qcd` case-insensitively. It must be run inside a CMSSW area that has `HiggsAnalysis/CombinedLimit` built, so that `combine` and `combineCards.py` are on `$PATH`.

For every channel it loads the stored yield/covariance bundles from `samples/{sample}/srN/{yield,stat_error,scale_error}`, `groups/{group}/srN/{yield,stat_error,scale_error}`, `qcd_predict/srN/{yield,stat_error,scale_error}`, and each bundle's `covariance_total`. It then iterates over scenarios:
- `combined`: signal = the sum of all signal groups from `groups/`; backgrounds = the remaining groups.
- `class`: one row per signal class; signal = that class's `groups/` block; backgrounds = every other group.
- `sample`: one row per signal sample; signal = that sample's `samples/` block; backgrounds = the remaining available sample blocks.

For each scenario combine is run twice: once using the MC-true group/sample yields directly from the ROOT file, once replacing all QCD classes (matched by `class_groups` names containing `qcd` case-insensitively) with the single merged `qcd_predict` yield while keeping the other backgrounds on their MC-true group/sample yields. Every channel is validated before running: all configured groups must exist, every configured signal sample must exist, and the code refuses to regularize, clamp, or synthesize missing/zero-rate inputs. By default `use_root_covariance=false`, so each SR is written as its own pure counting bin named `<channel>_sr<N>`; stored `covariance_total` blocks are validated and kept in the ROOT file but are not encoded as Gaussian shape nuisances. Combine's binned Poisson likelihood supplies the counting statistics from the weighted yields, matching the `signal_region.csv` Asimov significance convention, avoiding double-counting those same statistical fluctuations, and avoiding the old multi-bin shape-PDF factorization warning. If a `sample` scenario's signal process is identically zero in some channels, those channels are skipped with a warning; if the signal sample is identically zero in every channel, the wrapper logs a warning, writes `0` into the significance CSV field and `inf` into the limit CSV fields, and continues with the remaining scenarios. If `AsymptoticLimits` returns successfully but its ROOT output still lacks the expected quantiles, the wrapper logs a warning, writes `0` into the corresponding significance CSV field and `inf` into the limit CSV fields, and continues with the remaining scenarios.

When `use_root_covariance=true`, each process's `covariance_total` is additionally eigen-decomposed and every retained eigenmode (eigenvalue `> eigen_rel_cutoff × max(diag(cov))`) becomes one Gaussian shape nuisance. The templates start from `Up_i = rate_i + √λ_k · V_{i,k}` and `Down_i = rate_i − √λ_k · V_{i,k}`; if that would make any bin or whole-template norm non-positive and `rescale_shape_modes_to_positive=true`, the wrapper instead writes `Up_i = rate_i + a·√λ_k · V_{i,k}` and `Down_i = rate_i − a·√λ_k · V_{i,k}` with the largest safe `a < 1`, then compensates in the datacard with a `shape` coefficient of `1/a`. This optional mode represents extra process-yield covariance beyond the native Poisson counting statistics and writes one-bin shape templates for each SR. Per-channel datacards and optional covariance shape ROOT files are written under `output_dir/work/<qcd_mode>_<scope>_<name>/` and stitched together with `combineCards.py`. `combine` is then invoked as `-M Significance -t -1 --expectSignal 1` and `-M AsymptoticLimits -t -1 --run expected`, and the resulting `higgsCombine*.root` files are parsed to fill four CSVs in `output_dir`: `significance.csv`, `limits.csv`, `significance_abcd_mc.csv`, and `limits_abcd_mc.csv`. The MC-true and ABCD CSVs share the same row order (scope, name). When `keep_work: true` (default) the generated datacards, optional shape ROOT files, and combine outputs are left in place for inspection; with `keep_work: false` the work directory is removed after the final CSV is written.

### Data vs MC plotting
```bash
python3 plotting/data_mc.py
# Or with a custom plotting config:
PLOT_CONFIG_PATH=/path/to/config.json python3 plotting/data_mc.py
```

### Duplicate entry check
```bash
# Edit filePath / requestedTree inside systematics/find_duplicate_entries.C,
# then run it as a ROOT macro.
root -l -q systematics/find_duplicate_entries.C
```

`plotting/data_mc.py` reads [plotting/config.json](plotting/config.json) and [plotting/branch.json](plotting/branch.json), then for every tree in `submit_trees`:
- Loads the BDT-tree copies of `config.json` (to get `class_groups`, `input_root`, `input_pattern`) and `selection.json` (for `clip_ranges`, `thresholds`, `log_transform`) from the directory resolved by `bdt_root` with `{tree_name}` substituted. `input_root` is resolved relative to the BDT script directory (the parent of `bdt_root`), matching `train.py`'s convention.
- Discovers plottable variables from `selections/convert/branch.json`'s output entries for the tree (only `onlyMC: false`, expanding `slots` into `{name}_1 … {name}_N`, minus that tree's `skip_branches` from `plotting/branch.json`).
- For each MC sample listed in any `class_groups` entry, it globs the convert_branch ROOT files (`{sample}.root` + `{sample}_*.root`), sums the tree entries across those files, loads every event together with the tree's `event_reweight_branches` from `plotting/config.json` (default `{"fat2": ["weight_pu"], "fat3": ["weight_pu"]}`), computes `raw_w` as the product of those reweight branches on raw values (before any clip/log/threshold) and then drops them, and assigns per-event weight `raw_w × target_total / Σ raw_w_loaded` with `target_total = lumi_total × xsection × tree_entries_total / raw_entries`. Missing MC files, empty MC trees, empty MC classes before filtering, non-positive `raw_entries`, or non-positive `Σ raw_w_loaded` raise `RuntimeError`. `lumi_total` is the sum of `lumi` fields in `src/sample.json` for every sample name in `data_samples` (every listed data sample **must** already exist in `src/sample.json` with `is_MC: false`, otherwise data_mc.py errors out). Missing reweight branches in an MC file raise `KeyError`. The per-event weight is frozen before any cuts and is not recomputed later.
- For each data sample in `data_samples`, loads every event with `weight = 1` per event. Data samples do not read the reweight branches. Data and MC both use the configured BDT input pattern; missing sample files are treated as fatal errors rather than falling back to another directory.
- Applies threshold cuts, then clip ranges, both from the BDT tree's `selection.json`. `log_transform` is **not** applied — it only influences the default `logx` flag for plotting.
- Histograms each branch with the resolved binning: default `default_bins` bins, auto range from the union of MC + data, `logy` always on by default, `logx` on if the branch appears in `log_transform`. `logx` uses log-spaced bins and drops non-positive values; sentinel values (`< -990`) are excluded per-branch from that branch's histogram but the event stays in the sample for other branches. Per-branch overrides from `plotting/branch.json` (`bins`, `x_range`, `y_range`, `logx`, `logy`) take precedence.
- Emits stage-by-stage progress messages in the same concise style as `signal_region.py`, including config resolution, sample loading, filtering, per-branch plotting progress, `Wrote ...`, and `Runtime error: ...`.
- Saves one PDF per branch to `{output_root}/{tree_name}_{branch}.pdf` with a top panel (stacked MC + Data points + hatched MC uncertainty band) and a bottom Data/MC ratio panel.

## Configuration Architecture

All tools are driven by JSON config files. Sample definitions live centrally in [src/sample.json](src/sample.json); individual tool configs reference it via the `sample_config` key.

- **[src/sample.json](src/sample.json)** — master sample registry. Each entry has `name`, `path` (DAS path, string or list), `sample_ID`, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`. `convert_branch.C` updates `raw_entries` for the running sample by summing the chosen `tree_name` entries across all ROOT files resolved from that sample's configured `path` values.
- **[selections/convert/config.json](selections/convert/config.json)** — controls convert step: output paths, thread count, file size limits, pileup weight CSV path pattern.
- **[selections/mix/config.json](selections/mix/config.json)** — controls the optional mix step: selected trees, input/output ROOT roots and patterns, sample config, thread count (used for the OpenMP input-chunk scan), deterministic `random_state`, and the block-size bounds `min_block_entries` / `max_block_entries` (defaults `32` / `4096`) used for per-tree block shuffling while preserving chunk layout.
- **[selections/convert/selection.json](selections/convert/selection.json)** — physics selection: event preselection string, per-collection cuts/sorts, and `tree_selection` that splits output into `fat2` (exactly 2 AK8 jets) and `fat3` (≥3 AK8 jets) trees. Selections are parsed and JIT-compiled by the C++ expression engine.
- **[selections/convert/branch.json](selections/convert/branch.json)** — declares all input NanoAOD branches to read (scalars and collections with p4 definitions) and output branches to write.
- **[selections/weight/config.json](selections/weight/config.json)** — pileup reweighting settings: data pileup histogram files (nominal/low/high for systematics).
- **[selections/BDT/config.json](selections/BDT/config.json)** — controls BDT training inputs and outputs: `submit_trees`, `input_root`, `input_pattern`, `class_groups`, `output_root`, `model_pattern`, `entries_per_sample`, `train_fraction`, top-level `decor_loss_mode` (`smooth_cvm` by default, or legacy `cvm`), `decor_lambda`, `decor_n_bins`, `decor_n_thresholds`, `decor_score_tau`, `decor_bin_tau_scale`, and per-tree hyperparameters (`n_estimators`, `n_estimators_decorr`, `max_depth`, `learning_rate`, `learning_rate_decorr`, optional `min_learning_rate` / `lr_reduce_patience` for stage-2 dynamic lr reduction, `gamma`, `reg_lambda`, `reg_alpha`, `min_child_weight`, `subsample`, `colsample_bytree`, `early_stopping_rounds` with default 10) plus `decorrelate`, `decor_efficiencies` (fractions in `(0, 1]` for the decorrelation shape diagnostics), and `event_reweight_branches` (list of per-event reweight branches multiplied together before sample normalisation; default `["weight_pu"]`).
- **[selections/signal_region/config.json](selections/signal_region/config.json)** — controls signal_region.py: `lumi` (fb⁻¹), `N` / `n_signal_regions` (number of signal regions), `bdt_root` (trained tree output directory to read from, relative to `signal_region/`), `output_dir` (directory for saved PDFs and `signal_region.csv`, relative to `signal_region/`, defaulting to `bdt_root`), `score_axes` (`independent` / first `NUM_CLASSES − 1`, `all`, or an explicit list of class names/indices), `min_bkg_weight` / `min_signal_weight` (minimum weighted yields per region), optional `min_signal_entries` / `min_bkg_entries` event-count minima, `max_edge_candidates_per_axis` (maximum distinct edges per score axis, including `0.0` and `1.0`), candidate-generation controls `beam_width`, `top_intervals_per_axis`, `coordinate_rounds`, `seed_intervals_per_axis`, `multi_axis_seed_max_axes`, `multi_axis_seed_max_seeds`, `compatibility_seed_anchors`, `compatibility_seed_rounds`, `local_refine_rounds`, `local_refine_neighbor_edges`, `local_refine_top_candidates`, `local_refine_diverse_masks`, and `local_refine_candidate_overscan` (`0` scans the full pre-refinement pool), global-selection controls `global_beam_width`, `branch_bound_max_nodes` (`0` for no node cap), `branch_bound_time_limit_seconds` (`0` for no time cap), `deduplicate_event_masks`, `require_exact_n_regions`, `max_threads`, `progress_every_seconds`, and `seed_quantiles`. There is no candidate-pool or global-candidate count limit in the checked-in scan path.
- **[background_estimation/config.json](background_estimation/config.json)** — controls qcd_est.py: `lumi` (fb⁻¹), `bdt_root` (trained tree output directory to read from), `signal_region_csv` (path to the `signal_region.csv` written by `signal_region.py`), `output_dir` (directory for PDFs and ROOT outputs), and `root_file_name` (summary ROOT filename).
- **[plotting/config.json](plotting/config.json)** — controls data_mc.py: `submit_trees`, `sample_config`, `convert_branch_config`, `bdt_root` (per-tree path pattern, points at the BDT tree output dir that already contains the copied `config.json` / `selection.json`), `output_root` (per-tree output dir pattern), `data_samples` (list of data sample names whose entries must exist in `src/sample.json`), `default_bins` (default histogram bin count), and `event_reweight_branches` (per-tree dict `{tree_name: [branch, ...]}` of reweight branches multiplied into each MC event's weight; default `{"fat2": ["weight_pu"], "fat3": ["weight_pu"]}`; data samples are unaffected).
- **[plotting/branch.json](plotting/branch.json)** — plotting config split by tree (`fat2` / `fat3`). Each tree can define `skip_branches` and a `branches` map. Inside `branches`, each branch override can set `bins`, `x_range`, `y_range`, `logx`, `logy`; unset fields fall back to defaults. The file is intended to hold a few explicit examples that can be copied when adding new plot formatting rules later.
- **[combine/config.json](combine/config.json)** — `combine.C` settings: `channels` (list of `{name, root_file, bdt_root}` — each `root_file` is the `qcd_abcd_yields.root` output of `mode=5` for one channel, and each `bdt_root` is that channel's trained tree output directory; `combine.C` reads `class_groups` from every channel's copied `config.json`, resolves each config's `sample_config`, and requires the resulting class/sample signal/QCD definitions to match across channels), `output_dir`, optional `combine_cmd` and `combine_cards_cmd` (defaults `combine` / `combineCards.py`), `use_root_covariance` (default `false`; combine reads `srN/yield` one-bin histograms and writes each SR as an independent counting bin, without turning ROOT covariance into nuisances), `eigen_rel_cutoff` (used only when `use_root_covariance=true`, dropping eigenmodes with `λ ≤ cutoff × max(diag(cov))`; default `1e-10`), `rescale_shape_modes_to_positive` (used only when covariance nuisances are enabled; default `true`), and `keep_work` (keep the generated datacards under `output_dir/work/`; default `true`).

## BDT Training Details

- `selections/BDT/train.py` only trains on the samples explicitly listed in `class_groups`.
- The number of classes is inferred dynamically from `class_groups`; no fixed 5-class assumption is baked into training.
- A class is treated as `single` for plotting if every sample in that class has `is_signal=true`; otherwise it is treated as `background`.
- For each sample, the code first counts all entries across all matched ROOT files in the chosen tree, then defines a fixed sequential split: the first `train_fraction` goes to training and the remaining tail goes to testing. The split is defined on the concatenated sample entry order across files, so `test_ranges.json` can be used later to recover the exact test subset.
- `entries_per_sample` only caps how many events are actually loaded from the training side of that fixed split. The test side is always the full tail of the sample.
- Training events are shuffled after loading; test events keep their original order.
- Sample weights are computed separately for the training and test splits, immediately after reading and before any BDT threshold cuts:
  `target_total       = xsection * tree_entries_total / raw_entries`
  `per_event_weight   = raw_w * target_total / Σ raw_w_loaded`
  where `tree_entries_total` is the total number of entries in the chosen tree across all of that sample's ROOT files (same value for the training and test splits of a given sample), and `raw_w` is the per-event product of the tree's `event_reweight_branches` entries (default `weight_pu`) read on raw values before clip/log/threshold. The denominator `Σ raw_w_loaded` is summed over the events actually read on that side of the split, so the sample total `Σ weight = target_total` regardless of raw_w's magnitude — raw_w only reshapes the per-event distribution inside the sample. With an all-ones `raw_w` this reduces to a constant per-event weight.
- Within each class, the ratio of `target_total` between samples is proportional to `xsection * tree_entries_total / raw_entries`. These per-sample raw weights are frozen before any BDT threshold cuts, so threshold efficiency changes the post-cut class and sample composition naturally.
- The reweight branches are loaded alongside the feature branches, consumed solely to build `raw_w`, then dropped from the DataFrame — they never enter BDT training, standardisation, or threshold filtering. Missing reweight branches in the input ROOT files raise `KeyError`, and a non-positive `Σ raw_w_loaded` raises `RuntimeError`.
- Any branch that appears in `selection.json`'s `thresholds` block or in the per-tree `decorrelate` list but is NOT declared in `branch.json` is still read from the ROOT files (alongside the `branch.json` features and the reweight branches) so that `filter_X` can cut on it and the decorrelation step can reference it. After `filter_X`, threshold branches that are not in `decorrelate` are dropped from the DataFrame so the BDT input feature set remains strictly defined by `branch.json` (decorrelate branches stay in `X` because `train_multi_model` separates them into `Z` for the custom objective and trains only on the remaining `branch.json` columns).
- After threshold/sentinel filtering, class totals are rescaled separately within the training split and within the test split so every class has the same total weight inside that filtered split, with each class total scaled to `N_filtered / N_classes` so the overall XGBoost weight scale stays close to the filtered event count instead of the old large fixed constant.
- This post-threshold class balancing preserves the filtered per-sample proportions inside each class: each sample keeps the same relative share it had after the fixed pre-cut `raw_w × target_total / Σ raw_w_loaded` weighting and the threshold efficiency losses; only the class-level normalisation changes.
- Threshold/sentinel filtering in `filter_X` is applied **only** to branches that appear as keys in `selection.json`'s `thresholds` block. For each such branch, events with sentinel values (`< -990`) are dropped and the threshold condition is enforced. Branches that are *not* listed in `thresholds` are never inspected, so an event with a sentinel value in (for example) a lepton branch is still kept as long as the `thresholds` block doesn't target that lepton branch.
- If `decorrelate` is non-empty and `decor_lambda > 0`, `train.py` trains in two stages. Stage 1 uses native `multi:softprob` without decorrelation and chooses early stopping / best round on test `classification_loss`; stage 2 resumes from the saved stage-1 best model for `n_estimators_decorr` additional rounds and swaps in the custom multiclass objective with the configured decorrelation term, choosing early stopping / best round on test `total_loss`. If decorrelation is disabled, stage 2 is skipped and the stage-1 model is promoted to the main output path.
- In both stages, `classification_loss` is the same sum-scale weighted softprob loss built from the class-balanced unit-scale training weights (`Σ w·(-log p_true)`), while `mlogloss` is the native weighted average of that same quantity (`Σ w·(-log p_true) / Σ w`). Stage 1 receives transformed probabilities from XGBoost's builtin objective and evaluates both metrics directly on them; stage 2 receives raw margins from the custom objective, applies the multiclass softmax explicitly, and then evaluates the same formulas. This keeps the stage-1 and stage-2 logged `classification_loss` directly comparable while retaining `mlogloss` as the native weighted-average reference metric.
- `regularization_loss` remains a diagnostic quantity and is reconstructed from the current trees in the same native tree-solver parameterization as XGBoost: `gamma` times the number of leaves, plus the `reg_lambda` / `reg_alpha` penalties evaluated on the unshrunk leaf weights (`leaf / eta` for the stage that produced that boosting round). `total_loss = classification_loss + decorrelation_loss` is saved for comparison plots, while `regularization_loss` stays separate. Stage 1 uses `classification_loss` and stage 2 uses `total_loss` for early stopping and best-round selection; `regularization_loss` is not used for either.
- When stage 2 is active and both `lr_reduce_patience` and `min_learning_rate` are positive, the stage-2 monitor suppresses early stopping while `eta > min_learning_rate`; after `lr_reduce_patience` consecutive non-improving rounds it halves `eta` (floored at `min_learning_rate`), logs the reduction, resets both stale counters, and only begins enforcing the normal `early_stopping_rounds` threshold once that floor has been reached.
- In the custom-objective decorrelation modes (`smooth_cvm` and `cvm`), the decorrelation term is multiplied by one fixed train-split scale calibrated from the saved stage-1 best model, using `(cls_loss_stage1 + reg_loss_stage1) / max(raw_decor_loss_stage1, ε)`. `smooth_cvm` is the default differentiable mode: it matches the old CvM objective conceptually by comparing class-wise score-efficiency curves across decorrelation-variable bins, but replaces the hard bin/CDF steps with smooth sigmoid surrogates so the recorded decorrelation loss, total loss, and custom training gradient all refer to the same smooth objective. `cvm` keeps the original hard-bin CvM-style loss and its legacy surrogate gradient for backward-compatible studies. Early stopping remains stage-local, using test `classification_loss` in stage 1 and test `total_loss` in stage 2.
- `train.py` saves the stage-1 baseline model as `{tree}_model_stage1.json`, the final output as `{tree}_model.json`, and writes comparison plots with `_cls` / `_decorr` suffixes for the stage-1 and stage-2 models respectively. Downstream scripts load JSON models through `xgb.Booster`, take raw margins with `output_margin=True`, and apply the multiclass softmax explicitly before using the BDT scores.
- After writing the final model, `train.py` also writes `test_reference_signal_region.npz` and `test_reference_qcd_est.npz`. Each file stores the exact test-split feature order, sample order, class labels, pre-lumi physics weights, predicted class probabilities, and per-array tolerances for the corresponding downstream preprocessing chain. `signal_region.py` and `qcd_est.py` multiply the stored reference weights by their local configured `lumi` before comparing them to the reconstructed physical weights; any remaining mismatch is treated as fatal and the script exits instead of silently proceeding.
- ROC and score-distribution plots adapt to the configured classes and are written separately for the stage-1 baseline (`*_cls.pdf`) and stage-2 final model (`*_decorr.pdf`) when both stages exist. When both signal-like and background-like classes exist, one ROC PDF is written per signal class, overlaying every background class in a distinct color (test curves solid, train curves dashed), and one score PDF is written per signal-background pair. When there is no signal/background split, the code falls back to one ROC PDF per class (treating it as the "signal" against all other classes) and all-pair score distributions.
- After training, `train.py` copies `config.json`, `branch.json`, and `selection.json` into the tree output directory so that `signal_region.py` can run self-contained from that directory.

## Signal Region Scan Details (`signal_region.py`)

- Reads all parameters from `scan.json` (configurable via `SCAN_CONFIG_PATH` env var).
- Loads test events from `test_ranges.json` (exact same split as used in training).
- Physics weights: `target_total = lumi × xsection × total_tree_entries / raw_entries` and `per_event_weight = raw_w × target_total / Σ raw_w_loaded`, where `total_tree_entries` is all entries across all ROOT files for that sample in the chosen tree (train + test combined), and `raw_w` is the per-event product of the tree's `event_reweight_branches` in the BDT config copy (read on raw values, before any clip/log/threshold, and dropped from the DataFrame after `raw_w` is built). The sample total `Σ weight = target_total` regardless of raw_w; missing test split files/trees, failed segment reads, non-positive `raw_entries`, or non-positive `Σ raw_w_loaded` raise `RuntimeError`. No class rebalancing is applied.
- Preprocessing (clip, log transform, threshold filter) is identical to train.py's test-split logic.
- Threshold/sentinel filtering matches `train.py`: only branches that appear in `selection.json`'s `thresholds` block are inspected, and only those branches can drop events for sentinel values (`< -990`) or failed threshold conditions.
- Branch loading mirrors `train.py`: threshold or decorrelate branches that are not in `branch.json` are still read from the test split ROOT files for filtering and decorrelation, then the threshold-only ones (not in `decorrelate`) are dropped from `X` before the existing decorrelation/model-inference step runs.
- Decorrelated features (e.g., `msd8_1`) are excluded from the model input, same as during training.
- Scan dimensions come from `score_axes`: `independent` uses `p(class_0)` through `p(class_{NUM_CLASSES-2})`, while `all` includes every class probability, including the final class so direct upper/lower cuts on that class can be represented.
- Each signal region is a general axis-aligned high-dimensional rectangle: every axis has its own `[low, high)` (`high == 1.0` is open-ended). The previous "single slim slice + other axes open to 1" restriction has been removed; downstream `qcd_est.py` already consumes the per-axis `_low`/`_high` columns from `signal_region.csv`.
- Significance formula per region: `Z = sqrt(2 * [(S+B)*ln(1+S/B) − S])` (profile-likelihood). Error propagated via partial derivatives.
- Combined objective: `Z_comb = sqrt(Σ Z_i²)` over the N selected non-overlapping regions. The final selection is global over the generated candidate list rather than sequential: it searches K-region combinations and maximizes `Σ Z_i²`, which is exactly the script's existing combined-significance objective.
- Per-axis edge candidates: total, signal-only, background-only, and unweighted score quantiles are combined with tail-heavy quantiles, capped at `max_edge_candidates_per_axis` distinct edges while always including `0.0` and `1.0`. These edges are the global candidate boundary set.
- Beam coordinate search: seeds include the full box, low/mid/high-tail high-side and low-side single-axis seeds from `seed_quantiles`, and up to `seed_intervals_per_axis` exact single-axis bounded `[low, high)` interval seeds per score axis. `multi_axis_seed_max_axes` / `multi_axis_seed_max_seeds` can add bounded combinations of those one-axis seed intervals so the search starts from simultaneous lower/upper cuts on multiple score axes instead of relying only on later coordinate updates. Optional `forbidden_regions` are still supported for compatibility, but the normal workflow no longer performs one scan per SR. For up to `coordinate_rounds`, every beam state updates every axis by histogramming events that pass the other axes, sweeping all `(a < b)` intervals in `O(K²)`, and keeping the best `top_intervals_per_axis` exact intervals for that axis. Independent per-axis/per-state interval scans may use `max_threads`, but their outputs are merged back into the pool in deterministic scan order so threading does not change the finite candidate set. The beam keeps the strongest states while preserving some non-overlapping diversity, and every valid visited rectangle is added to the candidate pool.
- Compatibility expansion: after the first shared beam pass, the scan takes up to `compatibility_seed_anchors` high-Z anchors chosen by distinct event masks and creates seeds that are guaranteed to be geometrically non-overlapping with each anchor on one separating score axis. It then runs additional coordinate updates while keeping that separating axis fixed, so the pool contains candidates that are explicitly optimized as possible partners for a slightly lower first region. These candidates still enter the same global candidate list; they are not selected greedily.
- Local event-threshold refinement: up to `local_refine_top_candidates` regions are refined for `local_refine_rounds` passes. With `local_refine_diverse_masks=true`, the refinement seed list is chosen by distinct event masks rather than raw Z order, preventing hundreds of geometric variants of the same event set from consuming all refinement passes. Around each current low/high boundary, the scan injects nearby real event score values from the cached per-axis score values (`local_refine_neighbor_edges`) and again performs exact one-axis interval sweeps, adding each valid refined rectangle to the pool.
- Candidate pool: every region is deduped only by rounded `(low, high)` during generation and gated by `B ≥ min_bkg_weight`, `S > min_signal_weight`, and optional event-count minima. No candidate-pool or global-candidate count limit is applied, so candidates are not removed by single-region `Z`. Before global selection, every candidate is shrunk to an event-preserving minimal score box; exact event-mask/canonical-box deduplication then removes only candidates that select the same exact event mask and shrink to the same rounded canonical box. This keeps `S/B/Z` unchanged on the scan events while avoiding deletion of merely lower-ranked but complementary candidates.
- Global selection: `signal_region.py` compiles and loads `openmp_region_select.cpp` into `output_dir/.openmp/` when `max_threads > 1`. The helper first runs a bounded beam search with `global_beam_width` to get a strong incumbent, then runs branch-and-bound over the candidate list with an admissible upper bound that ignores future mutual overlaps but preserves compatibility with already selected regions. `branch_bound_max_nodes=0` disables the node cap, and `branch_bound_time_limit_seconds=0` disables the wall-time cap. If the search completes, the selected set is exact for the finite candidate list; if a cap is hit, the script reports `Z_best`, a conservative `Z_upper_bound`, and the resulting `delta_Z` certificate. If OpenMP build/load fails or `max_threads == 1`, the same search runs in Python. Progress logs report beam rounds, local refinement progress, candidate-pool size, global selector type, completion status, node count, overlap checks, and final `Z_comb`.
- Empty-bin score-axis expansion: after global selection and the non-overlap check, every selected SR is expanded in order (SR1 first, then SR2, …). For each SR, signal-class score axes have their upper bound pushed toward `1.0` and background-class score axes have their lower bound pushed toward `0.0`, but only into space that is empty in MC under that SR's other-axis cuts. Each expansion is constrained so that (a) the SR's exact selected event mask is preserved, and (b) the expanded box remains geometrically non-overlapping with every other selected SR (already-expanded earlier ones and not-yet-expanded later ones). The other expansion direction (signal-axis low / background-axis high) is intentionally **not** expanded. Per-bin `S`, `B`, `Z` are recomputed after expansion and stay numerically identical to the pre-expansion values; only the empty-side score-axis bounds change in the per-SR report and in `signal_region.csv`. The script logs how many signal-axis high bounds and background-axis low bounds were widened.
- Signal class: any class where all member samples have `is_signal=true`. Background: all other classes.
- Output files: `sr_score_*.pdf`, `scores_no_regions.pdf`, `scores.pdf`, and `signal_region.csv` all go to `output_dir`. The CSV stores one row per selected signal region and expands the high-dimensional bin definition into per-axis `{axis_name}_low` / `{axis_name}_high` columns.

## `run.sh` Behavior

[run.sh](run.sh) supports eight modes:

- `mode=0` compiles and runs `selections/convert/convert_branch.C`.
- `mode=1` compiles and runs `selections/weight/weight.C`.
- `mode=2` runs `selections/BDT/train.py` with `BDT_CONFIG_PATH` pointing to the chosen config file.
- `mode=3` runs `selections/signal_region/signal_region.py` with `SCAN_CONFIG_PATH` pointing to the chosen config file.
- `mode=4` runs `plotting/data_mc.py` with `PLOT_CONFIG_PATH` pointing to the chosen config file.
- `mode=5` runs `background_estimation/qcd_est.py` with `QCD_EST_CONFIG_PATH` pointing to the chosen config file.
- `mode=6` compiles and runs `selections/mix/mix.C` with `MIX_CONFIG_PATH` pointing to the chosen config file.
- `mode=7` compiles and runs `combine/combine.C` with `COMBINE_CONFIG_PATH` pointing to the chosen config file. It does not take sample arguments and runs once per invocation.

For `mode=0`, `mode=1`, and `mode=6`, the script resolves the sample list from JSON configs, then runs jobs with `MAX_CONCURRENT_JOBS` parallelism (default 1). Log output goes to `selections/convert/log.txt`, `selections/weight/log.txt`, `selections/BDT/log.txt`, `selections/signal_region/log.txt`, `plotting/log.txt`, `background_estimation/log.txt`, `selections/mix/log.txt`, or `combine/log.txt` depending on mode. All eight modes use the same timestamped run-log style in `run.sh`; `mode=0`, `mode=1`, and `mode=6` log per-sample `started` / `finished` records, while `mode=2`, `mode=3`, `mode=4`, `mode=5`, and `mode=7` log one `started` / `finished` record for the whole job with an explicit exit `status=`. Inside `selections/BDT/train.py`, `selections/signal_region/signal_region.py`, `plotting/data_mc.py`, and `background_estimation/qcd_est.py`, the script output follows the same concise stage-by-stage `Running ...`, `Wrote ...`, and `Runtime error: ...` style. The compiled binary is removed on exit for the C++ modes. OpenMP is auto-detected for intra-job parallelism when compiling the C++ tools.

## C++ Expression Engine (convert_branch.C)

The selection and branch formulas in JSON are parsed by a mini expression engine inside [selections/convert/convert_branch.C](selections/convert/convert_branch.C). Supported syntax:
- Arithmetic/logical operators, comparisons
- Functions: `pt(obj)`, `eta(obj)`, `phi(obj)`, `mass(obj)`, `abs()`, `min()`, `max()`, `size(coll)`, `deltaR(a,b)`, `min_deltaR(self, coll)`, `relPtDiff(a,b)`
- Collection indexing: `ak8[0]`, `ak4[1]`
- `self` / `other` keywords in per-object selection strings

## Physics Context

- **Signal:** VVV hadronic decays — WWW, WWZ, WZZ, ZZZ, and VH(->WW) associated production
- **Trigger:** `DST_PFScouting_JetHT` scouting stream
- **Jets:** AK8 fat jets (`ScoutingFatPFJetRecluster`) and AK4 jets (`ScoutingPFJetRecluster2`)
- **Signal regions:** `fat2` (2 AK8 jets, ≥2 AK4 jets) and `fat3` (≥3 AK8 jets)
- **QCD background:** estimated with ABCD method using BDT score as discriminant
- **Pileup corrections:** three variations (nominal/low/high) for systematic uncertainty
