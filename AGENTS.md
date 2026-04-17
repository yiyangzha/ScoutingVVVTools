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
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a CMS physics analysis toolkit for a VVV (triple vector boson) scouting search using Run 3 data. The workflow processes CMS NanoAOD-style ROOT files from DAS, applies selections, trains BDT models, and estimates backgrounds.

**Dependencies:** ROOT (with `root-config`), C++17, Python 3, OpenMP (optional), XRootD for remote files.

## Pipeline Overview

The analysis runs in this order:

1. **Skim** — reduce raw NanoAOD files to manageable size (`legacy/skim/`)
2. **Pileup weights** — derive MC pileup reweighting CSVs (`selections/weight/`)
3. **Convert** — apply physics selections, build BDT training trees (`selections/convert/`)
4. **B-veto** — derive AK4 b-jet veto working points (`selections/b_veto/`)
5. **BDT training** — train `fat2`/`fat3` classifiers (`selections/BDT/`)
6. **Background estimation** — QCD ABCD validation on the MC test split (`background_estimation/`)
7. **Data/MC plotting** — compare distributions of `fat2`/`fat3` variables in data vs MC (`plotting/`)
8. **Systematics** — trigger efficiency studies (`systematics/`)

## Convert_branch.C output layout (consumed downstream)

`convert_branch.C` writes per-sample ROOT files laid out as
`{output_root}/{sample_group}/{sample}.root`, where `sample_group ∈ {signal, bkg, data}` is derived from each sample's `is_MC`/`is_signal` flags. When a single output file exceeds `max_output_file_size_gb`, the writer rolls over into split files named `{sample}_1.root`, `{sample}_2.root`, …, alongside (or instead of) the base `{sample}.root`. Every ROOT file contains the `fat2` and `fat3` trees described by `selections/convert/branch.json`.

Downstream scripts — `selections/BDT/train.py`, `selections/BDT/signal_region/signal_region.py`, and `plotting/data_mc.py` — all consume these files via their config's `input_root` + `input_pattern` (`{input_root}/{sample_group}/{sample}.root`). They discover every file for a given sample by globbing the base pattern **and** the `{sample}_*.root` split variant, then concatenate their trees in sorted order.

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

### Compile and run C++ tools manually
```bash
# compile
c++ -O3 -std=c++17 $(root-config --cflags --libs) selections/convert/convert_branch.C -o convert_branch
# run one sample
CONVERT_CONFIG_PATH=selections/convert/config.json ./convert_branch www
```

When `convert_branch.C` runs a sample, it first sums the configured `tree_name` entries across all ROOT files resolved from that sample's `path` entries, then writes that total back to the active sample config's `raw_entries` field for the running sample before processing events.

### BDT training

BDT outputs are written under the per-tree `output_root` configured in `selections/BDT/config.json` (for example `selections/BDT/fat2/`). Models are saved there together with concise PDF summaries such as `importance.pdf`, `loss.pdf`, `feature_corr.pdf`, `decor_corr_train.pdf`, and pairwise `roc_*` / `score_*` plots. Each tree output directory also stores copies of `config.json`, `branch.json`, `selection.json`, and the fixed per-sample test split definition `test_ranges.json`.

### Signal region optimisation
```bash
# Edit selections/signal_region/config.json, then:
python3 selections/signal_region/signal_region.py
# Or with a custom scan config:
SCAN_CONFIG_PATH=/path/to/config.json python3 selections/signal_region/signal_region.py
```

`signal_region.py` reads all parameters from [selections/signal_region/config.json](selections/signal_region/config.json). It loads the saved model, branch, and selection configs from the `bdt_root` directory written by `train.py`, reloads the exact test split defined in `test_ranges.json`, applies physics-normalised weights (`lumi × σ × N_tree / N_raw`), and scans for N non-overlapping signal regions in the `NUM_CLASSES − 1` dimensional BDT score space that maximise `Z = sqrt(2[(S+B)ln(1+S/B) - S])`. Plots are saved as `sr_score_*.pdf` and `sr_regions_2d.pdf` in the configured `output_dir` (defaulting to `bdt_root` for backward compatibility), and the selected high-dimensional region definitions are written to `signal_region.csv` in the same directory.

### QCD ABCD estimation
```bash
python3 background_estimation/qcd_est.py
# Or with a custom config:
QCD_EST_CONFIG_PATH=/path/to/config.json python3 background_estimation/qcd_est.py
```

`background_estimation/qcd_est.py` reads [background_estimation/config.json](background_estimation/config.json), then reuses the trained `bdt_root` output in the same way as `signal_region.py`: it loads the saved `config.json`, `model`, `branch.json`, `selection.json`, and `test_ranges.json`, reads the full test split with the same per-event weight definition, removes only the non-mass thresholds before BDT inference, applies the same clip/log preprocessing, and uses the `signal_region.csv` produced by `signal_region.py` as the set of A-region score bins. It defines `A` as the union of those score bins with mass-pass, `B` as outside that union with mass-pass, `C` as inside that union with mass-fail, and `D` as outside that union with mass-fail, where mass-pass requires all `ScoutingFatPFJetRecluster_msoftdrop_*` thresholds to pass and mass-fail requires all of them to fail. A single QCD ABCD scale from the union-level `B/C/D` totals is then applied to all individual signal regions. The predicted QCD uncertainty in each SR is split into two pieces: an uncorrelated `stat_error` term from the SR fraction inside the A-union, and a fully correlated `scale_error` term from the global ABCD QCD scale. The ROOT output stores `yield`, `stat_error`, `scale_error`, and a full `covariance_total` matrix for every saved category, with the covariance matrix dimension matching the number of signal regions.

### Data vs MC plotting
```bash
python3 plotting/data_mc.py
# Or with a custom plotting config:
PLOT_CONFIG_PATH=/path/to/config.json python3 plotting/data_mc.py
```

`plotting/data_mc.py` reads [plotting/config.json](plotting/config.json) and [plotting/branch.json](plotting/branch.json), then for every tree in `submit_trees`:
- Loads the BDT-tree copies of `config.json` (to get `class_groups`, `input_root`, `input_pattern`) and `selection.json` (for `clip_ranges`, `thresholds`, `log_transform`) from the directory resolved by `bdt_root` with `{tree_name}` substituted. `input_root` is resolved relative to the BDT script directory (the parent of `bdt_root`), matching `train.py`'s convention.
- Discovers plottable variables from `selections/convert/branch.json`'s output entries for the tree (only `onlyMC: false`, expanding `slots` into `{name}_1 … {name}_N`, minus that tree's `skip_branches` from `plotting/branch.json`).
- For each MC sample listed in any `class_groups` entry, it globs the convert_branch ROOT files (`{sample}.root` + `{sample}_*.root`), sums the tree entries across those files, loads every event, and assigns a constant per-event weight so the sum equals `lumi_total × xsection × tree_entries_total / raw_entries`. `lumi_total` is the sum of `lumi` fields in `src/sample.json` for every sample name in `data_samples` (every listed data sample **must** already exist in `src/sample.json` with `is_MC: false`, otherwise data_mc.py errors out). The per-event weight is frozen before any cuts and is not recomputed later.
- For each data sample in `data_samples`, loads every event with `weight = 1` per event.
- Applies threshold cuts, then clip ranges, both from the BDT tree's `selection.json`. `log_transform` is **not** applied — it only influences the default `logx` flag for plotting.
- Histograms each branch with the resolved binning: default `default_bins` bins, auto range from the union of MC + data, `logy` always on by default, `logx` on if the branch appears in `log_transform`. `logx` uses log-spaced bins and drops non-positive values; sentinel values (`< -990`) are excluded per-branch from that branch's histogram but the event stays in the sample for other branches. Per-branch overrides from `plotting/branch.json` (`bins`, `x_range`, `y_range`, `logx`, `logy`) take precedence.
- Emits stage-by-stage progress messages in the same concise style as `signal_region.py`, including config resolution, sample loading, filtering, per-branch plotting progress, `Wrote ...`, and `Runtime error: ...`.
- Saves one PDF per branch to `{output_root}/{tree_name}_{branch}.pdf` with a top panel (stacked MC + Data points + hatched MC uncertainty band) and a bottom Data/MC ratio panel.

## Configuration Architecture

All tools are driven by JSON config files. Sample definitions live centrally in [src/sample.json](src/sample.json); individual tool configs reference it via the `sample_config` key.

- **[src/sample.json](src/sample.json)** — master sample registry. Each entry has `name`, `path` (DAS path, string or list), `sample_ID`, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`. `convert_branch.C` updates `raw_entries` for the running sample by summing the chosen `tree_name` entries across all ROOT files resolved from that sample's configured `path` values.
- **[selections/convert/config.json](selections/convert/config.json)** — controls convert step: output paths, thread count, file size limits, pileup weight CSV path pattern.
- **[selections/convert/selection.json](selections/convert/selection.json)** — physics selection: event preselection string, per-collection cuts/sorts, and `tree_selection` that splits output into `fat2` (exactly 2 AK8 jets) and `fat3` (≥3 AK8 jets) trees. Selections are parsed and JIT-compiled by the C++ expression engine.
- **[selections/convert/branch.json](selections/convert/branch.json)** — declares all input NanoAOD branches to read (scalars and collections with p4 definitions) and output branches to write.
- **[selections/weight/config.json](selections/weight/config.json)** — pileup reweighting settings: data pileup histogram files (nominal/low/high for systematics).
- **[selections/BDT/config.json](selections/BDT/config.json)** — controls BDT training inputs and outputs: `submit_trees`, `class_groups`, `output_root`, `model_pattern`, `entries_per_sample`, `train_fraction`, and per-tree hyperparameters plus `decorrelate`.
- **[selections/signal_region/config.json](selections/signal_region/config.json)** — controls signal_region.py: `lumi` (fb⁻¹), `N` / `n_signal_regions` (number of signal regions), `bdt_root` (trained tree output directory to read from, relative to `signal_region/`), `output_dir` (directory for saved PDFs and `signal_region.csv`, relative to `signal_region/`, defaulting to `bdt_root`), `n_thresholds` (scan grid points per axis), `min_bkg_weight` (minimum background weight per bin), and `rounds` (iterative refinement rounds).
- **[background_estimation/config.json](background_estimation/config.json)** — controls qcd_est.py: `lumi` (fb⁻¹), `bdt_root` (trained tree output directory to read from), `signal_region_csv` (path to the `signal_region.csv` written by `signal_region.py`), `output_dir` (directory for PDFs and ROOT outputs), and `root_file_name` (summary ROOT filename).
- **[plotting/config.json](plotting/config.json)** — controls data_mc.py: `submit_trees`, `sample_config`, `convert_branch_config`, `bdt_root` (per-tree path pattern, points at the BDT tree output dir that already contains the copied `config.json` / `selection.json`), `output_root` (per-tree output dir pattern), `data_samples` (list of data sample names whose entries must exist in `src/sample.json`), and `default_bins` (default histogram bin count).
- **[plotting/branch.json](plotting/branch.json)** — plotting config split by tree (`fat2` / `fat3`). Each tree can define `skip_branches` and a `branches` map. Inside `branches`, each branch override can set `bins`, `x_range`, `y_range`, `logx`, `logy`; unset fields fall back to defaults. The file is intended to hold a few explicit examples that can be copied when adding new plot formatting rules later.

## BDT Training Details

- `selections/BDT/train.py` only trains on the samples explicitly listed in `class_groups`.
- The number of classes is inferred dynamically from `class_groups`; no fixed 5-class assumption is baked into training.
- A class is treated as `single` for plotting if every sample in that class has `is_signal=true`; otherwise it is treated as `background`.
- For each sample, the code first counts all entries across all matched ROOT files in the chosen tree, then defines a fixed sequential split: the first `train_fraction` goes to training and the remaining tail goes to testing. The split is defined on the concatenated sample entry order across files, so `test_ranges.json` can be used later to recover the exact test subset.
- `entries_per_sample` only caps how many events are actually loaded from the training side of that fixed split. The test side is always the full tail of the sample.
- Training events are shuffled after loading; test events keep their original order.
- Sample weights are computed separately for the training and test splits, immediately after reading and before any BDT threshold cuts:
  `sample_total_weight(split) = xsection * N_split / raw_entries`
  where `N_split` is the total number of entries in that fixed train or test split before any threshold cuts.
- If `entries_per_sample` limits how many training events are actually loaded, per-event weight is scaled so the loaded training subset still sums to `sample_total_weight(train)`.
- Class totals are then rescaled separately within the training split and within the test split so every class has the same total weight inside each split.
- After those weights are assigned, later threshold cuts in `selections/BDT/selection.json` do **not** trigger any reweighting; the surviving events keep their original per-event weight. This means threshold efficiency naturally contributes to the effective sample composition seen by the trainer.
- Threshold/sentinel filtering in `filter_X` is applied **only** to branches that appear as keys in `selection.json`'s `thresholds` block. For each such branch, events with sentinel values (`< -990`) are dropped and the threshold condition is enforced. Branches that are *not* listed in `thresholds` are never inspected, so an event with a sentinel value in (for example) a lepton branch is still kept as long as the `thresholds` block doesn't target that lepton branch.
- ROC and score-distribution plots adapt to the configured classes. When both signal-like and background-like classes exist, the code produces all signal-background class pairs; otherwise it falls back to all class pairs.
- After training, `train.py` copies `config.json`, `branch.json`, and `selection.json` into the tree output directory so that `signal_region.py` can run self-contained from that directory.

## Signal Region Scan Details (`signal_region.py`)

- Reads all parameters from `scan.json` (configurable via `SCAN_CONFIG_PATH` env var).
- Loads test events from `test_ranges.json` (exact same split as used in training).
- Physics weights: `total_weight_per_sample = lumi × xsection × total_tree_entries / raw_entries`, where `total_tree_entries` is all entries across all ROOT files for that sample in the chosen tree (train + test combined). No class rebalancing is applied.
- Preprocessing (clip, log transform, threshold filter) is identical to train.py's test-split logic.
- Threshold/sentinel filtering matches `train.py`: only branches that appear in `selection.json`'s `thresholds` block are inspected, and only those branches can drop events for sentinel values (`< -990`) or failed threshold conditions.
- Decorrelated features (e.g., `msd8_1`) are excluded from the model input, same as during training.
- Scan dimensions: `D = NUM_CLASSES − 1`; axes are `p(class_0)` through `p(class_{D-1})` (the last class is excluded since probabilities sum to 1).
- Significance formula: `Z = sqrt(2 * [(S+B)*ln(1+S/B) - S])` (profile-likelihood). Error propagated via partial derivatives.
- Iterative refinement: for each of the N bins, runs `rounds` refinement iterations narrowing to top-3 candidates per axis, then picks the highest-Z non-overlapping bin satisfying `min_bkg_weight`.
- Combined significance: `Z_comb = √(Σ Z_i²)`.
- Signal class: any class where all member samples have `is_signal=true`. Background: all other classes.
- Output files: `sr_score_*.pdf`, `sr_regions_2d.pdf`, and `signal_region.csv` all go to `output_dir`. The CSV stores one row per selected signal region and expands the high-dimensional bin definition into per-axis `{axis_name}_low` / `{axis_name}_high` columns.

## `run.sh` Behavior

[run.sh](run.sh) supports six modes:

- `mode=0` compiles and runs `selections/convert/convert_branch.C`.
- `mode=1` compiles and runs `selections/weight/weight.C`.
- `mode=2` runs `selections/BDT/train.py` with `BDT_CONFIG_PATH` pointing to the chosen config file.
- `mode=3` runs `selections/signal_region/signal_region.py` with `SCAN_CONFIG_PATH` pointing to the chosen config file.
- `mode=4` runs `plotting/data_mc.py` with `PLOT_CONFIG_PATH` pointing to the chosen config file.
- `mode=5` runs `background_estimation/qcd_est.py` with `QCD_EST_CONFIG_PATH` pointing to the chosen config file.

For `mode=0` and `mode=1`, the script resolves the sample list from JSON configs, then runs jobs with `MAX_CONCURRENT_JOBS` parallelism (default 1). Log output goes to `selections/convert/log.txt`, `selections/weight/log.txt`, `selections/BDT/log.txt`, `selections/signal_region/log.txt`, `plotting/log.txt`, or `background_estimation/log.txt` depending on mode. All six modes use the same timestamped run-log style in `run.sh`; the C++ modes log per-sample `started` / `finished` records, while `mode=2`, `mode=3`, `mode=4`, and `mode=5` log one `started` / `finished` record for the whole Python job with an explicit exit `status=`. Inside `selections/BDT/train.py`, `selections/signal_region/signal_region.py`, `plotting/data_mc.py`, and `background_estimation/qcd_est.py`, the script output follows the same concise stage-by-stage `Running ...`, `Wrote ...`, and `Runtime error: ...` style. The compiled binary is removed on exit for the C++ modes. OpenMP is auto-detected for intra-job parallelism when compiling the C++ tools.

## C++ Expression Engine (convert_branch.C)

The selection and branch formulas in JSON are parsed by a mini expression engine inside [selections/convert/convert_branch.C](selections/convert/convert_branch.C). Supported syntax:
- Arithmetic/logical operators, comparisons
- Functions: `pt(obj)`, `eta(obj)`, `phi(obj)`, `mass(obj)`, `abs()`, `min()`, `max()`, `size(coll)`, `deltaR(a,b)`, `min_deltaR(self, coll)`, `relPtDiff(a,b)`
- Collection indexing: `ak8[0]`, `ak4[1]`
- `self` / `other` keywords in per-object selection strings

## Physics Context

- **Signal:** VVV hadronic decays — WWW, WWZ, WZZ, ZZZ, and VH(→WW) associated production
- **Trigger:** `DST_PFScouting_JetHT` scouting stream
- **Jets:** AK8 fat jets (`ScoutingFatPFJetRecluster`) and AK4 jets (`ScoutingPFJetRecluster2`)
- **Signal regions:** `fat2` (2 AK8 jets, ≥2 AK4 jets) and `fat3` (≥3 AK8 jets)
- **QCD background:** estimated with ABCD method using BDT score as discriminant
- **Pileup corrections:** three variations (nominal/low/high) for systematic uncertainty
