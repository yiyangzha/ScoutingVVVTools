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

## Overview

This is a CMS physics analysis toolkit for a VVV (triple vector boson) scouting search using Run 3 data. The workflow processes CMS NanoAOD-style ROOT files from DAS, applies selections, trains BDT models, and estimates backgrounds.

**Dependencies:** ROOT (with `root-config`), C++17, Python 3, OpenMP (optional), XRootD for remote files.

## Repository Maintenance

- Whenever you modify code or add a new repository-wide requirement, update this `AGENTS.md` to keep the project description and workflow notes in sync.

## Pipeline Overview

The analysis runs in this order:

1. **Skim** — reduce raw NanoAOD files to manageable size (`legacy/skim/`)
2. **Pileup weights** — derive MC pileup reweighting CSVs (`selections/weight/`)
3. **Convert** — apply physics selections, build BDT training trees (`selections/convert/`)
4. **B-veto** — derive AK4 b-jet veto working points (`selections/b_veto/`)
5. **BDT training** — train `fat2`/`fat3` classifiers (`selections/BDT/`)
6. **Background estimation** — QCD ABCD method and data/MC comparisons (`background_estimation/`)
7. **Systematics** — trigger efficiency studies (`systematics/`)

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

### ROOT macro tools
```bash
root -l -q 'systematics/efficiency.C("2024F", 200)'
root -l -q 'systematics/efficiency.C("vvv", 0)'
root -l -q 'selections/b_veto/b_veto.C'
```

### Python tools
```bash
python3 background_estimation/qcd_est.py \
  --base-dir /path/to/dataset \
  --model /path/to/model.json \
  --out-dir qcd_est

python3 background_estimation/data_mc.py group_hists_out.root

python3 selections/b_veto/ttbar_score.py
```

### BDT training
```bash
jupyter lab  # open selections/BDT/train.ipynb
# or run as script:
python3 selections/BDT/train.py
```

BDT outputs are written under the per-tree `output_root` configured in `selections/BDT/config.json` (for example `selections/BDT/fat2/`). Models are saved there together with concise PDF summaries such as `importance.pdf`, `loss.pdf`, `feature_corr.pdf`, `decor_corr_train.pdf`, and pairwise `roc_*` / `score_*` plots. When launched through `run.sh 2`, stdout/stderr are redirected to `selections/BDT/log.txt`, and both `run.sh` and `train.py` follow the same concise run-log style as the convert and weight programs: a `Running ...` header, thread information, `Wrote ...` output records, and `Runtime error: ...` on failure.

## Configuration Architecture

All tools are driven by JSON config files. Sample definitions live centrally in [src/sample.json](src/sample.json); individual tool configs reference it via the `sample_config` key.

- **[src/sample.json](src/sample.json)** — master sample registry. Each entry has `name`, `path` (DAS path, string or list), `sample_ID`, `is_MC`, `is_signal`, `xsection`, `lumi`, and `raw_entries`. For BDT training, `raw_entries` must be the total number of generated MC events before any convert-level filtering.
- **[selections/convert/config.json](selections/convert/config.json)** — controls convert step: output paths, thread count, file size limits, pileup weight CSV path pattern.
- **[selections/convert/selection.json](selections/convert/selection.json)** — physics selection: event preselection string, per-collection cuts/sorts, and `tree_selection` that splits output into `fat2` (exactly 2 AK8 jets) and `fat3` (≥3 AK8 jets) trees. Selections are parsed and JIT-compiled by the C++ expression engine.
- **[selections/convert/branch.json](selections/convert/branch.json)** — declares all input NanoAOD branches to read (scalars and collections with p4 definitions) and output branches to write.
- **[selections/weight/config.json](selections/weight/config.json)** — pileup reweighting settings: data pileup histogram files (nominal/low/high for systematics).
- **[selections/BDT/config.json](selections/BDT/config.json)** — controls BDT training inputs and outputs: `submit_trees`, `class_groups`, `output_root`, `model_pattern`, `entries_per_sample`, and per-tree hyperparameters plus `decorrelate`.
- **[selections/BDT/branch.json](selections/BDT/branch.json)** — lists the BDT feature branches. These names are aligned to the current `convert_branch` output names and include all non-`onlyMC` `fat2`/`fat3` branches except event identifiers, pileup bookkeeping, selection counts, and truth/meta labels.
- **[selections/BDT/selection.json](selections/BDT/selection.json)** — training-time feature preprocessing: clipping ranges, log transforms, and threshold cuts applied after reading the converted `fat2`/`fat3` trees.

## BDT Training Details

- `selections/BDT/train.py` only trains on the samples explicitly listed in `class_groups`.
- The number of classes is inferred dynamically from `class_groups`; no fixed 5-class assumption is baked into training.
- A class is treated as `single` for plotting if every sample in that class has `is_signal=true`; otherwise it is treated as `background`.
- Sample weights are computed once, immediately after reading each converted sample and before any BDT threshold cuts:
  `sample_total_weight(tree) = xsection * N_tree / raw_entries`
  where `N_tree` is the total number of entries in the converted `fat2` or `fat3` tree across all matched ROOT files.
- If `entries_per_sample` limits how many events are actually loaded, per-event weight is scaled so the loaded subset still sums to `sample_total_weight(tree)`.
- Class totals are then rescaled once so every class has the same total training weight.
- After those weights are assigned, later threshold cuts in `selections/BDT/selection.json` do **not** trigger any reweighting; the surviving events keep their original per-event weight. This means threshold efficiency naturally contributes to the effective sample composition seen by the trainer.
- ROC and score-distribution plots adapt to the configured classes. When both signal-like and background-like classes exist, the code produces all signal-background class pairs; otherwise it falls back to all class pairs.

## `run.sh` Behavior

[run.sh](run.sh) supports three modes:

- `mode=0` compiles and runs `convert/convert_branch.C`.
- `mode=1` compiles and runs `weight/weight.C`.
- `mode=2` runs `BDT/train.py` with `BDT_CONFIG_PATH` pointing to the chosen config file.

For `mode=0` and `mode=1`, the script resolves the sample list from JSON configs, then runs jobs with `MAX_CONCURRENT_JOBS` parallelism (default 1). Log output goes to `selections/convert/log.txt`, `selections/weight/log.txt`, or `selections/BDT/log.txt` depending on mode. All three modes use the same timestamped run-log style in `run.sh`; the C++ modes log per-sample `started` / `finished` records, while `mode=2` logs one `started` / `finished` record for the whole BDT job with an explicit exit `status=`. The compiled binary is removed on exit for the C++ modes. OpenMP is auto-detected for intra-job parallelism when compiling the C++ tools.

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
