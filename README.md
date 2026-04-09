## `data/skim`

### `skim_mc.C`

- Function: Skim MC ROOT files and merge the results.
- Edit: `inputDir`, `outputFile`, `cutString`.
- Run:
```bash
root -l -q 'data/skim/skim_mc.C'
```
- Input: ROOT files with an `Events` tree.
- Output: Merged skim ROOT file(s) with `cutflow`.

### `skim_condor.C`

- Function: Skim one remote ROOT file for a Condor-style job.
- Edit: `branchPatterns`, `cutString`.
- Run:
```bash
c++ -O2 $(root-config --cflags --libs) data/skim/skim_condor.C -o skim_condor
./skim_condor root://cms-xrd-global.cern.ch//store/...
```
- Input: One remote ROOT file.
- Output: `temp.root` with `Events` and `cutflow`.

### `skim_data.C`

- Function: Skim data files from a list and merge the results.
- Edit: `listFile`, `outputFile`, `cutString`, and prepare `x509up`.
- Run:
```bash
root -l -q 'data/skim/skim_data.C'
```
- Input: A text file with one ROOT path per line.
- Output: Merged skim ROOT file(s) with `cutflow`.

### `merge_data.C`

- Function: Merge existing ROOT files in batches and apply the event cut during merging.
- Edit: `inputDir`, `outputFile`, `cutString`.
- Run:
```bash
root -l -q 'data/skim/merge_data.C'
```
- Input: ROOT files with `Events` and `cutflow`.
- Output: Merged ROOT file(s) with `cutflow`.

## `selections/BDT/convert_branch.C`

- Function: Convert skimmed events into `fat2` and `fat3` training trees.
- Edit: Input paths if your skim files are stored elsewhere.
- Run:
```bash
root -l -q 'selections/BDT/convert_branch.C("www")'
root -l -q 'selections/BDT/convert_branch.C("qcd_ht2000toinf")'
root -l -q 'selections/BDT/convert_branch.C("2024H")'
```
- Input: One skimmed ROOT file selected by `typeArg`.
- Output: `dataset/signal/<type>.root`, `dataset/bkg/<type>.root`, or `dataset/data/<type>.root`.

## `selections/BDT/train.ipynb`

- Function: Train and evaluate the `fat2` and `fat3` BDT models.
- Edit: `BASE_DIR`, sample paths, model names, and training settings.
- Run:
```bash
jupyter lab
```
- Input: Dataset ROOT files in `signal`, `bkg`, and `data`.
- Output: Trained models and analysis plots saved by the notebook cells.

## `background_estimation/qcd_est.py`

- Function: Estimate the QCD yield in the `fat2` signal region.
- Edit: Dataset paths, model path, thresholds, and output directory if needed.
- Run:
```bash
python3 background_estimation/qcd_est.py \
  --base-dir /path/to/dataset \
  --model /path/to/model.json \
  --out-dir qcd_est
```
- Input: Dataset folders and one trained BDT model.
- Output: Region plots, scan plots, and `hists.root`.

## `background_estimation/data_mc.py`

- Function: Plot data-versus-MC comparisons from grouped ROOT histograms.
- Edit: Input file name, branch lists, or process grouping if needed.
- Run:
```bash
python3 background_estimation/data_mc.py group_hists_out.root
```
- Input: One grouped histogram ROOT file.
- Output: PDF plots in `pre-selection/`.

## `selections/b_veto/b_veto.C`

- Function: Build AK4 b-veto histograms from the selected samples.
- Edit: `SAMPLES`, `MAX_FILES_PER_SAMPLE`, and `OUTFILE` if needed.
- Run:
```bash
root -l -q 'selections/b_veto/b_veto.C'
```
- Input: Sample files collected from DAS.
- Output: `b_veto_hists.root`.

## `selections/b_veto/ttbar_score.py`

- Function: Derive b-veto working points, efficiencies, and ROC plots.
- Edit: `ROOT_FILE`, `OUT_DIR`, and `TARGET_LIGHT_EFFS` if needed.
- Run:
```bash
python3 selections/b_veto/ttbar_score.py
```
- Input: `b_veto_hists.root`.
- Output: CSV tables and PDF plots in `b_veto/`.

## `systematics/efficiency.C`

- Function: Measure `DST_PFScouting_JetHT` efficiency and related distributions.
- Edit: `TYPE`, `N_DATA_FILES`, `BASEDIR`, and sample settings if needed.
- Run:
```bash
root -l -q 'systematics/efficiency.C("2024F", 200)'
root -l -q 'systematics/efficiency.C("vvv", 0)'
```
- Input: MC samples or Run 2024 data.
- Output: `efficiency_<type>/` with one ROOT file and multiple PDFs.
