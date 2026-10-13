#!/usr/bin/env python3
"""score_kfold.py -- score every MC event of a k-fold tree with the model of its block.

    python3 score_kfold.py <bdt_tree_root> [sample ...] [--branches a,b,...] [--step-size 500000]

<bdt_tree_root> is the output directory of a k-fold training (fold0 ... fold<K-1>, train.py with
kfold = K). Every entry of every sample (default: all samples of the class_groups; the files and
block boundaries stored in the folds' test_ranges.json) is read in chunks; the events passing the
tree's thresholds except the msoftdrop windows are scored, each with the model of the rotation
whose test block holds it, and written to <bdt_tree_root>/kfold_scores/<sample>.npz:

  entry_index  global entry index of the event in the sample's concatenated entries
  fold         the rotation that scored it
  class_idx    class index of the sample
  proba        class probabilities (N x classes)
  weight       pre-lumi physics weight with every entry of the sample read: xsec x raw_w / S
               (raw_w = the tree's event_reweight_branches, genWeight x weight_pu; S = its sum
               over every processed generated event, sum_genweight_pu; mc_weights.py)
  b_<branch>   every threshold and decorrelate branch, plus --branches
  n_entries, sample, tree, branches (scalars / name list), model_fingerprint (of the K
               model files; kfold.load_scores refuses a cache written with other models)

The probabilities of every fold are checked against the rows of the fold's
test_reference_qcd_est_full.npz (matched by entry index). The SR scans and qcd_est.py read this
cache for k-fold models (all MC events, each with an unbiased score).
"""
import argparse
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import kfold  # noqa: E402


def log(msg):
    print(msg, flush=True)


def _sample_info(models):
    path = models.config["sample_config"]
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(_SCRIPT_DIR, path))
    return {s["name"]: s for s in kfold._load_json(path)["sample"]}


def _validate(models, sample, result):
    """Compare the probabilities with every fold's full test reference (rows of this sample that
    pass the thresholds, matched by entry index)."""
    for f, fold_dir in enumerate(models.fold_dirs):
        path = os.path.join(fold_dir, "test_reference_qcd_est_full.npz")
        ref = np.load(path, allow_pickle=False)
        if "entry_index" not in ref:
            raise RuntimeError(f"{path} has no entry_index: retrain with the current train.py")
        rows = np.flatnonzero(ref["sample_name"].astype(str) == sample)
        ref_idx = ref["entry_index"][rows]
        sel = np.flatnonzero(result["fold"] == f)
        cur_idx = result["entry_index"][sel]
        order = np.argsort(ref_idx, kind="stable")
        sorted_idx = ref_idx[order]
        pos = np.searchsorted(sorted_idx, cur_idx)
        inside = pos < len(sorted_idx)
        hit = np.zeros(len(cur_idx), dtype=bool)
        hit[inside] = sorted_idx[pos[inside]] == cur_idx[inside]
        if not np.any(hit):
            log(f"    fold {f}: no reference rows of {sample} pass the thresholds (reference has {len(rows)} rows)")
            continue
        ref_rows = rows[order][pos[hit]]
        cur_rows = sel[hit]
        if not np.array_equal(ref["class_idx"][ref_rows], result["class_idx"][cur_rows]):
            raise RuntimeError(f"fold {f}: class labels of {sample} differ from the reference")
        ref_proba = ref["proba"][ref_rows]
        cur_proba = result["proba"][cur_rows]
        rtol, atol = float(ref["proba_rtol"]), float(ref["proba_atol"])
        if not np.allclose(cur_proba, ref_proba, rtol=rtol, atol=atol):
            diff = float(np.max(np.abs(cur_proba - ref_proba)))
            raise RuntimeError(
                f"fold {f}: probabilities of {sample} differ from the reference by up to {diff:.3g} "
                f"(stored tolerances rtol={rtol:.3g}, atol={atol:.3g})"
            )
        log(f"    fold {f}: {len(cur_rows)} rows match the reference")


def score_sample(models, sample, info, extra_branches, step_size, out_dir):
    t0 = time.time()
    stored = sorted(set(models.thresholds) | set(models.decorrelate) | set(extra_branches))
    pre_thresholds = {k: v for k, v in models.thresholds.items() if not kfold.is_mass_threshold(k)}
    result = models.read_sample(sample, info, stored, pre_thresholds, step_size)
    class_idx = next(i for i, c in enumerate(models.class_names) if sample in models.config["class_groups"][c])
    result["class_idx"] = np.full(len(result["entry_index"]), class_idx, dtype=np.int16)
    _validate(models, sample, result)
    columns = result.pop("columns")
    n_tree = sum(e for _, e in models.files(sample))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{sample}.npz")
    tmp = os.path.join(out_dir, f".{sample}.tmp.npz")
    np.savez(tmp, n_entries=np.int64(n_tree), sample=np.asarray(sample), tree=np.asarray(models.tree),
             branches=np.asarray(stored), model_fingerprint=np.asarray(models.fingerprint()),
             **result, **{f"b_{b}": v for b, v in columns.items()})
    os.replace(tmp, out)
    log(f"  {sample}: {n_tree} entries, {len(result['entry_index'])} pass the non-mass thresholds, "
        f"weight sum {result['weight'].sum():.6g} (pre-lumi), {time.time() - t0:.0f} s -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bdt_root")
    parser.add_argument("samples", nargs="*")
    parser.add_argument("--branches", default="", help="extra branches to store, comma-separated")
    parser.add_argument("--step-size", type=int, default=500000)
    args = parser.parse_args()
    models = kfold.KFoldModels(os.path.abspath(args.bdt_root), log=log)
    infos = _sample_info(models)
    samples = args.samples or [s for members in models.config["class_groups"].values() for s in members]
    extra = [b for b in args.branches.split(",") if b]
    log(f"score_kfold.py: tree={models.tree}, k={models.k}, samples={len(samples)}, extra branches={extra}")
    for sample in samples:
        score_sample(models, sample, infos[sample], extra, args.step_size, os.path.join(models.root, "kfold_scores"))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log(f"Runtime error: {ex}")
        raise
