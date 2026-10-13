"""Shared helpers for k-fold models (train.py with kfold = K).

A k-fold tree output directory holds fold0 ... fold<K-1>, one model per rotation. Every sample's
concatenated entries (the file list stored in the folds' test_ranges.json) are cut into K blocks;
the model of rotation r neither trained nor monitored on block r, so an MC event at global entry
index i is scored with the model of the block holding i (fold_of_index). Data events, never
trained on, are assigned a fold by event number (data_fold). Used by score_kfold.py and by the
scripts that score MC events themselves (systematics, data/MC plots).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot

import model_io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mc_weight_common"))
import mc_weights  # noqa: E402

_BDT_DIR = os.path.dirname(os.path.abspath(__file__))


def is_kfold_root(bdt_root):
    """True if bdt_root is the output directory of a k-fold training (fold0/test_ranges.json)."""
    path = os.path.join(bdt_root, "fold0", "test_ranges.json")
    return os.path.exists(path) and "kfold" in json.load(open(path, encoding="utf-8"))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class KFoldModels:
    """The K rotations of a k-fold training: configs (identical in every fold), per-sample
    file lists and block boundaries, and the models (loaded on first use)."""

    def __init__(self, bdt_root, log=print):
        self.root = bdt_root
        self.log = log
        meta0 = _load_json(os.path.join(bdt_root, "fold0", "test_ranges.json"))
        if "kfold" not in meta0:
            raise RuntimeError(f"{bdt_root}/fold0 is not a k-fold rotation (no kfold in test_ranges.json)")
        self.k = int(meta0["kfold"])
        self.tree = meta0["tree_name"]
        self.fold_dirs = [os.path.join(bdt_root, f"fold{f}") for f in range(self.k)]
        self.config = _load_json(os.path.join(self.fold_dirs[0], "config.json"))
        self.branch_cfg = _load_json(os.path.join(self.fold_dirs[0], "branch.json"))
        self.selection = _load_json(os.path.join(self.fold_dirs[0], "selection.json"))
        self.samples = meta0["samples"]
        for f, d in enumerate(self.fold_dirs):
            meta = _load_json(os.path.join(d, "test_ranges.json"))
            if int(meta.get("fold", -1)) != f or int(meta.get("kfold", 0)) != self.k:
                raise RuntimeError(f"{d}/test_ranges.json is not rotation {f} of {self.k}")
            for name in ("config.json", "branch.json", "selection.json"):
                if f and _load_json(os.path.join(d, name)) != _load_json(os.path.join(self.fold_dirs[0], name)):
                    raise RuntimeError(f"{d}/{name} differs from fold0: the rotations must share one configuration")
            for sample, entry in meta["samples"].items():
                ref = self.samples.get(sample)
                if ref is None or ref["files"] != entry["files"] or ref["block_bounds"] != entry["block_bounds"]:
                    raise RuntimeError(f"{d}: file list / blocks of sample '{sample}' differ from fold0")
        self.class_names = list(self.config["class_groups"].keys())
        self.num_classes = len(self.class_names)
        self.features = [b["name"] for b in self.branch_cfg[self.tree]]
        sel = self.selection[self.tree]
        self.clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
        self.log_transform = list(sel.get("log_transform", []))
        self.thresholds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sel.get("thresholds", {}).items()}
        self.decorrelate = list(self.config.get(self.tree, {}).get("decorrelate", []))
        self._models = [None] * self.k

    # ---- event -> fold ----
    def current_files(self, sample, info):
        """The sample's files as train.py discovers them now (sorted base + split files)."""
        input_root = os.path.normpath(os.path.join(_BDT_DIR, self.config["input_root"]))
        group = "signal" if info["is_signal"] else "bkg"
        base = self.config["input_pattern"].format(input_root=input_root, sample_group=group, sample=sample)
        return sorted(glob.glob(base) + glob.glob(base[:-5] + "_*.root"))

    def files(self, sample):
        """[(path, entries)] of the sample in the stored (training) order."""
        return [(f["path"], int(f["entries"])) for f in self.samples[sample]["files"]]

    def file_offsets(self, sample):
        """{path: global index of the file's first entry}."""
        out, offset = {}, 0
        for path, entries in self.files(sample):
            out[path] = offset
            offset += entries
        return out

    def check_files(self, sample, current_files, tree_entries=None):
        """Error if the sample's current files differ from the ones the folds were trained on."""
        if sample not in self.samples:
            raise RuntimeError(
                f"sample '{sample}' is not part of the k-fold training of '{self.tree}' (not in its "
                "class_groups); leave it out of submit_samples"
            )
        stored = [p for p, _ in self.files(sample)]
        if list(current_files) != stored:
            raise RuntimeError(
                f"k-fold models of '{self.tree}' were trained on other files of sample '{sample}' "
                f"(stored {len(stored)}, now {len(current_files)}); retrain after a reconversion"
            )
        if tree_entries is not None and list(tree_entries) != [e for _, e in self.files(sample)]:
            raise RuntimeError(f"entry counts of sample '{sample}' changed since the k-fold training")

    def fold_of_index(self, sample, global_index):
        bounds = np.asarray(self.samples[sample]["block_bounds"], dtype=np.int64)
        return (np.searchsorted(bounds, np.asarray(global_index, dtype=np.int64), side="right") - 1).astype(np.int64)

    def data_fold(self, event):
        """Fold of data events (never trained on): event number modulo K."""
        return (np.asarray(event, dtype=np.int64) % self.k).astype(np.int64)

    # ---- models ----
    def model_base(self, fold):
        pattern = self.config.get("model_pattern", "{output_root}/{tree_name}_model")
        return pattern.format(output_root=self.fold_dirs[fold], tree_name=self.tree)

    def fingerprint(self):
        """SHA-256 (16 hex digits) of the K final model files, recorded in the score cache."""
        h = hashlib.sha256()
        for fold in range(self.k):
            files = sorted(glob.glob(self.model_base(fold) + ".*"))
            if not files:
                raise FileNotFoundError(f"no model file {self.model_base(fold)}.*")
            for path in files:
                with open(path, "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:16]

    def model(self, fold):
        if self._models[fold] is None:
            self._models[fold] = model_io.load_model(self.model_base(fold), self.config, self.num_classes,
                                                     log_message=self.log)
        return self._models[fold]

    def model_inputs(self, df):
        """Model input frame of the rows of df: branch.json features, clip/log standardised,
        decorrelated features dropped (as in train.py and qcd_est.py)."""
        X = standardize_X(df[self.features].copy(), self.clip_ranges, self.log_transform)
        drop = [c for c in self.decorrelate if c in X.columns]
        return X.drop(columns=drop) if drop else X

    def read_sample(self, sample, info, branches, thresholds, step_size=500000):
        """Every entry of a trained sample, read in chunks: the events passing `thresholds`
        (qcd_est.py semantics) with their out-of-fold class probabilities and pre-lumi physics weight
        with every entry read, xsec x raw_w / S (selections/mc_weight_common/mc_weights.py; raw_w =
        the tree's event_reweight_branches, genWeight x weight_pu, S = its sum over every processed
        generated event, `sum_genweight_pu`). `info` is the sample's src/sample.json entry. Returns
        {entry_index, fold, proba, weight, columns: {branch: array of `branches`}}."""
        self.check_files(sample, self.current_files(sample, info))
        reweight = list(self.config.get(self.tree, {}).get("event_reweight_branches", []))
        read = sorted(set(self.features) | set(branches) | set(thresholds) | set(reweight))
        n_tree = sum(e for _, e in self.files(sample))
        xsec = float(info["xsection"])
        # Every entry is read, so N_tree / N_read = 1.
        scale = xsec / mc_weights.generated_weight_sum(info, reweight) if xsec > 0.0 else 0.0
        parts = {k: [] for k in ("entry_index", "fold", "proba", "weight")}
        cols = {b: [] for b in branches}
        offsets = self.file_offsets(sample)
        n_read = 0
        for path, entries in self.files(sample):
            with uproot.open(path) as uf:
                tree = uf[self.tree]
                if tree.num_entries != entries:
                    raise RuntimeError(f"{path}:{self.tree} has {tree.num_entries} entries, the k-fold training saw {entries}")
                for arrays, report in tree.iterate(read, step_size=step_size, library="np", report=True):
                    n_read += len(arrays[read[0]])
                    mask = threshold_mask(arrays, thresholds)
                    if not mask.any():
                        continue
                    idx = offsets[path] + report.tree_entry_start + np.flatnonzero(mask)
                    folds = self.fold_of_index(sample, idx)
                    df = pd.DataFrame({b: arrays[b][mask] for b in self.features})
                    raw_w = mc_weights.event_weight_product({b: arrays[b][mask] for b in reweight}, reweight)
                    parts["entry_index"].append(idx)
                    parts["fold"].append(folds.astype(np.int8))
                    parts["proba"].append(self.predict(self.model_inputs(df), folds))
                    parts["weight"].append(raw_w * scale)
                    for b in branches:
                        cols[b].append(arrays[b][mask])
        if n_read != n_tree:
            raise RuntimeError(f"{sample}: read {n_read} entries, expected {n_tree}")
        empty = {"proba": np.zeros((0, self.num_classes)), "fold": np.zeros(0, np.int8), "entry_index": np.zeros(0, np.int64)}
        out = {k: (np.concatenate(v) if v else empty.get(k, np.zeros(0))) for k, v in parts.items()}
        out["columns"] = {b: (np.concatenate(v) if v else np.zeros(0)) for b, v in cols.items()}
        return out

    def predict(self, X, folds):
        """Class probabilities of the rows of X, each row with the model of its fold."""
        folds = np.asarray(folds, dtype=np.int64)
        out = np.empty((len(X), self.num_classes), dtype=np.float64)
        for f in np.unique(folds):
            rows = np.flatnonzero(folds == f)
            out[rows] = model_io.predict_model_proba(self.model(int(f)), X.iloc[rows], self.num_classes)
        return out


def load_scores(models, samples, branches=(), thresholds=None, log=print):
    """Rows of the score cache (score_kfold.py, <bdt_root>/kfold_scores/<sample>.npz) of the given
    samples that pass `thresholds` (qcd_est.py semantics; the cache holds the events passing the
    non-mass thresholds). Returns a dict with proba, weight (pre-lumi physics weight), class_idx,
    sample_name, entry_index, fold and columns ({branch: array} of `branches`)."""
    fingerprint = models.fingerprint()
    needed = sorted(set(branches) | set(thresholds or {}))
    parts = {k: [] for k in ("proba", "weight", "class_idx", "sample_name", "entry_index", "fold")}
    cols = {b: [] for b in branches}
    for sample in samples:
        path = os.path.join(models.root, "kfold_scores", f"{sample}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run selections/BDT/score_kfold.py {models.root} {sample}")
        z = np.load(path, allow_pickle=False)
        if str(z["model_fingerprint"]) != fingerprint:
            raise RuntimeError(f"{path} was scored with other models: rerun score_kfold.py for {sample}")
        missing = [b for b in needed if f"b_{b}" not in z]
        if missing:
            raise KeyError(f"{path} lacks the branches {missing}: rerun score_kfold.py with --branches")
        mask = (threshold_mask({b: z[f"b_{b}"] for b in thresholds}, thresholds)
                if thresholds else np.ones(len(z["weight"]), dtype=bool))
        for k in ("proba", "weight", "class_idx", "entry_index", "fold"):
            parts[k].append(z[k][mask])
        parts["sample_name"].append(np.full(int(mask.sum()), sample, dtype=object))
        for b in branches:
            cols[b].append(z[f"b_{b}"][mask])
        log(f"  {sample}: {int(mask.sum())} of {len(mask)} cached events (n_entries={int(z['n_entries'])})")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["columns"] = {b: np.concatenate(v) for b, v in cols.items()}
    return out


# ---- preprocessing shared with qcd_est.py (same semantics) ----
def standardize_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        valid = ~(arr < -990)
        if not valid.any():
            continue
        lo, hi = clip_ranges.get(col, (None, None))
        if lo is not None:
            arr[valid & (arr < lo)] = lo
        if hi is not None:
            arr[valid & (arr > hi)] = hi
        if col in log_set:
            pos = valid & (arr > 0)
            if pos.any():
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                arr[pos] = np.log(arr[pos])
        X[col] = arr
    return X


def mask_from_cond(values, cond):
    """qcd_est.py's _mask_from_cond on a numpy array: a number is a lower cut, [low, high] a
    window (None = open, exclusive bounds), a list of conditions their OR, and a
    {"&"/"and"/"|"/"or": [...]} dict their AND/OR."""
    if cond is None:
        return np.ones(values.shape[0], dtype=bool)
    if isinstance(cond, (int, float, np.integer, np.floating)):
        return values > float(cond)
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
        low, high = cond
        mask = np.ones(values.shape[0], dtype=bool)
        if low is not None:
            mask &= values > low
        if high is not None:
            mask &= values < high
        return mask
    if isinstance(cond, (list, tuple)):
        mask = np.zeros(values.shape[0], dtype=bool)
        for item in cond:
            mask |= mask_from_cond(values, item)
        return mask
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key in cond:
                mask = np.full(values.shape[0], is_and, dtype=bool)
                for item in cond[key]:
                    sub = mask_from_cond(values, item)
                    mask = (mask & sub) if is_and else (mask | sub)
                return mask
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported condition type: {type(cond)}")


def threshold_mask(columns, thresholds):
    """Rows passing every threshold, qcd_est.py semantics (a sentinel value < -990 fails).
    columns: mapping branch -> array (dict of numpy arrays or a DataFrame)."""
    n = len(next(iter(columns.values()))) if isinstance(columns, dict) else len(columns)
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        values = np.asarray(columns[branch], dtype=float)
        mask &= ~(values < -990)
        mask &= mask_from_cond(values, cond)
    return mask


def is_mass_threshold(name):
    """train.py's split of the thresholds: msoftdrop windows (applied downstream) vs the rest."""
    return str(name).startswith("ScoutingFatPFJetRecluster_msoftdrop_")
