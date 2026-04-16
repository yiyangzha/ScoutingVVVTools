import os
import glob
import json
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import xgboost as xgb
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier, plot_importance
from typing import Optional, Sequence, List, Union

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)

_EPS = 1e-12

# ── Config loading ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

cfg     = _load_json(os.path.join(_SCRIPT_DIR, "config.json"))
br_cfg  = _load_json(os.path.join(_SCRIPT_DIR, "branch.json"))
sel_cfg = _load_json(os.path.join(_SCRIPT_DIR, "selection.json"))

_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

# ── Constants from config ──────────────────────────────────────────────────────
RANDOM_STATE       = cfg.get("random_state", 42)
ENTRIES_PER_SAMPLE = cfg.get("entries_per_sample", 1_000_000)
DECOR_LAMBDA       = cfg.get("decor_lambda", 30)
SUBMIT_TREES       = cfg.get("submit_trees", ["fat2"])
INPUT_ROOT         = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN      = cfg["input_pattern"]
OUTPUT_ROOT        = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg.get("output_root", ".")))
MODEL_PATTERN      = cfg.get("model_pattern", "{output_root}/{tree_name}_model")

# ── Sample registry from sample.json ──────────────────────────────────────────
SAMPLE_INFO = {}
for _rule in sample_cfg["sample"]:
    SAMPLE_INFO[_rule["name"]] = {
        "xsection":    _rule["xsection"],
        "raw_entries": _rule.get("raw_entries", 100),
        "is_MC":       _rule["is_MC"],
        "is_signal":   _rule["is_signal"],
        "sample_ID":   _rule["sample_ID"],
    }

# ── Class groupings from config ────────────────────────────────────────────────
CLASS_GROUPS  = cfg["class_groups"]            # {"VVV": [...], "VH": [...], ...}
CLASS_NAMES   = list(CLASS_GROUPS.keys())      # ordered class name list
NUM_CLASSES   = len(CLASS_NAMES)

SAMPLE_TO_CLASS = {}
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    for _s in _members:
        SAMPLE_TO_CLASS[_s] = _idx

# Resolve training sample list
_submit_cfg = cfg.get("submit_samples", [])
if _submit_cfg:
    TRAINING_SAMPLES = [s for s in _submit_cfg if s in SAMPLE_TO_CLASS]
else:
    TRAINING_SAMPLES = [r["name"] for r in sample_cfg["sample"] if r["name"] in SAMPLE_TO_CLASS]


# ── File discovery ─────────────────────────────────────────────────────────────
def _sample_group(sample_name):
    return "signal" if SAMPLE_INFO[sample_name]["is_signal"] else "bkg"

def _input_files(sample_name):
    sg   = _sample_group(sample_name)
    base = INPUT_PATTERN.format(input_root=INPUT_ROOT, sample_group=sg, sample=sample_name)
    stem = base[:-5]  # strip .root
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


# ── Data loading ───────────────────────────────────────────────────────────────
def prepare_data(tree_name, branches):
    """Load training data for one tree (fat2 or fat3).

    Weight formula: per_evt_w = xsection * raw_entries / n_read
    Each class is then rescaled so its total weight sums to 1e10.
    """
    dfs = []

    for sample_name in TRAINING_SAMPLES:
        info  = SAMPLE_INFO[sample_name]
        files = _input_files(sample_name)
        if not files:
            print(f"  [WARN] no files for '{sample_name}', skipping")
            continue

        limit = ENTRIES_PER_SAMPLE
        parts = []
        n_read = 0
        for fpath in files:
            remain = limit - n_read
            if remain <= 0:
                break
            with uproot.open(fpath) as uf:
                if tree_name not in uf:
                    continue
                tree = uf[tree_name]
                n_to_read = min(remain, tree.num_entries)
                if n_to_read <= 0:
                    continue
                df_part = tree.arrays(branches, library="pd",
                                      entry_start=0, entry_stop=n_to_read)
                parts.append(df_part)
                n_read += len(df_part)

        if n_read == 0:
            print(f"  [WARN] zero entries read for '{sample_name}' in tree '{tree_name}', skipping")
            continue

        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()

        per_evt_w = (info["xsection"] * info["raw_entries"]) / float(n_read)
        df["weight"]    = per_evt_w
        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]

        dfs.append(df)
        print(f"  {sample_name}: {n_read} entries, w={per_evt_w:.4g}, "
              f"class={CLASS_NAMES[SAMPLE_TO_CLASS[sample_name]]}")

    if not dfs:
        raise RuntimeError(f"No data loaded for tree '{tree_name}'")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Per-class weight normalisation: scale each class total to 1e10
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask  = df_all["class_idx"] == cls_idx
        w_sum = df_all.loc[mask, "weight"].sum()
        if w_sum > 0:
            scale = 1e10 / w_sum
            df_all.loc[mask, "weight"] *= scale
            print(f"  [{cls_name}] total_w={w_sum:.4g}, scale={scale:.4g}")

    df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df_all[branches].copy()
    y = df_all["class_idx"].values.astype(int)
    w = df_all["weight"].values

    del df_all
    gc.collect()
    return X, y, w


# ── Event filtering ────────────────────────────────────────────────────────────
def filter_X(X: pd.DataFrame, y, w, branch: list,
             thresholds: dict = None, apply_to_sentinel: bool = True):
    """Apply per-branch threshold cuts; sentinel values (< -990) handled separately."""
    if thresholds is None:
        return X.copy(), y.copy(), w.copy()

    mask = pd.Series(True, index=X.index)

    def _combine(masks, op, idx):
        if not masks:
            return pd.Series(op == "&", index=idx)
        out = masks[0]
        for m in masks[1:]:
            out = (out & m) if op == "&" else (out | m)
        return out

    def _mask_from_cond(col, cond):
        idx = col.index
        if cond is None:
            return pd.Series(True, index=idx)
        if isinstance(cond, (int, float, np.integer, np.floating)):
            return col > float(cond)
        if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
            mn, mx = cond
            m = pd.Series(True, index=idx)
            if mn is not None:
                m &= col > mn
            if mx is not None:
                m &= col < mx
            return m
        if isinstance(cond, (list, tuple)):
            return _combine([_mask_from_cond(col, c) for c in cond], "|", idx)
        if isinstance(cond, dict):
            for op_key, op_sym in (("&", "&"), ("and", "&"), ("|", "|"), ("or", "|")):
                if op_key in cond:
                    return _combine([_mask_from_cond(col, c) for c in cond[op_key]], op_sym, idx)
            raise ValueError(f"Unsupported dict condition keys: {cond}")
        raise TypeError(f"Unsupported condition type: {type(cond)}")

    for b in branch:
        if b not in X.columns:
            raise KeyError(f"Column {b!r} not found in X")
        col      = X[b]
        cond     = thresholds.get(b, None)
        sentinel = col < -990

        if apply_to_sentinel:
            mask &= ~sentinel
            if cond is not None:
                mask &= _mask_from_cond(col, cond)
        else:
            if cond is not None:
                mask &= (_mask_from_cond(col, cond) | sentinel)

    return X.loc[mask].copy(), y[mask.values].copy(), w[mask.values].copy()


# ── Feature standardisation ────────────────────────────────────────────────────
def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    """Clip values and apply log transform in-place; sentinel values (< -990) are untouched."""
    log_set = set(log_transform)
    for col in X.columns:
        arr  = X[col].values.copy()
        mask = arr < -990   # sentinel
        valid = ~mask
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


# ── CvM flatness loss helpers ──────────────────────────────────────────────────
def _weighted_ecdf_positions(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0:
        return np.zeros_like(y)
    order    = np.argsort(y)
    w_sorted = w[order].astype(float)
    W        = float(np.sum(w_sorted)) + _EPS
    w_sorted /= W
    cum      = np.cumsum(w_sorted) - 0.5 * w_sorted
    pos      = np.empty_like(cum)
    pos[order] = cum
    return pos


def _cvm_flatness_value(y, Z, w, n_bins=10, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0 or Z.size == 0:
        return 0.0
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    _, n_decor = Z.shape
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return 0.0
    w_norm      = w / W_abs
    global_pos  = _weighted_ecdf_positions(y, w_norm)
    flat_penalty = 0.0
    for j in range(n_decor):
        zj    = Z[:, j]
        z_min = float(np.min(zj))
        z_max = float(np.max(zj))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            continue
        edges   = np.linspace(z_min, z_max, n_bins + 1)
        bin_idx = np.clip(np.searchsorted(edges, zj, side="right") - 1, 0, n_bins - 1)
        for b in range(n_bins):
            idx_b = np.nonzero(bin_idx == b)[0]
            if idx_b.size < 3:
                continue
            local_pos = _weighted_ecdf_positions(y[idx_b], w_norm[idx_b])
            diff      = local_pos - global_pos[idx_b]
            flat_penalty += float(np.sum(w_norm[idx_b] * (np.abs(diff) ** power)))
    return float(flat_penalty)


def _cvm_flatness_neg_grad_wrt_y(y, groups, w, power=2.0):
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0 or not groups:
        return np.zeros_like(y)
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return np.zeros_like(y)
    w_norm     = w / W_abs
    global_pos = _weighted_ecdf_positions(y, w_norm)
    neg_grad   = np.zeros_like(y)
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        if idx.size < 2:
            continue
        local_pos = _weighted_ecdf_positions(y[idx], w_norm[idx])
        diff      = local_pos - global_pos[idx]
        bin_grad  = power * np.sign(diff) * (np.abs(diff) ** (power - 1.0))
        neg_grad[idx] += bin_grad
    neg_grad *= w_norm
    return neg_grad


# ── Diagnostics ───────────────────────────────────────────────────────────────
def check_weights(w, name="w"):
    w = np.asarray(w, dtype=float).ravel()
    finite = np.isfinite(w)
    if not np.all(finite):
        bad = np.where(~finite)[0]
        print(f"[{name}] non-finite count: {bad.size}. e.g. indices: {bad[:10].tolist()}")
    else:
        print(f"[{name}] all finite")
    n     = w.size
    n_pos = int(np.sum(w > 0))
    n_neg = int(np.sum(w < 0))
    print(f"[{name}] N={n}, >0:{n_pos}, <0:{n_neg}, "
          f"sum={np.nansum(w):.4g}, min={np.nanmin(w):.4g}, max={np.nanmax(w):.4g}")


# ── Decorrelation helpers ──────────────────────────────────────────────────────
def _resolve_decor_indices(X, decorrelate_feature_names):
    if not decorrelate_feature_names:
        return []
    if isinstance(X, pd.DataFrame):
        name_to_idx = {c: i for i, c in enumerate(X.columns)}
        idx = []
        for key in decorrelate_feature_names:
            if isinstance(key, int):
                idx.append(key)
            else:
                if key not in name_to_idx:
                    raise ValueError(f"Decorrelation feature '{key}' not in DataFrame columns.")
                idx.append(name_to_idx[key])
        return sorted(set(idx))
    idx = []
    for key in decorrelate_feature_names:
        if isinstance(key, int):
            idx.append(key)
        else:
            raise ValueError("X is not a DataFrame; pass integer column indices for decorrelation.")
    return sorted(set(idx))


# ── Custom multiclass objective with CvM decorrelation ────────────────────────
def _make_multiclass_objective_with_decor(num_class, Z_train, w_train, lam):
    Z_train = np.asarray(Z_train, dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    n_train = w_train.shape[0]
    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
    n_samples, n_decor = Z_train.shape

    N_DECOR_BINS = 5
    decor_groups_list: List = []
    if lam > 0.0 and n_decor > 0:
        for j in range(n_decor):
            zj    = Z_train[:, j]
            z_min, z_max = float(np.min(zj)), float(np.max(zj))
            if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
                decor_groups_list.append(None)
                continue
            edges   = np.linspace(z_min, z_max, N_DECOR_BINS + 1)
            bin_idx = np.clip(np.searchsorted(edges, zj, side="right") - 1, 0, N_DECOR_BINS - 1)
            groups_j = [np.nonzero(bin_idx == b)[0] for b in range(N_DECOR_BINS)
                        if np.nonzero(bin_idx == b)[0].size >= 3]
            decor_groups_list.append(groups_j if groups_j else None)
    else:
        decor_groups_list = [None] * n_decor

    _scale       = float(n_train) / float(num_class) if n_train > 0 else 1.0
    _base_lr     = 0.05
    _iter_state  = {"t": 0}

    def obj(y_true, y_pred_in):
        t = int(_iter_state["t"])
        _iter_state["t"] = t + 1
        if t < 1000:
            lr_t = 0.01
        elif t < 1200:
            lr_t = 0.01 - (0.01 - 0.005) * (t - 1000) / 200
        else:
            lr_t = 0.005
        lr_mult = float(lr_t) / float(_base_lr)

        y_true    = np.asarray(y_true, dtype=int)
        n         = y_true.shape[0]
        y_pred_in = np.asarray(y_pred_in, dtype=float)
        if y_pred_in.ndim == 1:
            y_pred = y_pred_in.reshape(n, num_class)
        elif y_pred_in.shape == (num_class, n):
            y_pred = y_pred_in.T
        else:
            y_pred = y_pred_in.reshape(n, num_class)

        y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_y   = np.exp(y_shift)
        P       = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        w_used      = w_train
        class_w_sum = np.bincount(y_true, weights=w_used, minlength=num_class).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS
        w_eff       = (w_used / class_w_sum[y_true]) * _scale

        grad_cls = P.copy()
        grad_cls[np.arange(n), y_true] -= 1.0
        grad_cls *= w_eff[:, None]
        hess_cls  = (P * (1.0 - P)) * w_eff[:, None]

        grad_dec = np.zeros_like(grad_cls)
        if lam > 0.0 and n_decor > 0:
            for k in range(num_class):
                mask_k  = y_true == k
                if not np.any(mask_k):
                    continue
                yk      = y_pred[:, k]
                w_cls_k = np.zeros_like(w_used)
                w_cls_k[mask_k] = w_used[mask_k]
                gk = np.zeros(n, dtype=float)
                for j in range(n_decor):
                    groups_j = decor_groups_list[j]
                    if groups_j is None:
                        continue
                    gk += -_cvm_flatness_neg_grad_wrt_y(yk, groups_j, w_cls_k, power=2.0)
                grad_dec[:, k] = (lam * _scale) * gk

        grad = (grad_cls + grad_dec) * lr_mult
        hess = hess_cls + 1e-6
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


def _make_multiclass_total_loss_metric(num_class, Z_train, Z_test, w_train, w_test, lam):
    Z_train = np.asarray(Z_train, dtype=float)
    Z_test  = np.asarray(Z_test,  dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    w_test  = np.asarray(w_test,  dtype=float).ravel()
    n_train = w_train.shape[0]
    n_test  = w_test.shape[0]
    N_DECOR_BINS = 5

    def feval(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int)
        n      = y_true.shape[0]
        Z      = Z_train[:n] if n == n_train else Z_test[:n]
        w      = w_train[:n] if n == n_train else w_test[:n]

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(n, num_class)
        elif y_pred.shape == (num_class, n):
            y_pred = y_pred.T
        else:
            y_pred = y_pred.reshape(n, num_class)

        row_sum = np.sum(y_pred, axis=1)
        is_prob = (np.all(np.isfinite(y_pred)) and np.all(y_pred >= -1e-6)
                   and np.all(y_pred <= 1.0 + 1e-6)
                   and np.mean(np.abs(row_sum - 1.0)) < 1e-3)
        if is_prob:
            P = np.clip(y_pred, _EPS, 1.0)
            P = P / (np.sum(P, axis=1, keepdims=True) + _EPS)
        else:
            y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_y   = np.exp(y_shift)
            P       = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        class_w_sum = np.bincount(y_true, weights=w, minlength=num_class).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS
        w_eff  = w / class_w_sum[y_true]
        p_true = P[np.arange(n), y_true]
        ell    = -np.log(p_true + _EPS)
        logloss = float(np.sum(w_eff * ell))

        flat_penalty = 0.0
        if lam > 0.0 and Z.size > 0:
            for k in range(num_class):
                mask_k  = y_true == k
                if not np.any(mask_k):
                    continue
                w_cls_k = np.zeros_like(w)
                w_cls_k[mask_k] = w[mask_k]
                yk = y_pred[:, k] if not is_prob else np.log(P[:, k] + _EPS)
                flat_penalty += _cvm_flatness_value(yk, Z, w_cls_k,
                                                    n_bins=N_DECOR_BINS, power=2.0)
            flat_penalty *= lam

        return float(logloss + flat_penalty)

    return feval


# ── Training ───────────────────────────────────────────────────────────────────
def train_multi_model(X, y, w, model_name, tree_name,
                      decorrelate_feature_names=None):
    """Train a multiclass BDT with optional CvM decorrelation.

    Hyperparameters are read from config.json under the key matching tree_name.
    Returns clf and the full (pre-split) feature splits for plotting.
    """
    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)

    X_train_all, X_test_all, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    decor_idx = _resolve_decor_indices(X_train_all, decorrelate_feature_names)
    if decor_idx:
        all_idx  = np.arange(X_train_all.shape[1] if isinstance(X_train_all, np.ndarray)
                             else len(X_train_all.columns))
        keep_idx = np.setdiff1d(all_idx, decor_idx)
        if keep_idx.size == 0:
            raise ValueError("Decorrelation columns cover all features; nothing left to train on.")

        def _slice(Xlike, idx):
            return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

        X_train  = _slice(X_train_all, keep_idx)
        X_test   = _slice(X_test_all,  keep_idx)
        Z_train  = _slice(X_train_all, decor_idx)
        Z_test   = _slice(X_test_all,  decor_idx)
        Z_train  = np.asarray(Z_train, dtype=float)
        Z_test   = np.asarray(Z_test,  dtype=float)
    else:
        X_train, X_test = X_train_all, X_test_all
        Z_train = np.zeros((X_train_all.shape[0], 0), dtype=float)
        Z_test  = np.zeros((X_test_all.shape[0],  0), dtype=float)

    print(f"  X_train: {X_train.shape}, Z_train (decor): {Z_train.shape}")

    # Hyperparameters from config
    hp = cfg.get(tree_name, {})
    n_threads = max(1, min(16, os.cpu_count() or 1))
    common_kwargs = dict(
        num_class        = NUM_CLASSES,
        n_estimators     = hp.get("n_estimators",    200),
        max_depth        = hp.get("max_depth",         6),
        learning_rate    = hp.get("learning_rate",   0.1),
        gamma            = hp.get("gamma",             0),
        reg_lambda       = hp.get("reg_lambda",        1),
        reg_alpha        = hp.get("reg_alpha",         0),
        min_child_weight = hp.get("min_child_weight",  1),
        n_jobs           = n_threads,
        random_state     = RANDOM_STATE,
    )
    gpu_kwargs = dict(tree_method="hist", device="cuda")
    cpu_kwargs = dict(tree_method="hist")

    use_decor = Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0

    if use_decor:
        custom_obj    = _make_multiclass_objective_with_decor(
            NUM_CLASSES, Z_train, w_train, DECOR_LAMBDA)
        loss_metric   = _make_multiclass_total_loss_metric(
            NUM_CLASSES, Z_train, Z_test, w_train, w_test, DECOR_LAMBDA)
        extra = dict(objective=custom_obj, eval_metric=loss_metric,
                     early_stopping_rounds=10)
    else:
        extra = dict(objective="multi:softprob", early_stopping_rounds=10)

    try:
        clf = XGBClassifier(**common_kwargs, **gpu_kwargs, **extra)
    except xgb.core.XGBoostError:
        clf = XGBClassifier(**common_kwargs, **cpu_kwargs, **extra)

    fit_kwargs = dict(eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    if not use_decor:
        fit_kwargs["sample_weight"]          = w_train
        fit_kwargs["sample_weight_eval_set"] = [w_train, w_test]
    clf.fit(X_train, y_train, **fit_kwargs)

    if model_name.endswith(".json") or use_decor:
        save_path = model_name if model_name.endswith(".json") else model_name + ".json"
        clf.save_model(save_path)
    else:
        with open(model_name + ".pkl", "wb") as fout:
            pickle.dump(clf, fout)

    return clf, (X_train_all, X_test_all, y_train, y_test, w_train, w_test)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(clf, splits, tree_name, decorrelate_feature_names=None):
    """ROC curves, feature importance, score distributions, loss curve, decorrelation checks."""
    X_train_full, X_test_full, y_train, y_test, w_train, w_test = splits

    full_feature_names = list(X_train_full.columns) if hasattr(X_train_full, "columns") \
        else [f"f{i}" for i in range(X_train_full.shape[1])]

    def _resolve(names_or_idx):
        if not names_or_idx:
            return []
        name_to_idx = {c: i for i, c in enumerate(full_feature_names)}
        out = []
        for key in names_or_idx:
            if isinstance(key, int):
                if 0 <= key < len(full_feature_names):
                    out.append(key)
            else:
                if key in name_to_idx:
                    out.append(name_to_idx[key])
                else:
                    print(f"  [INFO] decor var '{key}' not in feature list, skipping")
        seen, res = set(), []
        for i in out:
            if i not in seen:
                seen.add(i)
                res.append(i)
        return res

    decor_idx_full = _resolve(decorrelate_feature_names)
    all_idx        = np.arange(len(full_feature_names))
    keep_idx       = np.setdiff1d(all_idx, decor_idx_full)

    def _slice(Xlike, idx):
        return Xlike.iloc[:, idx] if hasattr(Xlike, "iloc") else Xlike[:, idx]

    X_train_used = _slice(X_train_full, keep_idx)
    X_test_used  = _slice(X_test_full,  keep_idx)
    feat_names_used = [full_feature_names[i] for i in keep_idx]

    # Sanity check
    if hasattr(clf, "n_features_in_") and clf.n_features_in_ == X_train_full.shape[1]:
        X_train_used, X_test_used = X_train_full, X_test_full
        feat_names_used    = full_feature_names
        decor_idx_full     = []

    n_classes   = NUM_CLASSES
    class_names = CLASS_NAMES
    palette     = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    def _as_array(Xlike):
        return Xlike.to_numpy() if hasattr(Xlike, "to_numpy") else np.asarray(Xlike)

    def _safe_w(wv):
        return np.abs(np.asarray(wv, float).ravel())

    def _weighted_pearson(x, y_arr, wv, eps=1e-12):
        x  = np.asarray(x, float).ravel()
        y_arr = np.asarray(y_arr, float).ravel()
        wv = _safe_w(wv)
        m  = np.isfinite(x) & np.isfinite(y_arr) & np.isfinite(wv)
        if not np.any(m):
            return 0.0
        x, y_arr, wv = x[m], y_arr[m], wv[m]
        sw = wv.sum()
        if sw <= eps:
            return 0.0
        mx = (wv * x).sum() / (sw + eps)
        my = (wv * y_arr).sum() / (sw + eps)
        x0, y0 = x - mx, y_arr - my
        cov = (wv * x0 * y0).sum() / (sw + eps)
        vx  = (wv * x0 * x0).sum() / (sw + eps)
        vy  = (wv * y0 * y0).sum() / (sw + eps)
        return float(cov / (np.sqrt(vx * vy) + eps))

    def _weighted_quantile(x, q, wv, eps=1e-12):
        x  = np.asarray(x, float).ravel()
        wv = _safe_w(wv)
        q  = np.asarray(q, float)
        m  = np.isfinite(x) & np.isfinite(wv)
        if not np.any(m):
            return np.full_like(q, np.nan)
        x, wv = x[m], wv[m]
        sw = wv.sum()
        if sw <= eps:
            return np.quantile(x, q)
        order    = np.argsort(x)
        cw       = np.cumsum(wv[order])
        cw      /= (cw[-1] + eps)
        return np.interp(q, cw, x[order])

    # ── 1) ROC ─────────────────────────────────────────────────────────────────
    def _roc_1v1(Xs, ys, ws, ti, oi):
        probs = clf.predict_proba(Xs)
        mask  = (ys == ti) | (ys == oi)
        if not np.any(mask):
            return None
        y_bin = (ys[mask] == ti).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return None
        pt = probs[mask, ti]
        po = probs[mask, oi]
        sc = pt / np.clip(pt + po, _EPS, None)
        auc = roc_auc_score(y_bin, sc, sample_weight=ws[mask])
        fpr, tpr, _ = roc_curve(y_bin, sc, sample_weight=ws[mask])
        return fpr, tpr, auc

    def _plot_roc(target_idx, target_name):
        opponents = [i for i in range(n_classes) if i != target_idx]
        plt.figure(figsize=(10, 10))
        for color, opp_idx in zip(palette, opponents):
            opp = class_names[opp_idx]
            r_tst = _roc_1v1(X_test_used,  y_test,  w_test,  target_idx, opp_idx)
            r_trn = _roc_1v1(X_train_used, y_train, w_train, target_idx, opp_idx)
            if r_tst:
                fpr, tpr, auc = r_tst
                plt.plot(tpr, fpr, color=color, linestyle="-",
                         label=f"{opp} (Test AUC={auc:.3f})")
                print(f"  {tree_name} Test  AUC ({target_name} vs {opp}) = {auc:.4f}")
            if r_trn:
                fpr, tpr, auc = r_trn
                plt.plot(tpr, fpr, color=color, linestyle="--",
                         label=f"{opp} (Train AUC={auc:.3f})")
                print(f"  {tree_name} Train AUC ({target_name} vs {opp}) = {auc:.4f}")
        plt.xlabel(rf"$\epsilon_{{\rm {target_name}}}$", fontsize=20)
        plt.ylabel(r"$\epsilon_{\rm bkg}$", fontsize=20)
        plt.yscale("log")
        plt.ylim(1e-6, 1)
        plt.xlim(0, 1)
        plt.legend(loc="lower right", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{tree_name}_ROC_{target_name}_vs_all.png")
        plt.close()

    _plot_roc(0, class_names[0])  # VVV (index 0)

    # ── 2) Feature importance ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 20))
    plot_importance(clf, ax=ax, max_num_features=200)
    ax.set_title(f"{tree_name} Feature Importance", fontsize=16)
    ax.set_xscale("log")
    widths = [p.get_width() for p in ax.patches]
    if widths:
        nz = [w for w in widths if w > 0]
        if nz:
            ax.set_xlim(min(nz) / 2, max(widths) * 2)
    try:
        raw_labels = [t.get_text() for t in ax.get_yticklabels()]
        mapped = []
        for s in raw_labels:
            if isinstance(s, str) and s.startswith("f") and s[1:].isdigit():
                i = int(s[1:])
                mapped.append(feat_names_used[i] if i < len(feat_names_used) else s)
            else:
                mapped.append(s)
        if mapped:
            ax.set_yticklabels(mapped)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(f"{tree_name}_Feature_Importance.png")
    plt.close()

    # ── 3) Score distributions ────────────────────────────────────────────────
    probs_train = clf.predict_proba(X_train_used)
    probs_test  = clf.predict_proba(X_test_used)

    def _plot_score_dist(target_idx, target_name, bkg_label):
        p_tr = probs_train[:, target_idx]
        p_te = probs_test[:,  target_idx]
        bins = np.linspace(0, 1, 31)
        plt.figure()
        plt.xlim(0, 1)
        plt.hist(p_tr[y_train != target_idx], bins=bins,
                 weights=w_train[y_train != target_idx],
                 density=True, histtype="bar", alpha=0.5, label=f"Train {bkg_label}")
        plt.hist(p_tr[y_train == target_idx], bins=bins,
                 weights=w_train[y_train == target_idx],
                 density=True, histtype="bar", alpha=0.5, label=f"Train {target_name}")
        plt.hist(p_te[y_test != target_idx],  bins=bins,
                 weights=w_test[y_test != target_idx],
                 density=True, histtype="step", linewidth=2, color="lime",
                 label=f"Test {bkg_label}")
        plt.hist(p_te[y_test == target_idx],  bins=bins,
                 weights=w_test[y_test == target_idx],
                 density=True, histtype="step", linewidth=2, color="red",
                 label=f"Test {target_name}")
        plt.xlabel("BDT Score")
        plt.yscale("log")
        plt.ylim(1e-2,)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"{tree_name}_Score_Dist_{target_name}_vs_all.png")
        plt.close()

    rest_label = "+".join(n for n in class_names if n != class_names[0])
    _plot_score_dist(0, class_names[0], rest_label)
    if n_classes > 1:
        rest_label1 = "+".join(n for n in class_names if n != class_names[1])
        _plot_score_dist(1, class_names[1], rest_label1)

    # ── 4) Loss curve ─────────────────────────────────────────────────────────
    try:
        evals = clf.evals_result()
        v0    = evals.get("validation_0", {})
        loss_key = next((k for k in v0 if "total_loss" in k or "feval" in k or "logloss" in k), None)
        if loss_key and "validation_1" in evals:
            tr_loss = evals["validation_0"][loss_key]
            te_loss = evals["validation_1"][loss_key]
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(tr_loss) + 1), tr_loss, label="Train")
            plt.plot(range(1, len(te_loss) + 1), te_loss, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("total_loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(f"{tree_name}_Total_Loss_Curve.png")
            plt.close()
    except Exception:
        pass

    # ── 5) Feature correlation matrix ─────────────────────────────────────────
    X_tr_df = pd.DataFrame(_as_array(X_train_used), columns=feat_names_used)
    corr    = X_tr_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    plt.figure(figsize=(20, 20))
    plt.imshow(corr.values, aspect="equal", interpolation="none", vmin=-1, vmax=1, cmap="bwr")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.index)),   corr.index,   fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{tree_name}_Feature_Correlation_Matrix.png")
    plt.close()

    # ── 6) Decorrelation checks ───────────────────────────────────────────────
    if not decor_idx_full:
        return

    decor_var_names = [full_feature_names[i] for i in decor_idx_full]

    def _build_corr_matrix(probs, Xfull, wv):
        Xarr = _as_array(Xfull)
        wv   = _safe_w(wv)
        R    = np.zeros((n_classes, len(decor_idx_full)))
        for r, ci in enumerate(range(n_classes)):
            s = probs[:, ci]
            for c, j in enumerate(decor_idx_full):
                R[r, c] = _weighted_pearson(Xarr[:, j], s, wv)
        return R

    for tag, probs, Xfull, wv in [
        ("Train", probs_train, X_train_full, w_train),
        ("Test",  probs_test,  X_test_full,  w_test),
    ]:
        R = _build_corr_matrix(probs, Xfull, wv)
        plt.figure(figsize=(max(6, 1.2 * len(decor_idx_full) + 4), n_classes + 4))
        plt.imshow(R, aspect="auto", interpolation="none", vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(fraction=0.046, pad=0.04, label="weighted Pearson r")
        plt.xticks(range(len(decor_var_names)), decor_var_names, rotation=45, ha="right")
        plt.yticks(range(n_classes), class_names)
        for i in range(n_classes):
            for j in range(len(decor_var_names)):
                v = R[i, j]
                plt.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.5 else "black", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{tree_name}_Decor_Heatmap_{tag}.png")
        plt.close()


# ── Main execution loop ────────────────────────────────────────────────────────
for tree_name in SUBMIT_TREES:
    print(f"\n{'='*60}")
    print(f"  Training: {tree_name}")
    print(f"{'='*60}")

    branches   = [b["name"] for b in br_cfg[tree_name]]
    sel        = sel_cfg[tree_name]
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_tf      = sel.get("log_transform", [])
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    decorrelate = cfg.get(tree_name, {}).get("decorrelate", [])
    model_path  = MODEL_PATTERN.format(output_root=OUTPUT_ROOT, tree_name=tree_name)

    print("Loading data...")
    X, y, w = prepare_data(tree_name, branches)

    print("Applying thresholds...")
    X, y, w = filter_X(X, y, w, branches, thresholds, apply_to_sentinel=True)

    print("Standardising features...")
    X_std = standardize_X(X.copy(), clip_ranges, log_tf)

    print("Training model...")
    clf, splits = train_multi_model(X_std, y, w, model_path, tree_name,
                                    decorrelate_feature_names=decorrelate)

    print("Plotting results...")
    plot_results(clf, splits, tree_name, decorrelate_feature_names=decorrelate)

    print(f"  Done: {tree_name}")
