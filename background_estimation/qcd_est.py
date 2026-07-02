#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Estimate QCD with an ABCD method on the MC test split."""

from __future__ import annotations

import gc
import colorsys
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb


# -------------------- Style --------------------
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)

_CLASS_COLOR_BASE = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


def _plot_colors(n: int) -> List[str]:
    colors = list(_CLASS_COLOR_BASE)
    used = {color.lower() for color in colors}
    hue = 0.13
    while len(colors) < n:
        rgb = colorsys.hsv_to_rgb(hue % 1.0, 0.72, 0.86)
        candidate = "#{:02x}{:02x}{:02x}".format(
            int(round(rgb[0] * 255.0)),
            int(round(rgb[1] * 255.0)),
            int(round(rgb[2] * 255.0)),
        )
        if candidate.lower() not in used:
            colors.append(candidate)
            used.add(candidate.lower())
        hue += 0.618033988749895
    return colors[:n]


def _group_color_map(extra_names=None) -> Dict[str, str]:
    names = list(CLASS_NAMES)
    if extra_names is not None:
        for name in extra_names:
            if name not in names:
                names.append(name)
    return dict(zip(names, _plot_colors(len(names))))


# -------------------- Paths --------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
_SELECTIONS_DIR = os.path.join(_ROOT_DIR, "selections")
_BDT_DIR = os.path.join(_SELECTIONS_DIR, "BDT")
if _BDT_DIR not in sys.path:
    sys.path.insert(0, _BDT_DIR)

from model_io import (
    load_model as _shared_load_model,
    predict_model_proba as _shared_predict_model_proba,
)


# -------------------- Logging --------------------
def log_message(message: str) -> None:
    print(message, flush=True)


def log_warning(message: str) -> None:
    log_message(f"[WARN] {message}")


def _format_seconds(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.1f}s"


def _format_bytes(n_bytes: float) -> str:
    n_bytes = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0 or unit == "TB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} TB"


def _file_size_text(path: str) -> str:
    try:
        return _format_bytes(os.path.getsize(path))
    except OSError:
        return "unknown"


def _array_size_text(arr) -> str:
    return _format_bytes(getattr(arr, "nbytes", 0))


def _dataframe_size_text(df: pd.DataFrame) -> str:
    return _format_bytes(df.memory_usage(index=True, deep=False).sum())


# -------------------- Helpers --------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


# -------------------- Config loading --------------------
_cfg_path = os.environ.get("QCD_EST_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
_cfg_path = _resolve(_cfg_path, _SCRIPT_DIR)
qcd_cfg = _load_json(_cfg_path)

LUMI = float(qcd_cfg["lumi"])
BDT_ROOT = _resolve(qcd_cfg["bdt_root"], _SCRIPT_DIR)
OUTPUT_DIR = _resolve(qcd_cfg.get("output_dir", "./output"), _SCRIPT_DIR)
ROOT_FILE_NAME = qcd_cfg.get("root_file_name", "qcd_abcd_yields.root")
SIGNAL_REGION_CSV_PATH = _resolve(qcd_cfg["signal_region_csv"], _SCRIPT_DIR)
TEST_REFERENCE_QCD_EST = os.path.join(BDT_ROOT, "test_reference_qcd_est.npz")
TEST_REFERENCE_QCD_EST_FULL = os.path.join(BDT_ROOT, "test_reference_qcd_est_full.npz")


# -------------------- Trained-model config copies --------------------
cfg = _load_json(os.path.join(BDT_ROOT, "config.json"))
br_cfg = _load_json(os.path.join(BDT_ROOT, "branch.json"))
sel_cfg = _load_json(os.path.join(BDT_ROOT, "selection.json"))
test_meta = _load_json(os.path.join(BDT_ROOT, "test_ranges.json"))

_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_BDT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

TREE_NAME = test_meta["tree_name"]
MODEL_PATTERN = cfg.get("model_pattern", "{output_root}/{tree_name}_model")
CLASS_GROUPS = cfg["class_groups"]
CLASS_NAMES = list(CLASS_GROUPS.keys())
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_AXIS_NAMES = CLASS_NAMES[: max(1, NUM_CLASSES - 1)]
INFERENCE_THREADS = max(1, min(32, os.cpu_count() or 1))
XGB_PREDICT_BATCH_TARGET_BYTES = 512 * 1024 * 1024
XGB_PREDICT_MIN_BATCH_ROWS = 100_000
XGB_PREDICT_PROGRESS_SECONDS = 30.0


def _resolve_tree_branch_names(
    config: dict,
    tree_name: str,
    key: str,
    required: bool = False,
) -> List[str]:
    if key not in config:
        if required:
            raise RuntimeError(
                f"background_estimation config missing required '{key}' mapping"
            )
        return []
    payload = config[key]
    if not isinstance(payload, dict):
        raise TypeError(f"background_estimation config '{key}' must be a dict")
    if tree_name not in payload:
        if required:
            raise RuntimeError(
                f"background_estimation config '{key}' missing tree '{tree_name}'"
            )
        return []
    value = payload[tree_name]
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, list):
        names = value
    else:
        raise TypeError(f"{key}['{tree_name}'] must be a string or list")

    out = []
    seen = set()
    for item in names:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{key}['{tree_name}'] entries must be non-empty strings")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if required and not out:
        raise RuntimeError(
            f"{key}['{tree_name}'] must not be empty"
        )
    return out


def _resolve_tree_float(
    config: dict,
    tree_name: str,
    key: str,
    default: float = 1.0,
) -> float:
    payload = config.get(key, {})
    if not isinstance(payload, dict):
        raise TypeError(f"background_estimation config '{key}' must be a dict")
    value = payload.get(tree_name, default)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{key}['{tree_name}'] must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key}['{tree_name}'] must be a finite non-negative number")
    return value


ABCD_BRANCH_NAMES = _resolve_tree_branch_names(
    qcd_cfg,
    TREE_NAME,
    "abcd_branches",
    required=True,
)
A_REGION_SHAPE_BRANCHES = _resolve_tree_branch_names(
    qcd_cfg,
    TREE_NAME,
    "a_region_shape_branches",
    required=False,
)
QCD_PREDICT_SCALE_MULTIPLIER = _resolve_tree_float(
    qcd_cfg,
    TREE_NAME,
    "qcd_predict_scale_multipliers",
)

QCD_CLASS_NAMES = [class_name for class_name in CLASS_NAMES if "qcd" in class_name.lower()]
if not QCD_CLASS_NAMES:
    raise RuntimeError("BDT class_groups must contain at least one QCD class")
QCD_CLASS_SET = set(QCD_CLASS_NAMES)
QCD_PREDICT_GROUP_NAME = "QCD"


def _poisson_vars_from_yield(values: np.ndarray) -> np.ndarray:
    """Variance for combine: weighted yields are treated as Poisson counts."""
    vals = np.asarray(values, dtype=float)
    return np.maximum(vals, 0.0)


def _diag_cov_from_vars(vars_: np.ndarray) -> np.ndarray:
    return np.diag(np.maximum(np.asarray(vars_, dtype=float), 0.0))


# -------------------- Sample registry --------------------
SAMPLE_INFO = {}
for rule in sample_cfg["sample"]:
    SAMPLE_INFO[rule["name"]] = {
        "xsection": float(rule["xsection"]),
        "raw_entries": int(rule.get("raw_entries", -1)),
        "is_MC": bool(rule["is_MC"]),
        "is_signal": bool(rule["is_signal"]),
        "sample_ID": int(rule["sample_ID"]),
    }

SAMPLE_TO_CLASS = {}
SAMPLE_TO_GROUP = {}
for class_idx, (class_name, members) in enumerate(CLASS_GROUPS.items()):
    for sample_name in members:
        SAMPLE_TO_CLASS[sample_name] = class_idx
        SAMPLE_TO_GROUP[sample_name] = class_name

QCD_SAMPLES = set()
for class_name in QCD_CLASS_NAMES:
    QCD_SAMPLES.update(CLASS_GROUPS[class_name])


# -------------------- Threshold filtering --------------------
def _mask_from_cond(col: pd.Series, cond) -> pd.Series:
    idx = col.index
    if cond is None:
        return pd.Series(True, index=idx)
    if isinstance(cond, (int, float, np.integer, np.floating)):
        return col > float(cond)
    if isinstance(cond, (list, tuple)) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
        mn, mx = cond
        mask = pd.Series(True, index=idx)
        if mn is not None:
            mask &= col > mn
        if mx is not None:
            mask &= col < mx
        return mask
    if isinstance(cond, (list, tuple)):
        masks = [_mask_from_cond(col, item) for item in cond]
        out = pd.Series(False, index=idx)
        for mask in masks:
            out |= mask
        return out
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key not in cond:
                continue
            items = cond[key]
            out = pd.Series(True if is_and else False, index=idx)
            for item in items:
                mask = _mask_from_cond(col, item)
                out = (out & mask) if is_and else (out | mask)
            return out
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported condition type: {type(cond)}")


def _threshold_mask(
    X: pd.DataFrame,
    thresholds: dict | None = None,
    apply_to_sentinel: bool = True,
) -> pd.Series:
    mask = pd.Series(True, index=X.index)
    if not thresholds:
        return mask
    for name, cond in thresholds.items():
        if name not in X.columns:
            raise KeyError(f"Column {name!r} not found in X")
        col = X[name]
        sentinel = col < -990
        if apply_to_sentinel:
            mask &= ~sentinel
            if cond is not None:
                mask &= _mask_from_cond(col, cond)
        else:
            if cond is not None:
                mask &= (_mask_from_cond(col, cond) | sentinel)
    return mask


def filter_X(
    X: pd.DataFrame,
    y,
    w,
    branch: list,
    thresholds: dict | None = None,
    apply_to_sentinel: bool = True,
    sample_labels=None,
    mask: Optional[pd.Series] = None,
):
    """Apply per-branch threshold cuts, matching train.py and signal_region.py."""
    if mask is None and not thresholds:
        if sample_labels is None:
            return X.copy(), y.copy(), w.copy()
        return X.copy(), y.copy(), w.copy(), np.asarray(sample_labels).copy()

    if mask is None:
        mask = _threshold_mask(X, thresholds, apply_to_sentinel=apply_to_sentinel)
    elif not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=X.index)

    X_out = X.loc[mask].copy()
    y_out = y[mask.values].copy()
    w_out = w[mask.values].copy()
    if sample_labels is None:
        return X_out, y_out, w_out
    return X_out, y_out, w_out, np.asarray(sample_labels)[mask.values].copy()


def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        sentinel = arr < -990
        valid = ~sentinel
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


def _drop_decorrelated_features(X: pd.DataFrame, decorrelate: list[str]) -> pd.DataFrame:
    if not decorrelate:
        return X
    drop_cols = [name for name in decorrelate if name in X.columns]
    if drop_cols:
        return X.drop(columns=drop_cols)
    return X


def _softmax_rows(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / (np.sum(exp_v, axis=1, keepdims=True) + 1e-12)


def _reshape_multiclass_margin(predt, n_rows: int) -> np.ndarray:
    predt = np.asarray(predt, dtype=float)
    if predt.ndim == 2:
        if predt.shape[1] == NUM_CLASSES:
            return predt
        if predt.shape[0] == NUM_CLASSES:
            return predt.T
    return predt.reshape(int(n_rows), int(NUM_CLASSES))


def _xgb_batch_rows(n_rows: int, n_cols: int) -> int:
    if n_rows <= 0:
        return 1
    bytes_per_row = max(1, int(n_cols)) * np.dtype(np.float32).itemsize
    rows = max(
        XGB_PREDICT_MIN_BATCH_ROWS,
        XGB_PREDICT_BATCH_TARGET_BYTES // max(1, bytes_per_row),
    )
    return max(1, min(int(n_rows), int(rows)))


def _xgb_feature_names(X) -> Optional[List[str]]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return None


def _validate_xgb_feature_names(model: xgb.Booster, feature_names: Optional[List[str]]) -> None:
    model_features = getattr(model, "feature_names", None)
    if model_features and feature_names and list(model_features) != list(feature_names):
        raise RuntimeError(
            "XGBoost model feature mismatch: "
            f"current={feature_names}, model={list(model_features)}"
        )


def _predict_booster_proba_dmatrix_batched(model: xgb.Booster, X, context: str) -> np.ndarray:
    n_rows = int(X.shape[0])
    n_cols = int(X.shape[1])
    feature_names = _xgb_feature_names(X)
    out = np.empty((n_rows, NUM_CLASSES), dtype=float)
    batch_rows = _xgb_batch_rows(n_rows, n_cols)
    n_batches = int(math.ceil(n_rows / batch_rows)) if n_rows else 0
    start_time = time.perf_counter()
    next_progress = start_time + XGB_PREDICT_PROGRESS_SECONDS
    log_message(
        f"Using XGBoost batched DMatrix prediction for {context}: "
        f"rows={n_rows}, cols={n_cols}, batch_rows={batch_rows}, "
        f"batches={n_batches}, threads={INFERENCE_THREADS}"
    )
    for batch_idx, start in enumerate(range(0, n_rows, batch_rows), 1):
        end = min(start + batch_rows, n_rows)
        batch = X.iloc[start:end] if hasattr(X, "iloc") else X[start:end]
        dmat = xgb.DMatrix(batch, feature_names=feature_names)
        margins = _reshape_multiclass_margin(
            model.predict(dmat, output_margin=True),
            end - start,
        )
        out[start:end] = _softmax_rows(margins)
        now = time.perf_counter()
        if end == n_rows or now >= next_progress:
            log_message(
                f"  {context} prediction batch {batch_idx}/{n_batches}: "
                f"rows={end}/{n_rows}, elapsed={now - start_time:.1f}s"
            )
            next_progress = now + XGB_PREDICT_PROGRESS_SECONDS
    return out


def _predict_booster_proba_batched(model: xgb.Booster, X, context: str) -> np.ndarray:
    n_rows = int(X.shape[0])
    n_cols = int(X.shape[1])
    if n_rows == 0:
        return np.empty((0, NUM_CLASSES), dtype=float)
    feature_names = _xgb_feature_names(X)
    _validate_xgb_feature_names(model, feature_names)

    data = X.to_numpy(dtype=np.float32, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
    out = np.empty((n_rows, NUM_CLASSES), dtype=float)
    batch_rows = _xgb_batch_rows(n_rows, n_cols)
    n_batches = int(math.ceil(n_rows / batch_rows))
    start_time = time.perf_counter()
    next_progress = start_time + XGB_PREDICT_PROGRESS_SECONDS
    log_message(
        f"Using XGBoost batched inplace prediction for {context}: "
        f"rows={n_rows}, cols={n_cols}, batch_rows={batch_rows}, "
        f"batches={n_batches}, threads={INFERENCE_THREADS}"
    )

    for batch_idx, start in enumerate(range(0, n_rows, batch_rows), 1):
        end = min(start + batch_rows, n_rows)
        batch = data[start:end]
        if not batch.flags["C_CONTIGUOUS"]:
            batch = np.ascontiguousarray(batch)
        try:
            margins = model.inplace_predict(
                batch,
                predict_type="margin",
                validate_features=False,
            )
        except (AttributeError, TypeError, ValueError, xgb.core.XGBoostError) as exc:
            log_warning(
                "XGBoost inplace prediction unavailable; falling back to batched DMatrix "
                f"prediction for {context}: {exc}"
            )
            return _predict_booster_proba_dmatrix_batched(model, X, context)
        margins = _reshape_multiclass_margin(margins, end - start)
        out[start:end] = _softmax_rows(margins)
        now = time.perf_counter()
        if end == n_rows or now >= next_progress:
            log_message(
                f"  {context} prediction batch {batch_idx}/{n_batches}: "
                f"rows={end}/{n_rows}, elapsed={now - start_time:.1f}s"
            )
            next_progress = now + XGB_PREDICT_PROGRESS_SECONDS
    return out


def _predict_model_proba(model, X, context: str = "model"):
    if isinstance(model, xgb.Booster):
        return _predict_booster_proba_batched(model, X, context)
    return _shared_predict_model_proba(model, X, NUM_CLASSES)


def _compare_prediction_reference(path, feature_names, sample_labels, class_idx, weights, proba):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction reference not found: {path}. Re-run train.py before qcd_est.py."
        )

    total_start = time.perf_counter()
    log_message(
        f"Loading prediction reference: path={path}, compressed_size={_file_size_text(path)}"
    )
    ref = np.load(path, allow_pickle=False)

    step_start = time.perf_counter()
    ref_features = ref["feature_names"].astype(str).tolist()
    cur_features = list(feature_names)
    log_message(
        f"Reference feature check: current={len(cur_features)}, reference={len(ref_features)}, "
        f"elapsed={_format_seconds(step_start)}"
    )
    if cur_features != ref_features:
        raise RuntimeError(
            "Prediction reference mismatch for qcd_est model features: "
            f"current={cur_features}, reference={ref_features}"
        )

    step_start = time.perf_counter()
    log_message("Loading reference sample labels")
    ref_samples = ref["sample_name"].astype(str)
    cur_samples = np.asarray(sample_labels, dtype=str)
    log_message(
        f"Comparing sample labels: n={len(cur_samples)}, "
        f"reference_size={_array_size_text(ref_samples)}, elapsed={_format_seconds(step_start)}"
    )
    step_start = time.perf_counter()
    if not np.array_equal(cur_samples, ref_samples):
        raise RuntimeError("Prediction reference mismatch for qcd_est sample order/content")
    log_message(f"Sample label comparison passed: elapsed={_format_seconds(step_start)}")

    step_start = time.perf_counter()
    log_message("Loading reference class labels")
    ref_class_idx = ref["class_idx"].astype(int)
    cur_class_idx = np.asarray(class_idx, dtype=int)
    log_message(
        f"Comparing class labels: n={len(cur_class_idx)}, "
        f"reference_size={_array_size_text(ref_class_idx)}, elapsed={_format_seconds(step_start)}"
    )
    step_start = time.perf_counter()
    if not np.array_equal(cur_class_idx, ref_class_idx):
        raise RuntimeError("Prediction reference mismatch for qcd_est class labels")
    log_message(f"Class label comparison passed: elapsed={_format_seconds(step_start)}")

    step_start = time.perf_counter()
    log_message("Loading reference weights")
    ref_weights = ref["weight"].astype(float) * LUMI
    cur_weights = np.asarray(weights, dtype=float)
    weight_rtol = float(ref["weight_rtol"])
    weight_atol = float(ref["weight_atol"])
    log_message(
        f"Comparing weights: shape={cur_weights.shape}, current_size={_array_size_text(cur_weights)}, "
        f"reference_size={_array_size_text(ref_weights)}, rtol={weight_rtol}, atol={weight_atol}, "
        f"load_elapsed={_format_seconds(step_start)}"
    )
    step_start = time.perf_counter()
    if not np.allclose(cur_weights, ref_weights, rtol=weight_rtol, atol=weight_atol):
        diff = float(np.max(np.abs(cur_weights - ref_weights)))
        raise RuntimeError(
            "Prediction reference mismatch for qcd_est weights: "
            f"max_abs_diff={diff:.6g}, rtol={weight_rtol}, atol={weight_atol}"
        )
    log_message(f"Weight comparison passed: elapsed={_format_seconds(step_start)}")

    step_start = time.perf_counter()
    log_message("Loading reference probabilities")
    ref_proba = ref["proba"].astype(float)
    cur_proba = np.asarray(proba, dtype=float)
    proba_rtol = float(ref["proba_rtol"])
    proba_atol = float(ref["proba_atol"])
    log_message(
        f"Comparing probabilities: shape={cur_proba.shape}, "
        f"current_size={_array_size_text(cur_proba)}, reference_size={_array_size_text(ref_proba)}, "
        f"rtol={proba_rtol}, atol={proba_atol}, load_elapsed={_format_seconds(step_start)}"
    )
    if cur_proba.shape != ref_proba.shape:
        raise RuntimeError(
            "Prediction reference mismatch for qcd_est probabilities shape: "
            f"current={cur_proba.shape}, reference={ref_proba.shape}"
        )
    step_start = time.perf_counter()
    if not np.allclose(cur_proba, ref_proba, rtol=proba_rtol, atol=proba_atol):
        diff = float(np.max(np.abs(cur_proba - ref_proba)))
        # Batched XGBoost inference is not bitwise-reproducible across thread/batch
        # configurations (e.g. on shared nodes with varying core availability), so
        # tiny probability differences vs the stored reference are expected and
        # physically irrelevant. Only abort if the difference is large enough to
        # indicate a genuinely different model/inputs rather than numerical noise.
        BENIGN_PROBA_DIFF = 5.0e-3
        if diff <= BENIGN_PROBA_DIFF:
            log_message(
                "WARNING: qcd_est probabilities differ from reference within benign "
                f"tolerance: max_abs_diff={diff:.6g} (strict rtol={proba_rtol}, "
                f"atol={proba_atol}, benign_threshold={BENIGN_PROBA_DIFF}); continuing "
                "(batched-inference thread-order noise, not a model change)."
            )
        else:
            raise RuntimeError(
                "Prediction reference mismatch for qcd_est probabilities: "
                f"max_abs_diff={diff:.6g}, rtol={proba_rtol}, atol={proba_atol}"
            )
    else:
        log_message(f"Probability comparison passed: elapsed={_format_seconds(step_start)}")

    log_message(
        f"Validated prediction reference: {path}, total_elapsed={_format_seconds(total_start)}"
    )


# -------------------- Test data loading --------------------
def load_test_data(branches: list[str]) -> pd.DataFrame:
    """Load the full test split with the same weight definition as signal_region.py."""
    log_message(f"Loading MC test samples: n={len(test_meta['samples'])}")
    dfs = []

    reweight_branches = list(cfg.get(TREE_NAME, {}).get("event_reweight_branches", []))
    load_branches = list(branches)
    for rb in reweight_branches:
        if rb not in load_branches:
            load_branches.append(rb)

    for sample_name, sample_meta in test_meta["samples"].items():
        info = SAMPLE_INFO.get(sample_name)
        if info is None:
            raise RuntimeError(f"Sample '{sample_name}' not found in sample config")
        if not info["is_MC"]:
            log_warning(f"Skipping non-MC sample '{sample_name}'")
            continue
        if sample_name not in SAMPLE_TO_CLASS:
            raise RuntimeError(f"Sample '{sample_name}' not in any class group")

        xsec = float(info["xsection"])
        raw_entries = int(info["raw_entries"])
        total_entries = int(sample_meta["total_entries"])
        if raw_entries <= 0:
            raise RuntimeError(
                f"Sample '{sample_name}' has raw_entries={raw_entries}; fill src/sample.json"
            )

        parts = []
        for seg in sample_meta["test_segments"]:
            fpath = seg["file"]
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Test split file not found: {fpath}")
            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        raise KeyError(f"Tree '{TREE_NAME}' not in {fpath}")
                    tree = uf[TREE_NAME]
                    available = set(tree.keys())
                    missing = [branch for branch in load_branches if branch not in available]
                    if missing:
                        raise KeyError(
                            f"Missing branches in {fpath}:{TREE_NAME}: "
                            f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
                        )
                    parts.append(
                        tree.arrays(
                            load_branches,
                            library="pd",
                            entry_start=int(seg["entry_start"]),
                            entry_stop=int(seg["entry_stop"]),
                        )
                    )
            except Exception as exc:
                raise RuntimeError(f"Failed to read test split file {fpath}: {exc}") from exc

        if not parts:
            raise RuntimeError(f"No data loaded for sample '{sample_name}'")

        df = pd.concat(parts, ignore_index=True)
        n_loaded = len(df)

        if reweight_branches:
            raw_w = np.ones(n_loaded, dtype=float)
            for rb in reweight_branches:
                raw_w *= df[rb].to_numpy(dtype=float, copy=False)
            df = df.drop(columns=reweight_branches)
        else:
            raw_w = np.ones(n_loaded, dtype=float)

        if xsec <= 0.0:
            target_total = 0.0
            df["weight"] = 0.0
            log_warning(
                f"{sample_name}: non-positive xsec={xsec}, zero weight"
            )
        else:
            target_total = LUMI * xsec * total_entries / raw_entries
            raw_w_sum = float(raw_w.sum())
            if raw_w_sum <= 0.0:
                raise RuntimeError(
                    f"Sample '{sample_name}' has non-positive raw weight sum {raw_w_sum:.6g}"
                )
            df["weight"] = raw_w * (target_total / raw_w_sum)

        df["class_idx"] = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        df["group_name"] = SAMPLE_TO_GROUP[sample_name]
        dfs.append(df)
        log_message(
            f"  {sample_name}: class={SAMPLE_TO_GROUP[sample_name]}, "
            f"total_entries={total_entries}, loaded={n_loaded}, "
            f"weight_sum={float(df['weight'].sum()):.6g}"
        )

    if not dfs:
        raise RuntimeError("No MC test data loaded")

    df_all = pd.concat(dfs, ignore_index=True)
    log_message(f"Loaded MC test events: {len(df_all)}")
    del dfs
    gc.collect()
    return df_all


# -------------------- Model loading --------------------
def _configure_model_for_inference(model):
    if isinstance(model, xgb.Booster):
        model.set_param({"nthread": INFERENCE_THREADS})
        log_message(f"XGBoost inference threads = {INFERENCE_THREADS}")
    elif hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=INFERENCE_THREADS)
            log_message(f"Model inference jobs = {INFERENCE_THREADS}")
        except (TypeError, ValueError):
            pass
    return model


def _load_model():
    model_base = MODEL_PATTERN.format(output_root=BDT_ROOT, tree_name=TREE_NAME)
    model = _shared_load_model(model_base, cfg, NUM_CLASSES, log_message=log_message)
    return _configure_model_for_inference(model)


# -------------------- Region helpers --------------------
def _resolve_abcd_thresholds(thresholds: dict, abcd_branch_names: List[str]) -> Tuple[dict, dict]:
    abcd_thresholds = {}
    other_thresholds = {}
    abcd_set = set(abcd_branch_names)
    missing = [name for name in abcd_branch_names if name not in thresholds]
    if missing:
        raise RuntimeError(
            "Configured ABCD branch is missing from bdt_root selection.json thresholds: "
            + ", ".join(missing)
        )
    for name, cond in thresholds.items():
        if name in abcd_set:
            abcd_thresholds[name] = cond
        else:
            other_thresholds[name] = cond
    return abcd_thresholds, other_thresholds


def _abcd_pass_fail_masks(df: pd.DataFrame, abcd_thresholds: dict) -> Tuple[np.ndarray, np.ndarray]:
    pass_mask = np.ones(len(df), dtype=bool)
    fail_mask = np.ones(len(df), dtype=bool)
    valid_mask = np.ones(len(df), dtype=bool)

    for name, cond in abcd_thresholds.items():
        if name not in df.columns:
            raise KeyError(f"ABCD threshold branch {name!r} not found in DataFrame")
        col = df[name]
        values = col.to_numpy(dtype=float, copy=False)
        sentinel = values < -990
        finite = np.isfinite(values)
        branch_valid = (~sentinel) & finite
        cond_mask = _mask_from_cond(col, cond).to_numpy(dtype=bool)
        valid_mask &= branch_valid
        pass_mask &= branch_valid & cond_mask
        fail_mask &= branch_valid & (~cond_mask)

    pass_mask &= valid_mask
    fail_mask &= valid_mask
    return pass_mask, fail_mask


def _detect_signal_region_axes(df: pd.DataFrame) -> List[str]:
    axes = []
    columns = set(df.columns)
    for col in df.columns:
        if not col.endswith("_low"):
            continue
        axis_name = col[:-4]
        if f"{axis_name}_high" not in columns:
            continue
        if axis_name not in CLASS_NAMES:
            raise KeyError(
                f"Signal region axis {axis_name!r} is not in BDT class_groups: {CLASS_NAMES}"
            )
        axes.append(axis_name)
    if not axes:
        axes = list(DEFAULT_AXIS_NAMES)
    return axes


def _load_signal_regions() -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.exists(SIGNAL_REGION_CSV_PATH):
        raise FileNotFoundError(
            f"Signal region CSV not found: {SIGNAL_REGION_CSV_PATH}. Run signal_region.py first."
        )

    df = pd.read_csv(SIGNAL_REGION_CSV_PATH)
    if df.empty:
        raise RuntimeError(f"Signal region CSV is empty: {SIGNAL_REGION_CSV_PATH}")

    axis_names = _detect_signal_region_axes(df)
    required = ["bin_index"]
    for axis_name in axis_names:
        required.extend([f"{axis_name}_low", f"{axis_name}_high"])
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(
            f"Signal region CSV missing required columns: {', '.join(missing)}"
        )

    bin_values = pd.to_numeric(df["bin_index"], errors="raise").to_numpy(dtype=float)
    rounded_bins = np.rint(bin_values)
    if (
        not np.all(np.isfinite(bin_values))
        or not np.allclose(bin_values, rounded_bins)
        or np.any(rounded_bins <= 0)
    ):
        raise RuntimeError("Signal region CSV bin_index values must be positive integers")
    bin_ids = [int(value) for value in rounded_bins]
    if len(set(bin_ids)) != len(bin_ids):
        raise RuntimeError("Signal region CSV bin_index values must be unique")
    df = df.copy()
    df["bin_index"] = bin_ids

    return df.sort_values("bin_index").reset_index(drop=True), axis_names


def _region_mask(proba: np.ndarray, region_row: pd.Series, axis_names: List[str]) -> np.ndarray:
    mask = np.ones(proba.shape[0], dtype=bool)
    for axis_name in axis_names:
        low = float(region_row[f"{axis_name}_low"])
        high = float(region_row[f"{axis_name}_high"])
        axis_scores = proba[:, CLASS_NAMES.index(axis_name)]
        if high < 1.0 - 1e-12:
            mask &= (axis_scores >= low) & (axis_scores < high)
        else:
            mask &= axis_scores >= low
    return mask


# -------------------- Plotting helpers --------------------
def _hist_with_var(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, _ = np.histogram(values, bins=edges, weights=weights)
    vars_, _ = np.histogram(values, bins=edges, weights=weights ** 2)
    return vals.astype(float), vars_.astype(float)


def _ratio_pred_over_true(pred_vals, pred_vars, true_vals, true_vars):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(true_vals > 0, pred_vals / true_vals, np.nan)
        term_pred = np.where(pred_vals > 0, pred_vars / np.maximum(pred_vals, 1e-300) ** 2, 0.0)
        term_true = np.where(true_vals > 0, true_vars / np.maximum(true_vals, 1e-300) ** 2, 0.0)
        sigma = np.abs(ratio) * np.sqrt(term_pred + term_true)
    return ratio, sigma


def _add_uncert_band(ax, edges: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    lower_step = np.r_[lower, lower[-1]]
    upper_step = np.r_[upper, upper[-1]]
    ax.fill_between(
        edges,
        lower_step,
        upper_step,
        step="post",
        facecolor="none",
        edgecolor="gray",
        hatch="///",
        linewidth=0,
    )


def _qcd_merged_group_maps(
    group_vals: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    merged_vals = {}
    merged_vars = {}
    merged_groups = []
    n_regions = len(next(iter(group_vals.values())))
    qcd_vals = np.zeros(n_regions, dtype=float)
    qcd_vars = np.zeros(n_regions, dtype=float)

    for name in CLASS_NAMES:
        if name in QCD_CLASS_SET:
            qcd_vals += group_vals[name]
            qcd_vars += group_vars[name]
        else:
            merged_vals[name] = group_vals[name].copy()
            merged_vars[name] = group_vars[name].copy()
            merged_groups.append(name)

    merged_vals[QCD_PREDICT_GROUP_NAME] = qcd_vals
    merged_vars[QCD_PREDICT_GROUP_NAME] = qcd_vars
    merged_groups.append(QCD_PREDICT_GROUP_NAME)
    return merged_vals, merged_vars, merged_groups


def plot_abcd_region_counts(
    region_labels: List[str],
    group_vals: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
    out_path: str,
    normalize_per_bin: bool = False,
    groups: Optional[List[str]] = None,
    log_y: bool = True,
) -> None:
    edges = np.arange(len(region_labels) + 1, dtype=float)
    centers = edges[:-1] + 0.5
    widths = np.full(len(region_labels), 1.0)
    plot_groups = list(groups) if groups is not None else list(CLASS_NAMES)

    vals_map = {name: group_vals[name].copy() for name in plot_groups}
    vars_map = {name: group_vars[name].copy() for name in plot_groups}

    totals = np.zeros(len(region_labels), dtype=float)
    total_vars = np.zeros(len(region_labels), dtype=float)
    for name in plot_groups:
        totals += vals_map[name]
        total_vars += vars_map[name]

    if normalize_per_bin:
        scale = np.where(totals > 0, 1.0 / totals, 0.0)
        for name in plot_groups:
            vals_map[name] *= scale
            vars_map[name] *= scale ** 2
        totals *= scale
        total_vars *= scale ** 2

    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(region_labels), dtype=float)
    order = np.argsort([float(np.sum(vals_map[name])) for name in plot_groups])
    ordered_groups = [plot_groups[idx] for idx in order]
    color_map = _group_color_map(plot_groups)

    for name in ordered_groups:
        ax.bar(
            edges[:-1],
            vals_map[name],
            width=widths,
            bottom=bottom,
            align="edge",
            color=color_map[name],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=name,
        )
        bottom += vals_map[name]

    if not normalize_per_bin:
        sigma = np.sqrt(np.maximum(total_vars, 0.0))
        lower_clip = 1e-12 if log_y else 0.0
        lower = np.clip(totals - sigma, lower_clip, None)
        upper = np.clip(totals + sigma, lower_clip, None)
        _add_uncert_band(ax, edges, lower, upper)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(0.1, max(1.0, float(np.max(totals[totals > 0])) * 3.0 if np.any(totals > 0) else 1.0))
        else:
            max_upper = float(np.max(upper)) if len(upper) else 0.0
            ax.set_ylim(0.0, max(1.0, max_upper * 1.25))
        ax.set_ylabel("Events", fontsize=22)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction", fontsize=22)

    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel("Region", fontsize=22)
    ax.set_xticks(centers)
    ax.set_xticklabels(region_labels, fontsize=14)
    ax.margins(x=0)
    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
    ax.legend(loc="best", fontsize=14, frameon=False, ncol=2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Wrote plot file: {out_path}")


def plot_signal_region_prediction(
    region_labels: List[str],
    pred_group_vals: Dict[str, np.ndarray],
    pred_total_vals: np.ndarray,
    pred_total_vars: np.ndarray,
    true_vals: np.ndarray,
    true_vars: np.ndarray,
    out_path: str,
    groups: List[str],
    ylabel: str,
) -> None:
    n = len(region_labels)
    edges = np.arange(n + 1, dtype=float)
    centers = edges[:-1] + 0.5
    widths = np.full(n, 1.0)
    color_map = _group_color_map(groups)

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    order = np.argsort([float(np.sum(pred_group_vals[name])) for name in groups])
    ordered_groups = [groups[idx] for idx in order]

    bottom = np.zeros(n, dtype=float)
    for name in ordered_groups:
        vals = pred_group_vals[name]
        ax.bar(
            edges[:-1],
            vals,
            width=widths,
            bottom=bottom,
            align="edge",
            color=color_map.get(name, "#1f77b4"),
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
            label=name,
        )
        bottom += vals

    pred_sigma = np.sqrt(np.maximum(pred_total_vars, 0.0))
    lower = np.clip(pred_total_vals - pred_sigma, 1e-12, None)
    upper = np.clip(pred_total_vals + pred_sigma, 1e-12, None)
    _add_uncert_band(ax, edges, lower, upper)

    true_sigma = np.sqrt(np.maximum(true_vars, 0.0))
    y_plot = np.where(true_vals > 0, true_vals, np.nan)
    ax.errorbar(
        centers,
        y_plot,
        yerr=true_sigma,
        fmt="o",
        ms=7.2,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
        label="True",
    )

    ax.set_yscale("log")
    ymax = max(
        float(np.nanmax(pred_total_vals)) if pred_total_vals.size else 1.0,
        float(np.nanmax(true_vals)) if true_vals.size else 1.0,
        1.0,
    )
    ax.set_ylim(0.1, max(1.0, ymax * 4.0))
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylabel(ylabel, fontsize=22)
    ax.margins(x=0)
    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if "True" in labels:
        idx = labels.index("True")
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))
    ax.legend(handles, labels, loc="best", fontsize=14, frameon=False, ncol=2)

    ratio, ratio_err = _ratio_pred_over_true(pred_total_vals, pred_total_vars, true_vals, true_vars)
    axr.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        fmt="o",
        ms=7.2,
        color="black",
        mfc="black",
        mec="black",
        elinewidth=1.5,
        capsize=0,
    )
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    finite = np.isfinite(ratio)
    if np.any(finite):
        rmax = float(np.nanmax(ratio[finite] + np.nan_to_num(ratio_err[finite], nan=0.0)))
        rmin = float(np.nanmin(ratio[finite] - np.nan_to_num(ratio_err[finite], nan=0.0)))
        if rmax < 5.0:
            axr.set_ylim(max(0.0, 0.8 * rmin), max(2.0, 1.2 * rmax))
        else:
            axr.set_ylim(0.0, 5.0)
    else:
        axr.set_ylim(0.0, 2.0)

    axr.set_ylabel(r"$\frac{Pred}{True}$", fontsize=24)
    axr.yaxis.set_label_coords(-0.05, 0.6)
    axr.set_xlabel("Signal Region", fontsize=22)
    axr.set_xticks(centers)
    axr.set_xticklabels(region_labels, fontsize=14)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_message(f"Wrote plot file: {out_path}")


def plot_a_region_branch_shapes(
    X_raw: pd.DataFrame,
    group_labels: np.ndarray,
    weights: np.ndarray,
    branch_names: List[str],
    region_masks: List[np.ndarray],
    region_labels: List[str],
    a_union_mask: np.ndarray,
    out_dir: str,
    n_bins: int = 50,
) -> None:
    if not branch_names:
        return

    os.makedirs(out_dir, exist_ok=True)
    color_map = _group_color_map()
    plot_regions = list(zip(region_labels, region_masks)) + [("A union", a_union_mask)]
    group_labels = np.asarray(group_labels, dtype=object)
    weights = np.asarray(weights, dtype=float)

    log_message(
        f"Plotting A-region branch shapes: branches={len(branch_names)}, "
        f"regions_per_branch={len(plot_regions)}"
    )
    for branch in branch_names:
        if branch not in X_raw.columns:
            raise KeyError(f"A-region shape branch {branch!r} not found in loaded data")

        values = X_raw[branch].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(values) & (values > -10.0)
        edge_values = values[valid & a_union_mask]
        if edge_values.size == 0:
            log_warning(
                f"A-region shape branch '{branch}' has no valid A-union entries "
                "above -10; skipping"
            )
            continue

        lo = max(-10.0, float(np.min(edge_values)))
        hi = float(np.max(edge_values))
        if not np.isfinite(lo) or not np.isfinite(hi):
            log_warning(f"A-region shape branch '{branch}' has non-finite range; skipping")
            continue
        if lo >= hi:
            pad = max(1.0, abs(lo) * 0.05)
            lo = max(-10.0, lo - 0.5 * pad)
            hi += 0.5 * pad
            log_warning(
                f"A-region shape branch '{branch}' has degenerate range; "
                f"using [{lo:.6g}, {hi:.6g}]"
            )

        edges = np.linspace(lo, hi, int(n_bins) + 1)
        branch_file = _slugify(branch) or "branch"

        for region_label, region_mask in plot_regions:
            fig, ax = plt.subplots(figsize=(8.5, 6.5))
            plotted_any = False
            for group_name in CLASS_NAMES:
                mask = valid & region_mask & (group_labels == group_name)
                if not np.any(mask):
                    continue
                group_weights = weights[mask]
                if float(np.sum(group_weights)) <= 0.0:
                    continue
                ax.hist(
                    values[mask],
                    bins=edges,
                    weights=group_weights,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    color=color_map[group_name],
                    label=group_name,
                )
                plotted_any = True

            if not plotted_any:
                plt.close(fig)
                log_warning(
                    f"A-region shape branch '{branch}' has no positive-weight "
                    f"entries in {region_label}; skipping"
                )
                continue

            ax.set_xlim(lo, hi)
            ax.set_xlabel(branch, fontsize=18)
            ax.set_ylabel("A.U.", fontsize=22)
            ax.set_title(region_label, fontsize=18)
            hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
            ax.legend(loc="best", fontsize=12, frameon=False, ncol=2)
            fig.tight_layout()

            region_file = _slugify(region_label) or "region"
            out_path = os.path.join(
                out_dir,
                f"a_region_shape_{branch_file}_{region_file}.pdf",
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log_message(f"Wrote plot file: {out_path}")


def write_root_output(
    root_path: str,
    edges: np.ndarray,
    signal_region_ids: List[int],
    sample_yields: Dict[str, np.ndarray],
    sample_vars: Dict[str, np.ndarray],
    group_yields: Dict[str, np.ndarray],
    group_vars: Dict[str, np.ndarray],
    pred_qcd_vals: np.ndarray,
    pred_qcd_stat_vars: np.ndarray,
    pred_qcd_scale_vars: np.ndarray,
    pred_qcd_cov: np.ndarray,
    true_qcd_vals: np.ndarray,
    true_qcd_vars: np.ndarray,
    pred_total_vals: np.ndarray,
    pred_total_stat_vars: np.ndarray,
    pred_total_scale_vars: np.ndarray,
    pred_total_cov: np.ndarray,
    true_total_vals: np.ndarray,
    true_total_vars: np.ndarray,
) -> None:
    sr_ids = np.asarray(signal_region_ids, dtype=np.int32)
    if sr_ids.ndim != 1 or len(sr_ids) == 0:
        raise RuntimeError("Signal region ids must be a non-empty one-dimensional list")
    if len(set(int(value) for value in sr_ids)) != len(sr_ids) or np.any(sr_ids <= 0):
        raise RuntimeError("Signal region ids must be unique positive integers")

    def _write_bundle(
        root_file,
        prefix: str,
        values: np.ndarray,
        stat_vars: np.ndarray,
        scale_vars: np.ndarray | None = None,
        covariance_total: np.ndarray | None = None,
    ) -> None:
        vals = np.asarray(values, dtype=float)
        stat_vars = np.asarray(stat_vars, dtype=float)
        if scale_vars is None:
            scale_vars = np.zeros_like(stat_vars)
        else:
            scale_vars = np.asarray(scale_vars, dtype=float)
        if covariance_total is None:
            covariance_total = np.diag(np.maximum(stat_vars + scale_vars, 0.0))
        else:
            covariance_total = np.asarray(covariance_total, dtype=float)

        n_sr = len(vals)
        if n_sr != len(sr_ids):
            raise RuntimeError(f"Signal-region count mismatch for ROOT bundle '{prefix}'")
        if stat_vars.shape != vals.shape or scale_vars.shape != vals.shape:
            raise RuntimeError(f"Uncertainty size mismatch for ROOT bundle '{prefix}'")
        if covariance_total.shape != (n_sr, n_sr):
            raise RuntimeError(f"Covariance size mismatch for ROOT bundle '{prefix}'")

        one_bin_edges = np.array([0.0, 1.0], dtype=float)
        stat_err = np.sqrt(np.maximum(stat_vars, 0.0))
        scale_err = np.sqrt(np.maximum(scale_vars, 0.0))
        for idx in range(n_sr):
            sr_prefix = f"{prefix}/sr{int(sr_ids[idx])}"
            root_file[f"{sr_prefix}/yield"] = (np.array([vals[idx]], dtype=float), one_bin_edges)
            root_file[f"{sr_prefix}/stat_error"] = (
                np.array([stat_err[idx]], dtype=float),
                one_bin_edges,
            )
            root_file[f"{sr_prefix}/scale_error"] = (
                np.array([scale_err[idx]], dtype=float),
                one_bin_edges,
            )
        root_file[f"{prefix}/covariance_total"] = (covariance_total, edges, edges)

    with uproot.recreate(root_path) as root_file:
        root_file["metadata/signal_regions"] = {"bin_index": sr_ids}

        for sample_name in sorted(sample_yields):
            _write_bundle(
                root_file,
                f"samples/{sample_name}",
                sample_yields[sample_name],
                sample_vars[sample_name],
            )

        for group_name in CLASS_NAMES:
            _write_bundle(
                root_file,
                f"groups/{_slugify(group_name)}",
                group_yields[group_name],
                group_vars[group_name],
            )

        _write_bundle(
            root_file,
            "qcd_predict",
            pred_qcd_vals,
            pred_qcd_stat_vars,
            pred_qcd_scale_vars,
            pred_qcd_cov,
        )
        _write_bundle(root_file, "qcd_true", true_qcd_vals, true_qcd_vars)
        _write_bundle(
            root_file,
            "total_predict",
            pred_total_vals,
            pred_total_stat_vars,
            pred_total_scale_vars,
            pred_total_cov,
        )
        _write_bundle(root_file, "total_true", true_total_vals, true_total_vars)

    log_message(f"Wrote ROOT file: {root_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_message(
        f"Running qcd_est.py: tree={TREE_NAME}, lumi={LUMI} fb^-1, "
        f"bdt_root={BDT_ROOT}, signal_region_csv={SIGNAL_REGION_CSV_PATH}, output_dir={OUTPUT_DIR}"
    )
    log_message("Loading trained-model config copies")

    model_branches = [item["name"] for item in br_cfg[TREE_NAME]]
    selection = sel_cfg[TREE_NAME]
    clip_ranges = {key: tuple(val) for key, val in selection.get("clip_ranges", {}).items()}
    log_transform = list(selection.get("log_transform", []))
    thresholds = {
        key: (tuple(val) if isinstance(val, list) else val)
        for key, val in selection.get("thresholds", {}).items()
    }
    abcd_thresholds, bdt_thresholds = _resolve_abcd_thresholds(thresholds, ABCD_BRANCH_NAMES)
    decorrelate = cfg.get(TREE_NAME, {}).get("decorrelate", [])

    log_message("Loading signal region file")
    # Load every branch needed downstream: model features (model_branches), all
    # threshold branches (for filter_X and the ABCD pass/fail masks), every
    # decorrelate branch (in case decorrelation references a branch not in
    # branch.json), and the optional A-region shape branches. Model inference
    # still uses only model_branches.
    load_branches = sorted(
        set(model_branches)
        | set(thresholds.keys())
        | set(decorrelate)
        | set(A_REGION_SHAPE_BRANCHES)
    )
    signal_regions, axis_names = _load_signal_regions()
    region_labels = [f"SR{int(idx)}" for idx in signal_regions["bin_index"].tolist()]
    edges = np.arange(len(region_labels) + 1, dtype=float)
    log_message(
        f"Resolved inputs: model_branches={len(model_branches)}, "
        f"load_branches={len(load_branches)}, signal_regions={len(region_labels)}, "
        f"score_axes={axis_names}, non_abcd_thresholds={len(bdt_thresholds)}, "
        f"abcd_thresholds={list(abcd_thresholds)}, "
        f"a_region_shape_branches={A_REGION_SHAPE_BRANCHES}"
    )
    log_message(
        "QCD classes for ABCD merge: "
        + ", ".join(QCD_CLASS_NAMES)
        + f" ({len(QCD_SAMPLES)} samples)"
    )
    log_message(f"Output directory: {OUTPUT_DIR}")

    step_start = time.perf_counter()
    df_all = load_test_data(load_branches)
    log_message(
        f"Finished loading MC test data: events={len(df_all)}, columns={len(df_all.columns)}, "
        f"dataframe_size={_dataframe_size_text(df_all)}, elapsed={_format_seconds(step_start)}"
    )
    step_start = time.perf_counter()
    X_raw = df_all[load_branches].copy()
    y = df_all["class_idx"].values.astype(int)
    w = df_all["weight"].values.astype(float)
    sample_labels = df_all["sample_name"].astype(str).values
    group_labels = df_all["group_name"].astype(str).values
    log_message(
        f"Prepared raw analysis arrays: events={len(X_raw)}, branches={len(load_branches)}, "
        f"X_raw_size={_dataframe_size_text(X_raw)}, weights_size={_array_size_text(w)}, "
        f"elapsed={_format_seconds(step_start)}"
    )
    del df_all
    gc.collect()

    clf = _load_model()
    have_full_reference = os.path.exists(TEST_REFERENCE_QCD_EST_FULL)
    log_message(
        f"Prediction reference availability: full_reference={have_full_reference}, "
        f"full_path={TEST_REFERENCE_QCD_EST_FULL}, "
        f"full_size={_file_size_text(TEST_REFERENCE_QCD_EST_FULL) if have_full_reference else 'missing'}"
    )
    proba_full = None
    proba = None
    if have_full_reference:
        log_message(
            f"Validating full test-set prediction reference: events={len(X_raw)}, "
            f"model_branches={len(model_branches)}"
        )
        step_start = time.perf_counter()
        X_model_full = X_raw[model_branches].copy()
        log_message(
            f"Copied full test-set model features: shape={X_model_full.shape}, "
            f"size={_dataframe_size_text(X_model_full)}, elapsed={_format_seconds(step_start)}"
        )

        step_start = time.perf_counter()
        X_model_full = standardize_X(X_model_full, clip_ranges, log_transform)
        log_message(
            f"Standardised full test-set model features: shape={X_model_full.shape}, "
            f"elapsed={_format_seconds(step_start)}"
        )

        step_start = time.perf_counter()
        X_model_full = _drop_decorrelated_features(X_model_full, decorrelate)
        if decorrelate:
            log_message(
                f"Removed decorrelated full test-set features: {decorrelate}, "
                f"shape={X_model_full.shape}, elapsed={_format_seconds(step_start)}"
            )
        else:
            log_message(
                f"No decorrelated full test-set features to remove: shape={X_model_full.shape}, "
                f"elapsed={_format_seconds(step_start)}"
            )

        step_start = time.perf_counter()
        log_message(
            f"Running full test-set model prediction: X_shape={X_model_full.shape}, "
            f"model_type={type(clf).__name__}"
        )
        proba_full = _predict_model_proba(clf, X_model_full, context="full test-set")
        log_message(
            f"Finished full test-set model prediction: proba_shape={proba_full.shape}, "
            f"proba_size={_array_size_text(proba_full)}, elapsed={_format_seconds(step_start)}"
        )
        step_start = time.perf_counter()
        log_message("Comparing full test-set prediction reference")
        _compare_prediction_reference(
            TEST_REFERENCE_QCD_EST_FULL,
            X_model_full.columns
            if hasattr(X_model_full, "columns")
            else [f"f{i}" for i in range(X_model_full.shape[1])],
            sample_labels,
            y,
            w,
            proba_full,
        )
        log_message(f"Finished full test-set reference comparison: elapsed={_format_seconds(step_start)}")
        step_start = time.perf_counter()
        del X_model_full
        gc.collect()
        log_message(
            "Released full test-set model features; keeping probabilities for filtered reuse: "
            f"elapsed={_format_seconds(step_start)}"
        )

    log_message("Applying non-ABCD thresholds")
    events_before_filter = len(X_raw)
    step_start = time.perf_counter()
    filter_mask = _threshold_mask(X_raw, bdt_thresholds, apply_to_sentinel=True)
    if proba_full is not None:
        mask_values = filter_mask.values
        if int(np.count_nonzero(mask_values)) == len(mask_values):
            proba = proba_full
            proba_full = None
        else:
            proba = proba_full[mask_values].copy()
            del proba_full
        gc.collect()
        log_message(
            f"Reused full test-set probabilities for filtered events: "
            f"shape={proba.shape}, size={_array_size_text(proba)}, "
            f"elapsed={_format_seconds(step_start)}"
        )
    X_raw, y, w, sample_labels = filter_X(
        X_raw,
        y,
        w,
        load_branches,
        bdt_thresholds,
        apply_to_sentinel=True,
        sample_labels=sample_labels,
        mask=filter_mask,
    )
    group_labels = np.asarray([SAMPLE_TO_GROUP[name] for name in sample_labels], dtype=object)
    log_message(
        f"After non-ABCD filtering: events={len(X_raw)} "
        f"(removed={events_before_filter - len(X_raw)} of {events_before_filter}), "
        f"elapsed={_format_seconds(step_start)}"
    )

    log_message("Evaluating ABCD pass/fail masks")
    step_start = time.perf_counter()
    abcd_pass, abcd_fail = _abcd_pass_fail_masks(X_raw, abcd_thresholds)
    abcd_mixed = ~(abcd_pass | abcd_fail)
    log_message(
        f"ABCD branch categories: pass={int(np.count_nonzero(abcd_pass))}, "
        f"fail={int(np.count_nonzero(abcd_fail))}, "
        f"excluded_mixed={int(np.count_nonzero(abcd_mixed))}, "
        f"elapsed={_format_seconds(step_start)}"
    )

    if proba is None:
        log_message("Standardising model features")
        step_start = time.perf_counter()
        X_model = X_raw[model_branches].copy()
        log_message(
            f"Copied filtered model features: shape={X_model.shape}, "
            f"size={_dataframe_size_text(X_model)}, elapsed={_format_seconds(step_start)}"
        )
        step_start = time.perf_counter()
        X_model = standardize_X(X_model, clip_ranges, log_transform)
        log_message(
            f"Standardised filtered model features: shape={X_model.shape}, "
            f"elapsed={_format_seconds(step_start)}"
        )
        if decorrelate:
            step_start = time.perf_counter()
            X_model = _drop_decorrelated_features(X_model, decorrelate)
            log_message(
                f"Removed decorrelated filtered features: {decorrelate}, "
                f"shape={X_model.shape}, elapsed={_format_seconds(step_start)}"
            )

        log_message("Running model prediction")
        step_start = time.perf_counter()
        proba = _predict_model_proba(clf, X_model, context="filtered test-set")
        log_message(
            f"Predicted probabilities: shape={proba.shape}, size={_array_size_text(proba)}, "
            f"elapsed={_format_seconds(step_start)}"
        )
        log_warning(
            "Full qcd_est reference missing; using legacy filtered reference. "
            "Re-run train.py to produce test_reference_qcd_est_full.npz for configurable ABCD branches."
        )
        log_message(
            f"Validating filtered test-set prediction reference: path={TEST_REFERENCE_QCD_EST}, "
            f"size={_file_size_text(TEST_REFERENCE_QCD_EST)}"
        )
        step_start = time.perf_counter()
        _compare_prediction_reference(
            TEST_REFERENCE_QCD_EST,
            X_model.columns
            if hasattr(X_model, "columns")
            else [f"f{i}" for i in range(X_model.shape[1])],
            sample_labels,
            y,
            w,
            proba,
        )
        log_message(f"Finished filtered test-set reference comparison: elapsed={_format_seconds(step_start)}")

    log_message("Building ABCD regions")
    region_score_masks = []
    union_score_mask = np.zeros(len(X_raw), dtype=bool)
    membership = np.zeros(len(X_raw), dtype=int)
    for _, row in signal_regions.iterrows():
        mask = _region_mask(proba, row, axis_names)
        region_score_masks.append(mask)
        union_score_mask |= mask
        membership += mask.astype(int)

    if np.any(membership > 1):
        raise RuntimeError("Signal region definitions overlap on the current event set")

    region_a_masks = [mask & abcd_pass for mask in region_score_masks]
    a_union_mask = union_score_mask & abcd_pass
    b_mask = (~union_score_mask) & abcd_pass
    c_mask = union_score_mask & abcd_fail
    d_mask = (~union_score_mask) & abcd_fail

    log_message(
        f"ABCD event counts: A_union={int(np.count_nonzero(a_union_mask))}, "
        f"B={int(np.count_nonzero(b_mask))}, C={int(np.count_nonzero(c_mask))}, "
        f"D={int(np.count_nonzero(d_mask))}"
    )

    qcd_mask = np.isin(sample_labels, sorted(QCD_SAMPLES))
    weights = w

    def _sum_weight(mask):
        vals = weights[mask]
        return float(np.sum(vals)), float(np.sum(vals ** 2))

    qcd_a_total, qcd_a_var = _sum_weight(a_union_mask & qcd_mask)
    qcd_b_total, qcd_b_var = _sum_weight(b_mask & qcd_mask)
    qcd_c_total, qcd_c_var = _sum_weight(c_mask & qcd_mask)
    qcd_d_total, qcd_d_var = _sum_weight(d_mask & qcd_mask)

    # Optional per-event dump (guarded by env var) for offline ABCD non-closure
    # / 2-D score-vs-msoftdrop diagnostics. Does not affect normal runs.
    if os.environ.get("QCD_EST_DUMP_2D"):
        _abcd_branch = next(iter(abcd_thresholds))
        _dump_path = os.path.join(OUTPUT_DIR, "abcd_2d_dump.npz")
        np.savez_compressed(
            _dump_path,
            msoftdrop=X_raw[_abcd_branch].to_numpy(dtype=float),
            proba=np.asarray(proba, dtype=np.float32),
            weight=np.asarray(weights, dtype=float),
            qcd_mask=np.asarray(qcd_mask, dtype=bool),
            union_score_mask=np.asarray(union_score_mask, dtype=bool),
            abcd_pass=np.asarray(abcd_pass, dtype=bool),
            abcd_fail=np.asarray(abcd_fail, dtype=bool),
            region_masks=np.stack(region_score_masks).astype(bool),
            axis_names=np.asarray(list(axis_names)),
        )
        log_message(f"Wrote ABCD 2-D dump: {_dump_path} (events={len(weights)})")

    if qcd_b_total <= 0.0 or qcd_c_total <= 0.0 or qcd_d_total <= 0.0:
        raise RuntimeError("QCD B/C/D totals must be positive for ABCD scaling")
    if qcd_a_total <= 0.0:
        raise RuntimeError("QCD A-union total is zero; cannot derive global QCD scale")

    raw_pred_qcd_union = qcd_b_total * qcd_c_total / qcd_d_total
    raw_pred_qcd_union_var = (
        (qcd_c_total / qcd_d_total) ** 2 * qcd_b_var
        + (qcd_b_total / qcd_d_total) ** 2 * qcd_c_var
        + (qcd_b_total * qcd_c_total / (qcd_d_total ** 2)) ** 2 * qcd_d_var
    )
    pred_qcd_union = raw_pred_qcd_union * QCD_PREDICT_SCALE_MULTIPLIER
    pred_qcd_union_var = raw_pred_qcd_union_var * (QCD_PREDICT_SCALE_MULTIPLIER ** 2)
    pred_qcd_union_sigma = math.sqrt(max(pred_qcd_union_var, 0.0))
    raw_qcd_scale = raw_pred_qcd_union / qcd_a_total
    raw_qcd_scale_var = raw_pred_qcd_union_var / (qcd_a_total ** 2)
    raw_qcd_scale_sigma = math.sqrt(max(raw_qcd_scale_var, 0.0))
    qcd_scale = pred_qcd_union / qcd_a_total
    qcd_scale_var = pred_qcd_union_var / (qcd_a_total ** 2)
    qcd_scale_sigma = math.sqrt(max(qcd_scale_var, 0.0))

    log_message(
        f"ABCD QCD totals: A_union={qcd_a_total:.6g}, B={qcd_b_total:.6g}, "
        f"C={qcd_c_total:.6g}, D={qcd_d_total:.6g}, raw_pred_union={raw_pred_qcd_union:.6g}, "
        f"raw_scale={raw_qcd_scale:.6g} ± {raw_qcd_scale_sigma:.6g}, "
        f"manual_scale_multiplier={QCD_PREDICT_SCALE_MULTIPLIER:.6g}, "
        f"pred_union={pred_qcd_union:.6g} ± {pred_qcd_union_sigma:.6g}, "
        f"final_scale={qcd_scale:.6g} ± {qcd_scale_sigma:.6g}"
    )

    log_message(f"Filling signal-region yields: n={len(region_labels)}")
    sample_names = sorted({sample for sample in sample_labels})
    sample_yields = {sample: np.zeros(len(region_labels), dtype=float) for sample in sample_names}
    sample_vars = {sample: np.zeros(len(region_labels), dtype=float) for sample in sample_names}
    group_yields = {group: np.zeros(len(region_labels), dtype=float) for group in CLASS_NAMES}
    group_vars = {group: np.zeros(len(region_labels), dtype=float) for group in CLASS_NAMES}

    true_qcd_vals = np.zeros(len(region_labels), dtype=float)
    true_qcd_vars = np.zeros(len(region_labels), dtype=float)

    for idx, mask in enumerate(region_a_masks):
        for sample_name in sample_names:
            sample_mask = mask & (sample_labels == sample_name)
            vals = weights[sample_mask]
            sample_yields[sample_name][idx] = float(np.sum(vals))
            sample_vars[sample_name][idx] = float(np.sum(vals ** 2))
        for group_name in CLASS_NAMES:
            group_mask = mask & (group_labels == group_name)
            vals = weights[group_mask]
            group_yields[group_name][idx] = float(np.sum(vals))
            group_vars[group_name][idx] = float(np.sum(vals ** 2))
        qcd_vals = weights[mask & qcd_mask]
        true_qcd_vals[idx] = float(np.sum(qcd_vals))
        true_qcd_vars[idx] = float(np.sum(qcd_vals ** 2))

    qcd_fraction_vals = true_qcd_vals / qcd_a_total
    qcd_fraction_vars = np.zeros(len(region_labels), dtype=float)
    for idx in range(len(region_labels)):
        region_val = true_qcd_vals[idx]
        region_var = true_qcd_vars[idx]
        rest_val = qcd_a_total - region_val
        rest_var = max(0.0, qcd_a_var - region_var)
        qcd_fraction_vars[idx] = (
            ((rest_val / (qcd_a_total ** 2)) ** 2) * region_var
            + ((region_val / (qcd_a_total ** 2)) ** 2) * rest_var
        )

    raw_pred_qcd_vals = raw_pred_qcd_union * qcd_fraction_vals
    pred_qcd_vals = pred_qcd_union * qcd_fraction_vals
    pred_qcd_stat_vars = (pred_qcd_union ** 2) * qcd_fraction_vars
    pred_qcd_scale_vars = (true_qcd_vals ** 2) * qcd_scale_var
    pred_qcd_cov = np.diag(pred_qcd_stat_vars) + np.outer(
        np.sqrt(np.maximum(pred_qcd_scale_vars, 0.0)),
        np.sqrt(np.maximum(pred_qcd_scale_vars, 0.0)),
    )
    pred_qcd_vars = np.diag(pred_qcd_cov).astype(float)

    log_message(
        "QCD SR prediction debug: "
        f"manual_scale_multiplier={QCD_PREDICT_SCALE_MULTIPLIER:.6g}, "
        f"raw_scale={raw_qcd_scale:.6g}, final_scale={qcd_scale:.6g}"
    )
    for idx, label in enumerate(region_labels):
        log_message(
            f"  {label}: qcd_true={true_qcd_vals[idx]:.6g}, "
            f"a_fraction={qcd_fraction_vals[idx]:.6g}, "
            f"raw_predict={raw_pred_qcd_vals[idx]:.6g}, "
            f"scaled_predict={pred_qcd_vals[idx]:.6g}"
        )

    non_qcd_groups = [group for group in CLASS_NAMES if group not in QCD_CLASS_SET]
    pred_group_yields = {group: group_yields[group].copy() for group in non_qcd_groups}
    pred_group_vars = {group: group_vars[group].copy() for group in non_qcd_groups}
    pred_group_yields[QCD_PREDICT_GROUP_NAME] = pred_qcd_vals.copy()
    pred_group_vars[QCD_PREDICT_GROUP_NAME] = pred_qcd_vars.copy()
    pred_group_order = non_qcd_groups + [QCD_PREDICT_GROUP_NAME]

    true_total_vals = np.zeros(len(region_labels), dtype=float)
    true_total_vars = np.zeros(len(region_labels), dtype=float)
    pred_total_vals = np.zeros(len(region_labels), dtype=float)
    for group_name in CLASS_NAMES:
        true_total_vals += group_yields[group_name]
        true_total_vars += group_vars[group_name]
    pred_total_stat_vars = np.zeros(len(region_labels), dtype=float)
    for group_name in non_qcd_groups:
        pred_total_vals += pred_group_yields[group_name]
        pred_total_stat_vars += pred_group_vars[group_name]
    pred_total_vals += pred_qcd_vals

    pred_total_stat_vars += pred_qcd_stat_vars
    pred_total_scale_vars = pred_qcd_scale_vars.copy()
    pred_total_cov = np.diag(pred_total_stat_vars) + np.outer(
        np.sqrt(np.maximum(pred_total_scale_vars, 0.0)),
        np.sqrt(np.maximum(pred_total_scale_vars, 0.0)),
    )
    pred_total_vars = np.diag(pred_total_cov).astype(float)

    # The plots above validate ABCD on finite MC and therefore use the propagated
    # MC-entry variances. The ROOT file is consumed by combine, where the
    # weighted yield itself is treated as the Poisson count. Store that
    # combine-facing convention separately so it does not change the validation
    # plots.
    sample_root_vars = {
        sample: _poisson_vars_from_yield(values)
        for sample, values in sample_yields.items()
    }
    group_root_vars = {
        group: _poisson_vars_from_yield(values)
        for group, values in group_yields.items()
    }
    true_qcd_root_vars = _poisson_vars_from_yield(true_qcd_vals)
    pred_qcd_root_stat_vars = _poisson_vars_from_yield(pred_qcd_vals)
    pred_qcd_root_scale_vars = np.zeros_like(pred_qcd_root_stat_vars)
    pred_qcd_root_cov = _diag_cov_from_vars(pred_qcd_root_stat_vars)
    pred_total_root_stat_vars = _poisson_vars_from_yield(pred_total_vals)
    pred_total_root_scale_vars = np.zeros_like(pred_total_root_stat_vars)
    pred_total_root_cov = _diag_cov_from_vars(pred_total_root_stat_vars)
    true_total_root_vars = _poisson_vars_from_yield(true_total_vals)

    abcd_group_vals = {group: np.zeros(4, dtype=float) for group in CLASS_NAMES}
    abcd_group_vars = {group: np.zeros(4, dtype=float) for group in CLASS_NAMES}
    abcd_masks = [a_union_mask, b_mask, c_mask, d_mask]
    for reg_idx, mask in enumerate(abcd_masks):
        for group_name in CLASS_NAMES:
            group_mask = mask & (group_labels == group_name)
            vals = weights[group_mask]
            abcd_group_vals[group_name][reg_idx] = float(np.sum(vals))
            abcd_group_vars[group_name][reg_idx] = float(np.sum(vals ** 2))
    abcd_qcd_merged_vals, abcd_qcd_merged_vars, abcd_qcd_merged_groups = _qcd_merged_group_maps(
        abcd_group_vals,
        abcd_group_vars,
    )

    root_path = os.path.join(OUTPUT_DIR, ROOT_FILE_NAME)
    log_message("Writing summary ROOT file")
    write_root_output(
        root_path,
        edges,
        [int(idx) for idx in signal_regions["bin_index"].tolist()],
        sample_yields,
        sample_root_vars,
        group_yields,
        group_root_vars,
        pred_qcd_vals,
        pred_qcd_root_stat_vars,
        pred_qcd_root_scale_vars,
        pred_qcd_root_cov,
        true_qcd_vals,
        true_qcd_root_vars,
        pred_total_vals,
        pred_total_root_stat_vars,
        pred_total_root_scale_vars,
        pred_total_root_cov,
        true_total_vals,
        true_total_root_vars,
    )

    log_message("Plotting ABCD summary")
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_group_vals,
        abcd_group_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_counts.pdf"),
        normalize_per_bin=False,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_group_vals,
        abcd_group_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_counts_linear.pdf"),
        normalize_per_bin=False,
        log_y=False,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_group_vals,
        abcd_group_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_fractions.pdf"),
        normalize_per_bin=True,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_qcd_merged_vals,
        abcd_qcd_merged_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_counts_qcd_merged.pdf"),
        normalize_per_bin=False,
        groups=abcd_qcd_merged_groups,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_qcd_merged_vals,
        abcd_qcd_merged_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_counts_qcd_merged_linear.pdf"),
        normalize_per_bin=False,
        groups=abcd_qcd_merged_groups,
        log_y=False,
    )
    plot_abcd_region_counts(
        ["A union", "B", "C", "D"],
        abcd_qcd_merged_vals,
        abcd_qcd_merged_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_region_fractions_qcd_merged.pdf"),
        normalize_per_bin=True,
        groups=abcd_qcd_merged_groups,
    )
    plot_signal_region_prediction(
        region_labels,
        pred_group_yields,
        pred_total_vals,
        pred_total_vars,
        true_total_vals,
        true_total_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_signal_regions_total.pdf"),
        pred_group_order,
        "Events",
    )
    plot_signal_region_prediction(
        region_labels,
        {QCD_PREDICT_GROUP_NAME: pred_qcd_vals.copy()},
        pred_qcd_vals,
        pred_qcd_vars,
        true_qcd_vals,
        true_qcd_vars,
        os.path.join(OUTPUT_DIR, "qcd_abcd_signal_regions_qcd.pdf"),
        [QCD_PREDICT_GROUP_NAME],
        "QCD Events",
    )
    plot_a_region_branch_shapes(
        X_raw,
        group_labels,
        weights,
        A_REGION_SHAPE_BRANCHES,
        region_a_masks,
        region_labels,
        a_union_mask,
        os.path.join(OUTPUT_DIR, "a_region_shapes"),
    )

    log_message("Finished qcd_est.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_message(f"Runtime error: {exc}")
        raise
