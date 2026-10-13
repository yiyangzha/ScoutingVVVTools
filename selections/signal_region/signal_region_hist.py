"""Histogram-based coarse-to-fine signal-region optimizer (run.py mode 3).

It finds N non-overlapping two-sided score rectangles. Each region is rated with
the Asimov significance with a background uncertainty (G. Cowan, "Discovery
sensitivity for a counting experiment with background uncertainty"; sr.asimov_z)

    Z_A = sqrt(2 [(S+B) ln((S+B)(B+V) / (B^2+(S+B)V)) - (B^2/V) ln(1 + V S / (B (B+V)))])

whose background B and variance V follow the combine cards:

  * B is the background MC; with abcd_uncertainty (ABCD mode, k-fold models) its
    QCD part is the prediction from the region's control-region events,
    C_box x TF, as qcd_est.py abcd_mode per_sr and combine's control-region fit.
  * V = sum_k (sum_c delta_kc B_c)^2 + sum w^2 + floor (+ ABCD). delta_kc is the
    signed relative shift of background class c under nuisance row k of the cards:
    lumi, trigger, theory_pdf, theory_ps_isr, theory_ps_fsr, pileup, JES, JER,
    JMS, JMR (each correlated across the classes; the inclusive ratios of the
    configured *_syst_yields.json files, yield-weighted over the class's samples)
    and one theory_scale_<class> and one xsec_<class> row per class (combine.C's
    per-process rows; xsec_unc_json). sum w^2 is the MC statistics of the
    background MC; the optional zero-count floor (zero_count_floor) adds alpha_c^2
    for every background class c without MC events (or with a non-positive sum) in
    the region, as combine's `gmN 0 alpha` rows. In ABCD mode the QCD has no shifts,
    MC statistics or floor; its variance TF^2 x (control count + non-QCD MC w^2 in
    the control box) + (nonclosure x predicted QCD)^2 is added instead.

Steps:

  1. Coarse pass: coordinate-descent beam search on a uniform grid (a finer grid
     on the QCD axis), repeated with disjoint re-seeding.
  2. Deduplication: every candidate is shrunk to the smallest box holding the same
     scan events (and, in ABCD mode, control events), and candidates holding
     identical events are merged.
  3. Selection: branch-and-bound (OpenMP helper openmp_region_select.cpp, or
     Python) picking N mutually non-overlapping rectangles that maximise
     sum(Z_A,i^2); a stopped search reports its upper-bound certificate.
  4. Joint stage: one region at a time is replaced by the compatible candidate of
     highest Z_A that most increases the joint significance Z_joint of the set,
     a profile-likelihood Asimov fit of all regions with the correlated nuisance
     rows above (and the ABCD non-closure correlated across the regions), so the
     correlations of the systematics and their constraint by other regions enter
     the choice (joint_z).
  5. Fine pass: refine each selected boundary on the fine grid, locally, with the
     full acceptance constraints, kept only when it does not lower Z_joint.
  6. Boundaries inside empty score gaps: each region is shrunk to its events
     again, then signal-class axis upper bounds are raised and background-class
     axis lower bounds lowered into space without scan or control events, so
     signal-axis bounds sit as high and background-axis bounds as low as the same
     events allow, without overlapping another region.

Data loading, model inference, per-bin statistics layout, text reporting and all
plotting are reused from ``signal_region.py`` (imported as ``sr``).

The script and ``signal_region.py`` share one config file: the first
command-line argument, else ``SR_HIST_CONFIG_PATH``, else ``config.json`` next
to this file.
"""

import os
import sys
import json
import gc
import time
import ctypes
import fnmatch
import hashlib
import subprocess

# -- Config sharing: signal_region.py loads its config at import time from
#    SCAN_CONFIG_PATH and never reloads, so we must set it BEFORE importing. We
#    point it at our own config so both modules read identical shared keys
#    (bdt_root / score_axes / lumi / min_bkg_weight / ...). This script then
#    reopens the same file to read its extra histogram-search keys.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    HIST_CFG_PATH = os.path.abspath(sys.argv[1])
else:
    HIST_CFG_PATH = os.path.abspath(
        os.environ.get("SR_HIST_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
    )
os.environ["SCAN_CONFIG_PATH"] = HIST_CFG_PATH

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

import signal_region as sr  # triggers module-level config load from HIST_CFG_PATH


# -------------------- Histogram-search config --------------------
_hist_cfg = sr._load_json(HIST_CFG_PATH)

COARSE_W       = float(_hist_cfg.get("coarse_bin_width", 0.05))
FINE_W         = float(_hist_cfg.get("fine_bin_width", 0.01))
REFINE_WIN     = float(_hist_cfg.get("fine_refine_window", 0.05))
BEAM_WIDTH     = max(1, int(_hist_cfg.get("beam_width", 64)))
TOP_K          = max(1, int(_hist_cfg.get("top_intervals_per_axis", 8)))
ROUNDS         = max(1, int(_hist_cfg.get("coordinate_rounds", 6)))
FINE_PASSES    = max(1, int(_hist_cfg.get("fine_refine_passes", 2)))
GLOBAL_BEAM    = max(1, int(_hist_cfg.get("global_beam_width", 512)))
BNB_MAX_NODES  = max(0, int(_hist_cfg.get("branch_bound_max_nodes", 0)))
BNB_TIME_LIMIT = max(0.0, float(_hist_cfg.get("branch_bound_time_limit_seconds", 14400.0)))
MAX_THREADS    = max(1, int(_hist_cfg.get("max_threads", sr.MAX_THREADS)))
PROGRESS_EVERY = float(_hist_cfg.get("progress_every_seconds", 30.0))
VALIDATE_REF   = bool(_hist_cfg.get("validate_prediction_reference", True))

# Background systematic uncertainties of the objective: the inclusive (no signal
# region) *_syst_yields.json of modes 8, 9, 11-14 for this tree, plus the flat
# lumi and trigger fractions of the combine card.
BKG_SYST_JSONS = dict(_hist_cfg["background_syst_jsons"])
LUMI_UNC       = float(_hist_cfg["lumi_unc"])
TRIGGER_UNC    = float(_hist_cfg["trigger_unc"])
# Per-sample cross-section uncertainties of combine.C's xsec_<process> rows (fnmatch patterns).
XSEC_UNC_JSON  = str(_hist_cfg["xsec_unc_json"])
SYST_RATIO_KEYS = {
    "theory": (("pdf_up", "pdf_down"), ("scale_up", "scale_down"),
               ("ps_isr_up", "ps_isr_down"), ("ps_fsr_up", "ps_fsr_down")),
    "pileup": (("pu_up", "pu_down"),),
    "jes": (("jes_up", "jes_down"),),
    "jer": (("jer_up", "jer_down"),),
    "jms": (("jms_up", "jms_down"),),
    "jmr": (("jmr_up", "jmr_down"),),
}

# Optional finer grid just for the QCD axis (its score is sharply peaked near 0,
# and the exhaustive box search separates signal-rich bins with QCD cuts far
# below what the shared COARSE_W/FINE_W grid can resolve). Defaults fall back
# to the shared widths, so configs that don't set these are unaffected.
QCD_COARSE_W   = float(_hist_cfg.get("qcd_coarse_bin_width", COARSE_W))
QCD_FINE_W     = float(_hist_cfg.get("qcd_fine_bin_width", FINE_W))
QCD_REFINE_WIN = float(_hist_cfg.get("qcd_fine_refine_window", REFINE_WIN))

# Optional axis merging: sum two or more scanned class-score axes into a single
# derived discriminant before the search runs (e.g. p(VV)+p(VJets)), reducing
# the search dimensionality by one per merge. Config: a list of name-lists,
# e.g. "merge_score_axes": [["VV", "VJets"]]. Names refer to the axes already
# selected by the shared "score_axes" config key (sr.SCORE_AXIS_NAMES).
MERGE_AXES = list(_hist_cfg.get("merge_score_axes", []))


def _resolve_axes():
    """Axis names/underlying-class-index-groups after applying MERGE_AXES.

    Returns (axis_names, index_groups): index_groups[d] is the list of
    sr.CLASS_NAMES indices whose probabilities are summed to form scan axis d.
    Unmerged axes get a single-element group. Order follows the original
    sr.SCORE_AXIS_NAMES order (merged axes take the position of their first
    member).
    """
    base_names = list(sr.SCORE_AXIS_NAMES)
    base_indices = list(sr.SCORE_AXIS_INDICES)
    name_to_pos = {n: i for i, n in enumerate(base_names)}

    groups = []  # (label, [class_indices], sort_key)
    consumed = set()
    for merge_set in MERGE_AXES:
        positions = []
        for nm in merge_set:
            if nm not in name_to_pos:
                raise KeyError(
                    f"merge_score_axes name {nm!r} not in scanned axes {base_names}"
                )
            positions.append(name_to_pos[nm])
        if any(p in consumed for p in positions):
            raise ValueError(f"merge_score_axes group {merge_set!r} overlaps another group")
        consumed.update(positions)
        label = "+".join(merge_set)
        groups.append((label, [base_indices[p] for p in positions], min(positions)))

    for i, nm in enumerate(base_names):
        if i not in consumed:
            groups.append((nm, [base_indices[i]], i))

    groups.sort(key=lambda g: g[2])
    axis_names = [g[0] for g in groups]
    index_groups = [g[1] for g in groups]
    return axis_names, index_groups

# Reused constants / thresholds from signal_region.py.
MIN_BKG_WEIGHT     = sr.MIN_BKG_WEIGHT
MIN_SIGNAL_WEIGHT  = sr.MIN_SIGNAL_WEIGHT
MIN_SIGNAL_ENTRIES = sr.MIN_SIGNAL_ENTRIES
MIN_BKG_ENTRIES    = sr.MIN_BKG_ENTRIES
# Zero-count floor: in a region where a background class has no MC events (or a non-positive
# weighted sum), its typical event weight alpha is added in quadrature to sigma_B, as combine's
# `gmN 0 alpha` rows do for an empty process, so regions that are free of a class's MC events
# only by chance (e.g. heavy QCD events) are not rated as background-free.
ZERO_COUNT_FLOOR   = bool(sr.scan_cfg.get("zero_count_floor", False))
# Samples left out of the scan (exclude_samples, default none), as in qcd_est.py: e.g. the
# HT < 400 GeV QCD samples, whose few very heavy events set the ABCD transfer factor.
EXCLUDE_SAMPLES    = set(sr.scan_cfg.get("exclude_samples", []))
# ABCD-aware background uncertainty (abcd_uncertainty, default off; k-fold models only): QCD is
# predicted per SR from its control region, A = C x TF (qcd_est.py abcd_mode per_sr, combine's
# control-region fit), so in every box the QCD part of sigma_B^2 is TF^2 x (C + sum w^2 of the
# non-QCD background MC in C), C being the expected control-region count (every process), plus
# (nonclosure x QCD)^2, instead of the QCD MC statistics; QCD gets no zero-count floor.
#   {"branch": <ABCD branch, a threshold of selection.json>, "control_window": [low, high]
#    (exclusive, in the branch's fail region), "nonclosure": <fraction>}
ABCD_CFG = sr.scan_cfg.get("abcd_uncertainty")
if ABCD_CFG is not None:
    ABCD_BRANCH = str(ABCD_CFG["branch"])
    ABCD_CONTROL = tuple(float(v) for v in ABCD_CFG["control_window"])
    ABCD_NONCLOSURE = float(ABCD_CFG["nonclosure"])
    if len(ABCD_CONTROL) != 2 or not ABCD_CONTROL[0] < ABCD_CONTROL[1] or ABCD_NONCLOSURE < 0.0:
        raise ValueError(f"abcd_uncertainty needs control_window [low, high] with low < high and "
                         f"nonclosure >= 0, got {ABCD_CFG}")
    if not sr.KFOLD:
        raise RuntimeError("abcd_uncertainty needs a k-fold bdt_root (the score cache holds the "
                           "control-region events)")
EPS = 1e-12

log_message = sr.log_message
log_warning = sr.log_warning
log_info = sr.log_info


# -------------------- Input preparation (reuses sr.*) --------------------
def prepare_inputs():
    """Reproduce sr.main()'s preamble to obtain (proba, y, w, sample_labels, feats).

    Mirrors signal_region.py:main lines ~2342-2417 using only the importable
    helpers, so the test events, weights, feature standardisation and model
    inference are identical to the original tool.
    """
    os.makedirs(sr.OUTPUT_DIR, exist_ok=True)
    log_message(
        f"Running signal_region_hist.py: tree={sr.TREE_NAME}, lumi={sr.LUMI} fb^-1, "
        f"n_signal_regions={sr.N_SIGNAL_REGIONS}, bdt_root={sr.BDT_ROOT}, "
        f"output_dir={sr.OUTPUT_DIR}"
    )

    sel         = sr.sel_cfg[sr.TREE_NAME]
    branches    = [b["name"] for b in sr.br_cfg[sr.TREE_NAME]]
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_tf      = sel.get("log_transform", [])
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    decorrelate = sr.cfg.get(sr.TREE_NAME, {}).get("decorrelate", [])

    extra_cols = []
    for c in list(thresholds.keys()) + list(decorrelate):
        if c not in branches and c not in extra_cols:
            extra_cols.append(c)
    load_cols = branches + extra_cols
    drop_after_filter = [c for c in extra_cols if c not in decorrelate]

    log_message(f"SR optimisation split: {sr.OPT_SPLIT}")
    if sr.KFOLD and ABCD_CFG is not None:
        return load_kfold_with_control(thresholds)
    if sr.KFOLD:
        proba, y, w, sample_labels = sr.load_kfold_inputs(thresholds)
        proba, y, w, sample_labels = _exclude_samples(proba, y, w, sample_labels)
        return proba, y, w, sample_labels, [], None
    df_all = sr.load_split_data(load_cols, sr.OPT_SPLIT)
    X             = df_all[load_cols].copy()
    y             = df_all["class_idx"].values.astype(int)
    w             = df_all["weight"].values.astype(float)
    sample_labels = df_all["sample_name"].values
    del df_all
    gc.collect()

    log_message("Applying thresholds")
    X, y, w, sample_labels = sr.filter_X(
        X, y, w, load_cols, thresholds, apply_to_sentinel=True, sample_labels=sample_labels
    )
    log_message(f"After filtering: {len(X)} events")

    log_message("Standardising features")
    X = sr.standardize_X(X, clip_ranges, log_tf)

    if drop_after_filter:
        X = X.drop(columns=drop_after_filter, errors="ignore")

    all_feature_names = list(X.columns)
    if decorrelate:
        name_to_idx = {c: i for i, c in enumerate(all_feature_names)}
        decor_idx   = sorted(name_to_idx[k] for k in decorrelate if k in name_to_idx)
        keep_idx    = [i for i in range(len(all_feature_names)) if i not in decor_idx]
        X_model     = X.iloc[:, keep_idx]
        log_message(f"Removed decorrelated features: {decorrelate}")
    else:
        X_model = X

    model_base = sr.MODEL_PATTERN.format(output_root=sr.BDT_ROOT, tree_name=sr.TREE_NAME)
    clf = sr._shared_load_model(model_base, sr.cfg, sr.NUM_CLASSES, log_message=log_message)

    log_message("Running model prediction")
    proba = sr._predict_model_proba(clf, X_model)
    log_message(f"Predicted probabilities shape: {proba.shape}")

    if VALIDATE_REF:
        log_message(f"Validating {sr.OPT_SPLIT}-set prediction reference")
        sr._compare_prediction_reference(
            sr.OPT_REFERENCE_SIGNAL_REGION,
            X_model.columns if hasattr(X_model, "columns")
            else [f"f{i}" for i in range(X_model.shape[1])],
            sample_labels, y, w, proba,
        )

    proba, y, w, sample_labels = _exclude_samples(proba, y, w, sample_labels)
    return proba, y, w, sample_labels, list(X_model.columns), None


def _exclude_samples(proba, y, w, sample_labels):
    """Drop the events of EXCLUDE_SAMPLES (no-op without them)."""
    if not EXCLUDE_SAMPLES:
        return proba, y, w, sample_labels
    keep = ~np.isin(np.asarray(sample_labels, dtype=str), sorted(EXCLUDE_SAMPLES))
    log_message(f"Excluding samples {sorted(EXCLUDE_SAMPLES)}: {int(np.count_nonzero(~keep))} events "
                f"(weight {float(np.asarray(w)[~keep].sum()):.6g})")
    return proba[keep], np.asarray(y)[keep], np.asarray(w)[keep], np.asarray(sample_labels)[keep]


def load_kfold_with_control(thresholds):
    """k-fold score cache with the ABCD control region: the events passing every threshold (the
    scan events, as sr.load_kfold_inputs) and the events passing every threshold but the ABCD
    branch's that lie in its control window inside its fail region (qcd_est.py per_sr)."""
    if ABCD_BRANCH not in thresholds:
        raise KeyError(f"abcd_uncertainty branch {ABCD_BRANCH} has no threshold in selection.json")
    models = sr.kfold.KFoldModels(sr.BDT_ROOT, log=log_message)
    samples = [s for members in sr.CLASS_GROUPS.values() for s in members]
    others = {k: v for k, v in thresholds.items() if k != ABCD_BRANCH}
    log_message(f"Loading the k-fold score cache: {models.k} folds, {len(samples)} samples "
                f"(every threshold but {ABCD_BRANCH})")
    scores = sr.kfold.load_scores(models, samples, branches=[ABCD_BRANCH], thresholds=others, log=log_message)
    v = np.asarray(scores["columns"][ABCD_BRANCH], dtype=float)
    passed = sr.kfold.threshold_mask({ABCD_BRANCH: v}, {ABCD_BRANCH: thresholds[ABCD_BRANCH]})
    control = ~passed & (v > ABCD_CONTROL[0]) & (v < ABCD_CONTROL[1])
    w = scores["weight"] * sr.LUMI
    y = scores["class_idx"].astype(int)
    labels = scores["sample_name"]
    proba, yA, wA, lA = _exclude_samples(scores["proba"][passed], y[passed], w[passed], labels[passed])
    pC, yC, wC, _ = _exclude_samples(scores["proba"][control], y[control], w[control], labels[control])
    log_message(f"After thresholds: {len(wA)} events (all MC entries, k-fold scores); control window "
                f"{ABCD_CONTROL[0]:g} < {ABCD_BRANCH} < {ABCD_CONTROL[1]:g}: {len(wC)} events")
    return proba, yA, wA, lA, [], {"proba": pC, "y": yC, "w": wC}


# -------------------- Background systematic shifts --------------------
def _resolve_cfg_path(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_SCRIPT_DIR, path))


def _sample_xsec_unc(xsec_unc, sample):
    """Cross-section uncertainty of a sample: the value of the xsec_unc_json pattern(s) matching
    it (fnmatch, as combine.C's sampleXsecUnc); every matching pattern must give the same value."""
    value = None
    for pattern, unc in xsec_unc.items():
        if not fnmatch.fnmatchcase(sample, pattern):
            continue
        if value is not None and float(unc) != value:
            raise ValueError(f"{XSEC_UNC_JSON}: patterns with different values match sample '{sample}'")
        value = float(unc)
    if value is None:
        raise KeyError(f"{XSEC_UNC_JSON} has no cross-section uncertainty for sample '{sample}'")
    return value


def background_syst_shifts(y, w, sample_labels, qcd_data_driven):
    """Signed relative shifts delta[k, c] of the background classes c (columns follow
    sr.BACKGROUND_CLASS_INDICES) under the nuisances k of the combine cards, and the row names.

    Rows: lumi, trigger, theory_pdf, theory_scale_<class> (one per class, as combine's
    per-process QCD-scale rows), theory_ps_isr, theory_ps_fsr, pileup, jes, jer, jms, jmr
    (each correlated across the classes), and xsec_<class> (one per class, combine.C's
    xsec_<process> rows). The class ratio of a ratio nuisance is the yield-weighted mean of its
    samples' inclusive ratios (the combine kappa convention; a sample without theory weights
    counts with ratio 1 under the theory nuisances; a sample without an entry is skipped when
    its yield is not positive, otherwise an error). With u = R_up - 1 and d = R_down - 1,
    delta = u if |u| >= |d| else -d: the larger deviation, signed along the up variation. The
    xsec shift of a class is the mean of its samples' xsec_unc_json values weighted by their
    positive yields (combine.C writeXsecLnN). With qcd_data_driven (ABCD mode) the QCD classes
    get no shift, as the data-driven QCD of the cards has none of these rows. Rows without any
    shift are dropped. Yields are the scan weights after the thresholds.
    """
    missing = [name for name in SYST_RATIO_KEYS if name not in BKG_SYST_JSONS]
    if missing:
        raise KeyError(f"background_syst_jsons lacks {missing}")
    ratios = {name: sr._load_json(_resolve_cfg_path(BKG_SYST_JSONS[name])) for name in SYST_RATIO_KEYS}
    xsec_unc = sr._load_json(_resolve_cfg_path(XSEC_UNC_JSON))["uncertainties"]
    labels = np.asarray(sample_labels)
    classes = list(sr.BACKGROUND_CLASS_INDICES)
    n_cls = len(classes)
    ratio_nuisances = [(("theory_" + up_key[:-3]) if name == "theory" else name, name, up_key, down_key)
                       for name, pairs in SYST_RATIO_KEYS.items() for up_key, down_key in pairs]
    deltas = {nuis: np.zeros(n_cls) for nuis, _name, _up, _down in ratio_nuisances}
    xsec = np.zeros(n_cls)
    flat = np.ones(n_cls)
    for c, cls_idx in enumerate(classes):
        cls_name = sr.CLASS_NAMES[cls_idx]
        if qcd_data_driven and "qcd" in cls_name.lower():
            flat[c] = 0.0
            continue
        yields = {}
        for sample in sr.CLASS_GROUPS[cls_name]:
            m = labels == sample
            if np.any(m):
                yields[sample] = float(w[m].sum())
        total = float(sum(yields.values()))
        if total <= 0.0:
            log_warning(f"Background class {cls_name} has no positive yield; lumi/trigger shifts only")
            continue
        for nuis, name, up_key, down_key in ratio_nuisances:
            r_up = r_down = 0.0
            for sample, y_s in yields.items():
                entry = ratios[name].get(sample, {}).get(sr.TREE_NAME)
                if entry is None:
                    if name == "theory" and not sr.SAMPLE_INFO[sample]["rule"].get("has_theory_weights", False):
                        entry = {up_key: 1.0, down_key: 1.0}
                    elif y_s <= 0.0:
                        # The syst scripts write no entry for a non-positive signed sum;
                        # such a sample adds no positive yield (combine's convention).
                        continue
                    else:
                        raise KeyError(
                            f"{BKG_SYST_JSONS[name]} has no entry for sample {sample}, tree {sr.TREE_NAME}"
                        )
                r_up += y_s * float(entry[up_key])
                r_down += y_s * float(entry[down_key])
            u = r_up / total - 1.0
            d = r_down / total - 1.0
            deltas[nuis][c] = u if abs(u) >= abs(d) else -d
        positive = {sample: y_s for sample, y_s in yields.items() if y_s > 0.0}
        if positive:
            xsec[c] = (sum(y_s * _sample_xsec_unc(xsec_unc, sample) for sample, y_s in positive.items())
                       / sum(positive.values()))
        log_message(
            f"  Background shifts {cls_name}: lumi={LUMI_UNC:+.4f}, trigger={TRIGGER_UNC:+.4f}, "
            + ", ".join(f"{nuis}={deltas[nuis][c]:+.4f}" for nuis, _n, _u, _d in ratio_nuisances)
            + f", xsec={xsec[c]:+.4f}"
        )
    rows, names = [], []

    def _add(name, vec):
        if np.any(vec != 0.0):
            rows.append(np.asarray(vec, dtype=float))
            names.append(name)

    def _one_class(c, value):
        vec = np.zeros(n_cls)
        vec[c] = value
        return vec

    _add("lumi", LUMI_UNC * flat)
    _add("trigger", TRIGGER_UNC * flat)
    for nuis, _name, _up, _down in ratio_nuisances:
        if nuis == "theory_scale":
            for c, cls_idx in enumerate(classes):
                _add(f"theory_scale_{sr.CLASS_NAMES[cls_idx]}", _one_class(c, deltas[nuis][c]))
        else:
            _add(nuis, deltas[nuis])
    for c, cls_idx in enumerate(classes):
        _add(f"xsec_{sr.CLASS_NAMES[cls_idx]}", _one_class(c, xsec[c]))
    for name, vec in zip(names, rows):
        if np.any(vec <= -1.0):
            raise ValueError(f"Background shift {name} removes a whole class yield: {vec}")
    log_message(f"  Background nuisance rows ({len(names)}): {', '.join(names)}")
    shifts = np.vstack(rows) if rows else np.zeros((0, n_cls))
    return shifts, names


# -------------------- Scan context --------------------
class Ctx:
    """Per-run event arrays and precomputed reference tables for the scan."""

    def __init__(self, proba, y, w, bkg_shifts, control=None):
        n_cls = int(proba.shape[1])
        if n_cls != sr.NUM_CLASSES:
            raise RuntimeError(f"Model returned {n_cls} classes, expected {sr.NUM_CLASSES}")

        self.y = np.asarray(y, dtype=int)
        self.w = np.asarray(w, dtype=float)
        axis_names, index_groups = _resolve_axes()
        self.score_axes = np.column_stack(
            [proba[:, idxs].sum(axis=1) for idxs in index_groups]
        )  # (N, D)
        self.n_events, self.D = self.score_axes.shape
        self.axis_names = axis_names
        # A scan axis is a signal-score axis when every class it sums is a signal class.
        self.signal_axis = [all(i in sr.SIGNAL_CLASS_INDICES for i in idxs) for idxs in index_groups]

        self.is_sig = np.isin(self.y, sr.SIGNAL_CLASS_INDICES)
        self.is_bkg = np.isin(self.y, sr.BACKGROUND_CLASS_INDICES)
        self.w_sig = np.where(self.is_sig, self.w, 0.0)
        self.w_bkg = np.where(self.is_bkg, self.w, 0.0)  # every background MC event (reports)
        # ABCD mode (control = the control-region events, see ABCD_CFG): as in qcd_est.py per_sr
        # and combine's control-region fit, the QCD of a box is predicted from the control events
        # of the same box, C_box x TF, and the QCD classes (names containing "qcd") leave the
        # MC part of the background, its MC statistics, the zero-count floor and the systematic
        # shifts; the variance of the prediction (control statistics, non-closure) is added.
        self.abcd = control is not None
        self.qcd_classes = [c for c in sr.BACKGROUND_CLASS_INDICES if "qcd" in sr.CLASS_NAMES[c].lower()]
        self.is_qcd = np.isin(self.y, self.qcd_classes)
        # Background MC of the objective: in ABCD mode without the QCD.
        self.w_bkg_fit = np.where(self.is_bkg & ~self.is_qcd, self.w, 0.0) if self.abcd else self.w_bkg
        self.w2_bkg = self.w_bkg_fit * self.w_bkg_fit  # MC statistics of the objective's MC part
        if self.abcd:
            if not self.qcd_classes:
                raise RuntimeError("abcd_uncertainty needs a QCD class in class_groups")
            cy = np.asarray(control["y"], dtype=int)
            cw = np.asarray(control["w"], dtype=float)
            self.c_axes = np.column_stack([control["proba"][:, idxs].sum(axis=1) for idxs in index_groups])
            c_qcd = np.isin(cy, self.qcd_classes)
            c_mc_bkg = np.isin(cy, sr.BACKGROUND_CLASS_INDICES) & ~c_qcd
            # Per control event: its expected count (Poisson variance of the control region, every
            # process) plus w^2 for the non-QCD background MC (their MC statistics in the fit).
            self.c_var = cw + np.where(c_mc_bkg, cw * cw, 0.0)
            # Transfer factor window / control window of the QCD MC (qcd_est's B/D in complement
            # mode; the SR union holds a negligible part of the QCD).
            c_qcd_total = float(cw[c_qcd].sum())
            if c_qcd_total <= 0.0:
                raise RuntimeError(
                    f"abcd_uncertainty: no positive QCD yield in the control window {ABCD_CONTROL} of "
                    f"{ABCD_BRANCH}; the transfer factor is undefined"
                )
            self.abcd_tf = float(self.w[self.is_qcd].sum()) / c_qcd_total
            self.abcd_tf2 = self.abcd_tf ** 2
            self.abcd_nc2 = ABCD_NONCLOSURE ** 2
            # Predicted QCD in the signal window of every control event: its QCD weight x TF.
            self.c_qcd_pred = np.where(c_qcd, cw, 0.0) * self.abcd_tf
        else:
            self.c_axes = np.zeros((0, self.D))
        # Geometry points: the scan events and, in ABCD mode, the control events, whose boxes set
        # the QCD prediction; the deduplication, the boundary placement and the overlap check
        # keep both sets.
        self.geo_axes = np.vstack([self.score_axes, self.c_axes]) if self.abcd else self.score_axes
        self.geo_index = _AxisIndex(self.geo_axes)
        # Zero-count floor: per background class, its event mask and alpha^2, alpha = the typical
        # event weight (median |w|) of the class over all scan events, like qcd_est's
        # mean_event_weight for combine (the mean is dominated by a few very heavy events).
        self.floor_classes = [c for c in sr.BACKGROUND_CLASS_INDICES if not (self.abcd and c in self.qcd_classes)]
        self.bkg_class_masks = [self.y == c for c in self.floor_classes]
        self.floor_alpha2 = [
            (float(np.median(np.abs(self.w[mc]))) ** 2) if mc.any() else 0.0 for mc in self.bkg_class_masks
        ]
        self.S_total = float(self.w_sig.sum())
        self.B_total = float(self.w_bkg.sum())
        # Per background class weights of the objective's MC part (rows follow
        # sr.BACKGROUND_CLASS_INDICES; the QCD rows are empty in ABCD mode) and the signed
        # relative shifts of every nuisance row (nuisance x class).
        self.w_bkg_cls = np.vstack(
            [np.where(self.y == c, self.w_bkg_fit, 0.0) for c in sr.BACKGROUND_CLASS_INDICES]
        )
        self.bkg_shifts = np.asarray(bkg_shifts, dtype=float)

        # 1-D tail reference tables for per-bin tail efficiencies
        # (signal_region.py lines ~993-1011, T_REF=200, p_exp=0.005).
        self.T_REF = 200
        p_exp = 0.005
        self.thr_1d = np.clip(np.linspace(0.0, 1.0, self.T_REF) ** p_exp, 0.0, 1.0)
        self.S_tail_by_dim = np.zeros((self.D, self.T_REF))
        self.B_tail_by_dim = np.zeros((self.D, self.T_REF))
        for d in range(self.D):
            s = self.score_axes[:, d]
            order = np.argsort(s)
            s_sorted = s[order]
            cw_sig = np.cumsum(self.w_sig[order])
            cw_bkg = np.cumsum(self.w_bkg[order])
            idx = np.searchsorted(s_sorted, self.thr_1d, side="left")
            self.S_tail_by_dim[d] = (cw_sig[-1] if cw_sig.size else 0.0) - np.where(
                idx > 0, cw_sig[np.clip(idx - 1, 0, cw_sig.size - 1)], 0.0
            )
            self.B_tail_by_dim[d] = (cw_bkg[-1] if cw_bkg.size else 0.0) - np.where(
                idx > 0, cw_bkg[np.clip(idx - 1, 0, cw_bkg.size - 1)], 0.0
            )

        # Forbidden boxes (set per coarse pass for disjoint re-seeding).
        self.forbidden = []

        # Progress throttle.
        self._t0 = time.monotonic()
        self._last = [self._t0]

    def elapsed(self):
        return time.monotonic() - self._t0

    def progress(self, message, force=False):
        now = time.monotonic()
        if force or PROGRESS_EVERY <= 0.0 or now - self._last[0] >= PROGRESS_EVERY:
            log_message(f"  [{self.elapsed():.1f}s] {message}")
            self._last[0] = now


# -------------------- Significance --------------------
def za2(S, B, V):
    """Elementwise Z_A^2 with background variance V (sr.asimov_z, Cowan's Asimov
    significance with a background uncertainty); the statistics-only
    2[(S+B)ln(1+S/B) - S] where V is negligible; 0 where S <= 0 or B <= 0."""
    return np.asarray(sr.asimov_z(S, B, V), dtype=float) ** 2


def calc_Z_val(S, B, V=0.0):
    return float(sr.asimov_z(S, B, V))


def calc_Z(S, B, V, sS, sB):
    """Z_A and its error from the S and B statistical errors (numerical derivatives
    at fixed V)."""
    Z = calc_Z_val(S, B, V)
    if Z <= 0.0:
        return 0.0, 0.0
    hS = 1e-6 * max(abs(S), 1e-12)
    hB = 1e-6 * max(abs(B), 1e-12)
    dZ_dS = (calc_Z_val(S + hS, B, V) - calc_Z_val(S - hS, B, V)) / (2.0 * hS)
    dZ_dB = (calc_Z_val(S, B + hB, V) - calc_Z_val(S, B - hB, V)) / (2.0 * hB)
    sZ = float(np.sqrt((dZ_dS * sS) ** 2 + (dZ_dB * sB) ** 2))
    return Z, sZ


def floor_variance(ctx, m):
    """Zero-count floor added to the background variance for the region of event mask m
    (0 when disabled)."""
    if not ZERO_COUNT_FLOOR:
        return 0.0
    var = 0.0
    for mc, alpha2 in zip(ctx.bkg_class_masks, ctx.floor_alpha2):
        mr = m & mc
        if not mr.any() or float(ctx.w[mr].sum()) <= 0.0:
            var += alpha2
    return var


def abcd_qcd(ctx, lo, hi):
    """(QCD predicted in the box [lo, hi), control-region variance of the box): the predicted
    QCD is TF x the QCD of the box's control events (qcd_est.py per_sr A_i = C_i x B/D); the
    variance is the expected control count of every process plus the non-QCD MC w^2."""
    cm = box_mask(ctx.c_axes, lo, hi)
    return float(ctx.c_qcd_pred[cm].sum()), float(ctx.c_var[cm].sum())


def abcd_variance(ctx, lo, hi):
    """ABCD variance of the QCD prediction in the box [lo, hi):
    TF^2 x (control count + non-QCD MC w^2 in the control box) + (nonclosure x predicted QCD)^2."""
    qcd, c_var = abcd_qcd(ctx, lo, hi)
    return ctx.abcd_tf2 * max(c_var, 0.0) + ctx.abcd_nc2 * qcd * qcd


def background_variance(ctx, lo, hi, mask):
    """V of the box [lo, hi) with scan-event mask `mask`: sum_k (sum_c delta_kc B_c)^2, the
    MC statistics sum w^2 of the objective's background MC (without the QCD in ABCD mode),
    the zero-count floor, and in ABCD mode the variance of the QCD prediction."""
    B_cls = ctx.w_bkg_cls[:, mask].sum(axis=1)
    V = float(np.sum((ctx.bkg_shifts @ B_cls) ** 2) + ctx.w2_bkg[mask].sum()) + floor_variance(ctx, mask)
    if ctx.abcd:
        V += abcd_variance(ctx, lo, hi)
    return V


def _qcd_axis(ctx):
    """Index of the QCD axis within ctx.axis_names, or -1 if not scanned."""
    for i, name in enumerate(ctx.axis_names):
        if name.strip().upper() == "QCD":
            return i
    return -1


# -------------------- Geometry / membership (half-open [lo, hi)) --------------------
def _hi_to_open(h):
    return float(h) >= 1.0 - EPS


def rect_mask(ctx, lo, hi):
    return box_mask(ctx.score_axes, lo, hi)


class _AxisIndex:
    """Per-axis sorted order of a point set (the rows of axes): the points of a box are found from
    its narrowest axis instead of testing every point (deduplication of large candidate pools on
    the k-fold MC)."""

    def __init__(self, axes):
        self.axes = axes
        self.n = axes.shape[0]
        self.order = [np.argsort(axes[:, d], kind="stable") for d in range(axes.shape[1])]
        self.sorted = [axes[order, d] for d, order in enumerate(self.order)]

    def members(self, lo, hi):
        """Sorted indices of the points inside the half-open box [lo, hi) (box_mask semantics)."""
        best = None
        for d in range(self.axes.shape[1]):
            a = int(np.searchsorted(self.sorted[d], lo[d], side="left"))
            b = self.n if _hi_to_open(hi[d]) else int(np.searchsorted(self.sorted[d], hi[d], side="left"))
            if best is None or b - a < best[2] - best[1]:
                best = (d, a, b)
        d0, a, b = best
        if b <= a:
            return np.zeros(0, dtype=np.int64)
        idx = self.order[d0][a:b]
        pts = self.axes[idx]
        keep = np.ones(idx.size, dtype=bool)
        for d in range(self.axes.shape[1]):
            if d == d0:
                continue
            v = pts[:, d]
            keep &= (v >= lo[d]) if _hi_to_open(hi[d]) else ((v >= lo[d]) & (v < hi[d]))
        return np.sort(idx[keep]).astype(np.int64)


def box_mask(axes, lo, hi):
    """Events (rows of the score-axis array axes) inside the box [lo, hi)."""
    m = np.ones(axes.shape[0], dtype=bool)
    for d in range(axes.shape[1]):
        v = axes[:, d]
        if _hi_to_open(hi[d]):
            m &= v >= lo[d]
        else:
            m &= (v >= lo[d]) & (v < hi[d])
    return m


def rect_stats(ctx, lo, hi):
    """Return (S, sS, B, sB, S_entries, B_entries, V) for rectangle [lo, hi): B is the background
    of the objective (in ABCD mode the non-QCD MC plus the QCD predicted from the control box),
    sB the statistical error of its MC part, B_entries the background MC events."""
    m = rect_mask(ctx, lo, hi)
    ms = m & ctx.is_sig
    wS = ctx.w[ms]
    B = float(ctx.w_bkg_fit[m].sum())
    if ctx.abcd:
        B += abcd_qcd(ctx, lo, hi)[0]
    return (
        float(wS.sum()),
        float(np.sqrt((wS ** 2).sum())),
        B,
        float(np.sqrt(ctx.w2_bkg[m].sum())),
        int(ms.sum()),
        int((m & ctx.is_bkg).sum()),
        background_variance(ctx, lo, hi, m),
    )


def overlap(lo1, hi1, lo2, hi2, D):
    # Half-open [lo, hi) overlap test: abutting edges (hi_a == lo_b) do NOT overlap.
    for d in range(D):
        if not (lo1[d] < hi2[d] and lo2[d] < hi1[d]):
            return False
    return True


def _region_key(lo, hi):
    return (tuple(round(float(v), 10) for v in lo),
            tuple(round(float(v), 10) for v in hi))


def _valid_region(lo, hi, D):
    return all(float(lo[d]) < float(hi[d]) - EPS for d in range(D))


def _parallel_map_ordered(fn, items):
    items = list(items)
    if MAX_THREADS <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        return list(executor.map(fn, items))


# -------------------- Candidate evaluation --------------------
def evaluate_region(ctx, lo, hi, S_v=None, B_v=None, V_v=None, S_e=None, B_e=None):
    """Build a candidate dict, or None if it fails the acceptance constraints.

    Fast path: ``top_intervals_on_axis`` already returns the exact weighted S, B
    and background variance V of each child rectangle (the other-axes mask is
    applied and the interval is half-open identically to ``rect_mask``), and with
    entry-count thresholds also its signal / background MC event counts, so when
    they are supplied we skip the O(N*D) ``rect_stats`` recompute.
    """
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    if not _valid_region(lo, hi, ctx.D):
        return None
    if ctx.forbidden:
        for flo, fhi in ctx.forbidden:
            if overlap(lo, hi, flo, fhi, ctx.D):
                return None
    need_counts = MIN_SIGNAL_ENTRIES > 0 or MIN_BKG_ENTRIES > 0
    if S_v is None or B_v is None or V_v is None or (need_counts and (S_e is None or B_e is None)):
        S, _sS, B, _sB, S_e, B_e, V = rect_stats(ctx, lo, hi)
    else:
        S, B, V = float(S_v), float(B_v), float(V_v)
        S_e, B_e = (int(S_e), int(B_e)) if need_counts else (-1, -1)
    if B < MIN_BKG_WEIGHT or S <= MIN_SIGNAL_WEIGHT:
        return None
    if (MIN_SIGNAL_ENTRIES > 0 and S_e < MIN_SIGNAL_ENTRIES) or \
       (MIN_BKG_ENTRIES > 0 and B_e < MIN_BKG_ENTRIES):
        return None
    Z = calc_Z_val(S, B, V)
    if Z <= 0.0:
        return None
    return {"lo": lo, "hi": hi, "S": S, "B": B, "V": V, "Z": Z,
            "S_entries": S_e, "B_entries": B_e}


def top_intervals_on_axis(ctx, d, lo, hi, edges, top_n):
    """Top [edges[a], edges[b]) intervals on axis d, with the other axes fixed.

    Reimplementation of signal_region.py:_top_intervals_on_axis with the
    background-variance significance Z_A. Cost is O(n_events) for the other-axes
    mask plus K x K vectorised grids (K = len(edges)) of S, B, the background
    class yields, sum w^2, the zero-count floor and the ABCD variance. Each
    interval is returned as (low, high, S, B, V, Z, S_entries, B_entries); the MC
    event counts are histogrammed only when an entry-count threshold is active
    (else -1).
    """
    m = np.ones(ctx.n_events, dtype=bool)
    for dd in range(ctx.D):
        if dd == d:
            continue
        v = ctx.score_axes[:, dd]
        if _hi_to_open(hi[dd]):
            m &= v >= lo[dd]
        else:
            m &= (v >= lo[dd]) & (v < hi[dd])
    if not m.any():
        return []

    edges = np.unique(np.clip(np.asarray(edges, dtype=float), 0.0, 1.0))
    if edges.size < 2:
        return []

    v_d = ctx.score_axes[:, d]
    v_m = v_d[m]

    def _prefix(weights):
        h, _ = np.histogram(v_m, bins=edges, weights=weights[m])
        return np.r_[0.0, np.cumsum(h)]

    pS = _prefix(ctx.w_sig)
    pB = _prefix(ctx.w_bkg_fit)
    K = pS.size
    a_idx = np.arange(K).reshape(-1, 1)
    b_idx = np.arange(K).reshape(1, -1)
    tri = b_idx > a_idx
    S_mat = pS[b_idx] - pS[a_idx]
    B_mat = pB[b_idx] - pB[a_idx]
    if ctx.abcd:
        # QCD predicted from the control events of the same box (other axes fixed), C_box x TF,
        # and the box's control-region variance.
        mc_ = np.ones(ctx.c_axes.shape[0], dtype=bool)
        for dd in range(ctx.D):
            if dd == d:
                continue
            cv = ctx.c_axes[:, dd]
            mc_ &= (cv >= lo[dd]) if _hi_to_open(hi[dd]) else ((cv >= lo[dd]) & (cv < hi[dd]))
        cv_d = ctx.c_axes[mc_, d]
        hQ, _ = np.histogram(cv_d, bins=edges, weights=ctx.c_qcd_pred[mc_])
        hC, _ = np.histogram(cv_d, bins=edges, weights=ctx.c_var[mc_])
        pQ = np.r_[0.0, np.cumsum(hQ)]
        pC = np.r_[0.0, np.cumsum(hC)]
        Q_mat = pQ[b_idx] - pQ[a_idx]
        C_mat = pC[b_idx] - pC[a_idx]
        B_mat = B_mat + Q_mat
    valid = tri & (B_mat >= MIN_BKG_WEIGHT) & (S_mat > MIN_SIGNAL_WEIGHT)
    count_entries = MIN_SIGNAL_ENTRIES > 0 or MIN_BKG_ENTRIES > 0
    if count_entries:
        pNS = _prefix(ctx.is_sig.astype(float))
        pNB = _prefix(ctx.is_bkg.astype(float))
        NS_mat = np.rint(pNS[b_idx] - pNS[a_idx])
        NB_mat = np.rint(pNB[b_idx] - pNB[a_idx])
        valid &= (NS_mat >= MIN_SIGNAL_ENTRIES) & (NB_mat >= MIN_BKG_ENTRIES)
    if not valid.any():
        return []
    pW2 = _prefix(ctx.w2_bkg)
    V_mat = pW2[b_idx] - pW2[a_idx]
    B_cls_mats = []
    for c in range(ctx.w_bkg_cls.shape[0]):
        pC = _prefix(ctx.w_bkg_cls[c])
        B_cls_mats.append(pC[b_idx] - pC[a_idx])
    for k in range(ctx.bkg_shifts.shape[0]):
        shift_k = np.zeros_like(V_mat)
        for c, B_c in enumerate(B_cls_mats):
            shift_k += ctx.bkg_shifts[k, c] * B_c
        V_mat += shift_k ** 2
    if ZERO_COUNT_FLOOR:
        for mc, alpha2 in zip(ctx.bkg_class_masks, ctx.floor_alpha2):
            mm = m & mc
            hN, _ = np.histogram(v_d[mm], bins=edges)
            hW, _ = np.histogram(v_d[mm], bins=edges, weights=ctx.w[mm])
            pN = np.r_[0, np.cumsum(hN)]
            pW = np.r_[0.0, np.cumsum(hW)]
            empty = ((pN[b_idx] - pN[a_idx]) == 0) | ((pW[b_idx] - pW[a_idx]) <= 0.0)
            V_mat = V_mat + np.where(empty, alpha2, 0.0)
    if ctx.abcd:
        V_mat = V_mat + ctx.abcd_tf2 * np.maximum(C_mat, 0.0) + ctx.abcd_nc2 * Q_mat * Q_mat
    Z2 = np.where(valid, za2(S_mat, B_mat, V_mat), -np.inf)
    valid_count = int(np.count_nonzero(np.isfinite(Z2) & (Z2 > 0.0)))
    if valid_count == 0:
        return []
    take = min(max(1, int(top_n)), valid_count)
    flat_scores = Z2.ravel()
    if take >= flat_scores.size:
        flat_idx = np.argsort(flat_scores)[::-1]
    else:
        flat_idx = np.argpartition(flat_scores, -take)[-take:]
        flat_idx = flat_idx[np.argsort(flat_scores[flat_idx])[::-1]]

    intervals = []
    seen = set()
    for flat in flat_idx:
        if not np.isfinite(flat_scores[flat]) or flat_scores[flat] <= 0.0:
            continue
        a_best = int(flat // K)
        b_best = int(flat % K)
        if not valid[a_best, b_best]:
            continue
        key = (a_best, b_best)
        if key in seen:
            continue
        seen.add(key)
        intervals.append((
            float(edges[a_best]),
            float(edges[b_best]),
            float(S_mat[a_best, b_best]),
            float(B_mat[a_best, b_best]),
            float(V_mat[a_best, b_best]),
            float(np.sqrt(Z2[a_best, b_best])),
            int(NS_mat[a_best, b_best]) if count_entries else -1,
            int(NB_mat[a_best, b_best]) if count_entries else -1,
        ))
        if len(intervals) >= take:
            break
    return intervals


# -------------------- Coarse beam search --------------------
def _grid_edges(width):
    edges = np.round(np.arange(0.0, 1.0 + width / 2.0, width), 6)
    if edges[-1] < 1.0 - EPS:
        edges = np.r_[edges, 1.0]
    edges[-1] = 1.0
    edges[0] = 0.0
    return np.unique(edges)


_SEED_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _build_seeds(ctx, forbidden, axis_edges):
    """Starting rectangles for a coarse pass.

    Always probes every axis (per-axis high band [e,1) and low band [0,e) at a
    few coarse-grid thresholds, plus the full box), so each disjoint re-seeding
    pass can still discover a region near any class corner. Forbidden-box
    complement bands are added too. Seeds overlapping the forbidden set are
    rejected later by ``evaluate_region``. Mirrors the per-axis quantile seeding
    in signal_region.py (~1228-1336).

    ``axis_edges`` is a per-axis list of grid-edge arrays (axes may use
    different resolutions, e.g. a finer grid on the QCD axis).
    """
    D = ctx.D
    seeds = []
    seen = set()

    def _add(lo, hi):
        if not _valid_region(lo, hi, D):
            return
        key = _region_key(lo, hi)
        if key not in seen:
            seen.add(key)
            seeds.append((list(lo), list(hi)))

    _add([0.0] * D, [1.0] * D)  # full box (valid only when nothing is forbidden)
    for d in range(D):
        edges_d = axis_edges[d]
        last = len(edges_d) - 1
        for q in _SEED_QUANTILES:
            e = float(edges_d[int(round(q * last))])
            if e > EPS:
                lo = [0.0] * D; hi = [1.0] * D; lo[d] = e; _add(lo, hi)   # high band
            if e < 1.0 - EPS:
                lo = [0.0] * D; hi = [1.0] * D; hi[d] = e; _add(lo, hi)   # low band
    for flo, fhi in forbidden:
        for d in range(D):
            if flo[d] > EPS:
                lo = [0.0] * D; hi = [1.0] * D; hi[d] = float(flo[d]); _add(lo, hi)
            if fhi[d] < 1.0 - EPS:
                lo = [0.0] * D; hi = [1.0] * D; lo[d] = float(fhi[d]); _add(lo, hi)
    return seeds


def coarse_beam_search(ctx, forbidden_boxes=()):
    """Coordinate-descent beam search on the coarse (0.05) grid.

    Restricted to rectangles disjoint from ``forbidden_boxes``. Returns a pool
    (list of candidate dicts) covering every accepted rectangle evaluated across
    all rounds, not just the final beam. Empty if no valid region exists in the
    remaining space. The QCD axis (if scanned) uses its own, typically much
    finer, grid (``QCD_COARSE_W``) since its score is sharply peaked near 0.
    """
    coarse_edges = _grid_edges(COARSE_W)
    qcd_axis = _qcd_axis(ctx)
    coarse_edges_qcd = _grid_edges(QCD_COARSE_W) if qcd_axis >= 0 else coarse_edges
    axis_edges = [coarse_edges_qcd if d == qcd_axis else coarse_edges for d in range(ctx.D)]
    ctx.forbidden = list(forbidden_boxes or [])

    seed_boxes = _build_seeds(ctx, ctx.forbidden, axis_edges)
    initial = []
    pool = {}
    for item in _parallel_map_ordered(
        lambda s: evaluate_region(ctx, s[0], s[1]), seed_boxes
    ):
        if item is not None:
            key = _region_key(item["lo"], item["hi"])
            if key not in pool:
                pool[key] = item
                initial.append(item)
    if not initial:
        ctx.forbidden = []
        return []

    beam = sorted(initial, key=lambda it: -it["Z"])[:BEAM_WIDTH]
    ctx.progress(
        f"Coarse beam start: edges/axis={coarse_edges.size} "
        f"(qcd={coarse_edges_qcd.size}), seeds={len(initial)}, "
        f"beam={len(beam)}, best_Z={beam[0]['Z']:.4f}",
        force=True,
    )

    prev_beam_keys = None
    for r in range(ROUNDS):
        tasks = [(item, d) for item in beam for d in range(ctx.D)]

        def _task(task):
            item, d = task
            intervals = top_intervals_on_axis(
                ctx, d, item["lo"], item["hi"], axis_edges[d], TOP_K
            )
            children = []
            for low_d, high_d, S_v, B_v, V_v, _Z, S_e, B_e in intervals:
                lo = list(item["lo"])
                hi = list(item["hi"])
                lo[d] = low_d
                hi[d] = high_d
                children.append((lo, hi, S_v, B_v, V_v, S_e, B_e))
            return children

        produced = []
        for children in _parallel_map_ordered(_task, tasks):
            for lo, hi, S_v, B_v, V_v, S_e, B_e in children:
                key = _region_key(lo, hi)
                if key in pool:
                    produced.append(pool[key])
                    continue
                item = evaluate_region(ctx, lo, hi, S_v, B_v, V_v, S_e, B_e)
                if item is not None:
                    pool[key] = item
                    produced.append(item)

        merged = {}
        for item in beam + produced:
            key = _region_key(item["lo"], item["hi"])
            if key not in merged or item["Z"] > merged[key]["Z"]:
                merged[key] = item
        beam = sorted(merged.values(), key=lambda it: -it["Z"])[:BEAM_WIDTH]
        best_Z = beam[0]["Z"] if beam else 0.0
        ctx.progress(
            f"Coarse round {r + 1}/{ROUNDS} done: pool={len(pool)}, "
            f"beam={len(beam)}, best_Z={best_Z:.4f}",
            force=True,
        )

        beam_keys = tuple(_region_key(it["lo"], it["hi"]) for it in beam)
        if beam_keys == prev_beam_keys:
            ctx.progress("Coarse beam converged (beam unchanged)", force=True)
            break
        prev_beam_keys = beam_keys

    ctx.forbidden = []
    return list(pool.values())


def find_regions(ctx, target_n):
    """Generate a diverse candidate pool via disjoint re-seeding.

    Run the coarse beam ``target_n`` times; after each pass, forbid that pass's
    best region so the next pass explores space disjoint from all prior picks.
    The per-pass best regions form a mutually non-overlapping "chain" (a feasible
    target_n solution), and the union of all evaluated rectangles is the
    candidate pool handed to the exact selection.
    """
    forbidden = []
    pool = {}
    chain = []
    for k in range(target_n):
        cpool = coarse_beam_search(ctx, forbidden)
        if not cpool:
            ctx.progress(
                f"Diversity pass {k + 1}/{target_n}: no disjoint region remains",
                force=True,
            )
            break
        for it in cpool:
            key = _region_key(it["lo"], it["hi"])
            if key not in pool or it["Z"] > pool[key]["Z"]:
                pool[key] = it
        best = max(cpool, key=lambda it: it["Z"])
        chain.append(best)
        forbidden.append((best["lo"], best["hi"]))
        ctx.progress(
            f"Diversity pass {k + 1}/{target_n}: best_Z={best['Z']:.4f}, "
            f"chain={len(chain)}, pool={len(pool)}",
            force=True,
        )

    # The per-pass seeds cannot always reach a region disjoint from every prior
    # pick, but the accumulated pool may still contain one. Greedily complete the
    # chain from the pool so the incumbent handed to the exact selection is as
    # large (and high-Z) as the pool actually supports.
    chain_keys = {_region_key(c["lo"], c["hi"]) for c in chain}
    for it in sorted(pool.values(), key=lambda x: -x["Z"]):
        if len(chain) >= target_n:
            break
        key = _region_key(it["lo"], it["hi"])
        if key in chain_keys:
            continue
        if all(not overlap(it["lo"], it["hi"], c["lo"], c["hi"], ctx.D) for c in chain):
            chain.append(it)
            chain_keys.add(key)
    ctx.progress(
        f"Diversity done: pool={len(pool)}, completed_chain={len(chain)}/{target_n}",
        force=True,
    )
    return list(pool.values()), chain


# -------------------- Event-preserving boxes and event-mask deduplication --------------------
def _next_interval_hi(value, original_hi):
    if _hi_to_open(original_hi) and float(value) >= 1.0 - EPS:
        return 1.0
    return min(float(original_hi), float(np.nextafter(float(value), np.inf)))


def event_preserving_shrink(ctx, lo, hi, idx):
    """Smallest box holding the same geometry points (scan events and, in ABCD mode, control
    events; sorted indices idx into ctx.geo_axes) as [lo, hi) (port of signal_region.py's
    _event_preserving_shrink): every lower bound rises to its points' minimum and every upper
    bound drops to just above their maximum. The original box is returned when the shrunk one
    would hold other points."""
    if idx.size == 0:
        return list(map(float, lo)), list(map(float, hi))
    lo_new, hi_new = [], []
    min_width = 2.0 * EPS
    for d in range(ctx.D):
        orig_lo = float(lo[d])
        orig_hi = float(hi[d])
        vals = ctx.geo_axes[idx, d]
        sel_min = float(np.min(vals))
        sel_max = float(np.max(vals))
        new_hi = _next_interval_hi(sel_max, orig_hi)
        new_lo = max(orig_lo, min(sel_min, orig_hi - min_width))
        if new_hi - new_lo <= EPS:
            new_hi = min(orig_hi, max(new_hi, new_lo + min_width))
            if new_hi <= sel_max or new_hi - new_lo <= EPS:
                new_lo, new_hi = orig_lo, orig_hi
        lo_new.append(float(new_lo))
        hi_new.append(float(new_hi))
    if not np.array_equal(ctx.geo_index.members(lo_new, hi_new), idx):
        return list(map(float, lo)), list(map(float, hi))
    return lo_new, hi_new


def _hypervolume(lo, hi):
    return float(np.prod([max(0.0, float(h) - float(l)) for l, h in zip(lo, hi)]))


def dedupe_by_event_mask(ctx, items):
    """Shrink every candidate to its point-preserving box and keep one candidate per set of
    geometry points (the scan events and, in ABCD mode, the control events, which set the
    QCD prediction; identical points give identical S, B, V and Z; the smallest box is
    kept). Each item gets "mask_key", a digest of its point indices. Returns the unique
    items sorted by decreasing Z."""
    def _one(item):
        idx = ctx.geo_index.members(item["lo"], item["hi"])
        key = hashlib.blake2b(idx.tobytes(), digest_size=16).digest()
        lo, hi = event_preserving_shrink(ctx, item["lo"], item["hi"], idx)
        out = dict(item)
        out["lo"], out["hi"], out["mask_key"] = lo, hi, key
        return out

    best = {}
    done = 0
    for out in _parallel_map_ordered(_one, items):
        done += 1
        prev = best.get(out["mask_key"])
        if prev is None or _hypervolume(out["lo"], out["hi"]) < _hypervolume(prev["lo"], prev["hi"]) - 1e-15:
            best[out["mask_key"]] = out
        ctx.progress(f"Event-mask dedupe: processed {done}/{len(items)}, kept={len(best)}")
    unique = sorted(best.values(),
                    key=lambda it: (-it["Z"], _hypervolume(it["lo"], it["hi"]), _region_key(it["lo"], it["hi"])))
    log_message(f"  Event-mask dedupe: input={len(items)}, kept={len(unique)}")
    return unique


def place_boundaries(ctx, los, his):
    """Final boundaries of the selected regions inside empty score gaps: each region
    is shrunk to its events, then signal-axis upper bounds rise and background-axis
    lower bounds drop into space that is empty in MC (scan events and, in ABCD mode,
    control events) under the region's other-axis cuts, as far as the box stays
    non-overlapping with every other region (port of signal_region.py's empty-bin
    expansion). Signal-axis bounds thus sit as high and background-axis bounds as low
    as the same events allow."""
    n_sr = len(los)
    los = [list(map(float, l)) for l in los]
    his = [list(map(float, h)) for h in his]
    members = [ctx.geo_index.members(los[i], his[i]) for i in range(n_sr)]
    for i in range(n_sr):
        los[i], his[i] = event_preserving_shrink(ctx, los[i], his[i], members[i])

    def _other_axis_separates(lo_a, hi_a, lo_b, hi_b, skip):
        # Same strict half-open test as overlap().
        for dd in range(ctx.D):
            if dd != skip and (lo_a[dd] >= hi_b[dd] or lo_b[dd] >= hi_a[dd]):
                return True
        return False

    raised = lowered = 0
    for i in range(n_sr):
        lo_i, hi_i = los[i], his[i]
        for d in range(ctx.D):
            other = np.ones(ctx.geo_axes.shape[0], dtype=bool)
            for dd in range(ctx.D):
                if dd == d:
                    continue
                v_dd = ctx.geo_axes[:, dd]
                other &= (v_dd >= lo_i[dd]) if _hi_to_open(hi_i[dd]) else ((v_dd >= lo_i[dd]) & (v_dd < hi_i[dd]))
            v_d = ctx.geo_axes[:, d]
            if ctx.signal_axis[d]:
                if hi_i[d] >= 1.0 - EPS:
                    continue
                cand = other & (v_d >= hi_i[d])
                limit = float(np.min(v_d[cand])) if np.any(cand) else 1.0
                for j in range(n_sr):
                    if j != i and not _other_axis_separates(lo_i, hi_i, los[j], his[j], d) \
                            and los[j][d] >= hi_i[d] - EPS:
                        limit = min(limit, float(los[j][d]))
                if np.any(cand) and limit >= 1.0 - EPS:
                    # Events at the top edge: an upper bound there would be open and
                    # include them, so stop just below it.
                    limit = 1.0 - 2.0 * EPS
                if limit > hi_i[d] + EPS:
                    hi_i[d] = min(limit, 1.0)
                    raised += 1
            else:
                if lo_i[d] <= EPS:
                    continue
                cand = other & (v_d < lo_i[d])
                limit = float(np.nextafter(float(np.max(v_d[cand])), np.inf)) if np.any(cand) else 0.0
                for j in range(n_sr):
                    if j != i and not _other_axis_separates(lo_i, hi_i, los[j], his[j], d) \
                            and lo_i[d] >= his[j][d] - EPS:
                        limit = max(limit, float(his[j][d]))
                if limit < lo_i[d] - EPS:
                    lo_i[d] = max(limit, 0.0)
                    lowered += 1
        if not np.array_equal(ctx.geo_index.members(lo_i, hi_i), members[i]):
            raise RuntimeError(f"Boundary placement changed the events of SR{i + 1}")
    log_message(
        f"  Boundary placement: signal-axis high bounds raised={raised}, "
        f"background-axis low bounds lowered={lowered}"
    )
    return los, his


def check_no_overlap(ctx, los, his):
    """Geometric and event-level (scan and, in ABCD mode, control events) non-overlap of the
    selected regions."""
    members = [ctx.geo_index.members(los[i], his[i]) for i in range(len(los))]
    for ia in range(len(los)):
        for ib in range(ia + 1, len(los)):
            if overlap(los[ia], his[ia], los[ib], his[ib], ctx.D) or \
                    np.intersect1d(members[ia], members[ib], assume_unique=True).size:
                raise RuntimeError(f"Selected signal regions overlap ({ia + 1},{ib + 1})")


# -------------------- Fine refinement (selected regions only) --------------------
def fine_refine_selected(ctx, sel_los, sel_his):
    """Sharpen the chosen regions' boundaries onto the fine grid.

    Coarse selection fixes WHERE the regions are (coarse grid); this tunes
    their exact edges. For each region and axis, scan the fine grid within
    +/- REFINE_WIN of the current boundary and adopt the highest-Z interval
    that keeps the region non-overlapping with every other selected region.
    Cheap: n_regions x FINE_PASSES x D one-dimensional scans. The QCD axis (if
    scanned) uses its own finer grid/window (QCD_FINE_W/QCD_REFINE_WIN) so it
    can resolve the very tight tail cuts the exhaustive box search finds
    there; restricting to a local window keeps this cheap regardless of how
    fine QCD_FINE_W is.
    """
    fine_edges_full = _grid_edges(FINE_W)
    qcd_axis = _qcd_axis(ctx)
    fine_edges_full_qcd = _grid_edges(QCD_FINE_W) if qcd_axis >= 0 else fine_edges_full
    los = [list(map(float, l)) for l in sel_los]
    his = [list(map(float, h)) for h in sel_his]
    n = len(los)
    ctx.forbidden = []

    def _local_edges(boundary_lo, boundary_hi, d):
        is_qcd = (d == qcd_axis)
        base = fine_edges_full_qcd if is_qcd else fine_edges_full
        win = QCD_REFINE_WIN if is_qcd else REFINE_WIN
        keep = np.zeros(base.size, dtype=bool)
        for b in (boundary_lo, boundary_hi):
            keep |= np.abs(base - b) <= win + EPS
        edges = base[keep]
        edges = np.unique(np.r_[edges, boundary_lo, boundary_hi, 0.0, 1.0])
        return np.clip(edges, 0.0, 1.0)

    def _others_ok(i, lo_i, hi_i):
        for j in range(n):
            if j != i and overlap(lo_i, hi_i, los[j], his[j], ctx.D):
                return False
        return True

    ctx.progress(
        f"Fine refinement of {n} selected regions: window=+/-{REFINE_WIN} "
        f"(qcd=+/-{QCD_REFINE_WIN}), passes={FINE_PASSES}",
        force=True,
    )
    for i in range(n):
        for _ in range(FINE_PASSES):
            changed = False
            for d in range(ctx.D):
                current = evaluate_region(ctx, los[i], his[i])
                current_z = current["Z"] if current is not None else 0.0
                edges = _local_edges(los[i][d], his[i][d], d)
                intervals = top_intervals_on_axis(ctx, d, los[i], his[i], edges, TOP_K)
                for low_d, high_d, _S_v, _B_v, _V_v, _Z, _S_e, _B_e in intervals:
                    lo_alt = list(los[i]); hi_alt = list(his[i])
                    lo_alt[d] = low_d; hi_alt[d] = high_d
                    # Full acceptance (weight and entry minima of evaluate_region), and never
                    # below the region's current significance.
                    alt = evaluate_region(ctx, lo_alt, hi_alt)
                    if alt is None or alt["Z"] < current_z - 1e-12:
                        continue
                    if _others_ok(i, lo_alt, hi_alt):
                        if abs(low_d - los[i][d]) > 1e-12 or abs(high_d - his[i][d]) > 1e-12:
                            changed = True
                        los[i][d] = low_d
                        his[i][d] = high_d
                        break
            if not changed:
                break
        ctx.progress(f"  Refined SR{i + 1}/{n}", force=True)
    return los, his


# -------------------- Global selection (exact branch-and-bound) --------------------
def select_branch_bound(ctx, candidates, target_n, incumbent=None):
    """Pick target_n non-overlapping rectangles maximising sum(Z^2), exactly.

    Python fallback of select_regions_openmp: multi-start greedy incumbent plus
    an exact bitset branch-and-bound over all (deduplicated) candidates, sorted
    by Z^2 descending so the greedy optimistic bound is admissible.
    ``incumbent`` is a known feasible set (the disjoint chain, matched by event
    mask) used to seed pruning.
    """
    items = sorted(candidates, key=lambda it: -(it["Z"] ** 2))
    if len(items) < target_n:
        raise RuntimeError(
            f"Only {len(items)} candidate signal regions are available; "
            f"requested {target_n}"
        )
    n = len(items)
    los = [it["lo"] for it in items]
    his = [it["hi"] for it in items]
    Z2 = np.array([it["Z"] ** 2 for it in items], dtype=float)
    D = ctx.D
    key_to_idx = {it["mask_key"]: i for i, it in enumerate(items)}

    def _compatible(i, picks):
        return all(not overlap(los[i], his[i], los[j], his[j], D) for j in picks)

    # ---- Multi-start greedy incumbent. ----
    # Plain greedy from the single highest-Z region can fail: that region may
    # overlap the whole disjoint family. Trying each of the top-M regions as the
    # mandatory first pick reliably finds a feasible target_n set when one
    # exists, and reveals the largest achievable count. Every greedy prefix is a
    # valid disjoint set, so we record the best set found at each size.
    best_by_size = {}

    def _record(picks):
        for size in range(1, len(picks) + 1):
            sub = tuple(picks[:size])
            score = float(np.sum(Z2[list(sub)]))
            if size not in best_by_size or score > best_by_size[size][0]:
                best_by_size[size] = (score, sub)

    def _greedy_from_first(first):
        # Stop at target_n picks (checked before each addition: with target_n = 1 the old
        # check after the append never fired, and the greedy set of every compatible region
        # became the incumbent).
        picks = [first]
        for i in range(n):
            if len(picks) >= target_n:
                break
            if i == first:
                continue
            if _compatible(i, picks):
                picks.append(i)
        return picks

    M = min(n, max(8 * target_n, 64))
    for first in range(M):
        _record(_greedy_from_first(first))

    # Also try the supplied feasible chain (its regions are retained in items).
    if incumbent:
        chain_idx = [key_to_idx[it["mask_key"]] for it in incumbent if it["mask_key"] in key_to_idx]
        # Keep a mutually compatible subset: deduplication may have replaced a chain box.
        compatible_chain = []
        for i in chain_idx:
            if _compatible(i, compatible_chain):
                compatible_chain.append(i)
        if compatible_chain:
            _record(compatible_chain)

    if not best_by_size:
        raise RuntimeError("Global selection found no valid signal-region set")
    # ---- Exact branch-and-bound via compatibility bitsets. ----
    # Items are Z^2-descending, so candidate index order == Z^2 order, and the
    # lowest set bits of an "available" mask are the highest-Z^2 candidates.
    # Maximise (region_count, sum Z^2) lexicographically: prefer more
    # non-overlapping regions, then higher combined significance, up to target_n.
    LO = np.asarray(los, dtype=float)  # (n, D)
    HI = np.asarray(his, dtype=float)
    # One compatibility row at a time (memory O(n)): boxes i,j overlap iff
    # lo_i < hi_j and lo_j < hi_i on every axis (half-open); the row's bitset is
    # built from the packed boolean row.
    compat = [0] * n
    for i in range(n):
        overlap_row = np.ones(n, dtype=bool)
        for d in range(D):
            overlap_row &= (LO[i, d] < HI[:, d]) & (LO[:, d] < HI[i, d])
        compat_row = ~overlap_row
        compat_row[i] = False
        compat[i] = int.from_bytes(np.packbits(compat_row, bitorder="little").tobytes(), "little")
    full_mask = (1 << n) - 1

    seed_size = max(best_by_size)
    best = [seed_size, float(best_by_size[seed_size][0]), best_by_size[seed_size][1]]
    root_upper = float(np.sum(np.sort(Z2)[::-1][:target_n]))

    nodes = [0]
    stopped = [False]
    bnb_t0 = time.monotonic()

    def _consider(picks, score):
        s = len(picks)
        if s > best[0] or (s == best[0] and score > best[1] + 1e-12):
            best[0] = s
            best[1] = float(score)
            best[2] = picks

    def _dfs(avail, picks, score):
        if stopped[0]:
            return
        nodes[0] += 1
        if BNB_MAX_NODES > 0 and nodes[0] >= BNB_MAX_NODES:
            stopped[0] = True
            return
        if BNB_TIME_LIMIT > 0.0 and (nodes[0] & 0x3FFF) == 0 and \
                time.monotonic() - bnb_t0 >= BNB_TIME_LIMIT:
            stopped[0] = True
            return
        _consider(picks, score)
        s = len(picks)
        if s >= target_n:
            return
        cap = target_n - s
        cnt = bin(avail).count("1")
        bsize = s + min(cap, cnt)
        if bsize < best[0]:
            return
        # Score upper bound: top-cap Z^2 among available (lowest indices first).
        bscore = score
        need = cap
        tmp = avail
        while tmp and need > 0:
            b = tmp & (-tmp)
            i = b.bit_length() - 1
            tmp ^= b
            bscore += Z2[i]
            need -= 1
        if bsize == best[0] and bscore <= best[1] + 1e-12:
            return
        tmp = avail
        while tmp:
            b = tmp & (-tmp)
            i = b.bit_length() - 1
            tmp ^= b
            if stopped[0]:
                return
            # tmp now holds only indices > i, enforcing increasing-index order.
            _dfs(tmp & compat[i], picks + (i,), score + Z2[i])

    ctx.progress(
        f"Branch-and-bound start: candidates={n}, target_bins={target_n}, "
        f"incumbent_size={best[0]}, incumbent_Z={np.sqrt(best[1]):.4f}",
        force=True,
    )
    _dfs(full_mask, tuple(), 0.0)

    eff, best_score, best_picks = best[0], best[1], best[2]
    if not best_picks:
        raise RuntimeError("Global selection found no valid signal-region set")
    if eff < target_n:
        log_warning(
            f"Pool supports only {eff} mutually non-overlapping regions; "
            f"selecting {eff} (requested {target_n})"
        )

    completed = not stopped[0]
    upper = float(best_score) if completed else float(max(best_score, root_upper))
    if not completed:
        log_warning(
            f"Branch-and-bound stopped early (node cap {BNB_MAX_NODES} or "
            f"time limit {BNB_TIME_LIMIT}s); returning best incumbent "
            f"(may be sub-optimal)"
        )
    picks = list(best_picks)
    summary = {
        "selector": "Python branch-and-bound (hist, bitset)",
        "completed": completed,
        "nodes": int(nodes[0]),
        "objective_sum_z2": float(best_score),
        "objective_upper_bound_sum_z2": upper,
        "geometry_overlap_pairs": 0,
        "event_overlap_pairs": 0,
        "candidate_count": int(n),
    }
    return picks, [los[i] for i in picks], [his[i] for i in picks], summary


def _build_openmp_selector():
    """Compile (if needed) and load openmp_region_select.cpp, as signal_region.py does."""
    src = os.path.join(_SCRIPT_DIR, "openmp_region_select.cpp")
    build_dir = os.path.join(sr.OUTPUT_DIR, ".openmp")
    os.makedirs(build_dir, exist_ok=True)
    lib = os.path.join(build_dir, "openmp_region_select.so")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        attempts = [
            ("Homebrew libomp",
             ["-Xpreprocessor", "-fopenmp", "-D_OPENMP=201511", "-I/opt/homebrew/opt/libomp/include"],
             ["-L/opt/homebrew/opt/libomp/lib", "-lomp"]),
            ("Homebrew libomp (/usr/local)",
             ["-Xpreprocessor", "-fopenmp", "-D_OPENMP=201511", "-I/usr/local/opt/libomp/include"],
             ["-L/usr/local/opt/libomp/lib", "-lomp"]),
            ("generic -fopenmp", ["-fopenmp"], []),
        ]
        errors = []
        for label, cflags, ldflags in attempts:
            cmd = ["c++", "-O3", "-std=c++17", "-fPIC", "-shared"] + cflags + [src, "-o", lib] + ldflags
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as exc:
                errors.append(f"{label}: {exc}")
                continue
            if proc.returncode == 0:
                log_message(f"  OpenMP selector built with {label}")
                break
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            errors.append(f"{label}: {detail[-1] if detail else proc.returncode}")
        else:
            log_warning("OpenMP selector build failed; using the Python branch-and-bound. " + " | ".join(errors))
            return None
    try:
        fn = ctypes.CDLL(lib).select_regions_branch_bound_openmp
    except Exception as exc:
        log_warning(f"OpenMP selector load failed; using the Python branch-and-bound: {exc}")
        return None
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    return fn


def select_regions_openmp(ctx, candidates, target_n):
    """Beam incumbent plus branch-and-bound of openmp_region_select.cpp over all
    candidates (sorted by Z^2 descending); None when the helper is unavailable."""
    if MAX_THREADS <= 1 or target_n > 16:
        return None
    fn = _build_openmp_selector()
    if fn is None:
        return None
    items = sorted(candidates, key=lambda it: -(it["Z"] ** 2))
    n = len(items)
    lows = np.ascontiguousarray([it["lo"] for it in items], dtype=np.float64)
    highs = np.ascontiguousarray([it["hi"] for it in items], dtype=np.float64)
    z2 = np.ascontiguousarray([it["Z"] ** 2 for it in items], dtype=np.float64)
    out = np.full(target_n, -1, dtype=np.int32)
    stats = np.zeros(6, dtype=np.float64)
    ctx.progress(f"OpenMP branch-and-bound start: candidates={n}, target_bins={target_n}, "
                 f"beam_width={GLOBAL_BEAM}, time_limit={BNB_TIME_LIMIT}s", force=True)
    ret = fn(n, ctx.D, target_n, GLOBAL_BEAM, BNB_MAX_NODES, BNB_TIME_LIMIT,
             lows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             highs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             z2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
             stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), MAX_THREADS)
    if ret < 0:
        log_warning(f"OpenMP branch-and-bound returned {ret}; using the Python branch-and-bound")
        return None
    picks = [int(v) for v in out[:ret] if int(v) >= 0]
    summary = {
        "selector": "OpenMP branch-and-bound",
        "completed": bool(stats[3] > 0.5),
        "nodes": int(round(stats[2])),
        "objective_sum_z2": float(stats[0]),
        "objective_upper_bound_sum_z2": float(max(stats[0], stats[1])),
        "geometry_overlap_pairs": 0,
        "event_overlap_pairs": 0,
        "candidate_count": int(n),
    }
    if not summary["completed"]:
        log_warning(
            f"OpenMP branch-and-bound stopped before exhausting the search: "
            f"Z_best={np.sqrt(summary['objective_sum_z2']):.4f}, "
            f"Z_upper_bound<={np.sqrt(summary['objective_upper_bound_sum_z2']):.4f}"
        )
    return picks, [items[i]["lo"] for i in picks], [items[i]["hi"] for i in picks], summary


# -------------------- Joint significance of a set of regions --------------------
# The joint stage tries the JOINT_SWAP_POOL candidates of highest Z_A as replacements, for at
# most JOINT_SWAP_PASSES passes.
JOINT_SWAP_POOL = 256
JOINT_SWAP_PASSES = 10


def region_components(ctx, lo, hi):
    """Inputs of the joint significance for the box [lo, hi): (S, background MC per class
    (columns follow sr.BACKGROUND_CLASS_INDICES; the QCD columns are empty in ABCD mode), QCD
    predicted from the control box (ABCD mode, else 0), uncorrelated background variance: MC
    statistics of the background MC, zero-count floor and, in ABCD mode, the control-region
    statistics TF^2 x (control count + non-QCD MC w^2))."""
    m = rect_mask(ctx, lo, hi)
    S = float(ctx.w_sig[m].sum())
    b = ctx.w_bkg_cls[:, m].sum(axis=1)
    v = float(ctx.w2_bkg[m].sum()) + floor_variance(ctx, m)
    q = 0.0
    if ctx.abcd:
        q, c_var = abcd_qcd(ctx, lo, hi)
        v += ctx.abcd_tf2 * max(c_var, 0.0)
    return S, b, q, v


def joint_z(ctx, comps):
    """Asimov discovery significance of a set of regions fitted together with the nuisances of
    the combine cards (one channel). Region i: Asimov count n_i = S_i + B_i; background
    nu_i = gamma_i [sum_c b_ci prod_k (1 + delta_kc)^theta_k + q_i (1 + f_nc)^theta_nc], with
    theta_k ~ N(0, 1) for every row of ctx.bkg_shifts and, in ABCD mode, theta_nc for the
    non-closure f_nc of the QCD prediction, each correlated across the regions as in the cards
    (lnN rows; combine's qcd_abcd_nonclosure_<channel>), and gamma_i ~ N(1, v_i / B_i^2) for
    the uncorrelated variance. q0 = 2 sum_i [n_i ln(n_i / nu_i) - n_i + nu_i] + sum theta^2
    + sum (gamma_i - 1)^2 B_i^2 / v_i is minimised over theta (scipy L-BFGS-B) with every
    gamma_i profiled in closed form; Z = sqrt(q0). For a single region without correlated rows
    it is the Gaussian-constraint counterpart of Z_A."""
    S = np.array([c[0] for c in comps], dtype=float)
    b = np.vstack([c[1] for c in comps])
    q = np.array([c[2] for c in comps], dtype=float)
    v = np.array([c[3] for c in comps], dtype=float)
    B = b.sum(axis=1) + q
    if np.any(B <= 0.0):
        raise RuntimeError("joint_z needs a positive background in every region")
    n = S + B
    log_kappa = np.log1p(ctx.bkg_shifts)  # (rows, classes)
    n_rows = log_kappa.shape[0]
    with_nc = bool(ctx.abcd and ctx.abcd_nc2 > 0.0 and np.any(q > 0.0))
    log_nc = float(np.log1p(ABCD_NONCLOSURE)) if with_nc else 0.0
    has_v = v > 0.0
    inv_eps2 = np.where(has_v, B * B / np.where(has_v, v, 1.0), 0.0)  # 1 / eps_i^2

    def q0(theta):
        u = b @ np.exp(theta[:n_rows] @ log_kappa)
        u = u + (q * np.exp(theta[n_rows] * log_nc) if with_nc else q)
        if np.any(u <= 0.0):
            return 1e300
        # gamma minimises 2 (gamma u - n ln gamma) + (gamma - 1)^2 / eps^2, i.e. solves
        # gamma^2 / eps^2 + (u - 1 / eps^2) gamma - n = 0: the positive root, in a stable form.
        lin = u - inv_eps2
        root = np.sqrt(lin * lin + 4.0 * n * inv_eps2)
        gamma = np.where(
            has_v,
            np.where(lin <= 0.0, (root - lin) / np.where(has_v, 2.0 * inv_eps2, 1.0),
                     2.0 * n / np.where(lin + root > 0.0, lin + root, 1.0)),
            1.0,
        )
        nu = gamma * u
        terms = 2.0 * (n * np.log(n / nu) - n + nu) + inv_eps2 * (gamma - 1.0) ** 2
        return float(terms.sum() + theta @ theta)

    n_par = n_rows + (1 if with_nc else 0)
    start = np.zeros(n_par)
    value = q0(start)
    if n_par:
        res = minimize(q0, start, method="L-BFGS-B")
        value = min(value, float(res.fun))
    return float(np.sqrt(max(value, 0.0)))


def joint_refine_selection(ctx, items, picks):
    """Second stage of the selection: one region at a time is replaced by the compatible
    candidate (among the JOINT_SWAP_POOL candidates of highest Z_A) that most increases the
    joint significance, until no replacement increases it (at most JOINT_SWAP_PASSES passes).
    items: the candidates in the selectors' order; picks: indices of the selected set.
    Returns (picks, joint Z of the given set, joint Z of the returned set)."""
    order = sorted(range(len(items)), key=lambda i: -items[i]["Z"])[:JOINT_SWAP_POOL]
    pool = list(dict.fromkeys(list(picks) + order))
    comps = {}

    def _comp(i):
        if i not in comps:
            comps[i] = region_components(ctx, items[i]["lo"], items[i]["hi"])
        return comps[i]

    current = list(picks)
    z_start = z_cur = joint_z(ctx, [_comp(i) for i in current])
    for npass in range(JOINT_SWAP_PASSES):
        improved = False
        for pos in range(len(current)):
            others = current[:pos] + current[pos + 1:]
            best_j, best_z = None, z_cur
            for j in pool:
                if j in current:
                    continue
                if any(overlap(items[j]["lo"], items[j]["hi"], items[o]["lo"], items[o]["hi"], ctx.D)
                       for o in others):
                    continue
                z = joint_z(ctx, [_comp(o) for o in others] + [_comp(j)])
                if z > best_z + 1e-9:
                    best_j, best_z = j, z
            if best_j is not None:
                current[pos] = best_j
                z_cur = best_z
                improved = True
        ctx.progress(f"Joint stage pass {npass + 1}: Z_joint={z_cur:.4f}", force=True)
        if not improved:
            break
    return current, z_start, z_cur


# -------------------- Per-bin reports --------------------
def build_top_bins(ctx, sel_los, sel_his):
    """Build the top_bins list with the exact schema sr.* reporting expects.

    Port of signal_region.py's per-bin block; significance is Z_A (background
    variance V), significance_stat the statistics-only Z.
    """
    top_bins = []
    for k, (thr_low_vec, thr_high_vec) in enumerate(zip(sel_los, sel_his)):
        thr_low_vec = list(map(float, thr_low_vec))
        thr_high_vec = list(map(float, thr_high_vec))

        m_bin = rect_mask(ctx, thr_low_vec, thr_high_vec)
        # B is the objective's background (in ABCD mode with the predicted QCD), B_err the
        # statistical error of its MC part.
        S_bin, sS_bin, B_bin, sB_bin, S_e, B_e, V_bin = rect_stats(ctx, thr_low_vec, thr_high_vec)
        Z_bin, sZ_bin = calc_Z(S_bin, B_bin, V_bin, sS_bin, sB_bin)
        Z_stat_bin = calc_Z_val(S_bin, B_bin)
        if ctx.abcd:
            qcd_pred = abcd_qcd(ctx, thr_low_vec, thr_high_vec)[0]
            abcd_var = abcd_variance(ctx, thr_low_vec, thr_high_vec)
            log_message(
                f"  Bin {k + 1} ABCD: QCD predicted={qcd_pred:.4g} (MC: {float(ctx.w[m_bin & ctx.is_qcd].sum()):.4g}), "
                f"sigma_ABCD={abcd_var ** 0.5:.4g}, sigma_MC(non-QCD)={sB_bin:.4g}, "
                f"sigma_B={V_bin ** 0.5:.4g}"
            )

        # One-vs-rest table of the MC classes (reports).
        w_mc = ctx.w[m_bin & (ctx.is_sig | ctx.is_bkg)]
        W_bin = float(w_mc.sum())
        w2_bin = float((w_mc ** 2).sum())
        cat_data = []
        for cls_i, cls_name in enumerate(sr.CLASS_NAMES):
            mC = (ctx.y == cls_i) & m_bin
            wC = ctx.w[mC]
            S_j = float(wC.sum())
            sS_j = float(np.sqrt((wC ** 2).sum()))
            B_j = W_bin - S_j
            sB_j = float(np.sqrt(max(0.0, w2_bin - sS_j ** 2)))
            Z_j, sZ_j = calc_Z(S_j, B_j, 0.0, sS_j, sB_j)
            cat_data.append({
                "name": cls_name,
                "S": S_j, "S_err": sS_j,
                "B": B_j, "B_err": sB_j,
                "Z": Z_j, "Z_err": sZ_j,
            })

        bkg_data = []
        for bkg_i in sr.BACKGROUND_CLASS_INDICES:
            mC = (ctx.y == bkg_i) & m_bin
            wC = ctx.w[mC]
            bkg_data.append({
                "name": sr.CLASS_NAMES[bkg_i],
                "B": float(wC.sum()),
                "B_err": float(np.sqrt((wC ** 2).sum())),
            })

        bin_sig_eff = (S_bin / ctx.S_total) if ctx.S_total > 0 else float("nan")
        bin_bkg_eff = (float(ctx.w_bkg[m_bin].sum()) / ctx.B_total) if ctx.B_total > 0 else float("nan")

        tail_sig_eff, tail_bkg_eff = [], []
        for d in range(ctx.D):
            tidx = max(0, min(
                int(np.searchsorted(ctx.thr_1d, thr_low_vec[d], side="right") - 1),
                ctx.T_REF - 1,
            ))
            tail_sig_eff.append(
                (ctx.S_tail_by_dim[d, tidx] / ctx.S_total) if ctx.S_total > 0 else float("nan")
            )
            tail_bkg_eff.append(
                (ctx.B_tail_by_dim[d, tidx] / ctx.B_total) if ctx.B_total > 0 else float("nan")
            )

        top_bins.append({
            "bin_index":                 k + 1,
            "thr_low":                   np.array(thr_low_vec),
            "thr_high":                  np.array(thr_high_vec),
            "axis_names":                list(ctx.axis_names),
            "significance":              Z_bin,
            "significance_error":        sZ_bin,
            "significance_stat":         Z_stat_bin,
            "background_variance":       V_bin,
            "S":                         S_bin,  "S_err": sS_bin, "S_entries": S_e,
            "B":                         B_bin,  "B_err": sB_bin, "B_entries": B_e,
            "categories":                cat_data,
            "backgrounds":               bkg_data,
            "bin_signal_efficiency":     bin_sig_eff,
            "bin_background_efficiency": bin_bkg_eff,
            "tail_signal_efficiency":    tail_sig_eff,
            "tail_background_efficiency": tail_bkg_eff,
        })

        log_message(
            f"  Bin {k + 1}: Z_A={Z_bin:.4f}+/-{sZ_bin:.4f} (stat-only Z={Z_stat_bin:.4f}), "
            f"S={S_bin:.4g}+/-{sS_bin:.4g}, B={B_bin:.4g}+/-{sB_bin:.4g}, "
            f"sigma_B={np.sqrt(V_bin):.4g}"
        )

    return top_bins


# -------------------- Main --------------------
def main():
    proba, y, w, sample_labels, feature_names, control = prepare_inputs()

    log_message("Plotting score distributions")
    sr.plot_score_distributions(proba, y, w)

    log_message("Computing background systematic shifts")
    bkg_shifts, _shift_names = background_syst_shifts(y, w, sample_labels, qcd_data_driven=control is not None)
    ctx = Ctx(proba, y, w, bkg_shifts, control)
    if ctx.abcd:
        log_message(
            f"ABCD-aware sigma_B ON: control window {ABCD_CONTROL[0]:g} < {ABCD_BRANCH} < {ABCD_CONTROL[1]:g}, "
            f"TF (QCD window / control) = {ctx.abcd_tf:.4g}, nonclosure = {ABCD_NONCLOSURE:g}"
        )
    if ZERO_COUNT_FLOOR:
        log_message("Zero-count floor ON: alpha (median |event weight|) per background class: " + ", ".join(
            f"{sr.CLASS_NAMES[c]}={a2 ** 0.5:.4g}"
            for c, a2 in zip(ctx.floor_classes, ctx.floor_alpha2)))
    log_message(f"  S_total={ctx.S_total:.4g}, B_total={ctx.B_total:.4g}")
    log_message(f"  Scan dimensions D={ctx.D}, axes={ctx.axis_names}")
    log_message(
        f"  Histogram optimizer: coarse_w={COARSE_W}, fine_w={FINE_W}, "
        f"refine_win={REFINE_WIN}, beam_width={BEAM_WIDTH}, top_intervals={TOP_K}, "
        f"rounds={ROUNDS}, fine_passes={FINE_PASSES}, max_threads={MAX_THREADS}, "
        f"qcd_coarse_w={QCD_COARSE_W}, qcd_fine_w={QCD_FINE_W}, "
        f"qcd_refine_win={QCD_REFINE_WIN}"
    )

    target_n = max(1, int(sr.N_SIGNAL_REGIONS))

    # Coarse pass with disjoint re-seeding -> candidate pool + a feasible chain.
    candidates, chain = find_regions(ctx, target_n)
    log_message(
        f"  Candidate pool: {len(candidates)}, disjoint chain: {len(chain)}"
    )
    if not chain:
        raise RuntimeError(
            "No signal region found; lower min_bkg_weight or check inputs"
        )

    # Candidates with identical events are merged into their event-preserving boxes.
    candidates = dedupe_by_event_mask(ctx, candidates)
    chain = dedupe_by_event_mask(ctx, chain)

    # Global selection over every candidate: OpenMP branch-and-bound, else Python.
    selection = select_regions_openmp(ctx, candidates, target_n)
    if selection is None:
        selection = select_branch_bound(ctx, candidates, target_n, incumbent=chain)
    picks, sel_los, sel_his, summary = selection
    if len(picks) < target_n:
        message = (f"Global selection found only {len(picks)} non-overlapping regions; "
                   f"requested {target_n}")
        if sr.REQUIRE_EXACT_N_REGIONS:
            raise RuntimeError(message)
        log_warning(message)
    check_no_overlap(ctx, sel_los, sel_his)

    # Second stage: the joint significance of the set with the correlated nuisances of the
    # cards (the selectors' candidate order maps their picks to the candidates).
    ordered = sorted(candidates, key=lambda it: -(it["Z"] ** 2))
    picks, z_joint_selected, z_joint = joint_refine_selection(ctx, ordered, picks)
    sel_los = [list(ordered[i]["lo"]) for i in picks]
    sel_his = [list(ordered[i]["hi"]) for i in picks]
    log_message(f"  Joint stage: Z_joint {z_joint_selected:.4f} (sum Z_A^2 selection) -> {z_joint:.4f}")
    check_no_overlap(ctx, sel_los, sel_his)

    # Sharpen the chosen regions on the fine grid (non-overlap and acceptance preserved), kept
    # only when it does not lower the joint significance; then place the boundaries inside
    # empty score gaps.
    ref_los, ref_his = fine_refine_selected(ctx, sel_los, sel_his)
    z_joint_refined = joint_z(ctx, [region_components(ctx, l, h) for l, h in zip(ref_los, ref_his)])
    if z_joint_refined >= z_joint - 1e-9:
        sel_los, sel_his, z_joint = ref_los, ref_his, z_joint_refined
    else:
        log_message(f"  Fine refinement lowers Z_joint ({z_joint:.4f} -> {z_joint_refined:.4f}); "
                    f"the unrefined boundaries are kept")
    sel_los, sel_his = place_boundaries(ctx, sel_los, sel_his)
    check_no_overlap(ctx, sel_los, sel_his)
    log_message("  Non-overlap check passed (geometry and events)")
    summary["joint_significance"] = z_joint

    top_bins = build_top_bins(ctx, sel_los, sel_his)

    # Refresh the objective from the refined per-bin significances.
    refined_sum_z2 = float(sum(b["significance"] ** 2 for b in top_bins))
    summary["objective_sum_z2"] = refined_sum_z2
    summary["objective_upper_bound_sum_z2"] = float(
        max(refined_sum_z2, summary.get("objective_upper_bound_sum_z2", 0.0))
    )
    z_best = float(np.sqrt(refined_sum_z2))
    log_message(
        f"  Selected {len(top_bins)} signal regions, Z_joint={z_joint:.4f}, "
        f"sum(Z_A^2)={refined_sum_z2:.6g}, Z_comb={z_best:.4f}, "
        f"stat-only Z_comb={np.sqrt(sum(b['significance_stat'] ** 2 for b in top_bins)):.4f}, "
        f"nodes={summary['nodes']}, selector={summary['selector']}, completed={summary['completed']}"
    )

    result = sr._make_signal_region_result(top_bins, ctx.S_total, ctx.B_total, summary)

    # Write the numeric results before plotting: the reused 2-D simplex-outline
    # plotter assumes each scan axis name is a literal entry in sr.CLASS_NAMES
    # (it looks up a single class corner to draw each region's outline), which
    # does not hold for a merged axis (e.g. "VV+VJets"). That plot is cosmetic,
    # so a failure there must not cost the CSV/text report.
    sr.print_results(result)
    sr.write_signal_region_csv(result)

    log_message("Plotting signal regions")
    try:
        sr.plot_signal_regions_2d(result, proba, y, w)
    except Exception as ex:
        log_warning(
            f"Skipped 2-D region-outline plot (likely a merged scan axis not "
            f"present in sr.CLASS_NAMES): {ex}"
        )

    log_message(f"Finished signal_region_hist.py for tree={sr.TREE_NAME}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
