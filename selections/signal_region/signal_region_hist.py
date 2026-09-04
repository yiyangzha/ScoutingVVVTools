"""Histogram-based coarse-to-fine signal-region optimizer.

A standalone alternative to ``signal_region.py``. It finds N non-overlapping
two-sided score rectangles maximising the same asymptotic significance
``Z = sqrt(2[(S+B)ln(1+S/B) - S])``, but uses a lean grid-based search instead of
the original multi-stage heuristic:

  1. Coarse pass: coordinate-descent beam search on a uniform 0.05 grid.
  2. Fine pass: refine each surviving boundary on the 0.01 grid, locally, within
     +/- ``fine_refine_window`` of its coarse value.
  3. Selection: exact branch-and-bound picking N mutually non-overlapping
     rectangles that maximise sum(Z_i^2).

Data loading, model inference, per-bin statistics layout, text reporting and all
plotting are reused unchanged from ``signal_region.py`` (imported as ``sr``). Only
the candidate generator and the global selection are reimplemented here, because
in the original they live as nested closures that cannot be imported.

The script and ``signal_region.py`` share one config file. Pass its path as the
first command-line argument (``python signal_region_hist.py path/to/config.json``),
or set the env var ``SR_HIST_CONFIG_PATH`` (default: ``config_hist.json`` next to
this file) as a fallback.
"""

import os
import sys
import json
import gc
import time

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
        os.environ.get("SR_HIST_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config_hist.json"))
    )
os.environ["SCAN_CONFIG_PATH"] = HIST_CFG_PATH

import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
MAX_SEL_CANDS  = max(1, int(_hist_cfg.get("max_selection_candidates", 400)))
SEL_PER_AXIS   = max(0, int(_hist_cfg.get("selection_reps_per_axis", 60)))
BNB_MAX_NODES  = max(0, int(_hist_cfg.get("branch_bound_max_nodes", 2000000)))
BNB_TIME_LIMIT = max(0.0, float(_hist_cfg.get("branch_bound_time_limit_seconds", 60.0)))
MAX_THREADS    = max(1, int(_hist_cfg.get("max_threads", sr.MAX_THREADS)))
PROGRESS_EVERY = float(_hist_cfg.get("progress_every_seconds", 30.0))
VALIDATE_REF   = bool(_hist_cfg.get("validate_prediction_reference", False))

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

    df_all = sr.load_test_data(load_cols)
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
        log_message("Validating test-set prediction reference")
        sr._compare_prediction_reference(
            sr.TEST_REFERENCE_SIGNAL_REGION,
            X_model.columns if hasattr(X_model, "columns")
            else [f"f{i}" for i in range(X_model.shape[1])],
            sample_labels, y, w, proba,
        )

    return proba, y, w, sample_labels, list(X_model.columns)


# -------------------- Scan context --------------------
class Ctx:
    """Per-run event arrays and precomputed reference tables for the scan."""

    def __init__(self, proba, y, w):
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

        self.is_sig = np.isin(self.y, sr.SIGNAL_CLASS_INDICES)
        self.is_bkg = np.isin(self.y, sr.BACKGROUND_CLASS_INDICES)
        self.w_sig = np.where(self.is_sig, self.w, 0.0)
        self.w_bkg = np.where(self.is_bkg, self.w, 0.0)
        self.S_total = float(self.w_sig.sum())
        self.B_total = float(self.w_bkg.sum())

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


# -------------------- Significance (copies of sr closures) --------------------
def calc_Z_val(S, B):
    if S <= 0.0 or B <= 0.0:
        return 0.0
    f = (S + B) * np.log(1.0 + S / B) - S
    return float(np.sqrt(2.0 * max(0.0, f)))


def calc_Z(S, B, sS, sB):
    if S <= 0.0 or B <= 0.0:
        return 0.0, 0.0
    f = (S + B) * np.log(1.0 + S / B) - S
    if f <= 0.0:
        return 0.0, 0.0
    Z = float(np.sqrt(2.0 * f))
    ln1sb = np.log(1.0 + S / B)
    dZ_dS = ln1sb / Z
    dZ_dB = (ln1sb - S / B) / Z
    sZ = float(np.sqrt((dZ_dS * sS) ** 2 + (dZ_dB * sB) ** 2))
    return Z, sZ


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
    m = np.ones(ctx.n_events, dtype=bool)
    for d in range(ctx.D):
        v = ctx.score_axes[:, d]
        if _hi_to_open(hi[d]):
            m &= v >= lo[d]
        else:
            m &= (v >= lo[d]) & (v < hi[d])
    return m


def rect_stats(ctx, lo, hi):
    """Return (S, sS, B, sB, S_entries, B_entries) for rectangle [lo, hi)."""
    m = rect_mask(ctx, lo, hi)
    ms = m & ctx.is_sig
    mb = m & ctx.is_bkg
    wS = ctx.w[ms]
    wB = ctx.w[mb]
    return (
        float(wS.sum()),
        float(np.sqrt((wS ** 2).sum())),
        float(wB.sum()),
        float(np.sqrt((wB ** 2).sum())),
        int(ms.sum()),
        int(mb.sum()),
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
def evaluate_region(ctx, lo, hi, S_v=None, B_v=None):
    """Build a candidate dict, or None if it fails the acceptance constraints.

    Fast path: ``top_intervals_on_axis`` already returns the exact weighted S and
    B of each child rectangle (the other-axes mask is applied and the interval is
    half-open identically to ``rect_mask``), so when they are supplied and no
    entry-count thresholds are active we skip the O(N*D) ``rect_stats`` recompute.
    Mirrors signal_region.py:_evaluate_region(S_v, B_v) (~1043-1066).
    """
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    if not _valid_region(lo, hi, ctx.D):
        return None
    if ctx.forbidden:
        for flo, fhi in ctx.forbidden:
            if overlap(lo, hi, flo, fhi, ctx.D):
                return None
    need_stats = (S_v is None or B_v is None) or \
        MIN_SIGNAL_ENTRIES > 0 or MIN_BKG_ENTRIES > 0
    if need_stats:
        S, _sS, B, _sB, S_e, B_e = rect_stats(ctx, lo, hi)
    else:
        S, B, S_e, B_e = float(S_v), float(B_v), -1, -1
    if B < MIN_BKG_WEIGHT or S <= MIN_SIGNAL_WEIGHT:
        return None
    if (MIN_SIGNAL_ENTRIES > 0 and S_e < MIN_SIGNAL_ENTRIES) or \
       (MIN_BKG_ENTRIES > 0 and B_e < MIN_BKG_ENTRIES):
        return None
    Z = calc_Z_val(S, B)
    if Z <= 0.0:
        return None
    return {"lo": lo, "hi": hi, "S": S, "B": B, "Z": Z,
            "S_entries": S_e, "B_entries": B_e}


def top_intervals_on_axis(ctx, d, lo, hi, edges, top_n):
    """Top [edges[a], edges[b]) intervals on axis d, with the other axes fixed.

    Reimplementation of signal_region.py:_top_intervals_on_axis (~1080-1148).
    Cost is O(n_events) for the other-axes mask plus a small K x K vectorised
    Z^2 grid (K = len(edges)).
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
    hS, _ = np.histogram(v_d[m], bins=edges, weights=ctx.w_sig[m])
    hB, _ = np.histogram(v_d[m], bins=edges, weights=ctx.w_bkg[m])
    pS = np.r_[0.0, np.cumsum(hS)]
    pB = np.r_[0.0, np.cumsum(hB)]
    K = pS.size
    a_idx = np.arange(K).reshape(-1, 1)
    b_idx = np.arange(K).reshape(1, -1)
    tri = b_idx > a_idx
    S_mat = pS[b_idx] - pS[a_idx]
    B_mat = pB[b_idx] - pB[a_idx]
    valid = tri & (B_mat >= MIN_BKG_WEIGHT) & (S_mat > MIN_SIGNAL_WEIGHT)
    if not valid.any():
        return []
    Bsafe = np.where(valid, B_mat, 1.0)
    Ssafe = np.where(valid, S_mat, 0.0)
    f = (Ssafe + Bsafe) * np.log1p(Ssafe / Bsafe) - Ssafe
    f = np.where(valid & (f > 0.0), f, 0.0)
    Z2 = np.where(valid, 2.0 * f, -np.inf)
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
            float(np.sqrt(Z2[a_best, b_best])),
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
            for low_d, high_d, S_v, B_v, _Z in intervals:
                lo = list(item["lo"])
                hi = list(item["hi"])
                lo[d] = low_d
                hi[d] = high_d
                children.append((lo, hi, S_v, B_v))
            return children

        produced = []
        for children in _parallel_map_ordered(_task, tasks):
            for lo, hi, S_v, B_v in children:
                key = _region_key(lo, hi)
                if key in pool:
                    produced.append(pool[key])
                    continue
                item = evaluate_region(ctx, lo, hi, S_v, B_v)
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
                edges = _local_edges(los[i][d], his[i][d], d)
                intervals = top_intervals_on_axis(ctx, d, los[i], his[i], edges, TOP_K)
                for low_d, high_d, S_v, B_v, _Z in intervals:
                    if B_v < MIN_BKG_WEIGHT or S_v <= MIN_SIGNAL_WEIGHT:
                        continue
                    lo_alt = list(los[i]); hi_alt = list(his[i])
                    lo_alt[d] = low_d; hi_alt[d] = high_d
                    if not _valid_region(lo_alt, hi_alt, ctx.D):
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

    Standalone port of signal_region.py:_select_regions_beam_python (~1709) plus
    _select_regions_branch_bound_python (~1747-1835), minus the OpenMP path.
    Candidates are sorted by Z^2 descending so the greedy optimistic bound is
    admissible. ``incumbent`` is a known feasible set (the disjoint chain) used
    to seed pruning and guarantee a result; its regions are always kept through
    the candidate cap.
    """
    items_sorted = sorted(candidates, key=lambda it: -(it["Z"] ** 2))
    if len(items_sorted) < target_n:
        raise RuntimeError(
            f"Only {len(items_sorted)} candidate signal regions are available; "
            f"requested {target_n}"
        )
    # Diversity-preserving cap: top-Z reps would all cluster on one corner and
    # crowd out a low-Z-but-disjoint corner the selection needs. So also keep the
    # top reps per dominant high-cut axis, plus the incumbent chain.
    kept = list(items_sorted[:MAX_SEL_CANDS])
    present = {_region_key(it["lo"], it["hi"]) for it in kept}

    def _augment(extra):
        for it in extra:
            key = _region_key(it["lo"], it["hi"])
            if key not in present:
                kept.append(it)
                present.add(key)

    if SEL_PER_AXIS > 0:
        by_axis = {}
        for it in items_sorted:
            lo_ = it["lo"]
            d = int(np.argmax(lo_))
            if lo_[d] > EPS:
                bucket = by_axis.setdefault(d, [])
                if len(bucket) < SEL_PER_AXIS:
                    bucket.append(it)
        for d in by_axis:
            _augment(by_axis[d])
    if incumbent:
        _augment(incumbent)
    if len(kept) != len(items_sorted):
        log_message(
            f"  Selection pool: {len(items_sorted)} -> {len(kept)} "
            f"(top-Z + per-axis diversity + incumbent)"
        )
    items = sorted(kept, key=lambda it: -(it["Z"] ** 2))
    n = len(items)
    los = [it["lo"] for it in items]
    his = [it["hi"] for it in items]
    Z2 = np.array([it["Z"] ** 2 for it in items], dtype=float)
    D = ctx.D
    key_to_idx = {_region_key(it["lo"], it["hi"]): i for i, it in enumerate(items)}

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
        picks = [first]
        for i in range(n):
            if i == first:
                continue
            if _compatible(i, picks):
                picks.append(i)
                if len(picks) == target_n:
                    break
        return picks

    M = min(n, max(8 * target_n, 64))
    for first in range(M):
        _record(_greedy_from_first(first))

    # Also try the supplied feasible chain (its regions are retained in items).
    if incumbent:
        chain_idx = [key_to_idx[_region_key(it["lo"], it["hi"])]
                     for it in incumbent
                     if _region_key(it["lo"], it["hi"]) in key_to_idx]
        if chain_idx:
            _record(chain_idx)

    if not best_by_size:
        raise RuntimeError("Global selection found no valid signal-region set")
    # ---- Exact branch-and-bound via compatibility bitsets. ----
    # Items are Z^2-descending, so candidate index order == Z^2 order, and the
    # lowest set bits of an "available" mask are the highest-Z^2 candidates.
    # Maximise (region_count, sum Z^2) lexicographically: prefer more
    # non-overlapping regions, then higher combined significance, up to target_n.
    LO = np.asarray(los, dtype=float)  # (n, D)
    HI = np.asarray(his, dtype=float)
    overlap_all = np.ones((n, n), dtype=bool)
    for d in range(D):
        lod = LO[:, d]
        hid = HI[:, d]
        # boxes i,j overlap on axis d iff lo_i < hi_j and lo_j < hi_i (half-open).
        overlap_all &= (lod[:, None] < hid[None, :]) & (lod[None, :] < hid[:, None])
    compat_mat = ~overlap_all
    np.fill_diagonal(compat_mat, False)
    compat = [0] * n
    for i in range(n):
        m = 0
        for j in np.flatnonzero(compat_mat[i]):
            m |= (1 << int(j))
        compat[i] = m
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


# -------------------- Per-bin reports --------------------
def build_top_bins(ctx, sel_los, sel_his):
    """Build the top_bins list with the exact schema sr.* reporting expects.

    Port of signal_region.py per-bin block (~2141-2223). The optional empty-bin
    expansion (~2038-2139) is intentionally omitted.
    """
    top_bins = []
    for k, (thr_low_vec, thr_high_vec) in enumerate(zip(sel_los, sel_his)):
        thr_low_vec = list(map(float, thr_low_vec))
        thr_high_vec = list(map(float, thr_high_vec))

        m_bin = rect_mask(ctx, thr_low_vec, thr_high_vec)
        wS = ctx.w[m_bin & ctx.is_sig]
        wB = ctx.w[m_bin & ctx.is_bkg]
        S_bin = float(wS.sum())
        B_bin = float(wB.sum())
        sS_bin = float(np.sqrt((wS ** 2).sum()))
        sB_bin = float(np.sqrt((wB ** 2).sum()))
        S_e = int((m_bin & ctx.is_sig).sum())
        B_e = int((m_bin & ctx.is_bkg).sum())
        Z_bin, sZ_bin = calc_Z(S_bin, B_bin, sS_bin, sB_bin)

        W_bin = S_bin + B_bin
        w2_bin = sS_bin ** 2 + sB_bin ** 2
        cat_data = []
        for cls_i, cls_name in enumerate(sr.CLASS_NAMES):
            mC = (ctx.y == cls_i) & m_bin
            wC = ctx.w[mC]
            S_j = float(wC.sum())
            sS_j = float(np.sqrt((wC ** 2).sum()))
            B_j = W_bin - S_j
            sB_j = float(np.sqrt(max(0.0, w2_bin - sS_j ** 2)))
            Z_j, sZ_j = calc_Z(S_j, B_j, sS_j, sB_j)
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
        bin_bkg_eff = (B_bin / ctx.B_total) if ctx.B_total > 0 else float("nan")

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
            f"  Bin {k + 1}: Z={Z_bin:.4f}+/-{sZ_bin:.4f}, "
            f"S={S_bin:.4g}+/-{sS_bin:.4g}, B={B_bin:.4g}+/-{sB_bin:.4g}"
        )

    return top_bins


# -------------------- Main --------------------
def main():
    proba, y, w, sample_labels, feature_names = prepare_inputs()

    log_message("Plotting score distributions")
    sr.plot_score_distributions(proba, y, w)

    ctx = Ctx(proba, y, w)
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

    # Exact selection over the pool (multi-start incumbent + branch-and-bound).
    # The selector picks as many non-overlapping regions as the pool supports,
    # up to target_n. The chain is passed so its regions survive the candidate
    # cap and seed the incumbent.
    picks, sel_los, sel_his, summary = select_branch_bound(
        ctx, candidates, target_n, incumbent=chain
    )

    # Sharpen the chosen regions onto the 0.01 grid (non-overlap preserved).
    sel_los, sel_his = fine_refine_selected(ctx, sel_los, sel_his)

    # Verify non-overlap (geometry) after refinement.
    for ia in range(len(sel_los)):
        for ib in range(ia + 1, len(sel_los)):
            if overlap(sel_los[ia], sel_his[ia], sel_los[ib], sel_his[ib], ctx.D):
                raise RuntimeError(
                    f"Selected signal regions overlap ({ia + 1},{ib + 1})"
                )

    top_bins = build_top_bins(ctx, sel_los, sel_his)

    # Refresh the objective from the refined per-bin significances.
    refined_sum_z2 = float(sum(b["significance"] ** 2 for b in top_bins))
    summary["objective_sum_z2"] = refined_sum_z2
    summary["objective_upper_bound_sum_z2"] = float(
        max(refined_sum_z2, summary.get("objective_upper_bound_sum_z2", 0.0))
    )
    z_best = float(np.sqrt(refined_sum_z2))
    log_message(
        f"  Selected {len(top_bins)} signal regions, "
        f"sum(Z^2)={refined_sum_z2:.6g}, Z_comb={z_best:.4f}, "
        f"nodes={summary['nodes']}, selector={summary['selector']}"
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
