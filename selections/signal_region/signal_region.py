import os
import json
import gc
import pickle
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import xgboost as xgb
from itertools import product as iproduct

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)

_EPS = 1e-12
# _SCRIPT_DIR is selections/signal_region/.
# Trained BDT config copies still store paths relative to selections/BDT/, where train.py lives.
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_SELECTIONS_DIR = os.path.dirname(_SCRIPT_DIR)
_BDT_DIR        = os.path.join(_SELECTIONS_DIR, "BDT")


# ── Logging ────────────────────────────────────────────────────────────────────
def log_message(message):
    print(message, flush=True)

def log_warning(message):
    log_message(f"Warning: {message}")

def log_info(message):
    log_message(f"Info: {message}")

def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Load config.json ─────────────────────────────────────────────────────────────
_scan_cfg_path = os.environ.get("SCAN_CONFIG_PATH", os.path.join(_SCRIPT_DIR, "config.json"))
if not os.path.isabs(_scan_cfg_path):
    _scan_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _scan_cfg_path))

scan_cfg = _load_json(_scan_cfg_path)

LUMI           = float(scan_cfg["lumi"])
N_BINS         = int(scan_cfg["N"])
OUTPUT_ROOT    = scan_cfg["output_root"]
N_THRESHOLDS   = int(scan_cfg.get("n_thresholds", 10))
MIN_BKG_WEIGHT = float(scan_cfg.get("min_bkg_weight", 5.0))
ROUNDS         = int(scan_cfg.get("rounds", 5))

if not os.path.isabs(OUTPUT_ROOT):
    OUTPUT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, OUTPUT_ROOT))


# ── Load configs saved by train.py ────────────────────────────────────────────
cfg       = _load_json(os.path.join(OUTPUT_ROOT, "config.json"))
br_cfg    = _load_json(os.path.join(OUTPUT_ROOT, "branch.json"))
sel_cfg   = _load_json(os.path.join(OUTPUT_ROOT, "selection.json"))
test_meta = _load_json(os.path.join(OUTPUT_ROOT, "test_ranges.json"))

# sample_config paths in config.json are relative to _BDT_DIR (where train.py lives)
_sample_cfg_path = cfg["sample_config"]
if not os.path.isabs(_sample_cfg_path):
    _sample_cfg_path = os.path.normpath(os.path.join(_BDT_DIR, _sample_cfg_path))
sample_cfg = _load_json(_sample_cfg_path)

TREE_NAME     = test_meta["tree_name"]
MODEL_PATTERN = cfg.get("model_pattern", "{output_root}/{tree_name}_model")


# ── Sample registry ────────────────────────────────────────────────────────────
SAMPLE_INFO = {}
for _rule in sample_cfg["sample"]:
    SAMPLE_INFO[_rule["name"]] = {
        "xsection":    _rule["xsection"],
        "raw_entries": _rule.get("raw_entries", -1),
        "is_MC":       _rule["is_MC"],
        "is_signal":   _rule["is_signal"],
        "sample_ID":   _rule["sample_ID"],
    }

CLASS_GROUPS = cfg["class_groups"]
CLASS_NAMES  = list(CLASS_GROUPS.keys())
NUM_CLASSES  = len(CLASS_NAMES)

SIGNAL_CLASS_INDICES     = []
BACKGROUND_CLASS_INDICES = []
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    _flags = [SAMPLE_INFO[_s]["is_signal"] for _s in _members]
    if _flags and all(_flags):
        SIGNAL_CLASS_INDICES.append(_idx)
    else:
        BACKGROUND_CLASS_INDICES.append(_idx)

SAMPLE_TO_CLASS = {}
for _idx, (_cls, _members) in enumerate(CLASS_GROUPS.items()):
    for _s in _members:
        SAMPLE_TO_CLASS[_s] = _idx


# ── Data loading ───────────────────────────────────────────────────────────────
def load_test_data(branches):
    """Load test events from test_ranges.json with physics-normalised weights.

    For each sample:
      total_weight = lumi * xsec * total_tree_entries / raw_entries
      per_event_weight = total_weight / N_test_loaded

    Weights are fixed here; threshold filtering later does NOT rescale them.
    """
    log_message(f"Loading test data from: {os.path.join(OUTPUT_ROOT, 'test_ranges.json')}")
    dfs = []

    for sample_name, sample_meta in test_meta["samples"].items():
        info = SAMPLE_INFO.get(sample_name)
        if info is None:
            log_warning(f"Sample '{sample_name}' not in sample config, skipping")
            continue
        if sample_name not in SAMPLE_TO_CLASS:
            log_warning(f"Sample '{sample_name}' not in any class group, skipping")
            continue

        xsec          = float(info["xsection"])
        raw_entries   = int(info["raw_entries"])
        total_entries = int(sample_meta["total_entries"])

        parts = []
        for seg in sample_meta["test_segments"]:
            fpath = seg["file"]
            if not os.path.exists(fpath):
                log_warning(f"File not found: {fpath}, skipping segment")
                continue
            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        log_warning(f"Tree '{TREE_NAME}' not in {fpath}, skipping")
                        continue
                    df_part = uf[TREE_NAME].arrays(
                        branches,
                        library="pd",
                        entry_start=int(seg["entry_start"]),
                        entry_stop=int(seg["entry_stop"]),
                    )
                    parts.append(df_part)
            except Exception as exc:
                log_warning(f"Failed to read {fpath}: {exc}, skipping")
                continue

        if not parts:
            log_warning(f"No data loaded for '{sample_name}', skipping")
            continue

        df      = pd.concat(parts, ignore_index=True)
        n_loaded = len(df)

        if xsec <= 0.0 or raw_entries <= 0:
            target_total = 0.0
            df["weight"] = 0.0
            log_warning(
                f"  {sample_name}: non-positive xsec={xsec} or raw_entries={raw_entries}, zero weight"
            )
        else:
            # Normalise sample total weight to lumi * xsec * total_tree_entries / raw_entries.
            # This represents the expected event count after tree-level selection at the given lumi.
            target_total  = LUMI * xsec * total_entries / raw_entries
            df["weight"]  = target_total / n_loaded

        df["class_idx"]   = SAMPLE_TO_CLASS[sample_name]
        df["sample_name"] = sample_name
        dfs.append(df)

        log_message(
            f"  {sample_name}: n_loaded={n_loaded}, total_entries={total_entries}, "
            f"raw_entries={raw_entries}, xsec={xsec:.6g}, target_total={target_total:.6g}, "
            f"class={CLASS_NAMES[SAMPLE_TO_CLASS[sample_name]]}"
        )

    if not dfs:
        raise RuntimeError("No test data loaded")

    df_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df_all


# ── Feature standardisation (identical to train.py) ───────────────────────────
def standardize_X(X: pd.DataFrame, clip_ranges: dict, log_transform: list) -> pd.DataFrame:
    log_set = set(log_transform)
    for col in X.columns:
        arr   = X[col].values.copy()
        mask  = arr < -990
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


# ── Threshold filtering (identical to train.py) ───────────────────────────────
def filter_X(X: pd.DataFrame, y, w, branch: list,
             thresholds: dict = None, apply_to_sentinel: bool = True,
             sample_labels=None):
    if thresholds is None:
        if sample_labels is None:
            return X.copy(), y.copy(), w.copy()
        return X.copy(), y.copy(), w.copy(), np.asarray(sample_labels).copy()

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
        if isinstance(cond, (list, tuple)) and len(cond) == 2 and \
                not isinstance(cond[0], (list, dict, tuple)):
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

    X_out = X.loc[mask].copy()
    y_out = y[mask.values].copy()
    w_out = w[mask.values].copy()
    if sample_labels is None:
        return X_out, y_out, w_out
    return X_out, y_out, w_out, np.asarray(sample_labels)[mask.values].copy()


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _slugify(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")

def _savefig(stem):
    path = os.path.join(OUTPUT_ROOT, f"{stem}.pdf")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log_message(f"Wrote plot file: {path}")


def plot_score_distributions(proba, y, w):
    """Weighted BDT score distributions per scan axis (one plot per axis)."""
    D = max(1, proba.shape[1] - 1)
    palette = plt.cm.get_cmap("tab10", max(NUM_CLASSES, 3))(np.arange(max(NUM_CLASSES, 3)))
    bins = np.linspace(0.0, 1.0, 51)

    for d in range(D):
        axis_name = CLASS_NAMES[d]
        score = proba[:, d]
        plt.figure(figsize=(8, 6))
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            m = (y == cls_i)
            if not np.any(m):
                continue
            plt.hist(score[m], bins=bins, weights=w[m],
                     histtype="step", linewidth=2,
                     color=palette[cls_i], label=cls_name)
        plt.xlabel(f"p({axis_name})")
        plt.ylabel(f"Events / {LUMI:.0f} fb$^{{-1}}$")
        plt.yscale("log")
        plt.xlim(0, 1)
        plt.legend(fontsize=12)
        _savefig(f"sr_score_{_slugify(axis_name)}")


def plot_signal_regions_2d(result, proba, y, w):
    """2D scatter of the first two BDT axes with signal region boxes overlaid."""
    if proba.shape[1] < 3 or not result["top_bins"]:
        return

    is_sig = np.isin(y, SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, BACKGROUND_CLASS_INDICES)
    palette = plt.cm.get_cmap("Set1", max(len(result["top_bins"]), 1))

    plt.figure(figsize=(8, 8))
    plt.scatter(proba[is_bkg, 0], proba[is_bkg, 1],
                s=1, alpha=0.2, c="steelblue", label="Background", rasterized=True)
    plt.scatter(proba[is_sig, 0], proba[is_sig, 1],
                s=2, alpha=0.5, c="tomato", label="Signal", rasterized=True)

    for i, b in enumerate(result["top_bins"]):
        lo, hi = b["thr_low"], b["thr_high"]
        rect = plt.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=2, edgecolor=palette(i), facecolor="none",
            label=f"Bin {b['bin_index']} (Z={b['significance']:.2f})"
        )
        plt.gca().add_patch(rect)

    plt.xlabel(f"p({CLASS_NAMES[0]})")
    plt.ylabel(f"p({CLASS_NAMES[1]})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=10, markerscale=5)
    _savefig("sr_regions_2d")


# ── Signal region scan ────────────────────────────────────────────────────────
def find_signal_regions(proba, y, w):
    """Find N non-overlapping signal regions maximising Z = sqrt(2[(S+B)ln(1+S/B) - S]).

    Scan dimensions D = NUM_CLASSES - 1; axes are proba[:,0] … proba[:,D-1].
    Generalised from find_optimal_significance_combine in train.ipynb.
    """
    n_events, n_cls = proba.shape
    D = max(1, n_cls - 1)

    score_axes = [proba[:, d] for d in range(D)]
    axis_names = [CLASS_NAMES[d] for d in range(D)]

    is_sig = np.isin(y, SIGNAL_CLASS_INDICES)
    is_bkg = np.isin(y, BACKGROUND_CLASS_INDICES)

    S_total = float(w[is_sig].sum())
    B_total = float(w[is_bkg].sum())

    log_message(f"  S_total={S_total:.4g}, B_total={B_total:.4g}")
    log_message(f"  Scan dimensions D={D}, axes={axis_names}")

    def _calc_Z(S, B, sS, sB):
        """Profile-likelihood significance Z = sqrt(2[(S+B)ln(1+S/B) - S]) with propagated error."""
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

    def _calc_Z_val(S, B):
        """Fast significance for scan ranking."""
        if S <= 0.0 or B <= 0.0:
            return 0.0
        f = (S + B) * np.log(1.0 + S / B) - S
        return float(np.sqrt(2.0 * max(0.0, f)))

    # 1D tail scan (for efficiency references only)
    p_exp = 0.005
    thr_1d = np.clip(np.linspace(0.0, 1.0, max(2, N_THRESHOLDS)) ** p_exp, 0.0, 1.0)
    T = len(thr_1d)
    S_tail_by_dim = np.zeros((D, T))
    B_tail_by_dim = np.zeros((D, T))
    for d in range(D):
        s = score_axes[d]
        for it, thr in enumerate(thr_1d):
            m = s >= thr
            S_tail_by_dim[d, it] = float(w[m & is_sig].sum())
            B_tail_by_dim[d, it] = float(w[m & is_bkg].sum())

    def _overlap(lo1, hi1, lo2, hi2, eps=1e-12):
        for dim in range(len(lo1)):
            if not (lo1[dim] < hi2[dim] - eps and lo2[dim] < hi1[dim] - eps):
                return False
        return True

    def _build_tail(mask_sel, edges_):
        """Cumulative-sum tail tensor: tail[i,j,...] = weight for score[d] >= base[d][i]."""
        arrs  = tuple(score_axes[dim][mask_sel] for dim in range(D))
        Hw    = np.histogramdd(arrs, bins=edges_, weights=w[mask_sel])[0]
        T_arr = Hw.copy()
        for ax in range(D):
            T_arr = np.flip(T_arr, axis=ax)
            T_arr = np.cumsum(T_arr, axis=ax)
            T_arr = np.flip(T_arr, axis=ax)
        return T_arr

    def _rect_sum(tail, lows_idx, highs_idx_vec):
        """Inclusion-exclusion weighted sum over a rectangular bin."""
        tot = 0.0
        for bits in range(1 << D):
            idx = []
            pop = 0
            for dim in range(D):
                if (bits >> dim) & 1:
                    idx.append(int(highs_idx_vec[dim]))
                    pop += 1
                else:
                    idx.append(int(lows_idx[dim]))
            tot += (-1.0 if (pop & 1) else 1.0) * tail[tuple(idx)]
        return float(tot)

    selected_bins = []
    top_bins      = []

    for k in range(N_BINS):
        log_message(f"  Scanning bin {k + 1}/{N_BINS} ...")
        accum_map = {}
        unit_high = [1.0] * D

        cand_vals = [
            list(np.linspace(0.0, unit_high[d], max(2, N_THRESHOLDS), endpoint=False))
            for d in range(D)
        ]

        for r in range(ROUNDS):
            # Build per-axis grids for this refinement round
            base_lists = []
            val2idx    = []
            highs_idx  = []
            edges_r    = []
            for d in range(D):
                base = np.unique(np.r_[cand_vals[d], unit_high[d]])
                base_lists.append(base)
                val2idx.append({float(v): i for i, v in enumerate(base)})
                highs_idx.append(int(val2idx[-1][float(unit_high[d])]))
                edges_r.append(np.r_[base, base[-1] + 1e-9])

            S_tailD = _build_tail(is_sig, edges_r)
            B_tailD = _build_tail(is_bkg, edges_r)

            low_idx_lists = []
            for d in range(D):
                idxs = [val2idx[d].get(float(v)) for v in cand_vals[d]]
                idxs = [int(i) for i in idxs if i is not None and i < highs_idx[d]]
                if not idxs:
                    idxs = [0]
                low_idx_lists.append(sorted(set(idxs)))

            combos_round = []
            for slice_dim in range(D):
                for lows in iproduct(*low_idx_lists):
                    lows = list(lows)
                    hi_idx_vec = []
                    valid = True
                    for d in range(D):
                        if d == slice_dim:
                            li = lows[d]
                            if li + 1 <= highs_idx[d]:
                                hi_idx_vec.append(li + 1)
                            else:
                                valid = False
                                break
                        else:
                            hi_idx_vec.append(highs_idx[d])
                    if not valid:
                        continue

                    S_bin = _rect_sum(S_tailD, lows, hi_idx_vec)
                    B_bin = _rect_sum(B_tailD, lows, hi_idx_vec)
                    Z     = _calc_Z_val(S_bin, B_bin)

                    thr_low_vals  = [float(base_lists[d][lows[d]]) for d in range(D)]
                    thr_high_vals = []
                    for d in range(D):
                        if d == slice_dim:
                            thr_high_vals.append(float(base_lists[d][lows[d] + 1]))
                        else:
                            thr_high_vals.append(float(unit_high[d]))

                    non_overlap = all(
                        not _overlap(thr_low_vals, thr_high_vals, lo_s, hi_s)
                        for (lo_s, hi_s) in selected_bins
                    )
                    if not non_overlap:
                        continue

                    combos_round.append((Z, thr_low_vals, thr_high_vals, S_bin, B_bin))

            combos_round.sort(key=lambda x: x[0], reverse=True)

            for Z, lows_vals, highs_vals, S_est, B_est in combos_round:
                key = (tuple(round(v, 12) for v in lows_vals),
                       tuple(round(v, 12) for v in highs_vals))
                if key not in accum_map or Z > accum_map[key]["Z"]:
                    accum_map[key] = {
                        "Z": Z, "S": S_est, "B": B_est,
                        "lows": tuple(lows_vals), "highs": tuple(highs_vals),
                    }

            # Pick top-3 thresholds per axis for next-round refinement
            accum_sorted = sorted(accum_map.values(), key=lambda d: d["Z"], reverse=True)
            pick_per_dim = [[] for _ in range(D)]
            seen_per_dim = [set() for _ in range(D)]
            for item in accum_sorted:
                for d in range(D):
                    v = float(item["lows"][d])
                    if v not in seen_per_dim[d]:
                        seen_per_dim[d].add(v)
                        pick_per_dim[d].append(v)
                if all(len(pick_per_dim[d]) >= 3 for d in range(D)):
                    break

            new_cands = []
            for d in range(D):
                picks = sorted(pick_per_dim[d]) if pick_per_dim[d] else [0.0, unit_high[d]]
                lo_d  = float(max(0.0, min(picks)))
                hi_d  = float(min(unit_high[d], max(picks)))
                if hi_d <= lo_d + 1e-12:
                    span = max(1e-3, 0.1 * unit_high[d])
                    lo_d = max(0.0, unit_high[d] - span)
                    hi_d = unit_high[d]
                new_cands.append(
                    list(np.linspace(lo_d, hi_d, max(2, N_THRESHOLDS), endpoint=False))
                )
            cand_vals = new_cands

        # Select best valid bin from accumulated candidates
        accum_sorted = sorted(accum_map.values(), key=lambda d: d["Z"], reverse=True)
        chosen = None
        for item in accum_sorted:
            if item["B"] >= MIN_BKG_WEIGHT and all(
                not _overlap(item["lows"], item["highs"], lo_s, hi_s)
                for (lo_s, hi_s) in selected_bins
            ):
                chosen = item
                break

        if chosen is None:
            log_message(f"  No valid bin found for bin {k + 1}, stopping early")
            break

        thr_low_vec  = list(map(float, chosen["lows"]))
        thr_high_vec = list(map(float, chosen["highs"]))

        # Exact event-level statistics for the chosen bin
        def _mask_dim(dim_, lo_, hi_):
            s_ = score_axes[dim_]
            return (s_ >= lo_) & (s_ < hi_) if hi_ < 1.0 - 1e-12 else s_ >= lo_

        m_bin = np.ones(n_events, dtype=bool)
        for d in range(D):
            m_bin &= _mask_dim(d, thr_low_vec[d], thr_high_vec[d])

        wS = w[m_bin & is_sig]
        wB = w[m_bin & is_bkg]
        S_bin  = float(wS.sum())
        B_bin  = float(wB.sum())
        sS_bin = float(np.sqrt((wS ** 2).sum()))
        sB_bin = float(np.sqrt((wB ** 2).sum()))
        S_e    = int((m_bin & is_sig).sum())
        B_e    = int((m_bin & is_bkg).sum())
        Z_bin, sZ_bin = _calc_Z(S_bin, B_bin, sS_bin, sB_bin)

        selected_bins.append((thr_low_vec[:], thr_high_vec[:]))

        # Per-class breakdown (each class treated as "signal vs all others in bin")
        W_bin  = S_bin + B_bin
        w2_bin = sS_bin ** 2 + sB_bin ** 2

        cat_data = []
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            mC   = (y == cls_i) & m_bin
            wC   = w[mC]
            S_j  = float(wC.sum())
            sS_j = float(np.sqrt((wC ** 2).sum()))
            B_j  = W_bin - S_j
            sB_j = float(np.sqrt(max(0.0, w2_bin - sS_j ** 2)))
            Z_j, sZ_j = _calc_Z(S_j, B_j, sS_j, sB_j)
            cat_data.append({
                "name":  cls_name,
                "S":     S_j,  "S_err": sS_j,
                "B":     B_j,  "B_err": sB_j,
                "Z":     Z_j,  "Z_err": sZ_j,
            })

        bkg_data = []
        for bkg_i in BACKGROUND_CLASS_INDICES:
            mC = (y == bkg_i) & m_bin
            wC = w[mC]
            bkg_data.append({
                "name":  CLASS_NAMES[bkg_i],
                "B":     float(wC.sum()),
                "B_err": float(np.sqrt((wC ** 2).sum())),
            })

        bin_sig_eff = (S_bin / S_total) if S_total > 0 else float("nan")
        bin_bkg_eff = (B_bin / B_total) if B_total > 0 else float("nan")

        tail_sig_eff, tail_bkg_eff = [], []
        for d in range(D):
            idx = max(0, min(
                int(np.searchsorted(thr_1d, thr_low_vec[d], side="right") - 1), T - 1
            ))
            tail_sig_eff.append(
                (S_tail_by_dim[d, idx] / S_total) if S_total > 0 else float("nan")
            )
            tail_bkg_eff.append(
                (B_tail_by_dim[d, idx] / B_total) if B_total > 0 else float("nan")
            )

        top_bins.append({
            "bin_index":                 k + 1,
            "thr_low":                   np.array(thr_low_vec),
            "thr_high":                  np.array(thr_high_vec),
            "axis_names":                axis_names,
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
            f"  Bin {k + 1}: Z={Z_bin:.4f}±{sZ_bin:.4f}, "
            f"S={S_bin:.4g}±{sS_bin:.4g}, B={B_bin:.4g}±{sB_bin:.4g}"
        )

    # Combined significance: Z_comb = sqrt(sum Z_i^2)
    if top_bins:
        S_comb  = float(sum(b["S"]       for b in top_bins))
        B_comb  = float(sum(b["B"]       for b in top_bins))
        sS_comb = float(np.sqrt(sum(b["S_err"] ** 2 for b in top_bins)))
        sB_comb = float(np.sqrt(sum(b["B_err"] ** 2 for b in top_bins)))
        Z_vec   = np.array([b["significance"]       for b in top_bins])
        sZ_vec  = np.array([b["significance_error"] for b in top_bins])
        Z_comb  = float(np.sqrt(np.sum(Z_vec ** 2)))
        sZ_comb = (float(np.sqrt(np.sum((Z_vec * sZ_vec) ** 2))) / Z_comb
                   if Z_comb > 0 else float(np.sqrt(np.sum(sZ_vec ** 2))))
    else:
        S_comb = B_comb = sS_comb = sB_comb = Z_comb = sZ_comb = 0.0

    return {
        "top_bins":                    top_bins,
        "combined_S":                  S_comb,
        "combined_B":                  B_comb,
        "combined_S_err":              sS_comb,
        "combined_B_err":              sB_comb,
        "combined_significance":       Z_comb,
        "combined_significance_error": sZ_comb,
        "S_total":                     S_total,
        "B_total":                     B_total,
    }


# ── Result printing ────────────────────────────────────────────────────────────
def print_results(result):
    top_bins = result["top_bins"]

    for b in top_bins:
        log_message(f"\n-- Bin {b['bin_index']} --")
        region_parts = [
            f"{b['axis_names'][d]}: [{b['thr_low'][d]:.6f}, {b['thr_high'][d]:.6f})"
            for d in range(len(b["axis_names"]))
        ]
        log_message(f"  Region: " + "  ".join(region_parts))
        log_message(f"  Significance: {b['significance']:.4f} ± {b['significance_error']:.4f}")
        log_message(
            f"  S (weighted): {b['S']:.4g} ± {b['S_err']:.4g}  |  S (entries): {b['S_entries']}"
        )
        log_message(
            f"  B (weighted): {b['B']:.4g} ± {b['B_err']:.4g}  |  B (entries): {b['B_entries']}"
        )
        log_message(
            f"  Signal eff. (bin): {b['bin_signal_efficiency']:.6f}  |  "
            f"Bkg eff. (bin): {b['bin_background_efficiency']:.6f}"
        )
        if b["tail_signal_efficiency"]:
            tse = "  ".join(
                f"{b['axis_names'][d]}={b['tail_signal_efficiency'][d]:.6f}"
                for d in range(len(b["axis_names"]))
            )
            tbe = "  ".join(
                f"{b['axis_names'][d]}={b['tail_background_efficiency'][d]:.6f}"
                for d in range(len(b["axis_names"]))
            )
            log_message(f"  Tail signal eff.:   {tse}")
            log_message(f"  Tail bkg eff.:      {tbe}")

        log_message("  Per-class significance:")
        for cat in b["categories"]:
            log_message(f"    {cat['name']}: {cat['Z']:.4f} ± {cat['Z_err']:.4f}")

        log_message("  Per-class breakdown (weighted ± error):")
        for cat in b["categories"]:
            log_message(
                f"    {cat['name']}: S={cat['S']:.4g} ± {cat['S_err']:.4g}  |  "
                f"B(rest)={cat['B']:.4g} ± {cat['B_err']:.4g}"
            )

        log_message("  Background breakdown (weighted ± error):")
        for bkg in b["backgrounds"]:
            log_message(f"    {bkg['name']}: B={bkg['B']:.4g} ± {bkg['B_err']:.4g}")

    log_message(f"\n==== Combined (top {len(top_bins)} bins) ====")
    log_message(f"  S total:               {result['combined_S']:.4g} ± {result['combined_S_err']:.4g}")
    log_message(f"  B total:               {result['combined_B']:.4g} ± {result['combined_B_err']:.4g}")
    log_message(
        f"  Combined significance: {result['combined_significance']:.4f} ± "
        f"{result['combined_significance_error']:.4f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log_message(
        f"Running signal_region.py: tree={TREE_NAME}, lumi={LUMI} fb^-1, "
        f"N={N_BINS}, output={OUTPUT_ROOT}"
    )

    sel         = sel_cfg[TREE_NAME]
    branches    = [b["name"] for b in br_cfg[TREE_NAME]]
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_tf      = sel.get("log_transform", [])
    thresholds  = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in sel.get("thresholds", {}).items()}
    decorrelate = cfg.get(TREE_NAME, {}).get("decorrelate", [])

    # Load test data; physics weights are assigned here and do not change afterwards
    df_all = load_test_data(branches)
    X             = df_all[branches].copy()
    y             = df_all["class_idx"].values.astype(int)
    w             = df_all["weight"].values.astype(float)
    sample_labels = df_all["sample_name"].values
    del df_all
    gc.collect()

    # Apply threshold filtering — weights unchanged by this step
    log_message("Applying thresholds ...")
    X, y, w, sample_labels = filter_X(
        X, y, w, branches, thresholds, apply_to_sentinel=True, sample_labels=sample_labels
    )
    log_message(f"After filtering: {len(X)} events")

    # Apply feature standardisation (same as train.py)
    log_message("Standardising features ...")
    X = standardize_X(X, clip_ranges, log_tf)

    # Remove decorrelated features before model input (same exclusion as during training)
    all_feature_names = list(X.columns)
    if decorrelate:
        name_to_idx = {c: i for i, c in enumerate(all_feature_names)}
        decor_idx   = sorted(name_to_idx[k] for k in decorrelate if k in name_to_idx)
        keep_idx    = [i for i in range(len(all_feature_names)) if i not in decor_idx]
        X_model     = X.iloc[:, keep_idx]
        log_message(f"Removed decorrelated features: {decorrelate}")
    else:
        X_model = X

    # Load trained model
    model_base = MODEL_PATTERN.format(output_root=OUTPUT_ROOT, tree_name=TREE_NAME)
    if os.path.exists(model_base + ".json"):
        model_path = model_base + ".json"
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        log_message(f"Loaded model: {model_path}")
    elif os.path.exists(model_base + ".pkl"):
        model_path = model_base + ".pkl"
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        log_message(f"Loaded model: {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_base}(.json/.pkl)")

    # BDT prediction
    log_message("Running BDT prediction ...")
    proba = clf.predict_proba(X_model)
    log_message(f"Predicted probabilities shape: {proba.shape}")

    # Plot weighted score distributions
    log_message("Plotting score distributions ...")
    plot_score_distributions(proba, y, w)

    # Scan for N non-overlapping signal regions
    log_message(f"Scanning for {N_BINS} signal regions ...")
    result = find_signal_regions(proba, y, w)

    # Plot 2D signal regions (first two axes, if D >= 2)
    log_message("Plotting signal regions ...")
    plot_signal_regions_2d(result, proba, y, w)

    # Print text summary
    print_results(result)

    log_message(f"Finished signal_region.py for tree={TREE_NAME}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_message(f"Runtime error: {ex}")
        raise
