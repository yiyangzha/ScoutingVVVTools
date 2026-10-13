#!/usr/bin/env python3
"""theory_syst.py  (mode 8)

Computes theory weight variation yield ratios for all samples that have
``has_theory_weights: true`` in sample.json.  The converted ROOT files must
include the branches added by mode 0:
  genWeight      (scalar)
  LHEPdfWeight   [101]
  LHEScaleWeight [9]
  PSWeight       [4]

For each (sample, tree) it always reports the **inclusive** (post-preselection)
ratios, and -- when a signal-region CSV and the matching BDT model are
configured -- it ALSO reports the ratios **per signal region**.  Region
membership is taken from the nominal BDT scores (theory weights do not change
which region an event falls in), using the same per-class rectangle definition
that ``qcd_est.py`` applies, so the regions match the ABCD method exactly.

PDF uncertainty: the method must match the PDF set type, set via the config key
  ``pdf_uncertainty_method``:
    "hessian_symmetric" (DEFAULT) -- for symmetric-Hessian sets such as
      NNPDF31_nnlo_hessian_pdfas (LHA 306000) or
      NNPDF31_nnlo_as_0118_nf_4_mc_hessian (LHA 325500, used by the v5 VVV/VV
      samples).  sigma = sqrt(sum_i (Y_i/Y_0 - 1)^2)
      over the 100 Hessian members, applied as +/- around the central member.
    "mc_replica" -- PDF4LHC replica prescription (arXiv:1510.03865 §6.2):
      sort the 100 replica yields, take the 16th and 84th as the 68% CL interval.
  Using the replica prescription on a Hessian set underestimates the uncertainty
  by ~sqrt(N) ~ 10x, so the default is hessian_symmetric.
  alpha_s ("+as"): when the LHEPdfWeightAlphaS branch is present (mode 0 now
  stores the two alpha_s members 306101/306102), the alpha_s variation
  (Y_up - Y_down)/2 is folded into the PDF uncertainty in quadrature. Files
  produced before that convert change get the Hessian PDF part only. Sets
  without alpha_s members (e.g. LHA 325500, 101 weights) leave the branch at
  the 1.0 placeholder; such samples are reported with alphas_available=false.
Scale uncertainty: envelope of 7 (mu_R, mu_F) combinations excluding the two
  anti-correlated corners (LHEScaleWeight indices 2 and 6).
Normalisation to the generated events (2026-10-09): every member's yield ratio Y_i / Y_0 is divided
  by the member's value over all generated events, sum(genWeight * w_i) / sum(genWeight) from the
  NanoAOD Runs trees (generator_sums.py, config key generator_sums_json, required for every theory
  sample), before the envelope / quadrature, so the variations change only the selection
  efficiency: the samples are normalised to their sample.json cross-sections, whose inclusive
  uncertainty enters combine separately (combine.C xsec_unc_json). Before, the ratios also held
  the change of the inclusive cross-section.
PS uncertainty: PSWeight is [0] isr.murfac=2.0; [1] fsr.murfac=2.0;
  [2] isr.murfac=0.5; [3] fsr.murfac=0.5 (branch documentation). ps_isr_up /
  ps_isr_down are the isr.murfac = 0.5 / 2.0 variations (more / less ISR), and
  ps_fsr_up / ps_fsr_down the same for FSR, so each direction means the same
  variation in every region.
Regions: the BDT thresholds (the ABCD window included) are applied with
  qcd_est.py's semantics (sentinel values fail), so each SR is the ABCD A
  region; a configured but missing input is an error.
Weights: every selected-event ratio is sum(w * w_var) / sum(w) with the signed event weight
  w = genWeight * weight_pu (selections/mc_weight_common/mc_weights.py; its normalisation
  cancels in the ratio), divided by the generated-event ratio above, so it is the variation's
  change of the acceptance only.

Outputs (written to output_dir):
  theory_syst_yields.json  -- per-sample/tree dict; inclusive ratios at the top
                              level (back-compatible with combine.C), plus a
                              ``regions`` sub-dict keyed by signal-region
                              bin_index.
  theory_syst_yields.csv   -- flat CSV with a ``region`` column ("inclusive" or
                              the bin_index).
"""

import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# model_io lives under selections/BDT; make it importable for inference.
_BDT_TOOLS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "BDT"))
if _BDT_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _BDT_TOOLS_DIR)
import model_io  # noqa: E402
import kfold  # noqa: E402


def log(msg):
    print(msg, flush=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_path = os.environ.get(
    "THEORY_CONFIG_PATH",
    os.path.join(_SCRIPT_DIR, "config.json"),
)
if not os.path.isabs(_cfg_path):
    _cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, _cfg_path))

cfg = _load_json(_cfg_path)

SUBMIT_TREES  = cfg.get("submit_trees", ["fat2", "fat3"])
INPUT_ROOT    = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["input_root"]))
INPUT_PATTERN = cfg["input_pattern"]
OUTPUT_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg.get("output_dir", ".")))
CHUNK_SIZE    = cfg.get("chunk_size", "200 MB")
BDT_ROOT_CFG  = cfg.get("bdt_root", None)            # str or {tree: str}
SR_CSV_CFG    = cfg.get("signal_region_csv", None)   # str/pattern or {tree: str}
# Regions with fewer than this many post-preselection events fall back to the
# inclusive ratio (tight SRs can be statistically empty for a given sample).
MIN_REGION_EVENTS = int(cfg.get("min_region_events", 1))

# PDF combination method (must match the stored PDF set type).  These samples
# use the symmetric-Hessian set NNPDF31_nnlo_hessian_pdfas (LHA 306000), so the
# default is the Hessian quadrature; "mc_replica" is the 16/84 percentile method.
PDF_METHOD = str(cfg.get("pdf_uncertainty_method", "hessian_symmetric")).lower()
if PDF_METHOD not in ("hessian_symmetric", "mc_replica"):
    raise ValueError(
        f"pdf_uncertainty_method must be 'hessian_symmetric' or 'mc_replica', "
        f"got {PDF_METHOD!r}"
    )

# Per-member theory-weight sums over all generated events (generator_sums.py).
if "generator_sums_json" not in cfg:
    raise KeyError("config generator_sums_json is required: run generator_sums.py (the variations are "
                   "normalised to the generated events)")
_gen_sums_path = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["generator_sums_json"]))
if not os.path.exists(_gen_sums_path):
    raise FileNotFoundError(f"{_gen_sums_path} missing: run generator_sums.py and generator_sums.py --merge")
GENERATOR_SUMS = _load_json(_gen_sums_path)

_sample_cfg_path = os.path.normpath(os.path.join(_SCRIPT_DIR, cfg["sample_config"]))
sample_cfg   = _load_json(_sample_cfg_path)
SAMPLE_INFO  = {s["name"]: s for s in sample_cfg["sample"]}

THEORY_SAMPLES = [
    s["name"]
    for s in sample_cfg["sample"]
    if s.get("has_theory_weights", False) and s.get("is_MC", True)
]

# Optional restriction to a subset of samples (e.g. one sample per SLURM
# shard); absent/empty means "all theory-weight samples", preserving prior
# behaviour.
_submit_samples = cfg.get("submit_samples") or []
if _submit_samples:
    _submit_set = set(_submit_samples)
    THEORY_SAMPLES = [s for s in THEORY_SAMPLES if s in _submit_set]

# LHEScaleWeight indices to include (exclude anti-correlated corners 2 and 6).
SCALE_VALID = [0, 1, 3, 4, 5, 7, 8]

N_PDF_TOTAL    = 101
N_ALPHAS       = 2
N_SCALE        = 9
N_PS           = 4


# ---------------------------------------------------------------------------
# Per-tree config resolution
# ---------------------------------------------------------------------------

def _per_tree(value, tree):
    if isinstance(value, dict):
        return value.get(tree)
    return value


def _resolve_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_SCRIPT_DIR, path))


def _bdt_dir_for(tree):
    return _resolve_path(_per_tree(BDT_ROOT_CFG, tree))


def _sr_csv_for(tree):
    raw = _per_tree(SR_CSV_CFG, tree)
    if raw is None:
        return None
    raw = raw.format(tree_name=tree)  # allow a {tree_name} pattern
    return _resolve_path(raw)


# ---------------------------------------------------------------------------
# Sample I/O helpers
# ---------------------------------------------------------------------------

def _sample_group(name):
    info = SAMPLE_INFO[name]
    if not info.get("is_MC", True):
        return "data"
    return "signal" if info.get("is_signal", False) else "bkg"


def _input_files(sample_name):
    sg   = _sample_group(sample_name)
    base = INPUT_PATTERN.format(
        input_root=INPUT_ROOT, sample_group=sg, sample=sample_name
    )
    stem = base[:-5] if base.endswith(".root") else base
    return sorted(glob.glob(base) + glob.glob(stem + "_*.root"))


# ---------------------------------------------------------------------------
# Feature standardization (mirrors signal_region.py / train.py)
# ---------------------------------------------------------------------------

def _standardize_X(X, clip_ranges, log_transform):
    log_set = set(log_transform)
    for col in X.columns:
        arr = X[col].values.copy()
        mask = arr < -990
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


# ---------------------------------------------------------------------------
# Signal-region handling (mirrors qcd_est.py)
# ---------------------------------------------------------------------------

def _detect_signal_region_axes(df, class_names):
    axes = []
    columns = set(df.columns)
    for col in df.columns:
        if not col.endswith("_low"):
            continue
        axis_name = col[:-4]
        if f"{axis_name}_high" not in columns:
            continue
        if axis_name not in class_names:
            raise KeyError(
                f"Signal region axis {axis_name!r} is not in BDT class_groups: {class_names}. "
                f"theory_syst supports the per-class rectangle CSV (same as qcd_est)."
            )
        axes.append(axis_name)
    return axes


def _region_mask(proba, region_row, axis_names, class_names):
    mask = np.ones(proba.shape[0], dtype=bool)
    for axis_name in axis_names:
        low = float(region_row[f"{axis_name}_low"])
        high = float(region_row[f"{axis_name}_high"])
        axis_scores = proba[:, class_names.index(axis_name)]
        if high < 1.0 - 1e-12:
            mask &= (axis_scores >= low) & (axis_scores < high)
        else:
            mask &= axis_scores >= low
    return mask


# ---------------------------------------------------------------------------
# Per-tree context (thresholds always; model + regions when configured)
# ---------------------------------------------------------------------------

def _load_thresholds(bdt_dir, tree_name):
    if bdt_dir is None:
        return {}
    sel_path = os.path.join(bdt_dir, "selection.json")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"bdt_root has no selection.json: {sel_path}")
    sel = _load_json(sel_path)
    return sel.get(tree_name, {}).get("thresholds", {})


def _build_tree_context(tree):
    """Return a dict: always has 'thresholds'; has model/region info if enabled."""
    bdt_dir = _bdt_dir_for(tree)
    # k-fold models (train.py kfold = K): configs from fold0, every MC event scored with the model
    # of the k-fold block holding it (selections/BDT/kfold.py).
    kf = kfold.KFoldModels(bdt_dir, log=log) if bdt_dir is not None and kfold.is_kfold_root(bdt_dir) else None
    if kf is not None:
        bdt_dir = kf.fold_dirs[0]
    ctx = {"thresholds": _load_thresholds(bdt_dir, tree), "regions_enabled": False, "kfold": kf}

    sr_csv = _sr_csv_for(tree)
    if bdt_dir is None or sr_csv is None:
        log(f"  [{tree}] no signal_region_csv/bdt_root configured -> inclusive only")
        return ctx

    bcfg_path = os.path.join(bdt_dir, "config.json")
    brj_path  = os.path.join(bdt_dir, "branch.json")
    sel_path  = os.path.join(bdt_dir, "selection.json")
    for p in (sr_csv, bcfg_path, brj_path, sel_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{tree}] configured input not found: {p}")

    bcfg = _load_json(bcfg_path)
    brj  = _load_json(brj_path)
    selj = _load_json(sel_path)
    class_names = list(bcfg["class_groups"].keys())
    feature_cols = [b["name"] for b in brj[tree]]
    sel = selj.get(tree, {})
    clip_ranges = {k: tuple(v) for k, v in sel.get("clip_ranges", {}).items()}
    log_transform = sel.get("log_transform", [])
    decorrelate = bcfg.get(tree, {}).get("decorrelate", [])

    model_pattern = bcfg.get("model_pattern", "{output_root}/{tree_name}_model")
    model_base = model_pattern.format(output_root=bdt_dir, tree_name=tree)
    model = None if kf is not None else model_io.load_model(model_base, bcfg, len(class_names), log_message=log)

    sr_df = pd.read_csv(sr_csv)
    if sr_df.empty:
        raise ValueError(f"[{tree}] signal_region CSV is empty: {sr_csv}")
    axes = _detect_signal_region_axes(sr_df, class_names)
    if not axes:
        raise ValueError(f"[{tree}] no per-class axes in signal_region CSV: {sr_csv}")
    bin_ids = [int(round(float(v))) for v in sr_df["bin_index"].tolist()]

    ctx.update(
        regions_enabled=True, model=model, class_names=class_names,
        num_classes=len(class_names), feature_cols=feature_cols,
        clip_ranges=clip_ranges, log_transform=log_transform,
        decorrelate=decorrelate, sr_df=sr_df, axes=axes, bin_ids=bin_ids,
        sr_csv=sr_csv,
        # SR provenance for combine: first 13 hex digits of the SHA-256 of the SR CSV bytes.
        sr_csv_hash=hashlib.sha256(open(sr_csv, "rb").read()).hexdigest()[:13],
    )
    log(f"  [{tree}] regions enabled: {len(bin_ids)} SRs from {os.path.basename(sr_csv)}, "
        f"axes={axes}")
    return ctx


# ---------------------------------------------------------------------------
# Inference + masks
# ---------------------------------------------------------------------------

def _mask_from_cond(values, cond):
    """qcd_est.py's _mask_from_cond on a numpy array: a number is a lower cut,
    [low, high] a window (None = open), a list of conditions their OR, and a
    {"&"/"and"/"|"/"or": [...]} dict their AND/OR."""
    if cond is None:
        return np.ones(values.shape[0], dtype=bool)
    if isinstance(cond, (int, float)):
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
            mask |= _mask_from_cond(values, item)
        return mask
    if isinstance(cond, dict):
        for key, is_and in (("&", True), ("and", True), ("|", False), ("or", False)):
            if key not in cond:
                continue
            mask = np.full(values.shape[0], is_and, dtype=bool)
            for item in cond[key]:
                sub = _mask_from_cond(values, item)
                mask = (mask & sub) if is_and else (mask | sub)
            return mask
        raise ValueError(f"Unsupported dict condition keys: {cond}")
    raise TypeError(f"Unsupported condition type: {type(cond)}")


def _threshold_mask(data_np, thresholds):
    """Events passing every threshold, with qcd_est.py's semantics: a sentinel
    value (< -990) fails, otherwise the condition decides."""
    n = len(next(iter(data_np.values())))
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        if branch not in data_np:
            raise KeyError(f"Threshold branch {branch!r} was not loaded from the input tree")
        values = np.asarray(data_np[branch], dtype=float)
        mask &= ~(values < -990)
        mask &= _mask_from_cond(values, cond)
    return mask


def _infer_proba(chunk, mask, ctx, folds=None):
    cols = {}
    for name in ctx["feature_cols"]:
        if name not in chunk:
            raise KeyError(
                f"BDT feature branch {name!r} missing from input file; "
                f"cannot evaluate signal regions for this sample/tree."
            )
        cols[name] = np.asarray(chunk[name])[mask]
    X = pd.DataFrame(cols)
    X = _standardize_X(X, ctx["clip_ranges"], ctx["log_transform"])
    if ctx["decorrelate"]:
        drop = [c for c in ctx["decorrelate"] if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
    if ctx.get("kfold") is not None:
        return ctx["kfold"].predict(X, folds)
    return model_io.predict_model_proba(ctx["model"], X, ctx["num_classes"])


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def _generator_sums(sample_name):
    """The sample's per-member sums over all generated events, in the stored per-event layout."""
    g = GENERATOR_SUMS.get(sample_name)
    if g is None:
        raise KeyError(f"{sample_name} has no entry in {_gen_sums_path}: run generator_sums.py {sample_name} "
                       f"and generator_sums.py --merge")
    return {"pdf": np.asarray(g["lhe_pdf_sumw"], dtype=np.float64),
            "alphas": np.asarray(g["lhe_alphas_sumw"], dtype=np.float64),
            "scale": np.asarray(g["lhe_scale_sumw"], dtype=np.float64),
            "ps": np.asarray(g["ps_sumw"], dtype=np.float64)}


def _ratios_from_sums(nom_sum, pdf_sums, scale_sums, ps_sums, n_events, gen, alphas_sums=None):
    """Return the ratio dict for one accumulation slot, or None if empty. Every member's ratio is
    divided by its value over all generated events (gen, _generator_sums)."""
    if nom_sum <= 0.0 or n_events < 1:
        return None
    member_ratios = pdf_sums[1:101] / nom_sum / gen["pdf"][1:101]   # 100 PDF members (i = 1..100)
    central = float(pdf_sums[0] / nom_sum / gen["pdf"][0])          # central member, ~1.0
    if PDF_METHOD == "hessian_symmetric":
        # Symmetric-Hessian set (NNPDF31_nnlo_hessian_pdfas): the uncertainty is
        # the quadrature sum of the per-eigenvector deviations from the central.
        sigma_pdf = float(np.sqrt(np.sum((member_ratios - central) ** 2)))
        pdf_center = central
    else:
        # MC replica (PDF4LHC 68% CL): sort the 100 replicas, take 16th/84th.
        sorted_r = np.sort(member_ratios)
        lo = float(sorted_r[15]); hi = float(sorted_r[83])
        pdf_center = 0.5 * (hi + lo)
        sigma_pdf = 0.5 * (hi - lo)

    # alpha_s variation (NNPDF31 "+as": members 306101/306102 = alpha_s down/up),
    # folded into the PDF uncertainty in quadrature when the branch is present.
    sigma_as = 0.0
    if alphas_sums is not None:
        as_ratios = alphas_sums / nom_sum / gen["alphas"]   # [alpha_s_down, alpha_s_up]
        sigma_as = 0.5 * abs(float(as_ratios[1]) - float(as_ratios[0]))
    sigma_tot = float(np.sqrt(sigma_pdf ** 2 + sigma_as ** 2))
    pdf_up = pdf_center + sigma_tot
    pdf_down = pdf_center - sigma_tot
    pdf_central = pdf_center

    scale_ratios = scale_sums / nom_sum / gen["scale"]
    valid_s = scale_ratios[SCALE_VALID]
    scale_up   = float(valid_s.max())
    scale_down = float(valid_s.min())

    # PSWeight: [0] isr.murfac=2.0; [1] fsr.murfac=2.0; [2] isr.murfac=0.5;
    # [3] fsr.murfac=0.5. up = murfac 0.5 (more radiation), down = murfac 2.0.
    ps_ratios = ps_sums / nom_sum / gen["ps"]
    ps_isr_up   = float(ps_ratios[2])
    ps_isr_down = float(ps_ratios[0])
    ps_fsr_up   = float(ps_ratios[3])
    ps_fsr_down = float(ps_ratios[1])

    return {
        "n_events":    int(n_events),
        "nom_sum":     float(nom_sum),
        "generator_normalised": True,
        "pdf_method":  PDF_METHOD,
        "pdf_alphas_unc": float(sigma_as),
        "alphas_available": alphas_sums is not None,
        "pdf_up":      pdf_up,      "pdf_down":     pdf_down,  "pdf_central": pdf_central,
        "scale_up":    scale_up,    "scale_down":   scale_down,
        "ps_isr_up":   ps_isr_up,   "ps_isr_down":  ps_isr_down,
        "ps_fsr_up":   ps_fsr_up,   "ps_fsr_down":  ps_fsr_down,
    }


def _compute_ratios(sample_name, ctx):
    """Accumulate inclusive (slot 0) and per-region (slots 1..N) yield sums.

    Returns {"inclusive": ratios_or_None, "regions": {bin_id: ratios_or_None}};
    a slot is None when it has no nominal events.
    """
    files = _input_files(sample_name)
    if not files:
        raise FileNotFoundError(
            f"No converted files for theory sample {sample_name} under {INPUT_ROOT} "
            f"(pattern {INPUT_PATTERN}); convert it or leave it out of submit_samples"
        )

    thresholds = ctx["thresholds"]
    enabled = ctx["regions_enabled"]
    n_slots = 1 + (len(ctx["bin_ids"]) if enabled else 0)
    gen = _generator_sums(sample_name)

    # LHEPdfWeightAlphaS is optional (only present after re-running mode 0 with
    # the alpha_s branch); folded into the PDF uncertainty when available.
    theory_branches = ["LHEPdfWeight", "LHEScaleWeight", "PSWeight"]
    load_set = {"genWeight", "weight_pu", *theory_branches, *thresholds.keys()}
    if enabled:
        load_set.update(ctx["feature_cols"])
    load_list = sorted(load_set)

    nom_sum     = np.zeros(n_slots, dtype=np.float64)
    pdf_sums    = np.zeros((n_slots, N_PDF_TOTAL), dtype=np.float64)
    alphas_sums = np.zeros((n_slots, N_ALPHAS),    dtype=np.float64)
    scale_sums  = np.zeros((n_slots, N_SCALE),     dtype=np.float64)
    ps_sums     = np.zeros((n_slots, N_PS),        dtype=np.float64)
    n_events    = np.zeros(n_slots, dtype=np.int64)
    has_alphas = [False]
    kf = ctx.get("kfold") if enabled else None
    if kf is not None:
        kf.check_files(sample_name, files)
        offsets = kf.file_offsets(sample_name)

    def _accumulate(slot, w, pdf_w, scale_w, ps_w, as_w):
        nom_sum[slot]    += float(w.sum())
        pdf_sums[slot]   += (w[:, None] * pdf_w).sum(axis=0)
        scale_sums[slot] += (w[:, None] * scale_w).sum(axis=0)
        ps_sums[slot]    += (w[:, None] * ps_w).sum(axis=0)
        n_events[slot]   += int(w.shape[0])
        if as_w is not None:
            alphas_sums[slot] += (w[:, None] * as_w).sum(axis=0)

    for fpath in files:
        with uproot.open(fpath) as uf:
            if ctx["_tree"] not in uf:
                raise KeyError(f"{fpath} has no tree {ctx['_tree']!r}; re-run mode 0 (convert)")
            tree = uf[ctx["_tree"]]
            avail = set(tree.keys())
            missing = [b for b in load_list if b not in avail]
            if missing:
                raise KeyError(f"{fpath}:{ctx['_tree']} lacks the branches {missing}; "
                               f"re-run mode 0 (convert)")
            # LHEPdfWeightAlphaS and nLHEPdfWeight are optional (files converted before
            # these branches existed); alpha_s is folded into the PDF uncertainty when available.
            actual = load_list + [b for b in ("LHEPdfWeightAlphaS", "nLHEPdfWeight") if b in avail]
            for chunk, report in tree.iterate(expressions=actual, step_size=CHUNK_SIZE, library="np",
                                              report=True):
                mask = _threshold_mask(chunk, thresholds)
                if not mask.any():
                    continue
                # Signed event weight genWeight * weight_pu (selections/mc_weight_common/mc_weights.py);
                # the theory member weights are w_var / w_nominal.
                w_pu    = (np.asarray(chunk["genWeight"], dtype=np.float64)
                           * np.asarray(chunk["weight_pu"], dtype=np.float64))[mask]
                pdf_w   = np.asarray(chunk["LHEPdfWeight"],   dtype=np.float64)[mask]
                scale_w = np.asarray(chunk["LHEScaleWeight"], dtype=np.float64)[mask]
                ps_w    = np.asarray(chunk["PSWeight"],       dtype=np.float64)[mask]
                if "LHEPdfWeightAlphaS" in chunk:
                    as_w = np.asarray(chunk["LHEPdfWeightAlphaS"], dtype=np.float64)[mask]
                    # convert_branch.C fills 1.0 when the PDF set has no alpha_s
                    # members (e.g. LHA 325500 with 101 weights). It now also stores
                    # nLHEPdfWeight: alpha_s members exist iff >= 103 weights; older
                    # files fall back to treating non-placeholder values as present.
                    if "nLHEPdfWeight" in chunk:
                        if np.any(np.asarray(chunk["nLHEPdfWeight"])[mask] >= 103):
                            has_alphas[0] = True
                    elif np.any(as_w != 1.0):
                        has_alphas[0] = True
                else:
                    as_w = None

                _accumulate(0, w_pu, pdf_w, scale_w, ps_w, as_w)  # inclusive

                if enabled:
                    folds = (kf.fold_of_index(sample_name, offsets[fpath] + report.tree_entry_start
                                              + np.flatnonzero(mask)) if kf is not None else None)
                    proba = _infer_proba(chunk, mask, ctx, folds)
                    for j in range(len(ctx["bin_ids"])):
                        rmask = _region_mask(
                            proba, ctx["sr_df"].iloc[j], ctx["axes"], ctx["class_names"]
                        )
                        if rmask.any():
                            _accumulate(j + 1, w_pu[rmask], pdf_w[rmask],
                                        scale_w[rmask], ps_w[rmask],
                                        as_w[rmask] if as_w is not None else None)

    if not has_alphas[0]:
        log(f"    WARNING [{sample_name}] no alpha_s variations stored (LHEPdfWeightAlphaS absent "
            f"or placeholder 1.0): PDF uncertainty excludes alpha_s (alphas_available=false)")

    def _slot_alphas(slot):
        return alphas_sums[slot] if has_alphas[0] else None

    inclusive = _ratios_from_sums(nom_sum[0], pdf_sums[0], scale_sums[0], ps_sums[0],
                                  n_events[0], gen, _slot_alphas(0))
    regions = None
    if enabled:
        regions = {}
        for j, bid in enumerate(ctx["bin_ids"]):
            slot = j + 1
            regions[bid] = _ratios_from_sums(
                nom_sum[slot], pdf_sums[slot], scale_sums[slot], ps_sums[slot],
                n_events[slot], gen, _slot_alphas(slot)
            )
    return {"inclusive": inclusive, "regions": regions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYST_FIELDS = [
    ("pdf",    "pdf_up",    "pdf_down"),
    ("scale",  "scale_up",  "scale_down"),
    ("ps_isr", "ps_isr_up", "ps_isr_down"),
    ("ps_fsr", "ps_fsr_up", "ps_fsr_down"),
]


def _pct(r):
    return f"{100. * (r - 1.):+.2f}%"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("theory_syst.py — theory weight variations (inclusive + per signal region)")
    log(f"  pdf_method: {PDF_METHOD}")
    log(f"  variations normalised to the generated events: {_gen_sums_path}")
    log(f"  trees:   {SUBMIT_TREES}")
    log(f"  samples: {THEORY_SAMPLES}")
    log(f"  output:  {OUTPUT_DIR}\n")

    all_results = {}  # (sample, tree) -> entry dict

    for tree_name in SUBMIT_TREES:
        log(f"=== Tree: {tree_name} ===")
        ctx = _build_tree_context(tree_name)
        ctx["_tree"] = tree_name
        if ctx["thresholds"]:
            log(f"  pre-selection cuts: {list(ctx['thresholds'].keys())}")
        else:
            log("  no pre-selection cuts applied")

        for sample_name in THEORY_SAMPLES:
            log(f"  {sample_name} ...")
            res = _compute_ratios(sample_name, ctx)
            incl = res["inclusive"]
            if incl is None:
                log(f"    [{sample_name}/{tree_name}] no nominal events after the thresholds -> no entry")
                continue

            entry = dict(incl)  # inclusive ratios at top level (back-compatible)
            log(
                f"    inclusive: n={incl['n_events']}"
                f"  pdf=[{_pct(incl['pdf_down'])}, {_pct(incl['pdf_up'])}]"
                f"  scale=[{_pct(incl['scale_down'])}, {_pct(incl['scale_up'])}]"
                f"  ps_isr=[{_pct(incl['ps_isr_down'])}, {_pct(incl['ps_isr_up'])}]"
                f"  ps_fsr=[{_pct(incl['ps_fsr_down'])}, {_pct(incl['ps_fsr_up'])}]"
            )

            if res["regions"] is not None:
                region_out = {}
                for bid, rr in res["regions"].items():
                    if rr is None or rr["n_events"] < MIN_REGION_EVENTS:
                        n_have = 0 if rr is None else rr["n_events"]
                        log(f"    SR{bid}: only {n_have} events "
                            f"(< {MIN_REGION_EVENTS}); taking the inclusive ratios")
                        region_out[str(bid)] = {
                            **incl, "n_events": n_have,
                            "nom_sum": 0.0 if rr is None else rr["nom_sum"],
                            "fallback_to_inclusive": True,
                        }
                    else:
                        region_out[str(bid)] = rr
                        log(f"    SR{bid}: n={rr['n_events']}"
                            f"  pdf=[{_pct(rr['pdf_down'])}, {_pct(rr['pdf_up'])}]"
                            f"  scale=[{_pct(rr['scale_down'])}, {_pct(rr['scale_up'])}]")
                entry["regions"] = region_out
                entry["sr_csv_hash"] = ctx["sr_csv_hash"]

            all_results[(sample_name, tree_name)] = entry

    # ---- Write JSON ----
    json_path = os.path.join(OUTPUT_DIR, "theory_syst_yields.json")
    nested = {}
    for (sample, tree), entry in all_results.items():
        nested.setdefault(sample, {})[tree] = entry
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2)
    log(f"\nWrote {json_path}")

    # ---- Write CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "theory_syst_yields.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "tree", "region", "systematic", "up_ratio", "down_ratio"])
        for (sample, tree) in sorted(all_results.keys()):
            entry = all_results[(sample, tree)]
            for syst, up_key, down_key in SYST_FIELDS:
                writer.writerow([sample, tree, "inclusive", syst,
                                 f"{entry[up_key]:.6f}", f"{entry[down_key]:.6f}"])
            for bid, rr in entry.get("regions", {}).items():
                for syst, up_key, down_key in SYST_FIELDS:
                    writer.writerow([sample, tree, bid, syst,
                                     f"{rr[up_key]:.6f}", f"{rr[down_key]:.6f}"])
    log(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
