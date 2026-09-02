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
      NNPDF31_nnlo_hessian_pdfas (LHA 306000), which is what these samples use
      (LHE combine type "symmhessian+as").  sigma = sqrt(sum_i (Y_i/Y_0 - 1)^2)
      over the 100 Hessian members, applied as +/- around the central member.
    "mc_replica" -- PDF4LHC replica prescription (arXiv:1510.03865 §6.2):
      sort the 100 replica yields, take the 16th and 84th as the 68% CL interval.
  Using the replica prescription on a Hessian set underestimates the uncertainty
  by ~sqrt(N) ~ 10x, so the default is hessian_symmetric.
  alpha_s ("+as"): when the LHEPdfWeightAlphaS branch is present (mode 0 now
  stores the two alpha_s members 306101/306102), the alpha_s variation
  (Y_up - Y_down)/2 is folded into the PDF uncertainty in quadrature. Files
  produced before that convert change get the Hessian PDF part only.
Scale uncertainty: envelope of 7 (mu_R, mu_F) combinations excluding the two
  anti-correlated corners (LHEScaleWeight indices 2 and 6).
PS uncertainty: ISR and FSR envelope from PSWeight[0..3].

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
        return {}
    sel = _load_json(sel_path)
    return sel.get(tree_name, {}).get("thresholds", {})


def _build_tree_context(tree):
    """Return a dict: always has 'thresholds'; has model/region info if enabled."""
    bdt_dir = _bdt_dir_for(tree)
    ctx = {"thresholds": _load_thresholds(bdt_dir, tree), "regions_enabled": False}

    sr_csv = _sr_csv_for(tree)
    if bdt_dir is None or sr_csv is None:
        log(f"  [{tree}] no signal_region_csv/bdt_root configured -> inclusive only")
        return ctx
    if not os.path.exists(sr_csv):
        log(f"  [{tree}] signal_region CSV not found ({sr_csv}) -> inclusive only")
        return ctx

    bcfg_path = os.path.join(bdt_dir, "config.json")
    brj_path  = os.path.join(bdt_dir, "branch.json")
    sel_path  = os.path.join(bdt_dir, "selection.json")
    for p in (bcfg_path, brj_path, sel_path):
        if not os.path.exists(p):
            log(f"  [{tree}] missing {os.path.basename(p)} in bdt_root -> inclusive only")
            return ctx

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
    model = model_io.load_model(model_base, bcfg, len(class_names), log_message=log)

    sr_df = pd.read_csv(sr_csv)
    if sr_df.empty:
        log(f"  [{tree}] signal_region CSV is empty -> inclusive only")
        return ctx
    axes = _detect_signal_region_axes(sr_df, class_names)
    if not axes:
        log(f"  [{tree}] no per-class axes detected in CSV -> inclusive only")
        return ctx
    bin_ids = [int(round(float(v))) for v in sr_df["bin_index"].tolist()]

    ctx.update(
        regions_enabled=True, model=model, class_names=class_names,
        num_classes=len(class_names), feature_cols=feature_cols,
        clip_ranges=clip_ranges, log_transform=log_transform,
        decorrelate=decorrelate, sr_df=sr_df, axes=axes, bin_ids=bin_ids,
        sr_csv=sr_csv,
    )
    log(f"  [{tree}] regions enabled: {len(bin_ids)} SRs from {os.path.basename(sr_csv)}, "
        f"axes={axes}")
    return ctx


# ---------------------------------------------------------------------------
# Inference + masks
# ---------------------------------------------------------------------------

def _threshold_mask(data_np, thresholds):
    if not thresholds:
        n = len(next(iter(data_np.values()))) if data_np else 0
        return np.ones(n, dtype=bool)
    n = len(next(iter(data_np.values())))
    mask = np.ones(n, dtype=bool)
    for branch, cond in thresholds.items():
        if branch not in data_np:
            continue
        arr = np.asarray(data_np[branch], dtype=float)
        lo = hi = None
        if isinstance(cond, (list, tuple)) and len(cond) == 2:
            lo, hi = cond[0], cond[1]
        elif isinstance(cond, (int, float)):
            lo = cond
        if lo is not None:
            mask &= arr > float(lo)
        if hi is not None:
            mask &= arr < float(hi)
    return mask


def _infer_proba(chunk, mask, ctx):
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
    return model_io.predict_model_proba(ctx["model"], X, ctx["num_classes"])


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def _ratios_from_sums(nom_sum, pdf_sums, scale_sums, ps_sums, n_events, alphas_sums=None):
    """Return the ratio dict for one accumulation slot, or None if empty."""
    if nom_sum <= 0.0 or n_events < 1:
        return None
    member_ratios = pdf_sums[1:101] / nom_sum    # 100 PDF members (i = 1..100)
    central = float(pdf_sums[0] / nom_sum)        # central member, ~1.0
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
        as_ratios = alphas_sums / nom_sum         # [alpha_s_down, alpha_s_up]
        sigma_as = 0.5 * abs(float(as_ratios[1]) - float(as_ratios[0]))
    sigma_tot = float(np.sqrt(sigma_pdf ** 2 + sigma_as ** 2))
    pdf_up = pdf_center + sigma_tot
    pdf_down = pdf_center - sigma_tot
    pdf_central = pdf_center

    scale_ratios = scale_sums / nom_sum
    valid_s = scale_ratios[SCALE_VALID]
    scale_up   = float(valid_s.max())
    scale_down = float(valid_s.min())

    ps_ratios = ps_sums / nom_sum
    ps_isr_up   = float(max(ps_ratios[0], ps_ratios[1]))
    ps_isr_down = float(min(ps_ratios[0], ps_ratios[1]))
    ps_fsr_up   = float(max(ps_ratios[2], ps_ratios[3]))
    ps_fsr_down = float(min(ps_ratios[2], ps_ratios[3]))

    return {
        "n_events":    int(n_events),
        "nom_sum":     float(nom_sum),
        "pdf_method":  PDF_METHOD,
        "pdf_alphas_unc": float(sigma_as),
        "pdf_up":      pdf_up,      "pdf_down":     pdf_down,  "pdf_central": pdf_central,
        "scale_up":    scale_up,    "scale_down":   scale_down,
        "ps_isr_up":   ps_isr_up,   "ps_isr_down":  ps_isr_down,
        "ps_fsr_up":   ps_fsr_up,   "ps_fsr_down":  ps_fsr_down,
    }


def _compute_ratios(sample_name, ctx):
    """Accumulate inclusive (slot 0) and per-region (slots 1..N) yield sums.

    Returns {"inclusive": ratios_or_None, "regions": {bin_id: ratios_or_None}}
    or None if the sample has no usable theory data.
    """
    files = _input_files(sample_name)
    if not files:
        log(f"    no input files for {sample_name}, skipping")
        return None

    thresholds = ctx["thresholds"]
    enabled = ctx["regions_enabled"]
    n_slots = 1 + (len(ctx["bin_ids"]) if enabled else 0)

    # LHEPdfWeightAlphaS is optional (only present after re-running mode 0 with
    # the alpha_s branch); folded into the PDF uncertainty when available.
    theory_branches = ["LHEPdfWeight", "LHEPdfWeightAlphaS", "LHEScaleWeight", "PSWeight"]
    load_set = {"weight_pu", *theory_branches, *thresholds.keys()}
    if enabled:
        load_set.update(ctx["feature_cols"])
    load_list = list(load_set)

    nom_sum     = np.zeros(n_slots, dtype=np.float64)
    pdf_sums    = np.zeros((n_slots, N_PDF_TOTAL), dtype=np.float64)
    alphas_sums = np.zeros((n_slots, N_ALPHAS),    dtype=np.float64)
    scale_sums  = np.zeros((n_slots, N_SCALE),     dtype=np.float64)
    ps_sums     = np.zeros((n_slots, N_PS),        dtype=np.float64)
    n_events    = np.zeros(n_slots, dtype=np.int64)
    found_theory = False
    has_alphas = [False]

    def _accumulate(slot, w, pdf_w, scale_w, ps_w, as_w):
        nom_sum[slot]    += float(w.sum())
        pdf_sums[slot]   += (w[:, None] * pdf_w).sum(axis=0)
        scale_sums[slot] += (w[:, None] * scale_w).sum(axis=0)
        ps_sums[slot]    += (w[:, None] * ps_w).sum(axis=0)
        n_events[slot]   += int(w.shape[0])
        if as_w is not None:
            alphas_sums[slot] += (w[:, None] * as_w).sum(axis=0)

    for fpath in files:
        try:
            with uproot.open(fpath) as uf:
                if ctx["_tree"] not in uf:
                    continue
                tree = uf[ctx["_tree"]]
                avail = set(tree.keys())
                if "LHEPdfWeight" not in avail:
                    log(f"    LHEPdfWeight missing in {os.path.basename(fpath)} — re-run mode 0")
                    continue
                found_theory = True
                actual = [b for b in load_list if b in avail]
                for chunk in tree.iterate(expressions=actual, step_size=CHUNK_SIZE, library="np"):
                    mask = _threshold_mask(chunk, thresholds)
                    if not mask.any():
                        continue
                    w_pu    = np.asarray(chunk["weight_pu"],      dtype=np.float64)[mask]
                    pdf_w   = np.asarray(chunk["LHEPdfWeight"],   dtype=np.float64)[mask]
                    scale_w = np.asarray(chunk["LHEScaleWeight"], dtype=np.float64)[mask]
                    ps_w    = np.asarray(chunk["PSWeight"],       dtype=np.float64)[mask]
                    if "LHEPdfWeightAlphaS" in chunk:
                        as_w = np.asarray(chunk["LHEPdfWeightAlphaS"], dtype=np.float64)[mask]
                        has_alphas[0] = True
                    else:
                        as_w = None

                    _accumulate(0, w_pu, pdf_w, scale_w, ps_w, as_w)  # inclusive

                    if enabled:
                        proba = _infer_proba(chunk, mask, ctx)
                        for j in range(len(ctx["bin_ids"])):
                            rmask = _region_mask(
                                proba, ctx["sr_df"].iloc[j], ctx["axes"], ctx["class_names"]
                            )
                            if rmask.any():
                                _accumulate(j + 1, w_pu[rmask], pdf_w[rmask],
                                            scale_w[rmask], ps_w[rmask],
                                            as_w[rmask] if as_w is not None else None)
        except Exception as exc:
            log(f"    error reading {fpath}: {exc}")
            continue

    if not found_theory:
        log(f"    [{sample_name}] no theory branches found in any file")
        return None

    def _slot_alphas(slot):
        return alphas_sums[slot] if has_alphas[0] else None

    inclusive = _ratios_from_sums(nom_sum[0], pdf_sums[0], scale_sums[0], ps_sums[0],
                                  n_events[0], _slot_alphas(0))
    regions = None
    if enabled:
        regions = {}
        for j, bid in enumerate(ctx["bin_ids"]):
            slot = j + 1
            regions[bid] = _ratios_from_sums(
                nom_sum[slot], pdf_sums[slot], scale_sums[slot], ps_sums[slot],
                n_events[slot], _slot_alphas(slot)
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
            if res is None:
                continue
            incl = res["inclusive"]
            if incl is None:
                log(f"    [{sample_name}/{tree_name}] nominal weight sum is zero, skipping")
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
                            f"(< {MIN_REGION_EVENTS}); falling back to inclusive ratio")
                        region_out[str(bid)] = {**incl, "fallback_to_inclusive": True}
                    else:
                        region_out[str(bid)] = rr
                        log(f"    SR{bid}: n={rr['n_events']}"
                            f"  pdf=[{_pct(rr['pdf_down'])}, {_pct(rr['pdf_up'])}]"
                            f"  scale=[{_pct(rr['scale_down'])}, {_pct(rr['scale_up'])}]")
                entry["regions"] = region_out

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
