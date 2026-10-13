"""MC event-weight helpers shared by every event-weight builder.

Signed generator weights: MC event weights carry g = genWeight / genweight_mean, where
genweight_mean in src/sample.json is the mean signed generator weight of the generated
sample (sum genEventSumw / sum genEventCount over the NanoAOD Runs trees, filled by
``convert_branch <sample> --update-genweight-mean``).  Since raw_entries counts generated
events, the normalisation xsec * N_tree / raw_entries keeps its meaning, while
negative-weight (NLO) events now enter with negative sign.  For a sample with constant
positive genWeight, g == 1.

Absolute reweight normalisation: the per-event reweight product (event_reweight_branches,
in production just weight_pu) is divided by its mean over all generated events of the
sample (``<branch>_mean`` in src/sample.json, written by convert_branch when it merges the
sample) instead of by its sum over the loaded events, so the effect of the reweighting on
the selection efficiency is kept.
"""

import math

import numpy as np

GEN_WEIGHT_BRANCH = "genWeight"


def require_genweight_mean(value, sample_name):
    """Return the sample's genweight_mean as float; raise if missing or not positive."""
    if value is None:
        raise RuntimeError(
            f"MC sample '{sample_name}' has no genweight_mean in src/sample.json; run "
            f"'CONVERT_CONFIG_PATH=selections/convert/config.json ./convert_branch "
            f"{sample_name} --update-genweight-mean' first"
        )
    mean = float(value)
    if not math.isfinite(mean) or mean <= 0.0:
        raise RuntimeError(f"MC sample '{sample_name}' has invalid genweight_mean={value!r}")
    return mean


def gen_weight_factor(gen_weight, mean, sample_name=None):
    """Per-event signed factor genWeight / genweight_mean.

    Ntuples converted before convert_branch stored the real genWeight for every MC sample
    hold the placeholder 1.0 for samples without theory weights (e.g. QCD); dividing that by
    the real mean would mis-scale the sample, so all-1.0 values with a mean != 1 are refused.
    """
    values = np.asarray(gen_weight, dtype=float)
    if values.size and abs(float(mean) - 1.0) > 1e-6 and np.all(values == 1.0):
        label = f"MC sample '{sample_name}'" if sample_name else "An MC sample"
        raise RuntimeError(
            f"{label} has the placeholder genWeight = 1.0 in its converted ntuples while "
            f"genweight_mean = {float(mean):.6g}; reconvert it with the current convert_branch"
        )
    return values / float(mean)


def reweight_mean(sample_entry, reweight_branches, sample_name):
    """All-generated-event mean of the event reweight product (absolute normalisation).

    convert_branch stores the mean per reweight branch (weight_pu, weight_pu_up,
    weight_pu_down), so only a single reweight branch is supported; no branch means 1.
    """
    branches = list(reweight_branches or [])
    if not branches:
        return 1.0
    if len(branches) > 1:
        raise RuntimeError(
            f"Absolute normalisation needs the all-event mean of the reweight product, which "
            f"convert_branch stores per branch only; configure a single event_reweight_branch "
            f"(got {branches})"
        )
    key = f"{branches[0]}_mean"
    value = sample_entry.get(key)
    if value is None:
        raise RuntimeError(
            f"MC sample '{sample_name}' has no {key} in src/sample.json; it is written when "
            f"convert_branch merges the sample (reconvert it with the current convert_branch)"
        )
    mean = float(value)
    if not math.isfinite(mean) or mean <= 0.0:
        raise RuntimeError(f"MC sample '{sample_name}' has invalid {key}={value!r}")
    return mean
