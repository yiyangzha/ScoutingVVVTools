"""Signed, absolutely normalized MC event weights shared by the analysis scripts
(train.py, signal_region.py, qcd_est.py, data_mc.py, jms_jmr_fit.py and the
systematics scripts).

A simulated event of sample s carries the physics weight

    w = lumi * xsection * genWeight * weight_pu / S_s * N_tree / N_read

  S_s     sum of genWeight * weight_pu over every processed generated event of
          the sample, before any selection, written into src/sample.json as
          `sum_genweight_pu` by convert_branch's final merge. The denominator
          covers all generated events, so the selected events carry the
          selection efficiency, and negative generator weights enter with
          their sign.
  N_tree  entries of the tree in all of the sample's converted files.
  N_read  entries actually read (a train/test split or a capped read); the
          factor N_tree / N_read scales that subset to the whole tree, and is
          1 for full reads.

The reweight-branch product of an event and its generated-event denominator
must match: WEIGHT_DENOMINATOR_KEYS maps every supported product to its
sample.json field (the pileup variations of pileup_syst.py use their own sum).

Classifier training uses rectified weights |w| * sum(w) / sum(|w|) per sample
and split (training_weights), because XGBoost accepts only non-negative
weights; every physics output uses the signed weights.
"""

import numpy as np

# Reweight-branch product -> sample.json field holding its sum over every
# processed generated event of the sample (written by convert_branch).
WEIGHT_DENOMINATOR_KEYS = {
    ("genWeight", "weight_pu"): "sum_genweight_pu",
    ("genWeight", "weight_pu_up"): "sum_genweight_pu_up",
    ("genWeight", "weight_pu_down"): "sum_genweight_pu_down",
    ("genWeight",): "sum_genweight",
}


def denominator_key(reweight_branches):
    """sample.json field normalizing the product of reweight_branches."""
    key = tuple(sorted(reweight_branches))
    if key not in WEIGHT_DENOMINATOR_KEYS:
        raise ValueError(
            f"Unsupported MC reweight branches {list(reweight_branches)}; supported products: "
            + ", ".join(str(list(k)) for k in WEIGHT_DENOMINATOR_KEYS)
        )
    return WEIGHT_DENOMINATOR_KEYS[key]


def generated_weight_sum(sample_rule, reweight_branches):
    """Sum of the reweight-branch product over every processed generated event."""
    key = denominator_key(reweight_branches)
    value = sample_rule.get(key)
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise RuntimeError(
            f"Sample '{sample_rule.get('name')}' has no positive '{key}' in the sample config; "
            "run the convert_branch merge of this campaign first."
        )
    return float(value)


def event_weight_product(columns, reweight_branches):
    """Per-event product of the reweight branches (columns: mapping branch -> array)."""
    if not reweight_branches:
        raise ValueError("MC event weights need reweight branches (genWeight, weight_pu)")
    product = None
    for branch in reweight_branches:
        values = np.asarray(columns[branch], dtype=float)
        product = values.copy() if product is None else product * values
    return product


def physics_weights(raw_w, xsection, generated_sum, n_tree, n_read, lumi=1.0):
    """lumi * xsection * raw_w / generated_sum * n_tree / n_read (signed)."""
    if n_read <= 0:
        raise ValueError("physics_weights needs at least one read entry")
    scale = float(lumi) * float(xsection) / float(generated_sum) * float(n_tree) / float(n_read)
    return np.asarray(raw_w, dtype=float) * scale


def training_weights(signed_weights):
    """|w| * sum(w) / sum(|w|): non-negative weights keeping the signed total."""
    signed_weights = np.asarray(signed_weights, dtype=float)
    abs_weights = np.abs(signed_weights)
    abs_sum = float(abs_weights.sum())
    if abs_sum <= 0.0:
        return abs_weights
    return abs_weights * (float(signed_weights.sum()) / abs_sum)
