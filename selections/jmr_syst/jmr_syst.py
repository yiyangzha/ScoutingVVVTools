#!/usr/bin/env python3
"""jmr_syst.py  (mode 14)

Jet-mass-resolution (JMR) yield ratios of all MC samples, per tree and per signal region, for
the stochastic AK8 soft-drop mass smearing with JMR +- its uncertainty
(selections/jms_jmr/output/jms_jmr_results.json) instead of the nominal JMR;
a JMR below 1 means no smearing.
convert_branch.C writes this variation as the <tree>__jmr_up and
<tree>__jmr_down trees next to the nominal trees, and
selections/jet_syst_common/jet_variation_common.py computes

  jmr_up   = sum(weight_branch on <tree>__jmr_up)   / sum(weight_branch on <tree>)
  jmr_down = sum(weight_branch on <tree>__jmr_down) / sum(weight_branch on <tree>)

in the same regions as qcd_est.py.

Outputs (written to output_dir):
  jmr_syst_yields.json  -- {sample: {tree: {nom_sum, n_events, jmr_up, jmr_down,
                                            regions: {bin_id: {...}}}}}
  jmr_syst_yields.csv   -- flat CSV: sample,tree,region,jmr_up,jmr_down,n_events
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402


if __name__ == "__main__":
    jvc.run("jmr", _SCRIPT_DIR, "JMR_SYST_CONFIG_PATH")
