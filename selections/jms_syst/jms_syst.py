#!/usr/bin/env python3
"""jms_syst.py  (mode 13)

Jet-mass-scale (JMS) yield ratios of all MC samples, per tree and per signal region, for
the AK8 soft-drop mass scale JMS +- its uncertainty (selections/jms_jmr/output/
jms_jmr_results.json) instead of the nominal JMS.
convert_branch.C writes this variation as the <tree>__jms_up and
<tree>__jms_down trees next to the nominal trees, and
selections/jet_syst_common/jet_variation_common.py computes

  jms_up   = sum(weight_branch on <tree>__jms_up)   / sum(weight_branch on <tree>)
  jms_down = sum(weight_branch on <tree>__jms_down) / sum(weight_branch on <tree>)

in the same regions as qcd_est.py.

Outputs (written to output_dir):
  jms_syst_yields.json  -- {sample: {tree: {nom_sum, n_events, jms_up, jms_down,
                                            regions: {bin_id: {...}}}}}
  jms_syst_yields.csv   -- flat CSV: sample,tree,region,jms_up,jms_down,n_events
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402


if __name__ == "__main__":
    jvc.run("jms", _SCRIPT_DIR, "JMS_SYST_CONFIG_PATH")
