#!/usr/bin/env python3
"""jer_syst.py  (mode 12)

Jet-energy-resolution (JER) yield ratios of all MC samples, per tree and per signal region, for
the JER smearing of every MC AK4/AK8 jet with the AK4PFPuppi JER scale factor
up/down instead of nom, with the same random numbers as the nominal smearing.
convert_branch.C writes this variation as the <tree>__jer_up and
<tree>__jer_down trees next to the nominal trees, and
selections/jet_syst_common/jet_variation_common.py computes

  jer_up   = sum(weight_branch on <tree>__jer_up)   / sum(weight_branch on <tree>)
  jer_down = sum(weight_branch on <tree>__jer_down) / sum(weight_branch on <tree>)

in the same regions as qcd_est.py.

Outputs (written to output_dir):
  jer_syst_yields.json  -- {sample: {tree: {nom_sum, n_events, jer_up, jer_down,
                                            regions: {bin_id: {...}}}}}
  jer_syst_yields.csv   -- flat CSV: sample,tree,region,jer_up,jer_down,n_events
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402


if __name__ == "__main__":
    jvc.run("jer", _SCRIPT_DIR, "JER_SYST_CONFIG_PATH")
