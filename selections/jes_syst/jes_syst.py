#!/usr/bin/env python3
"""jes_syst.py  (mode 11)

Jet-energy-scale (JES) yield ratios of all MC samples, per tree and per signal region, for
the flat +-jes_shift (10%) per-jet JES variation that convert_branch.C applies
after the JER smearing to every AK4/AK8 jet pT, mass, and soft-drop mass (and,
through Type-1, to the MET).
convert_branch.C writes this variation as the <tree>__jes_up and
<tree>__jes_down trees next to the nominal trees, and
selections/jet_syst_common/jet_variation_common.py computes

  jes_up   = sum(weight_branch on <tree>__jes_up)   / sum(weight_branch on <tree>)
  jes_down = sum(weight_branch on <tree>__jes_down) / sum(weight_branch on <tree>)

in the same regions as qcd_est.py.

Outputs (written to output_dir):
  jes_syst_yields.json  -- {sample: {tree: {nom_sum, n_events, jes_up, jes_down,
                                            regions: {bin_id: {...}}}}}
  jes_syst_yields.csv   -- flat CSV: sample,tree,region,jes_up,jes_down,n_events
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402


if __name__ == "__main__":
    jvc.run("jes", _SCRIPT_DIR, "JES_SYST_CONFIG_PATH")
