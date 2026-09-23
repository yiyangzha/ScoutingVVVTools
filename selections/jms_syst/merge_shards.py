#!/usr/bin/env python3
"""Merge per-sample jms_syst.py SLURM shard outputs into the final
jms_syst_yields.json/.csv, matching the script's own output exactly (see
jet_variation_common.merge_shards)."""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "jet_syst_common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import jet_variation_common as jvc  # noqa: E402


if __name__ == "__main__":
    sys.exit(jvc.merge_shards("jms", _SCRIPT_DIR))
