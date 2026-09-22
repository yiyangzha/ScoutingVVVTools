#!/usr/bin/env python3
"""Merge per-sample jer_syst.py SLURM shard outputs into the final
jer_syst_yields.json/.csv, matching main()'s own output exactly (the
merge is a plain dict union -- each shard's JSON already contains only its
own sample's entries keyed by tree, with zero cross-sample aggregation)."""
import csv
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARDS_DIR = os.path.join(_SCRIPT_DIR, "output", "shards")
OUT_DIR = os.path.join(_SCRIPT_DIR, "output")


def main():
    manifest = [
        line.strip()
        for line in open(os.path.join(_SCRIPT_DIR, "shards", "manifest.txt"))
        if line.strip()
    ]

    nested = {}
    missing = []
    for sample in manifest:
        path = os.path.join(SHARDS_DIR, sample, "jer_syst_yields.json")
        if not os.path.exists(path):
            missing.append(sample)
            continue
        shard = json.load(open(path))
        for s, per_tree in shard.items():
            nested.setdefault(s, {}).update(per_tree)

    if missing:
        print(f"WARNING: {len(missing)} shard(s) missing (job failed or "
              f"sample had zero nominal weight in all trees): {missing}")

    json_path = os.path.join(OUT_DIR, "jer_syst_yields.json")
    json.dump(nested, open(json_path, "w"), indent=2)
    print(f"Wrote {json_path}  ({len(nested)} samples)")

    csv_path = os.path.join(OUT_DIR, "jer_syst_yields.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "tree", "region", "jer_up", "jer_down", "n_events"])
        for sample in sorted(nested):
            for tree in sorted(nested[sample]):
                entry = nested[sample][tree]
                writer.writerow([sample, tree, "inclusive",
                                  f"{entry['jer_up']:.6f}", f"{entry['jer_down']:.6f}",
                                  entry["n_events"]])
                for bid, rr in entry.get("regions", {}).items():
                    writer.writerow([sample, tree, bid,
                                      f"{rr['jer_up']:.6f}", f"{rr['jer_down']:.6f}",
                                      rr["n_events"]])
    print(f"Wrote {csv_path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
