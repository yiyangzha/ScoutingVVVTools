#!/usr/bin/env python3
"""Audit zero-entry Scouting MC files and their normalization contribution.

The ntuple production must preserve ``Runs/genEventSumw`` from every input
file, including a file whose ``Events`` tree has zero entries.  This script
resolves the MC samples selected in ``ntuple_config.json``, reads that metadata
from each source ROOT file, and reports the normalization bias caused by
discarding zero-entry files.

Run this on a CMS host with ``dasgoclient``, PyROOT, and XRootD access, e.g.:

  python3 systematics/scale_factor/check_empty_mc_inputs.py \\
    --workers 8 \\
    --xsec-weights /path/to/1_templates/xsec_weight.json \\
    --report empty_mc_report.json

``xsec_weight.json`` is optional.  When supplied, the script compares its
stored ``genEventSumw`` (the denominator actually used by topwsf) with the
source-file sums measured here.  It also reports both the output ``Events``
count and ``Runs/genEventCount`` when present; neither replaces
``Runs/genEventSumw`` for weighted MC normalization.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "ntuple_config.json"
DEFAULT_REDIRECTOR = "root://cms-xrd-global.cern.ch/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="ntuple JSON configuration")
    parser.add_argument("--year", default="2024", help="year key under ntuple.samples.by_year")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="restrict to one MC sample nickname; may be given more than once",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel ROOT readers (default: 4)")
    parser.add_argument("--dasgoclient", default="dasgoclient", help="DAS client executable")
    parser.add_argument("--redirector", default=DEFAULT_REDIRECTOR, help="redirector prepended to /store LFNs")
    parser.add_argument("--tree-name", default="Events", help="event tree name (default: Events)")
    parser.add_argument(
        "--xsec-weights",
        type=Path,
        help="optional topwsf 1_templates/xsec_weight.json to compare against the measured sums",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def selected_mc_names(config: dict[str, Any], year: str) -> list[str]:
    try:
        groups = config["ntuple"]["samples"]["by_year"][year]["mc_groups"]
    except KeyError as error:
        raise SystemExit(f"No ntuple MC groups configured for year {year}.") from error

    names: list[str] = []
    for group_names in groups.values():
        names.extend(str(name) for name in group_names)
    return names


def to_root_url(path: str, redirector: str) -> str:
    if path.startswith("root://"):
        return path
    if path.startswith("/store/"):
        return redirector.rstrip("/") + "/" + path
    return path


def run_das_query(dataset: str, dasgoclient: str) -> list[str]:
    queries = [f"file dataset={dataset}"]
    if dataset.endswith("/USER"):
        queries = [
            f"file dataset={dataset} instance=prod/phys03",
            f"file dataset={dataset} system=rucio",
            f"file dataset={dataset} system=dbs3",
            *queries,
        ]

    errors: list[str] = []
    for query in queries:
        completed = subprocess.run(
            [dasgoclient, "-query", query],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        files = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode == 0 and files:
            return sorted(set(files))
        errors.append(f"{query}: {completed.stderr.strip() or 'no files returned'}")
    raise RuntimeError("DAS failed for " + dataset + "\n  " + "\n  ".join(errors))


def resolve_sample_files(sample: dict[str, Any], dasgoclient: str, redirector: str) -> list[str]:
    files: list[str] = []
    for entry in as_list(sample.get("path")):
        if entry.endswith(".root") or entry.startswith("root://") or entry.startswith("/store/"):
            files.append(to_root_url(entry, redirector))
            continue
        if entry.startswith("/") and entry.count("/") >= 3:
            files.extend(to_root_url(path, redirector) for path in run_das_query(entry, dasgoclient))
            continue
        raise RuntimeError(f"Unsupported sample path entry: {entry}")
    return sorted(set(files))


def inspect_root_file(task: tuple[str, str, str]) -> dict[str, Any]:
    sample, path, tree_name = task
    result: dict[str, Any] = {"sample": sample, "path": path}
    root_file = None
    try:
        import ROOT  # Imported inside worker processes so PyROOT state is not shared.

        ROOT.gROOT.SetBatch(True)
        root_file = ROOT.TFile.Open(path, "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError("TFile.Open failed")
        events = root_file.Get(tree_name)
        if not events:
            raise RuntimeError(f"missing {tree_name} tree")
        runs = root_file.Get("Runs")
        if not runs:
            raise RuntimeError("missing Runs tree")
        if not runs.GetBranch("genEventSumw"):
            raise RuntimeError("missing Runs/genEventSumw")

        sumw = 0.0
        has_gen_event_count = bool(runs.GetBranch("genEventCount"))
        gen_event_count = 0
        for entry in runs:
            sumw += float(entry.genEventSumw)
            if has_gen_event_count:
                gen_event_count += int(entry.genEventCount)
        result.update(
            status="ok",
            events_entries=int(events.GetEntries()),
            runs_entries=int(runs.GetEntries()),
            gen_event_sumw=sumw,
            gen_event_count=gen_event_count if has_gen_event_count else None,
        )
    except Exception as error:  # Keep scanning to show every inaccessible input.
        result.update(status="error", error=str(error))
    finally:
        if root_file:
            root_file.Close()
    return result


def load_xsec_weights(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected an object in xsec weights file: {path}")
    by_sample: dict[str, dict[str, Any]] = {}
    for value in payload.values():
        if isinstance(value, dict) and "sample" in value:
            by_sample[str(value["sample"])] = value
    return by_sample


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) <= 1e-20 else numerator / denominator


def summarize_sample(sample: str, records: list[dict[str, Any]], used_weight: dict[str, Any] | None) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    errors = [record for record in records if record.get("status") != "ok"]
    empty = [record for record in ok if record["events_entries"] == 0]
    nonempty = [record for record in ok if record["events_entries"] > 0]
    sumw_all = sum(float(record["gen_event_sumw"]) for record in ok)
    sumw_empty = sum(float(record["gen_event_sumw"]) for record in empty)
    sumw_nonempty = sum(float(record["gen_event_sumw"]) for record in nonempty)
    events_all = sum(int(record["events_entries"]) for record in ok)
    has_gen_event_count = bool(ok) and all(record["gen_event_count"] is not None for record in ok)
    gen_event_count_all = sum(int(record["gen_event_count"]) for record in ok) if has_gen_event_count else None
    gen_event_count_empty = sum(int(record["gen_event_count"]) for record in empty) if has_gen_event_count else None
    used_sumw = None if used_weight is None else float(used_weight.get("genEventSumw", math.nan))
    return {
        "sample": sample,
        "files_total": len(records),
        "files_read_ok": len(ok),
        "files_error": len(errors),
        "files_empty_events": len(empty),
        "files_nonempty_events": len(nonempty),
        "events_entries_all_files": events_all,
        "gen_event_count_all_files": gen_event_count_all,
        "gen_event_count_empty_events_files": gen_event_count_empty,
        "events_over_gen_event_count": (
            None if gen_event_count_all is None else safe_ratio(float(events_all), float(gen_event_count_all))
        ),
        "sumw_all_files": sumw_all,
        "sumw_empty_events_files": sumw_empty,
        "sumw_nonempty_events_files": sumw_nonempty,
        "predicted_mc_inflation_if_empty_files_are_dropped": safe_ratio(sumw_all, sumw_nonempty),
        "topwsf_used_genEventSumw": used_sumw,
        "used_over_all_sumw": None if used_sumw is None else safe_ratio(used_sumw, sumw_all),
        "used_over_nonempty_sumw": None if used_sumw is None else safe_ratio(used_sumw, sumw_nonempty),
        "errors": errors,
        "files": records,
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.6g}"


def print_summary(summaries: list[dict[str, Any]], has_xsec_weights: bool) -> None:
    print()
    print("sample                         files  empty  Events(all)  Runs count  Events/Runs")
    print("-" * 94)
    for summary in summaries:
        print(
            f"{summary['sample']:<30} {summary['files_read_ok']:>5}/{summary['files_total']:<4}"
            f" {summary['files_empty_events']:>5} {summary['events_entries_all_files']:>12}"
            f" {fmt(summary['gen_event_count_all_files']):>11} {fmt(summary['events_over_gen_event_count']):>12}"
        )
    print()
    print("sample                         sumw(all)       sumw(nonempty)  predicted MC scale")
    print("-" * 90)
    for summary in summaries:
        print(
            f"{summary['sample']:<30} {fmt(summary['sumw_all_files']):>15}"
            f" {fmt(summary['sumw_nonempty_events_files']):>15}"
            f" {fmt(summary['predicted_mc_inflation_if_empty_files_are_dropped']):>18}"
        )
    if has_xsec_weights:
        print()
        print("topwsf denominator cross-check (a correct current run should use all-file sumw):")
        print("sample                         used/all       used/nonempty")
        print("-" * 68)
        for summary in summaries:
            print(
                f"{summary['sample']:<30} {fmt(summary['used_over_all_sumw']):>14}"
                f" {fmt(summary['used_over_nonempty_sumw']):>18}"
            )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    config_path = args.config.resolve()
    config = load_json(config_path)
    sample_config_path = (config_path.parent / config["sample_config"]).resolve()
    samples_by_name = {sample["name"]: sample for sample in load_json(sample_config_path)["sample"]}
    names = selected_mc_names(config, str(args.year))
    if args.sample:
        requested = set(args.sample)
        unknown = sorted(requested - set(names))
        if unknown:
            raise SystemExit("Requested sample(s) are not selected by this ntuple config: " + ", ".join(unknown))
        names = [name for name in names if name in requested]

    xsec_weights = load_xsec_weights(args.xsec_weights)
    all_records: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for name in names:
        sample = samples_by_name.get(name)
        if sample is None:
            raise SystemExit(f"Selected MC sample is missing from {sample_config_path}: {name}")
        if not sample.get("is_MC", False):
            raise SystemExit(f"Selected sample is not marked MC: {name}")
        print(f"Resolving {name} with DAS ...", flush=True)
        files = resolve_sample_files(sample, args.dasgoclient, args.redirector)
        print(f"  {len(files)} files", flush=True)
        all_records[name] = [{"sample": name, "path": path} for path in files]

    tasks = [(name, record["path"], args.tree_name) for name, records in all_records.items() for record in records]
    results_by_sample: OrderedDict[str, list[dict[str, Any]]] = OrderedDict((name, []) for name in names)
    print(f"Inspecting {len(tasks)} ROOT files with {args.workers} worker(s) ...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(inspect_root_file, tasks):
            results_by_sample[result["sample"]].append(result)

    summaries = [summarize_sample(name, results_by_sample[name], xsec_weights.get(name)) for name in names]
    print_summary(summaries, bool(xsec_weights))

    report = {
        "config": str(config_path),
        "sample_config": str(sample_config_path),
        "year": str(args.year),
        "tree_name": args.tree_name,
        "redirector": args.redirector,
        "summaries": summaries,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"Wrote report: {args.report}")

    failed = sum(summary["files_error"] for summary in summaries)
    if failed:
        print(f"ERROR: {failed} files could not be inspected; do not interpret incomplete sums as final.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
