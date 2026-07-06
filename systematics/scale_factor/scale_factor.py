#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path, base=HERE):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def scalar_yaml(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    typed_literals = {"null", "true", "false", "yes", "no", "on", "off"}
    parses_as_number = False
    try:
        float(text)
        parses_as_number = True
    except ValueError:
        pass
    if text.lower() in typed_literals or parses_as_number:
        return json.dumps(text)
    if text == "" or any(ch in text for ch in ":#{}[],&*?!|>'\"%@`") or text.strip() != text:
        return json.dumps(text)
    return text


def dump_yaml_value(value, indent=0):
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            key_text = scalar_yaml(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key_text}:")
                lines.append(dump_yaml_value(item, indent + 2))
            else:
                lines.append(f"{pad}{key_text}: {scalar_yaml(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return f"{pad}[" + ", ".join(scalar_yaml(item) for item in value) + "]"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(dump_yaml_value(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{pad}- " + dump_yaml_value(item, 0).strip())
            else:
                lines.append(f"{pad}- {scalar_yaml(item)}")
        return "\n".join(lines)
    return f"{pad}{scalar_yaml(value)}"


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dump_yaml_value(payload))
        handle.write("\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def sample_map(cfg):
    sample_path = resolve_path(cfg.get("sample_config", "../../src/sample.json"))
    payload = load_json(sample_path)
    out = {}
    for item in payload.get("sample", []):
        name = item.get("name")
        if name:
            if name in out:
                raise SystemExit(f"Duplicate sample name in {sample_path}: {name}")
            out[name] = item
    if not out:
        raise SystemExit(f"No samples found in {sample_path}")
    return out


def require_samples(samples, names):
    missing = [name for name in names if name not in samples]
    if missing:
        raise SystemExit("Unknown sample(s) in scale-factor config: " + ", ".join(missing))


def sample_paths(sample):
    return [str(path) for path in as_list(sample.get("path"))]


def selected_sample_groups(cfg, samples, section_name):
    section = cfg[section_name]
    selected = section["samples"]
    data = list(selected.get("data", []))
    mc_groups = {group: list(names) for group, names in selected.get("mc_groups", {}).items()}
    require_samples(samples, data)
    for names in mc_groups.values():
        require_samples(samples, names)
    validate_sample_info(samples, data, mc_groups)
    return data, mc_groups


def validate_sample_info(samples, data_names, mc_groups):
    for name in data_names:
        sample = samples[name]
        if sample.get("is_MC", True):
            raise SystemExit(f"Configured data sample is marked is_MC=true in sample.json: {name}")
        if not sample_paths(sample):
            raise SystemExit(f"Configured data sample has no path in sample.json: {name}")
        if float(sample.get("lumi", 0.0)) <= 0.0:
            raise SystemExit(f"Configured data sample has missing/non-positive lumi in sample.json: {name}")
    for group, names in mc_groups.items():
        if not names:
            raise SystemExit(f"Configured MC group is empty: {group}")
        for name in names:
            sample = samples[name]
            if not sample.get("is_MC", True):
                raise SystemExit(f"Configured MC sample is marked is_MC=false in sample.json: {name}")
            if not sample_paths(sample):
                raise SystemExit(f"Configured MC sample has no path in sample.json: {name}")
            if float(sample.get("xsection", 0.0)) <= 0.0:
                raise SystemExit(f"Configured MC sample has missing/non-positive xsection in sample.json: {name}")


def run_targets(cfg):
    targets = cfg.get("run_targets", cfg.get("targets"))
    if not isinstance(targets, list) or not targets:
        raise SystemExit("scale-factor config must define a non-empty run_targets list")
    out = []
    for index, raw in enumerate(targets):
        jet_type = str(raw.get("jet_type", "")).lower()
        jet_category = str(raw.get("jet_category", ""))
        taggers = list(raw.get("taggers", []))
        if not jet_type or not jet_category:
            raise SystemExit(f"run_targets[{index}] must define jet_type and jet_category")
        if not taggers:
            raise SystemExit(f"run_targets[{index}] must define the taggers to run")
        out.append({"jet_type": jet_type, "jet_category": jet_category, "taggers": taggers})
    return out


def require_local_path(path, label):
    text = str(path)
    if text.startswith("root://") or text.startswith("/store/") or text.startswith("/eos/"):
        raise SystemExit(f"{label} must be a local path, got: {text}")


def year_output_dir(cfg):
    ntuple = cfg["ntuple"]
    sample_base = resolve_path(ntuple["sample_base"])
    require_local_path(sample_base, "ntuple.sample_base")
    suffix = f"_{cfg['year']}_{cfg['nano_version']}"
    text = str(sample_base)
    if not sample_base.name.endswith(suffix):
        text += suffix
    return Path(text)


def variation_names(ntuple_cfg):
    variations = ["nominal"]
    requested = ntuple_cfg.get("variations", {})
    for syst in ("jes", "jer", "met"):
        if requested.get(syst, False):
            variations.extend([f"{syst}_up", f"{syst}_down"])
    return variations


def sample_yaml_payload(samples, names):
    return {name: sample_paths(samples[name]) for name in names}


def run_command(cmd, cwd=None, dry_run=False):
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run([str(part) for part in cmd], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmake_prefix_path_arg():
    prefixes = []
    for value in (os.environ.get("CMAKE_PREFIX_PATH", ""), os.environ.get("CONDA_PREFIX", ""), sys.prefix):
        for part in str(value).split(os.pathsep):
            if part and part not in prefixes:
                prefixes.append(part)
    return [f"-DCMAKE_PREFIX_PATH={os.pathsep.join(prefixes)}"] if prefixes else []


def correctionlib_dir_arg():
    candidates = []
    try:
        import correctionlib
    except ImportError:
        pass
    else:
        candidates.append(Path(correctionlib.__file__).resolve().parent / "cmake")

    for key in ("purelib", "platlib"):
        site_path = sysconfig.get_paths().get(key)
        if site_path:
            candidates.append(Path(site_path) / "correctionlib" / "cmake")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(Path(conda_prefix).glob("lib/python*/site-packages/correctionlib/cmake"))

    for path in candidates:
        if (path / "correctionlibConfig.cmake").exists() or (path / "correctionlib-config.cmake").exists():
            return [f"-Dcorrectionlib_DIR={path}"]
    return []


def yaml_cpp_dir_arg():
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.extend([
            prefix / "lib" / "cmake" / "yaml-cpp",
            prefix / "share" / "cmake" / "yaml-cpp",
        ])
    for path in candidates:
        if (path / "yaml-cpp-config.cmake").exists() or (path / "yaml-cppConfig.cmake").exists():
            return [f"-Dyaml-cpp_DIR={path}"]
    return []


def make_ntuple(cfg, args):
    samples = sample_map(cfg)
    data_names, mc_groups = selected_sample_groups(cfg, samples, "ntuple")
    ntuple = cfg["ntuple"]
    nano_repo = resolve_path(ntuple.get("repo", "nano.cpp"))
    sample_dir = resolve_path(ntuple.get("generated_sample_dir", "generated/samples"))
    out_dir = year_output_dir(cfg)
    require_local_path(resolve_path(ntuple.get("job_dir", "jobs/ntuples")), "ntuple.job_dir")
    variations = ",".join(variation_names(ntuple))
    targets = run_targets(cfg)

    mc_names = []
    for group_names in mc_groups.values():
        for name in group_names:
            if name not in mc_names:
                mc_names.append(name)

    commands = []
    if ntuple.get("build_before_make_condor", True):
        commands.append([
            "cmake", "-S", str(nano_repo), "-B", str(nano_repo / "build"),
            *cmake_prefix_path_arg(),
            *correctionlib_dir_arg(),
            *yaml_cpp_dir_arg(),
        ])
        commands.append(["cmake", "--build", str(nano_repo / "build"), "-j"])

    binary = nano_repo / "build" / "nano_make_condor"
    config_card = resolve_path(ntuple["config"])
    for target in targets:
        tokens = {
            "jet_type": target["jet_type"],
            "jet_category": target["jet_category"],
            "year": cfg["year"],
        }
        tagger_override = "stored_tagger_names=" + dump_yaml_value(target["taggers"]).strip()
        sample_files = []
        if mc_names:
            path = sample_dir / f"{tokens['jet_type']}_{tokens['jet_category']}_{cfg['year']}_mc.yaml"
            write_yaml(path, sample_yaml_payload(samples, mc_names))
            sample_files.append(("mc", path, False))
        if data_names:
            path = sample_dir / f"{tokens['jet_type']}_{tokens['jet_category']}_{cfg['year']}_data.yaml"
            write_yaml(path, sample_yaml_payload(samples, data_names))
            sample_files.append(("data", path, True))

        for sample_set, sample_yaml, is_data in sample_files:
            job_dir_pattern = ntuple.get("job_dir", "jobs/ntuples/{jet_type}_{jet_category}_{year}_{sample_set}")
            job_dir = resolve_path(job_dir_pattern.format(**tokens, sample_set=sample_set))
            require_local_path(job_dir, "ntuple.job_dir")
            cmd = [
                binary,
                "--input-yaml", sample_yaml,
                "--job-dir", job_dir,
                "--output-dir", out_dir,
                "--config", config_card,
                "--channel", ntuple.get("channel", "scouting_muon"),
                "--tree-name", ntuple.get("tree_name", "Events"),
                "--nfiles-per-job", int(ntuple.get("nfiles_per_job", 1)),
                "--num-events", int(ntuple.get("num_events", -1)),
                "--variations", variations,
                "--use-sample-key-nickname",
                "--set", tagger_override,
            ]
            if is_data:
                cmd.append("--run-data")
            if ntuple.get("download_remote_inputs", False):
                cmd.append("--download-remote-inputs")
            else:
                cmd.append("--no-download-remote-inputs")
            commands.append(cmd)
            if ntuple.get("submit_condor", False):
                commands.append(["condor_submit", "submit.jdl", {"cwd": job_dir}])

    command_log = []
    for cmd in commands:
        cwd = None
        if cmd and isinstance(cmd[-1], dict):
            meta = cmd.pop()
            cwd = meta.get("cwd")
        command_log.append(" ".join(str(part) for part in cmd))
        if args.prepare_only or (not ntuple.get("run_make_condor", True) and cmd[0] == binary):
            print(" ".join(str(part) for part in cmd), flush=True)
            continue
        run_command(cmd, cwd=cwd or (nano_repo if Path(cmd[0]) == binary else None), dry_run=args.dry_run)

    write_text(resolve_path("generated/ntuple_commands.sh"), "\n".join(command_log) + "\n")
    print(f"Wrote generated sample YAMLs under {sample_dir}")
    print(f"Expected merged ntuples under {out_dir}")


def score_bin_name(lo, hi):
    def fmt(x):
        return f"{float(x):.3g}".replace("-", "m").replace(".", "p")
    return f"score{fmt(lo)}to{fmt(hi)}"


def topwsf_groups_and_xsecs(cfg, samples, mc_groups):
    scale = float(cfg.get("xsection_to_pb", 1.0))
    out = {}
    for group, names in mc_groups.items():
        group_out = {}
        for name in names:
            sample = samples[name]
            if not sample.get("is_MC", True):
                continue
            xsec = float(sample.get("xsection", 0.0))
            if xsec <= 0.0:
                continue
            group_out[name] = xsec * scale
        if group_out:
            out[group] = group_out
    return out


def data_lumi(samples, data_names):
    total = 0.0
    for name in data_names:
        lumi = float(samples[name].get("lumi", 0.0))
        if lumi > 0.0:
            total += lumi
    return total


def generated_card(cfg, samples, data_names, mc_groups, target, tagger_name, tagger_cfg):
    cal = cfg["calibration"]
    card_dir = resolve_path(cal.get("generated_card_dir", "generated/topwsf"))
    require_local_path(card_dir, "calibration.generated_card_dir")
    boohft_base = resolve_path(cal.get("repo", "boohft-calib")) / "cards" / "topwsf" / "base.yml"
    extends = [os.path.relpath(boohft_base, card_dir)]
    score_bins = {
        score_bin_name(lo, hi): [float(lo), float(hi)]
        for lo, hi in tagger_cfg["score_bins"]
    }
    year = str(cfg["year"])
    lumi = data_lumi(samples, data_names)
    if lumi <= 0.0:
        raise SystemExit("data sample lumi is missing/non-positive in sample.json")
    fit_cfg = cal.get("fit", {})
    systematics = list(cal.get("systematics", {}).get("enabled", []))
    input_sample_base = resolve_path(cal["input_sample_base"])
    require_local_path(input_sample_base, "calibration.input_sample_base")
    payload = {
        "extends": extends,
        "routine_name": f"{cal.get('routine_name_prefix', 'scouting_topwsf')}_{target['jet_type']}_{target['jet_category'].lower()}_{tagger_name}",
        "category": cal.get("category", "w"),
        "year": year,
        "nano_version": str(cfg["nano_version"]),
        "sample_base": str(input_sample_base),
        "sample_scan_wp": str(input_sample_base),
        "workers": list(cal.get("workers", [20, 20])),
        "run_step": str(cal.get("run_step", "11")),
        "skip_coffea": False,
        "apply_toppt_weight": True,
        "systematics": systematics,
        "selection": cal["selection"],
        "skip_fit": bool(fit_cfg.get("skip_fit", False)),
        "fit_run_impact": bool(fit_cfg.get("fit_run_impact", True)),
        "fit_impact_parallel": int(fit_cfg.get("fit_impact_parallel", 8)),
        "fit_auto_mc_stats": int(fit_cfg.get("fit_auto_mc_stats", 0)),
        "lumi_dict": {year: lumi},
        "lumi_uncertainty": {year: 1.025},
        "data_samples": list(data_names),
        "enabled_sample_groups": list(mc_groups.keys()),
        "mc_sample_groups_and_xsecs": topwsf_groups_and_xsecs(cfg, samples, mc_groups),
        "tagger": {
            "label": tagger_cfg.get("label", tagger_name),
            "type": cal.get("category", "w"),
            "expr": tagger_cfg["expr"],
            "span": [0.0, 1.0],
            "wps": score_bins,
        },
        "fit_pt_bins": tagger_cfg["pt_bins"],
    }
    card_path = card_dir / f"{year}_{target['jet_type']}_{target['jet_category']}_{tagger_name}.yml"
    write_yaml(card_path, payload)
    return card_path


def compute_sf(cfg, args):
    samples = sample_map(cfg)
    data_names, mc_groups = selected_sample_groups(cfg, samples, "calibration")
    cal = cfg["calibration"]
    boohft_repo = resolve_path(cal.get("repo", "boohft-calib"))
    targets = run_targets(cfg)

    cards = []
    for target in targets:
        jet_type = target["jet_type"]
        jet_category = target["jet_category"]
        taggers = cal["binning"].get(jet_type, {}).get(jet_category, {})
        if not taggers:
            raise SystemExit(f"No tagger binning configured for {jet_type}/{jet_category}")
        for tagger_name in target["taggers"]:
            if args.tagger and args.tagger != tagger_name:
                continue
            if tagger_name not in taggers:
                raise SystemExit(f"Tagger {tagger_name} is listed in run_targets but missing from calibration.binning.{jet_type}.{jet_category}")
            tagger_cfg = taggers[tagger_name]
            cards.append(generated_card(cfg, samples, data_names, mc_groups, target, tagger_name, tagger_cfg))

    if not cards:
        raise SystemExit("No scale-factor cards were generated")

    for card in cards:
        cmd = [
            "python3", "launcher.py", str(card),
            "--routine", cal.get("routine", "topwsf"),
            "--run-step", str(cal.get("run_step", "11")),
            "--workers", *[str(x) for x in cal.get("workers", [20, 20])],
        ]
        if args.generate_only or not cal.get("run_launcher", True):
            print(" ".join(cmd), flush=True)
            continue
        run_command(cmd, cwd=boohft_repo, dry_run=args.dry_run)
    print("Generated topwsf cards:")
    for card in cards:
        print(f"  {card}")


def parse_args():
    parser = argparse.ArgumentParser(description="Control AK8/AK4 scale-factor ntuple and fit workflows.")
    parser.add_argument("command", choices=["make-ntuple", "compute-sf"])
    parser.add_argument("--config", default=os.environ.get("SCALE_FACTOR_CONFIG_PATH"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="For make-ntuple: write generated files and commands only.")
    parser.add_argument("--generate-only", action="store_true", help="For compute-sf: write boohft cards only.")
    parser.add_argument("--tagger", default="", help="For compute-sf: run only one configured tagger.")
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = "ntuple_config.json" if args.command == "make-ntuple" else "sf_config.json"
    config_path = resolve_path(args.config or default_config, Path.cwd())
    cfg = load_json(config_path)
    if args.command == "make-ntuple":
        make_ntuple(cfg, args)
    elif args.command == "compute-sf":
        compute_sf(cfg, args)


if __name__ == "__main__":
    main()
