#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


HERE = Path(__file__).resolve().parent
_PIXI_ENV_EXPORTS = None
_LCG_ENV_EXPORTS = None
FIXED_LCG_VIEW = Path("/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt")
LCG_CVMFS_ROOT = Path("/cvmfs/sft.cern.ch/lcg")


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


def config_years(cfg):
    years = [str(year) for year in as_list(cfg.get("year"))]
    if not years:
        raise SystemExit("scale-factor config must define year as a string or a non-empty list")
    return years


def normalize_mc_groups(selected):
    return {group: list(names) for group, names in selected.get("mc_groups", {}).items()}


def selected_sample_groups(cfg, samples, section_name):
    section = cfg[section_name]
    selected = section["samples"]
    data = list(selected.get("data", []))
    mc_groups = normalize_mc_groups(selected)
    require_samples(samples, data)
    for names in mc_groups.values():
        require_samples(samples, names)
    validate_sample_info(samples, data, mc_groups)
    return data, mc_groups


def selected_sample_groups_by_year(cfg, samples, section_name):
    selected = cfg[section_name]["samples"]
    years = config_years(cfg)
    by_year = selected.get("by_year")
    if by_year is None:
        if len(years) != 1:
            raise SystemExit(f"{section_name}.samples.by_year is required when year lists multiple years")
        data, mc_groups = selected_sample_groups(cfg, samples, section_name)
        return [{"year": years[0], "data": data, "mc_groups": mc_groups}]

    out = []
    for year in years:
        if str(year) not in by_year:
            raise SystemExit(f"{section_name}.samples.by_year does not define samples for selected year {year}")
        year_selected = by_year.get(str(year), {})
        data = list(year_selected.get("data", []))
        mc_groups = normalize_mc_groups(year_selected)
        require_samples(samples, data)
        for names in mc_groups.values():
            require_samples(samples, names)
        validate_sample_info(samples, data, mc_groups)
        out.append({"year": str(year), "data": data, "mc_groups": mc_groups})
    return out


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


def run_targets(cfg, require_taggers=True):
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
        if require_taggers and not taggers:
            raise SystemExit(f"run_targets[{index}] must define the taggers to run")
        out.append({"jet_type": jet_type, "jet_category": jet_category, "taggers": taggers})
    return out


def require_local_path(path, label):
    text = str(path)
    if text.startswith("root://") or text.startswith("/store/"):
        raise SystemExit(f"{label} must be a local path, got: {text}")


def safe_path_token(value):
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in ("_", "-", ".") else "_")
    return "".join(out).strip("_") or "value"


def join_remote_path(base, *parts):
    text = str(base).rstrip("/")
    for part in parts:
        item = str(part).strip("/")
        if item:
            text += "/" + item
    return text


def target_storage_name(cfg, target, ntuple, year=None):
    pieces = [
        "scouting_vvv_scale_factor",
        ntuple.get("channel", "scouting_muon"),
        target["jet_type"],
        target["jet_category"],
    ]
    if not split_by_era(ntuple):
        pieces.append(year or config_years(cfg)[0])
    pieces.append(cfg["nano_version"])
    return safe_path_token("_".join(str(piece) for piece in pieces))


def sample_era_name(sample_name, default_year):
    text = str(sample_name)
    if text.endswith("_official"):
        text = text[: -len("_official")]
    if len(text) >= 5 and text[:4].isdigit() and text[4].isalpha():
        return text[:5]
    return str(default_year)


def era_year(era):
    text = str(era)
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else text


def data_era_groups(data_names, default_year):
    groups = []
    by_era = {}
    for name in data_names:
        era = sample_era_name(name, default_year)
        if era not in by_era:
            info = {"era": era, "year": era_year(era), "data": []}
            groups.append(info)
            by_era[era] = info
        by_era[era]["data"].append(name)
    if not groups:
        era = str(default_year)
        groups.append({"era": era, "year": era_year(era), "data": []})
    return groups


def split_by_era(section_cfg):
    return bool(section_cfg.get("split_by_era", False))


def sample_base_output_dir(base_path, cfg, era, section_cfg, year=None):
    base = resolve_path(base_path)
    require_local_path(base, "sample_base")
    if split_by_era(section_cfg):
        return base / era_year(era) / f"{era}_{cfg['nano_version']}"
    suffix = f"_{year or era_year(era)}_{cfg['nano_version']}"
    text = str(base)
    if not base.name.endswith(suffix):
        text += suffix
    return Path(text)


def mc_output_dir(base_path, cfg, year, section_cfg):
    base = resolve_path(base_path)
    require_local_path(base, "sample_base")
    year = str(year)
    if split_by_era(section_cfg):
        return base / year / f"{year}_mc_{cfg['nano_version']}"
    suffix = f"_{year}_mc_{cfg['nano_version']}"
    text = str(base)
    if not base.name.endswith(suffix):
        text += suffix
    return Path(text)


def ntuple_output_dir(cfg, era):
    return sample_base_output_dir(cfg["ntuple"]["sample_base"], cfg, era, cfg["ntuple"])


def ntuple_mc_output_dir(cfg, year):
    return mc_output_dir(cfg["ntuple"]["sample_base"], cfg, year, cfg["ntuple"])


def calibration_input_dir(cfg, era):
    return sample_base_output_dir(cfg["calibration"]["input_sample_base"], cfg, era, cfg["calibration"])


def calibration_mc_input_dir(cfg, year):
    return mc_output_dir(cfg["calibration"]["input_sample_base"], cfg, year, cfg["calibration"])


def tier_ntuple_output_dir(cfg, target, era, sample_set="data"):
    ntuple = cfg["ntuple"]
    if sample_set not in ("data", "mc"):
        raise SystemExit(f"Unknown ntuple sample set: {sample_set}")
    base = ntuple.get(
        "tier_storage_base",
        "root://cceos.ihep.ac.cn//eos/ihep/cms/store/user/yiyangz/Research/VVV/ScoutingVVVTools_sf/scale_factor/ntuples",
    )
    if not str(base).startswith("root://"):
        raise SystemExit(f"ntuple.tier_storage_base must be a root:// path, got: {base}")
    if split_by_era(ntuple):
        suffix = "_mc" if sample_set == "mc" else ""
        return join_remote_path(base, target_storage_name(cfg, target, ntuple), era_year(era), f"{era}{suffix}_{cfg['nano_version']}")
    if sample_set == "mc":
        return join_remote_path(
            base,
            target_storage_name(cfg, target, ntuple, era_year(era)),
            f"{era}_mc_{cfg['nano_version']}",
        )
    return join_remote_path(base, target_storage_name(cfg, target, ntuple, era_year(era)))


def year_output_dir(cfg):
    return ntuple_output_dir(cfg, config_years(cfg)[0])


def is_official_sample(sample_name):
    return str(sample_name).endswith("_official")


def ntuple_config_card(ntuple, year, official=False):
    config_key = "official_configs" if official else "configs"
    configs = ntuple.get(config_key, {})
    if isinstance(configs, dict) and str(year) in configs:
        return resolve_path(configs[str(year)])
    if official:
        raise SystemExit(f"ntuple.official_configs does not define a card for official data in year {year}")
    if "config" not in ntuple:
        raise SystemExit(f"ntuple.configs does not define a card for year {year}")
    config = str(ntuple["config"]).replace("{year}", str(year)).replace("$YEAR", str(year))
    return resolve_path(config)


def variation_names(ntuple_cfg):
    variations = ["nominal"]
    requested = ntuple_cfg.get("variations", {})
    for syst in ("jes", "jer", "met"):
        if requested.get(syst, False):
            variations.extend([f"{syst}_up", f"{syst}_down"])
    return variations


def build_jobs(value):
    if value is None:
        return os.cpu_count() or 1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "all", "available"):
            return os.cpu_count() or 1
        value = text
    jobs = int(value)
    if jobs <= 0:
        raise SystemExit("ntuple.build_jobs must be a positive integer or 'auto'")
    return jobs


def env_truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def sample_yaml_payload(samples, names):
    return {name: sample_paths(samples[name]) for name in names}


def mc_sample_names(mc_groups):
    names = []
    for group_names in mc_groups.values():
        for name in group_names:
            if name not in names:
                names.append(name)
    return names


def is_under(path, base):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
    except ValueError:
        return False
    return True


def require_conda_prefix():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise SystemExit("Mode 11 must run inside the pixi/conda environment so C++ dependencies all come from one prefix.")
    path = Path(conda_prefix)
    if not path.exists():
        raise SystemExit(f"CONDA_PREFIX does not exist: {conda_prefix}")
    if not is_under(sys.executable, path):
        raise SystemExit(
            "Mode 11 must be run with the python3 from the active pixi/conda environment. "
            f"Current python3 is {sys.executable}, CONDA_PREFIX is {path}."
        )
    return path


def unique_paths(paths):
    out = []
    for path in paths:
        path = Path(path)
        if path not in out:
            out.append(path)
    return out


def prepend_env_path(env, key, paths):
    values = [str(path) for path in paths if path and Path(path).exists()]
    if not values:
        return
    current = env.get(key, "")
    if current:
        values.append(current)
    env[key] = os.pathsep.join(values)


def resolve_dasgoclient():
    override = os.environ.get("SCALE_FACTOR_DASGOCLIENT")
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(f"SCALE_FACTOR_DASGOCLIENT does not exist: {override}")
        return str(path)

    cvmfs_dasgoclient = Path("/cvmfs/cms.cern.ch/common/dasgoclient")
    if cvmfs_dasgoclient.exists() and os.access(cvmfs_dasgoclient, os.X_OK):
        return str(cvmfs_dasgoclient)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry) / "dasgoclient"
        if path.exists() and os.access(path, os.X_OK):
            candidates.append(path)
    for path in candidates:
        if conda_prefix and is_under(path, conda_prefix):
            continue
        return str(path)
    found = shutil.which("dasgoclient")
    return found or ""


def resolve_das_home():
    override = os.environ.get("SCALE_FACTOR_DAS_HOME")
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(f"SCALE_FACTOR_DAS_HOME does not exist: {override}")
        return str(path)

    current_home = os.environ.get("HOME", "")
    if current_home.startswith("/afs/") and Path(current_home).exists():
        return current_home

    names = []
    for key in ("USER", "LOGNAME"):
        value = os.environ.get(key)
        if value and value not in names:
            names.append(value)
    if current_home:
        home_name = Path(current_home).name
        if home_name and home_name not in names:
            names.append(home_name)

    candidates = [Path("/afs/ihep.ac.cn/users") / name[0] / name for name in names if name]
    for path in candidates:
        if path.exists():
            return str(path)

    if current_home.startswith("/publicfs/cms/user/") and candidates:
        tried = ", ".join(str(path) for path in candidates)
        raise SystemExit(
            "Could not resolve an AFS HOME for DAS key definitions. "
            f"Set SCALE_FACTOR_DAS_HOME explicitly. Tried: {tried}"
        )

    return current_home


def clean_path_value(value, conda_prefix):
    if not value:
        return ""
    cleaned = []
    for item in value.split(os.pathsep):
        if not item:
            continue
        if conda_prefix and is_under(item, conda_prefix):
            continue
        cleaned.append(item)
    return os.pathsep.join(cleaned)


def clean_single_path_value(value, conda_prefix):
    if not value:
        return ""
    path = Path(value).expanduser()
    if conda_prefix and is_under(path, conda_prefix):
        return ""
    return value


def das_env_prefix():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    das_home = resolve_das_home()
    unset_vars = [
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_SHLVL",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "PIXI_PROJECT_NAME",
        "PIXI_PROJECT_ROOT",
        "PIXI_ENVIRONMENT_NAME",
        "PIXI_EXE",
        "PIXI_HOME",
        "PIXI_IN_SHELL",
    ]
    path_vars = [
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "CMAKE_PREFIX_PATH",
    ]
    single_path_vars = [
        "PYTHONHOME",
        "ROOTSYS",
        "CC",
        "CXX",
    ]
    unsets = list(unset_vars)
    assignments = []
    for name in path_vars:
        clean_value = clean_path_value(os.environ.get(name, ""), conda_prefix)
        if clean_value:
            assignments.append(f"{name}=" + shlex.quote(clean_value))
        else:
            unsets.append(name)
    for name in single_path_vars:
        clean_value = clean_single_path_value(os.environ.get(name, ""), conda_prefix)
        if clean_value:
            assignments.append(f"{name}=" + shlex.quote(clean_value))
        else:
            unsets.append(name)
    if das_home:
        assignments.append("HOME=" + shlex.quote(das_home))
    parts = ["env"]
    parts.extend(f"-u {name}" for name in unsets)
    parts.extend(assignments)
    cms_setup = "[ -r /cvmfs/cms.cern.ch/cmsset_default.sh ] && source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1; exec \"$@\""
    parts.extend([shlex.quote("/bin/bash"), "-lc", shlex.quote(cms_setup), "scale-factor-das"])
    return " ".join(parts)


def base_command_env():
    env = os.environ.copy()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    include_dirs = []
    if conda_prefix:
        prefix = Path(conda_prefix)
        prepend_env_path(env, "PATH", [prefix / "bin"])
        include_dirs.append(prefix / "include")
        env["ROOTSYS"] = str(prefix)
        prepend_env_path(env, "LD_LIBRARY_PATH", [prefix / "lib"])
        prepend_env_path(env, "LIBRARY_PATH", [prefix / "lib"])
    prepend_env_path(env, "CPATH", include_dirs)
    prepend_env_path(env, "CPLUS_INCLUDE_PATH", include_dirs)
    return env


def command_env():
    env = base_command_env()
    if not env.get("X509_USER_PROXY"):
        default_proxy = Path(f"/tmp/x509up_u{os.getuid()}")
        if default_proxy.exists():
            env["X509_USER_PROXY"] = str(default_proxy)
    dasgoclient = resolve_dasgoclient()
    if dasgoclient:
        env["SCALE_FACTOR_DASGOCLIENT"] = dasgoclient
        das_home = resolve_das_home()
        if das_home:
            env["SCALE_FACTOR_DAS_HOME"] = das_home
        env["SCALE_FACTOR_DAS_ENV_PREFIX"] = das_env_prefix()
    if os.environ.get("CONDA_PREFIX") or discover_lcg_setup():
        exports = cpp_env_exports()
        setup_env = exports.pop("SCALE_FACTOR_SETUP_ENV", None)
        if isinstance(setup_env, dict):
            for name in (
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "CONDA_SHLVL",
                "CONDA_EXE",
                "CONDA_PYTHON_EXE",
                "PIXI_PROJECT_NAME",
                "PIXI_PROJECT_ROOT",
                "PIXI_ENVIRONMENT_NAME",
                "PIXI_EXE",
                "PIXI_HOME",
                "PIXI_IN_SHELL",
            ):
                env.pop(name, None)
            env.update(setup_env)
        env.update(exports)
        if exports.get("SCALE_FACTOR_CC_COMPILER"):
            env["CC"] = exports["SCALE_FACTOR_CC_COMPILER"]
        if exports.get("SCALE_FACTOR_CXX_COMPILER"):
            env["CXX"] = exports["SCALE_FACTOR_CXX_COMPILER"]
        if exports.get("SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV"):
            env["CMAKE_PREFIX_PATH"] = exports["SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV"]
        lib_dirs = [exports.get("SCALE_FACTOR_CONDA_LIB_DIR"), exports.get("SCALE_FACTOR_CORRECTIONLIB_LIB_DIR")]
        prepend_env_path(env, "LD_LIBRARY_PATH", lib_dirs)
        prepend_env_path(env, "LIBRARY_PATH", lib_dirs)
    return env


def run_command(cmd, cwd=None, dry_run=False, env=None):
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run([str(part) for part in cmd], cwd=cwd, env=env or command_env())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def topwsf_command_env():
    env = os.environ.copy()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env.setdefault(name, "1")
    if conda_prefix:
        prefix = Path(conda_prefix)
        prepend_env_path(env, "PATH", [prefix / "bin"])
        prepend_env_path(env, "LD_LIBRARY_PATH", [prefix / "lib"])
        prepend_env_path(env, "LIBRARY_PATH", [prefix / "lib"])
    return env


def correctionlib_cmake_dir():
    conda_prefix = require_conda_prefix()
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

    candidates.extend(conda_prefix.glob("lib/python*/site-packages/correctionlib/cmake"))

    for path in candidates:
        if (path / "correctionlibConfig.cmake").exists() or (path / "correctionlib-config.cmake").exists():
            if not is_under(path, conda_prefix):
                raise SystemExit(f"Resolved correctionlib outside CONDA_PREFIX: {path}")
            return path
    raise SystemExit("Could not find correctionlib CMake config under the active pixi/conda environment.")


def correctionlib_package_dir():
    try:
        import correctionlib
    except ImportError:
        return None
    package_dir = Path(correctionlib.__file__).resolve().parent
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and not is_under(package_dir, conda_prefix):
        raise SystemExit(f"Resolved correctionlib Python package outside CONDA_PREFIX: {package_dir}")
    return package_dir


def yaml_cpp_cmake_dir():
    prefix = require_conda_prefix()
    candidates = [
        prefix / "lib" / "cmake" / "yaml-cpp",
        prefix / "share" / "cmake" / "yaml-cpp",
    ]
    for path in candidates:
        if (path / "yaml-cpp-config.cmake").exists() or (path / "yaml-cppConfig.cmake").exists():
            return path
    for pattern in ("lib/**/yaml-cpp-config.cmake", "lib/**/yaml-cppConfig.cmake", "share/**/yaml-cpp-config.cmake", "share/**/yaml-cppConfig.cmake"):
        for config_path in sorted(prefix.glob(pattern)):
            parent = config_path.parent
            if is_under(parent, prefix):
                return parent
    raise SystemExit("Could not find yaml-cpp CMake config under the active pixi/conda environment.")


def yaml_cpp_include_dir(prefix):
    candidates = [prefix / "include"]
    for path in candidates:
        if (path / "yaml-cpp" / "yaml.h").exists():
            return path
    raise SystemExit("Could not find yaml-cpp headers under the active pixi/conda environment.")


def yaml_cpp_library(prefix):
    candidates = []
    for lib_dir in (prefix / "lib", prefix / "lib64"):
        candidates.extend([
            lib_dir / "libyaml-cpp.so",
            lib_dir / "libyaml-cpp.dylib",
            lib_dir / "libyaml-cpp.a",
        ])
        candidates.extend(sorted(lib_dir.glob("libyaml-cpp.so*")))
    path = first_existing_path(candidates, prefix)
    if path:
        return path
    raise SystemExit("Could not find libyaml-cpp under the active pixi/conda environment.")


def root_config_value(option):
    try:
        result = subprocess.run(["root-config", option], capture_output=True, text=True, env=base_command_env())
    except OSError:
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return Path(text) if text else None


def root_cmake_dir():
    prefix = require_conda_prefix()
    cmake_dir = root_config_value("--cmakedir")
    if cmake_dir:
        if not is_under(cmake_dir, prefix):
            raise SystemExit(f"root-config --cmakedir resolves ROOT outside CONDA_PREFIX: {cmake_dir}")
        if (cmake_dir / "ROOTConfig.cmake").exists():
            return cmake_dir
    root_prefix = root_config_value("--prefix")
    if root_prefix and not is_under(root_prefix, prefix):
        raise SystemExit(f"root-config resolves ROOT outside CONDA_PREFIX: {root_prefix}")
    search_roots = [prefix]
    if root_prefix and root_prefix != prefix:
        search_roots.append(root_prefix)
    candidates = []
    for base in search_roots:
        candidates.extend([
            base / "lib" / "cmake" / "ROOT",
            base / "lib" / "cmake",
            base / "cmake" / "ROOT",
            base / "cmake",
            base / "lib" / "root" / "cmake",
            base / "share" / "root" / "cmake",
            base / "etc" / "root" / "cmake",
        ])
    candidates = unique_paths(candidates)
    for path in candidates:
        if (path / "ROOTConfig.cmake").exists():
            return path
    for pattern in ("lib/**/ROOTConfig.cmake", "cmake/**/ROOTConfig.cmake", "share/**/ROOTConfig.cmake", "etc/**/ROOTConfig.cmake"):
        for config_path in sorted(prefix.glob(pattern)):
            parent = config_path.parent
            if is_under(parent, prefix):
                return parent
    tried = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "Could not find ROOTConfig.cmake under the active pixi/conda environment. "
        "mode 11 must not mix the pixi compiler with system ROOT. Checked: " + tried
    )


def validate_cached_cmake_path(build_dir, key, expected, label, required_files=None, any_required_files=None):
    cached = read_cmake_cache_value(build_dir, key)
    if not cached:
        return
    cached_path = Path(cached)
    expected_path = Path(expected)
    if cached_path.resolve() != expected_path.resolve():
        raise SystemExit(
            f"nano.cpp build cache uses {label}={cached}, but the active C++ dependency environment resolves "
            f"{label}={expected_path}. Move systematics/scale_factor/nano.cpp/build aside and rerun mode 11."
        )
    for name in required_files or []:
        if not (cached_path / name).exists():
            raise SystemExit(
                f"nano.cpp build cache uses {label}={cached}, but required file {name} is missing. "
                "Move systematics/scale_factor/nano.cpp/build aside and rerun mode 11."
            )
    if any_required_files and not any((cached_path / name).exists() for name in any_required_files):
        names = ", ".join(any_required_files)
        raise SystemExit(
            f"nano.cpp build cache uses {label}={cached}, but none of these required files exists: {names}. "
            "Move systematics/scale_factor/nano.cpp/build aside and rerun mode 11."
        )


def validate_cached_compiler(build_dir, cache_key, expected, label):
    cached = read_cmake_cache_value(build_dir, cache_key)
    if not cached:
        return
    if Path(cached).resolve() != Path(expected).resolve():
        raise SystemExit(
            f"nano.cpp build cache uses {label}={cached}, but the active C++ dependency environment resolves "
            f"{label}={expected}. Move systematics/scale_factor/nano.cpp/build aside and rerun mode 11."
        )


def python3_executable(prefix):
    candidate = prefix / "bin" / "python3"
    return candidate if candidate.exists() else Path(sys.executable)


def pixi_env_exports():
    global _PIXI_ENV_EXPORTS
    if _PIXI_ENV_EXPORTS is not None:
        return dict(_PIXI_ENV_EXPORTS)
    prefix = require_conda_prefix()
    cc_compiler = conda_c_compiler(prefix)
    cxx_compiler = conda_cxx_compiler(prefix)
    if not cc_compiler:
        raise SystemExit(
            "Could not find a pixi/conda C compiler wrapper under CONDA_PREFIX. "
            "Run pixi install after pulling the updated pixi.toml."
        )
    if not cxx_compiler:
        raise SystemExit(
            "Could not find a pixi/conda C++ compiler wrapper under CONDA_PREFIX. "
            "Run pixi install after pulling the updated pixi.toml."
        )
    root_dir = root_cmake_dir()
    correction_dir = correctionlib_cmake_dir()
    correction_package_dir = correctionlib_package_dir()
    if not correction_package_dir:
        raise SystemExit("Could not import correctionlib from the active pixi/conda environment.")
    correction_lib_dir = correction_package_dir / "lib"
    yaml_dir = yaml_cpp_cmake_dir()
    yaml_include_dir = yaml_cpp_include_dir(prefix)
    yaml_library = yaml_cpp_library(prefix)
    conda_lib_dir = prefix / "lib"
    lib_dirs = [conda_lib_dir]
    if correction_lib_dir.exists():
        lib_dirs.append(correction_lib_dir)
    link_flags = []
    for path in lib_dirs:
        link_flags.extend([f"-L{path}", f"-Wl,-rpath,{path}", f"-Wl,-rpath-link,{path}"])
    cmake_prefixes = [str(prefix), str(prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr")]
    _PIXI_ENV_EXPORTS = {
        "SCALE_FACTOR_CPP_RUNTIME": "pixi",
        "SCALE_FACTOR_CONDA_PREFIX": str(prefix),
        "SCALE_FACTOR_RUNTIME_PREFIX": str(prefix),
        "SCALE_FACTOR_CONDA_LIB_DIR": str(conda_lib_dir),
        "SCALE_FACTOR_CC_COMPILER": str(cc_compiler),
        "SCALE_FACTOR_CXX_COMPILER": str(cxx_compiler),
        "SCALE_FACTOR_PYTHON3_EXECUTABLE": str(python3_executable(prefix)),
        "SCALE_FACTOR_ROOT_DIR": str(root_dir),
        "SCALE_FACTOR_CORRECTIONLIB_DIR": str(correction_dir),
        "SCALE_FACTOR_CORRECTIONLIB_LIB_DIR": str(correction_lib_dir) if correction_lib_dir.exists() else "",
        "SCALE_FACTOR_YAML_CPP_DIR": str(yaml_dir),
        "SCALE_FACTOR_YAML_CPP_INCLUDE_DIR": str(yaml_include_dir),
        "SCALE_FACTOR_YAML_CPP_LIBRARY": str(yaml_library),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH": ";".join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV": os.pathsep.join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_LINK_FLAGS": " ".join(link_flags),
        "SCALE_FACTOR_CMAKE_RPATH": ";".join(str(path) for path in lib_dirs),
    }
    return dict(_PIXI_ENV_EXPORTS)


def pixi_cmake_args(build_dir):
    exports = cpp_env_exports()
    validate_cached_compiler(build_dir, "CMAKE_C_COMPILER", exports["SCALE_FACTOR_CC_COMPILER"], "CMAKE_C_COMPILER")
    validate_cached_compiler(build_dir, "CMAKE_CXX_COMPILER", exports["SCALE_FACTOR_CXX_COMPILER"], "CMAKE_CXX_COMPILER")
    validate_cached_compiler(build_dir, "Python3_EXECUTABLE", exports["SCALE_FACTOR_PYTHON3_EXECUTABLE"], "Python3_EXECUTABLE")
    validate_cached_cmake_path(build_dir, "ROOT_DIR", exports["SCALE_FACTOR_ROOT_DIR"], "ROOT_DIR", ["ROOTConfig.cmake"])
    validate_cached_cmake_path(
        build_dir,
        "correctionlib_DIR",
        exports["SCALE_FACTOR_CORRECTIONLIB_DIR"],
        "correctionlib_DIR",
        any_required_files=["correctionlibConfig.cmake", "correctionlib-config.cmake"],
    )
    validate_cached_cmake_path(
        build_dir,
        "yaml-cpp_DIR",
        exports["SCALE_FACTOR_YAML_CPP_DIR"],
        "yaml-cpp_DIR",
        any_required_files=["yaml-cpp-config.cmake", "yaml-cppConfig.cmake", "yamlcpp-config.cmake", "yamlcppConfig.cmake"],
    )
    validate_cached_cmake_path(
        build_dir,
        "NANO_YAML_CPP_INCLUDE_DIR",
        exports["SCALE_FACTOR_YAML_CPP_INCLUDE_DIR"],
        "NANO_YAML_CPP_INCLUDE_DIR",
        required_files=["yaml-cpp/yaml.h"],
    )
    validate_cached_compiler(build_dir, "NANO_YAML_CPP_LIBRARY", exports["SCALE_FACTOR_YAML_CPP_LIBRARY"], "NANO_YAML_CPP_LIBRARY")
    return [
        f"-DCMAKE_CXX_COMPILER={exports['SCALE_FACTOR_CXX_COMPILER']}",
        f"-DPython3_EXECUTABLE={exports['SCALE_FACTOR_PYTHON3_EXECUTABLE']}",
        f"-DCMAKE_PREFIX_PATH={exports['SCALE_FACTOR_CMAKE_PREFIX_PATH']}",
        f"-DROOT_DIR={exports['SCALE_FACTOR_ROOT_DIR']}",
        f"-Dcorrectionlib_DIR={exports['SCALE_FACTOR_CORRECTIONLIB_DIR']}",
        f"-Dyaml-cpp_DIR={exports['SCALE_FACTOR_YAML_CPP_DIR']}",
        f"-DNANO_YAML_CPP_INCLUDE_DIR={exports['SCALE_FACTOR_YAML_CPP_INCLUDE_DIR']}",
        f"-DNANO_YAML_CPP_LIBRARY={exports['SCALE_FACTOR_YAML_CPP_LIBRARY']}",
        f"-DCMAKE_EXE_LINKER_FLAGS={exports['SCALE_FACTOR_CMAKE_LINK_FLAGS']}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={exports['SCALE_FACTOR_CMAKE_LINK_FLAGS']}",
        f"-DCMAKE_MODULE_LINKER_FLAGS={exports['SCALE_FACTOR_CMAKE_LINK_FLAGS']}",
        f"-DCMAKE_BUILD_RPATH={exports['SCALE_FACTOR_CMAKE_RPATH']}",
        f"-DCMAKE_INSTALL_RPATH={exports['SCALE_FACTOR_CMAKE_RPATH']}",
    ]

def read_cmake_cache_value(build_dir, key):
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    with open(cache, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(f"{key}:"):
                return line.split("=", 1)[1].strip()
    return None


def first_existing_path(paths, base=None):
    for path in paths:
        if not path:
            continue
        path = Path(path).expanduser()
        if not path.exists():
            continue
        if base and not is_under(path, base):
            continue
        return path
    return None


def conda_c_compiler(conda_prefix):
    bin_dir = Path(conda_prefix) / "bin"
    candidates = []
    if os.environ.get("CC"):
        candidates.append(Path(os.environ["CC"]))
    for pattern in (
        "*-conda-linux-gnu-cc",
        "*-conda_cos*-linux-gnu-cc",
        "*-conda-linux-gnu-gcc",
        "*-conda_cos*-linux-gnu-gcc",
    ):
        candidates.extend(sorted(bin_dir.glob(pattern)))
    candidates.extend(bin_dir / name for name in ("gcc", "cc"))
    return first_existing_path(candidates, conda_prefix)


def conda_cxx_compiler(conda_prefix):
    bin_dir = Path(conda_prefix) / "bin"
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(Path(os.environ["CXX"]))
    for pattern in (
        "*-conda-linux-gnu-c++",
        "*-conda_cos*-linux-gnu-c++",
        "*-conda-linux-gnu-g++",
        "*-conda_cos*-linux-gnu-g++",
    ):
        candidates.extend(sorted(bin_dir.glob(pattern)))
    candidates.extend(bin_dir / name for name in ("g++", "c++"))
    return first_existing_path(candidates, conda_prefix)


def lcg_setup_from_value(value):
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path
    setup = path / "setup.sh"
    if setup.is_file():
        return setup
    return None


def discover_lcg_setup():
    setup = lcg_setup_from_value(FIXED_LCG_VIEW)
    if setup:
        return setup
    if FIXED_LCG_VIEW.parent.exists():
        raise SystemExit(f"Fixed CVMFS LCG view is required but missing: {FIXED_LCG_VIEW}")
    return None


def source_setup_environment(setup):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    env = os.environ.copy()
    for name in ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH", "PYTHONPATH"):
        clean_value = clean_path_value(env.get(name, ""), conda_prefix)
        if clean_value:
            env[name] = clean_value
        else:
            env.pop(name, None)
    for name in ("PYTHONHOME", "ROOTSYS", "CC", "CXX"):
        clean_value = clean_single_path_value(env.get(name, ""), conda_prefix)
        if clean_value:
            env[name] = clean_value
        else:
            env.pop(name, None)
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_SHLVL",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "PIXI_PROJECT_NAME",
        "PIXI_PROJECT_ROOT",
        "PIXI_ENVIRONMENT_NAME",
        "PIXI_EXE",
        "PIXI_HOME",
        "PIXI_IN_SHELL",
    ):
        env.pop(name, None)

    command = f"set -e; source {shlex.quote(str(setup))} >/dev/null 2>&1; env -0"
    result = subprocess.run(["/bin/bash", "-lc", command], capture_output=True, env=env)
    if result.returncode != 0:
        raise SystemExit(f"Failed to source LCG setup script: {setup}")

    out = {}
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        out[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return out


def which_in_env(name, env):
    found = shutil.which(name, path=env.get("PATH", ""))
    return Path(found) if found else None


def env_path_entries(env, key):
    return [Path(item) for item in env.get(key, "").split(os.pathsep) if item]


def is_lcg_path(path):
    return is_under(Path(path).resolve(), LCG_CVMFS_ROOT)


def lcg_root_config_value(root_config, option, env):
    result = subprocess.run([str(root_config), option], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return Path(text) if text else None


def root_cmake_candidate_dirs(base):
    return [
        base,
        base / "ROOT",
        base / "root",
        base / "lib" / "cmake" / "ROOT",
        base / "lib" / "cmake" / "root",
        base / "lib64" / "cmake" / "ROOT",
        base / "lib64" / "cmake" / "root",
        base / "lib" / "cmake",
        base / "lib64" / "cmake",
        base / "cmake" / "ROOT",
        base / "cmake" / "root",
        base / "cmake",
        base / "share" / "root" / "cmake",
        base / "share" / "ROOT" / "cmake",
        base / "etc" / "root" / "cmake",
    ]


def first_root_cmake_dir(candidates):
    for path in unique_paths(path for path in candidates if path):
        if (path / "ROOTConfig.cmake").exists():
            return path
    return None


def lcg_root_cmake_dir(env, view_root):
    root_config = which_in_env("root-config", env)
    if not root_config:
        raise SystemExit("LCG/CVMFS C++ runtime was selected, but root-config is not available after sourcing the setup script.")
    search_roots = [
        lcg_root_config_value(root_config, "--cmakedir", env),
        lcg_root_config_value(root_config, "--prefix", env),
        Path(env["ROOTSYS"]) if env.get("ROOTSYS") else None,
        view_root,
    ]
    search_roots.extend(env_path_entries(env, "CMAKE_PREFIX_PATH"))
    candidates = []
    for base in search_roots:
        if base:
            candidates.extend(root_cmake_candidate_dirs(base))
    root_dir = first_root_cmake_dir(candidates)
    if root_dir:
        return root_dir

    tried = ", ".join(str(path) for path in unique_paths(path for path in candidates if path))
    raise SystemExit(f"Could not resolve ROOTConfig.cmake from fixed LCG root-config: {root_config}. Checked: {tried}")


def cmake_package_candidate_dirs(base, package_names):
    candidates = [base, base / "cmake"]
    for pkg_name in package_names:
        candidates.extend([
            base / pkg_name,
            base / pkg_name / "cmake",
            base / "lib" / "cmake" / pkg_name,
            base / "lib64" / "cmake" / pkg_name,
            base / "share" / pkg_name / "cmake",
            base / "share" / "cmake" / pkg_name,
            base / "share" / pkg_name,
        ])
    for site_parent in [base, base / "lib", base / "lib64"]:
        for site_packages in site_parent.glob("python*/site-packages"):
            for pkg_name in package_names:
                candidates.extend([
                    site_packages / pkg_name,
                    site_packages / pkg_name / "cmake",
                    site_packages / pkg_name / "lib" / "cmake" / pkg_name,
                ])
    if base.name == "site-packages":
        for pkg_name in package_names:
            candidates.extend([base / pkg_name, base / pkg_name / "cmake"])
    return candidates


def cmake_config_dir(candidates, config_names):
    names = {name.lower() for name in config_names}
    for path in unique_paths(path for path in candidates if path):
        if not path.exists():
            continue
        if not is_lcg_path(path):
            continue
        for name in config_names:
            if (path / name).exists():
                return path
        for pattern in ("*Config.cmake", "*-config.cmake"):
            for child in path.glob(pattern):
                if child.name.lower() in names:
                    return path
    return None


def lcg_search_roots(env, view_root):
    roots = [view_root]
    roots.extend(env_path_entries(env, "CMAKE_PREFIX_PATH"))
    roots.extend(env_path_entries(env, "PYTHONPATH"))
    for base in [view_root, *env_path_entries(env, "CMAKE_PREFIX_PATH")]:
        roots.extend([base / "lib", base / "lib64", base / "python"])
        roots.extend(base.glob("lib/python*/site-packages"))
        roots.extend(base.glob("lib64/python*/site-packages"))
        roots.extend(base.glob("python/lib/python*/site-packages"))
        roots.extend(base.glob("python/python*/site-packages"))
    return unique_paths(path for path in roots if path)


def find_lcg_cmake_package_dir(env, view_root, package_names, config_names, label):
    candidates = []
    for base in lcg_search_roots(env, view_root):
        candidates.extend(cmake_package_candidate_dirs(base, package_names))
    package_dir = cmake_config_dir(candidates, config_names)
    if package_dir:
        return package_dir
    tried = ", ".join(str(path) for path in unique_paths(path for path in candidates if path)[:80])
    raise SystemExit(f"Could not find {label} CMake config in fixed LCG setup. Checked: {tried}")


def find_lcg_include_dir(env, view_root, header, label):
    candidates = []
    for base in lcg_search_roots(env, view_root):
        candidates.extend([base, base / "include"])
    candidates.extend(env_path_entries(env, "CPATH"))
    candidates.extend(env_path_entries(env, "CPLUS_INCLUDE_PATH"))
    for path in unique_paths(path for path in candidates if path):
        header_path = path / header
        if header_path.exists() and is_lcg_path(header_path):
            include_root = header_path.resolve()
            for _ in header.parts:
                include_root = include_root.parent
            return include_root
    tried = ", ".join(str(path) for path in unique_paths(path for path in candidates if path)[:80])
    raise SystemExit(f"Could not find {label} include directory in fixed LCG setup. Checked: {tried}")


def find_lcg_library(env, view_root, names, label):
    candidates = []
    library_roots = [view_root]
    library_roots.extend(env_path_entries(env, "CMAKE_PREFIX_PATH"))
    library_roots.extend(env_path_entries(env, "LD_LIBRARY_PATH"))
    library_roots.extend(env_path_entries(env, "LIBRARY_PATH"))
    for base in unique_paths(path for path in library_roots if path):
        dirs = [base]
        if base.name not in ("lib", "lib64"):
            dirs.extend([base / "lib", base / "lib64"])
        for lib_dir in dirs:
            for name in names:
                candidates.extend([
                    lib_dir / f"lib{name}.so",
                    lib_dir / f"lib{name}.dylib",
                    lib_dir / f"lib{name}.a",
                ])
                candidates.extend(sorted(lib_dir.glob(f"lib{name}.so*")))
    for path in unique_paths(candidates):
        if path.exists() and is_lcg_path(path):
            return path.resolve()
    tried = ", ".join(str(path) for path in unique_paths(candidates)[:80])
    raise SystemExit(f"Could not find {label} library in fixed LCG setup. Checked: {tried}")


def lcg_env_exports():
    global _LCG_ENV_EXPORTS
    if _LCG_ENV_EXPORTS is not None:
        return dict(_LCG_ENV_EXPORTS)

    setup = discover_lcg_setup()
    if not setup:
        return None

    env = source_setup_environment(setup)
    view_root = setup.parent
    cc_compiler = which_in_env("gcc", env) or which_in_env("cc", env)
    cxx_compiler = which_in_env("g++", env) or which_in_env("c++", env)
    python3 = which_in_env("python3", env) or which_in_env("python", env)
    if not cc_compiler or not cxx_compiler:
        raise SystemExit(f"LCG setup does not provide gcc/g++ compilers: {setup}")
    if not python3:
        raise SystemExit(f"LCG setup does not provide python3: {setup}")

    root_dir = lcg_root_cmake_dir(env, view_root)
    correction_dir = find_lcg_cmake_package_dir(
        env,
        view_root,
        ("correctionlib", "correctionlib-cpp"),
        ("correctionlibConfig.cmake", "correctionlib-config.cmake"),
        "correctionlib",
    )
    yaml_dir = find_lcg_cmake_package_dir(
        env,
        view_root,
        ("yaml-cpp", "yamlcpp", "yaml_cpp"),
        ("yaml-cpp-config.cmake", "yaml-cppConfig.cmake", "yamlcpp-config.cmake", "yamlcppConfig.cmake"),
        "yaml-cpp",
    )
    yaml_include_dir = find_lcg_include_dir(env, view_root, Path("yaml-cpp") / "yaml.h", "yaml-cpp")
    yaml_library = find_lcg_library(env, view_root, ("yaml-cpp", "yamlcpp"), "yaml-cpp")

    cmake_prefixes = []
    for item in env.get("CMAKE_PREFIX_PATH", "").split(os.pathsep):
        if item and item not in cmake_prefixes:
            cmake_prefixes.append(item)
    if str(view_root) not in cmake_prefixes:
        cmake_prefixes.insert(0, str(view_root))

    _LCG_ENV_EXPORTS = {
        "SCALE_FACTOR_CPP_RUNTIME": "lcg",
        "SCALE_FACTOR_SETUP_ENV": env,
        "SCALE_FACTOR_RUNTIME_PREFIX": "",
        "SCALE_FACTOR_CC_COMPILER": str(cc_compiler),
        "SCALE_FACTOR_CXX_COMPILER": str(cxx_compiler),
        "SCALE_FACTOR_PYTHON3_EXECUTABLE": str(python3),
        "SCALE_FACTOR_ROOT_DIR": str(root_dir),
        "SCALE_FACTOR_CORRECTIONLIB_DIR": str(correction_dir),
        "SCALE_FACTOR_CORRECTIONLIB_LIB_DIR": "",
        "SCALE_FACTOR_YAML_CPP_DIR": str(yaml_dir),
        "SCALE_FACTOR_YAML_CPP_INCLUDE_DIR": str(yaml_include_dir),
        "SCALE_FACTOR_YAML_CPP_LIBRARY": str(yaml_library),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH": ";".join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV": os.pathsep.join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_LINK_FLAGS": "",
        "SCALE_FACTOR_CMAKE_RPATH": "",
    }
    return dict(_LCG_ENV_EXPORTS)


def cpp_env_exports():
    exports = lcg_env_exports()
    if exports:
        return dict(exports)
    return pixi_env_exports()


def make_ntuple(cfg, args):
    if not discover_lcg_setup():
        require_conda_prefix()
    samples = sample_map(cfg)
    year_sample_groups = selected_sample_groups_by_year(cfg, samples, "ntuple")
    ntuple = cfg["ntuple"]
    nano_repo = resolve_path(ntuple.get("repo", "nano.cpp"))
    sample_dir = resolve_path(ntuple.get("generated_sample_dir", "generated/samples"))
    use_tier_storage = bool(ntuple.get("use_tier_storage", False))
    require_local_path(resolve_path(ntuple.get("job_dir", "jobs/ntuples")), "ntuple.job_dir")
    variations = ",".join(variation_names(ntuple))
    targets = run_targets(cfg, require_taggers=False)

    commands = []
    build_dir = nano_repo / "build"
    if ntuple.get("build_before_make_condor", True):
        commands.append([
            "cmake", "-S", str(nano_repo), "-B", str(build_dir),
            *pixi_cmake_args(build_dir),
        ])
        commands.append(["cmake", "--build", str(build_dir), "-j", build_jobs(ntuple.get("build_jobs", "auto"))])

    binary = nano_repo / "build" / "nano_make_condor"
    expected_outputs = []
    for year_samples in year_sample_groups:
        year = str(year_samples["year"])
        mc_names = mc_sample_names(year_samples["mc_groups"])
        eras = data_era_groups(year_samples["data"], year)

        for target in targets:
            planned_jobs = []

            if mc_names:
                tokens = {
                    "jet_type": target["jet_type"],
                    "jet_category": target["jet_category"],
                    "year": year,
                    "era": year,
                }
                sample_yaml = sample_dir / f"{tokens['jet_type']}_{tokens['jet_category']}_{year}_mc.yaml"
                write_yaml(sample_yaml, sample_yaml_payload(samples, mc_names))
                out_dir = ntuple_mc_output_dir(cfg, year)
                planned_jobs.append((
                    "mc",
                    sample_yaml,
                    False,
                    tokens,
                    out_dir,
                    tier_ntuple_output_dir(cfg, target, year, "mc") if use_tier_storage else str(out_dir),
                    ntuple_config_card(ntuple, year),
                ))
                expected_outputs.append(("MC", year, out_dir))

            for era_info in eras:
                if not era_info["data"]:
                    continue
                official_flags = [is_official_sample(name) for name in era_info["data"]]
                if any(official_flags) and not all(official_flags):
                    raise SystemExit(
                        f"Data era {era_info['era']} mixes official and non-official samples; "
                        "generate them as separate data-era jobs because they require different branch cards"
                    )
                era = era_info["era"]
                tokens = {
                    "jet_type": target["jet_type"],
                    "jet_category": target["jet_category"],
                    "year": year,
                    "era": era,
                }
                sample_yaml = sample_dir / f"{tokens['jet_type']}_{tokens['jet_category']}_{era}_data.yaml"
                write_yaml(sample_yaml, sample_yaml_payload(samples, era_info["data"]))
                out_dir = ntuple_output_dir(cfg, era)
                planned_jobs.append((
                    "data",
                    sample_yaml,
                    True,
                    tokens,
                    out_dir,
                    tier_ntuple_output_dir(cfg, target, era, "data") if use_tier_storage else str(out_dir),
                    ntuple_config_card(ntuple, year, official=all(official_flags)),
                ))
                expected_outputs.append(("data", era, out_dir))

            for sample_set, sample_yaml, is_data, tokens, out_dir, ntuple_remote_dir, config_card in planned_jobs:
                job_dir_pattern = ntuple.get("job_dir", "jobs/ntuples/{jet_type}_{jet_category}_{era}_{sample_set}")
                job_dir = resolve_path(job_dir_pattern.format(**tokens, sample_set=sample_set))
                require_local_path(job_dir, "ntuple.job_dir")
                cmd = [
                    binary,
                    "--input-yaml", sample_yaml,
                    "--job-dir", job_dir,
                    "--output-dir", ntuple_remote_dir,
                    "--merge-output-dir", out_dir,
                    "--config", config_card,
                    "--channel", ntuple.get("channel", "scouting_muon"),
                    "--tree-name", ntuple.get("tree_name", "Events"),
                    "--nfiles-per-job", int(ntuple.get("nfiles_per_job", 1)),
                    "--num-events", int(ntuple.get("num_events", -1)),
                    "--request-disk-mb", int(ntuple.get("request_disk_mb", 50000)),
                    "--variations", variations,
                    "--use-sample-key-nickname",
                ]
                if target["taggers"]:
                    cmd.extend(["--set", "stored_tagger_names=" + dump_yaml_value(target["taggers"]).strip()])
                if is_data:
                    cmd.append("--run-data")
                if ntuple.get("download_remote_inputs", False):
                    cmd.append("--download-remote-inputs")
                else:
                    cmd.append("--no-download-remote-inputs")
                commands.append(cmd)
                if ntuple.get("submit_condor", False):
                    commands.append(["./submit.sh", {"cwd": job_dir}])

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
    for sample_set, label, output_dir in expected_outputs:
        print(f"Expected merged {sample_set} ntuples for {label} under {output_dir}")


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


def calibration_input_variations(calibration):
    variations = ["nominal"]
    enabled_systematics = calibration.get("systematics", {}).get("enabled", [])
    for systematic in ("jes", "jer", "met"):
        if systematic in enabled_systematics:
            variations.extend([f"{systematic}_up", f"{systematic}_down"])
    return variations


def validate_annual_input_link(source, destination):
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f"Missing merged ntuple required for the annual fit input: {source}")
    if destination.is_symlink():
        if destination.resolve() == source:
            return source
        raise SystemExit(
            f"Annual fit input link already points elsewhere: {destination} -> {destination.resolve()}"
        )
    if destination.exists():
        raise SystemExit(f"Annual fit input would overwrite an existing file: {destination}")
    return source


def link_annual_input_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise SystemExit(
            f"Annual fit input link already points elsewhere: {destination} -> {destination.resolve()}"
        )
    if destination.exists():
        raise SystemExit(f"Annual fit input would overwrite an existing file: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def prepare_annual_fit_input(cfg, year_samples):
    year = str(year_samples["year"])
    calibration = cfg["calibration"]
    annual_dir = calibration_input_dir(cfg, year)
    mc_dir = calibration_mc_input_dir(cfg, year)
    mc_names = mc_sample_names(year_samples["mc_groups"])
    links = []

    for data_name in year_samples["data"]:
        era = sample_era_name(data_name, year)
        source = calibration_input_dir(cfg, era) / "nominal" / f"{data_name}_nominal.root"
        destination = annual_dir / "nominal" / source.name
        links.append((source, destination))

    for variation in calibration_input_variations(calibration):
        for mc_name in mc_names:
            source = mc_dir / variation / f"{mc_name}_{variation}.root"
            destination = annual_dir / variation / source.name
            links.append((source, destination))

    destinations = {}
    for source, destination in links:
        existing_source = destinations.get(destination)
        if existing_source is not None and existing_source != source:
            raise SystemExit(
                f"Annual fit input has colliding source files for {destination}: {existing_source}, {source}"
            )
        destinations[destination] = source

    validated_links = [
        (validate_annual_input_link(source, destination), destination)
        for source, destination in links
    ]
    for source, destination in validated_links:
        link_annual_input_file(source, destination)

    print(f"Prepared annual fit input for {year}: {annual_dir}")


def generated_card(cfg, samples, data_names, mc_groups, year, target, tagger_name, tagger_cfg):
    cal = cfg["calibration"]
    enabled_mc_groups = list(cal.get("fit_enabled_mc_groups", mc_groups.keys()))
    unknown_enabled_groups = sorted(set(enabled_mc_groups) - set(mc_groups))
    if unknown_enabled_groups:
        raise SystemExit(
            "calibration.fit_enabled_mc_groups contains unknown MC group(s): "
            + ", ".join(unknown_enabled_groups)
        )
    year = str(year)
    card_dir = resolve_path(cal.get("generated_card_dir", "generated/topwsf")) / year
    require_local_path(card_dir, "calibration.generated_card_dir")
    boohft_base = resolve_path(cal.get("repo", "boohft-calib")) / "cards" / "topwsf" / "base.yml"
    extends = [os.path.relpath(boohft_base, card_dir)]
    score_bins = {
        score_bin_name(lo, hi): [float(lo), float(hi)]
        for lo, hi in tagger_cfg["score_bins"]
    }
    lumi = data_lumi(samples, data_names)
    if lumi <= 0.0:
        raise SystemExit(f"data sample lumi is missing/non-positive in sample.json for {year}")
    fit_cfg = cal.get("fit", {})
    systematics = list(cal.get("systematics", {}).get("enabled", []))
    input_sample_base = calibration_input_dir(cfg, year)
    require_local_path(input_sample_base, "calibration.input_sample_base")
    payload = {
        "extends": extends,
        "routine_name": f"{cal.get('routine_name_prefix', 'scouting_topwsf')}_{target['jet_type']}_{target['jet_category'].lower()}_{tagger_name}",
        "category": cal.get("category", "w"),
        "year": year,
        "nano_version": str(cfg["nano_version"]),
        "sample_base": str(input_sample_base),
        "sample_scan_wp": str(input_sample_base),
        "sample_base_append_year_version": False,
        "sample_scan_wp_append_year_version": False,
        "workers": list(cal.get("workers", [20, 20])),
        "run_step": str(cal.get("run_step", "11")),
        "skip_coffea": False,
        "apply_toppt_weight": True,
        "systematics": systematics,
        "selection": cal["selection"],
        "template_pt_bins": list(cal.get("template_pt_bins", [100, 200.0, 1200.0])),
        "skip_fit": bool(fit_cfg.get("skip_fit", False)),
        "fit_run_impact": bool(fit_cfg.get("fit_run_impact", True)),
        "fit_impact_parallel": int(fit_cfg.get("fit_impact_parallel", 8)),
        "fit_auto_mc_stats": int(fit_cfg.get("fit_auto_mc_stats", 0)),
        "lumi_dict": {year: lumi},
        "lumi_uncertainty": {year: 1.025},
        "data_samples": list(data_names),
        "enabled_sample_groups": enabled_mc_groups,
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
    year_sample_groups = selected_sample_groups_by_year(cfg, samples, "calibration")
    cal = cfg["calibration"]
    boohft_repo = resolve_path(cal.get("repo", "boohft-calib"))
    targets = run_targets(cfg)

    cards = []
    for year_samples in year_sample_groups:
        year = str(year_samples["year"])
        mc_groups = year_samples["mc_groups"]
        if not args.generate_only and not args.dry_run:
            prepare_annual_fit_input(cfg, year_samples)
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
                cards.append(generated_card(cfg, samples, year_samples["data"], mc_groups, year, target, tagger_name, tagger_cfg))

    if not cards:
        raise SystemExit("No scale-factor cards were generated")

    launcher_env = topwsf_command_env()
    for card in cards:
        cmd = [
            sys.executable, "launcher.py", str(card),
            "--routine", cal.get("routine", "topwsf"),
            "--run-step", str(cal.get("run_step", "11")),
            "--workers", *[str(x) for x in cal.get("workers", [20, 20])],
        ]
        if args.generate_only or not cal.get("run_launcher", True):
            print(" ".join(cmd), flush=True)
            continue
        run_command(cmd, cwd=boohft_repo, dry_run=args.dry_run, env=launcher_env)
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
