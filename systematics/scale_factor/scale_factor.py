#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


HERE = Path(__file__).resolve().parent
_PIXI_ENV_EXPORTS = None


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


def sample_yaml_payload(samples, names):
    return {name: sample_paths(samples[name]) for name in names}


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
    dasgoclient = resolve_dasgoclient()
    if dasgoclient:
        env["SCALE_FACTOR_DASGOCLIENT"] = dasgoclient
    if os.environ.get("CONDA_PREFIX"):
        exports = pixi_env_exports()
        env.update(exports)
        env["CC"] = exports["SCALE_FACTOR_CC_COMPILER"]
        env["CXX"] = exports["SCALE_FACTOR_CXX_COMPILER"]
        env["CMAKE_PREFIX_PATH"] = exports["SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV"]
        lib_dirs = [exports.get("SCALE_FACTOR_CONDA_LIB_DIR"), exports.get("SCALE_FACTOR_CORRECTIONLIB_LIB_DIR")]
        prepend_env_path(env, "LD_LIBRARY_PATH", lib_dirs)
        prepend_env_path(env, "LIBRARY_PATH", lib_dirs)
    return env


def run_command(cmd, cwd=None, dry_run=False):
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run([str(part) for part in cmd], cwd=cwd, env=command_env())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


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
            f"nano.cpp build cache uses {label}={cached}, but the active pixi/conda environment resolves "
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
            f"nano.cpp build cache uses {label}={cached}, but the active pixi/conda environment resolves "
            f"{label}={expected}. Move systematics/scale_factor/nano.cpp/build aside and rerun mode 11."
        )


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
    conda_lib_dir = prefix / "lib"
    lib_dirs = [conda_lib_dir]
    if correction_lib_dir.exists():
        lib_dirs.append(correction_lib_dir)
    link_flags = []
    for path in lib_dirs:
        link_flags.extend([f"-L{path}", f"-Wl,-rpath,{path}", f"-Wl,-rpath-link,{path}"])
    cmake_prefixes = [str(prefix), str(prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr")]
    _PIXI_ENV_EXPORTS = {
        "SCALE_FACTOR_CONDA_PREFIX": str(prefix),
        "SCALE_FACTOR_CONDA_LIB_DIR": str(conda_lib_dir),
        "SCALE_FACTOR_CC_COMPILER": str(cc_compiler),
        "SCALE_FACTOR_CXX_COMPILER": str(cxx_compiler),
        "SCALE_FACTOR_ROOT_DIR": str(root_dir),
        "SCALE_FACTOR_CORRECTIONLIB_DIR": str(correction_dir),
        "SCALE_FACTOR_CORRECTIONLIB_LIB_DIR": str(correction_lib_dir) if correction_lib_dir.exists() else "",
        "SCALE_FACTOR_YAML_CPP_DIR": str(yaml_dir),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH": ";".join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_PREFIX_PATH_ENV": os.pathsep.join(cmake_prefixes),
        "SCALE_FACTOR_CMAKE_LINK_FLAGS": " ".join(link_flags),
        "SCALE_FACTOR_CMAKE_RPATH": ";".join(str(path) for path in lib_dirs),
    }
    return dict(_PIXI_ENV_EXPORTS)


def pixi_cmake_args(build_dir):
    exports = pixi_env_exports()
    validate_cached_compiler(build_dir, "CMAKE_C_COMPILER", exports["SCALE_FACTOR_CC_COMPILER"], "CMAKE_C_COMPILER")
    validate_cached_compiler(build_dir, "CMAKE_CXX_COMPILER", exports["SCALE_FACTOR_CXX_COMPILER"], "CMAKE_CXX_COMPILER")
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
        any_required_files=["yaml-cpp-config.cmake", "yaml-cppConfig.cmake"],
    )
    return [
        f"-DCMAKE_CXX_COMPILER={exports['SCALE_FACTOR_CXX_COMPILER']}",
        f"-DCMAKE_PREFIX_PATH={exports['SCALE_FACTOR_CMAKE_PREFIX_PATH']}",
        f"-DROOT_DIR={exports['SCALE_FACTOR_ROOT_DIR']}",
        f"-Dcorrectionlib_DIR={exports['SCALE_FACTOR_CORRECTIONLIB_DIR']}",
        f"-Dyaml-cpp_DIR={exports['SCALE_FACTOR_YAML_CPP_DIR']}",
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

def make_ntuple(cfg, args):
    require_conda_prefix()
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
    build_dir = nano_repo / "build"
    if ntuple.get("build_before_make_condor", True):
        commands.append([
            "cmake", "-S", str(nano_repo), "-B", str(build_dir),
            *pixi_cmake_args(build_dir),
        ])
        commands.append(["cmake", "--build", str(build_dir), "-j", build_jobs(ntuple.get("build_jobs", "auto"))])

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
