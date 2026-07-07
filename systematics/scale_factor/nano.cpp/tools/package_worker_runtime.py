#!/usr/bin/env python3
"""Build a relocatable runtime bundle for nano.cpp Condor workers.

The bundle intentionally contains only the private runtime needed by nano_run:
nano_run, local build products, their shared-library closure from the active
pixi/conda prefix, and ROOT runtime metadata. Site tools such as hadd, xrdcp,
and xrdfs are resolved on the worker from the node or CVMFS environment instead
of being copied from the submitter's pixi environment.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tarfile
import time
from collections import deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, help="Active pixi/conda prefix for private-runtime builds")
    parser.add_argument("--build-dir", required=True, type=Path, help="nano.cpp build directory")
    parser.add_argument("--output", required=True, type=Path, help="Output worker_runtime.tar.gz")
    return parser.parse_args()


def is_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def arcname(path: Path, prefix: Path | None, build_dir: Path) -> Path:
    resolved = path.resolve()
    if prefix is not None and is_under(resolved, prefix):
        return Path("env") / resolved.relative_to(prefix.resolve())
    if is_under(resolved, build_dir):
        return Path("build") / resolved.relative_to(build_dir.resolve())
    raise RuntimeError(f"Cannot place path outside prefix/build in runtime bundle: {path}")


def add_file(files: dict[Path, Path], path: Path, prefix: Path | None, build_dir: Path) -> None:
    if not path.exists():
        return
    if not (path.is_file() or path.is_symlink()):
        return
    resolved = path.resolve()
    if not ((prefix is not None and is_under(resolved, prefix)) or is_under(resolved, build_dir)):
        raise RuntimeError(f"Runtime bundle path resolves outside prefix/build: {path} -> {resolved}")
    files[arcname(path, prefix, build_dir)] = path


def add_tree(files: dict[Path, Path], root: Path, prefix: Path | None, build_dir: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        add_file(files, path, prefix, build_dir)


def ldd_paths(path: Path) -> list[Path]:
    result = subprocess.run(["ldd", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ldd failed for {path}:\n{result.stderr.strip()}")

    out: list[Path] = []
    for line in result.stdout.splitlines():
        match = re.search(r"=>\s+(/[^ ]+)", line)
        if not match:
            match = re.match(r"\s*(/[^ ]+)\s+\(", line)
        if match:
            out.append(Path(match.group(1)))
    return out


def add_shared_library_siblings(files: dict[Path, Path], path: Path, prefix: Path | None, build_dir: Path) -> None:
    name = path.name
    if ".so" not in name:
        return
    lib_prefix = name.split(".so", 1)[0] + ".so"
    for sibling in path.parent.glob(lib_prefix + "*"):
        add_file(files, sibling, prefix, build_dir)


def add_dependency_closure(files: dict[Path, Path], seeds: list[Path], prefix: Path | None, build_dir: Path) -> None:
    queue: deque[Path] = deque(seeds)
    seen: set[Path] = set()
    while queue:
        path = queue.popleft().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not ((prefix is not None and is_under(path, prefix)) or is_under(path, build_dir)):
            continue
        add_file(files, path, prefix, build_dir)
        add_shared_library_siblings(files, path, prefix, build_dir)
        for dep in ldd_paths(path):
            dep = dep.resolve()
            if ((prefix is not None and is_under(dep, prefix)) or is_under(dep, build_dir)) and dep not in seen:
                queue.append(dep)


def add_root_runtime_metadata(files: dict[Path, Path], prefix: Path, build_dir: Path) -> None:
    lib_dir = prefix / "lib"
    plugin_seeds: list[Path] = []
    if lib_dir.exists():
        for pattern in ("*.pcm", "*.rootmap", "libNet*.so*", "libROOT*.so*", "libXrd*.so*"):
            for path in lib_dir.glob(pattern):
                add_file(files, path, prefix, build_dir)
                add_shared_library_siblings(files, path, prefix, build_dir)
                if ".so" in path.name:
                    plugin_seeds.append(path)

    add_dependency_closure(files, plugin_seeds, prefix, build_dir)

    for path in (
        prefix / "etc" / "root",
        prefix / "etc" / "system.rootrc",
        prefix / "share" / "root",
        prefix / "lib" / "root",
    ):
        if path.is_dir():
            add_tree(files, path, prefix, build_dir)
        else:
            add_file(files, path, prefix, build_dir)


def add_core_runtime_libraries(files: dict[Path, Path], prefix: Path, build_dir: Path) -> None:
    """Add compiler runtime libraries that ldd can miss through wrapper indirection."""
    lib_roots = [
        prefix / "lib",
        prefix / "lib64",
        prefix / "x86_64-conda-linux-gnu" / "lib",
    ]
    patterns = (
        "libstdc++.so*",
        "libgcc_s.so*",
        "libgomp.so*",
        "libquadmath.so*",
        "libatomic.so*",
    )
    for root in lib_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                add_file(files, path, prefix, build_dir)


def add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    args = parse_args()
    prefix = args.prefix.resolve() if args.prefix else None
    build_dir = args.build_dir.resolve()
    output = args.output

    if prefix is not None and not prefix.exists():
        raise RuntimeError(f"Prefix does not exist: {prefix}")
    if not build_dir.exists():
        raise RuntimeError(f"Build directory does not exist: {build_dir}")

    required_build = [
        build_dir / "nano_run",
        build_dir / "external" / "CMSJMECalculators" / "libCMSJMECalculators.so",
    ]
    optional_build = [
        build_dir / "libnanoaodtools_cpp.so",
    ]
    missing = [str(path) for path in required_build if not path.exists()]
    if missing:
        raise RuntimeError("Missing runtime bundle inputs:\n  " + "\n  ".join(missing))

    files: dict[Path, Path] = {}
    seeds = [*required_build, *[path for path in optional_build if path.exists()]]
    add_dependency_closure(files, seeds, prefix, build_dir)
    if prefix is not None:
        add_core_runtime_libraries(files, prefix, build_dir)
        add_root_runtime_metadata(files, prefix, build_dir)

    lib_dirs = {
        "build",
        "build/external/CMSJMECalculators",
        "env/lib",
    }
    for arc in files:
        if ".so" in arc.name:
            lib_dirs.add(str(arc.parent))

    manifest = [
        "nano.cpp Condor worker runtime bundle",
        f"prefix={prefix if prefix is not None else '<site-runtime>'}",
        f"build_dir={build_dir}",
        f"files={len(files)}",
        "seeds:",
        *[f"  {path}" for path in seeds],
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", dereference=True) as tar:
        for arc, path in sorted(files.items(), key=lambda item: str(item[0])):
            tar.add(path, arcname=str(arc), recursive=False)
        add_text(tar, "runtime_ld_paths.txt", "\n".join(sorted(lib_dirs)) + "\n")
        add_text(tar, "runtime_manifest.txt", "\n".join(manifest) + "\n")

    print(f"Wrote worker runtime bundle: {output}")
    print(f"Runtime files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
