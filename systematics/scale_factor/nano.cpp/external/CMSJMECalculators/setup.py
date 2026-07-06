import os, os.path
from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup

cmake_args = [
        f"-DCMSJMECALCULATOR_VERSION:STRING={get_version()}",
        ]


correction_lib = os.popen("correction config --cmake").read().strip()
cmake_args.append(f"{correction_lib}")

if os.getenv("CMSSW_BASE") and os.getenv("ROOTSYS"):
    rootdir= os.path.join(os.getenv("ROOTSYS"), "cmake")
    if os.path.isdir(rootdir):
        cmake_args.append(f"-DROOT_DIR={rootdir}")

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_install_dir="python",
    cmake_args=cmake_args
)
