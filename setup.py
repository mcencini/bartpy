# Copyright 2021. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2021 Max Litster <litster@berkeley.edu>
# 2024 bartorch contributors
#
# This file now serves as the CMake-based build entry point for the
# _bartorch_ext C++ extension (PyTorch ↔ BART zero-copy bridge).
#
# Legacy bartpy installation (subprocess-based wrappers) is still available;
# see bartpy/ for the original package.
#
# Usage
# -----
#   CPU-only:
#       pip install -e .
#
#   With CUDA:
#       CMAKE_ARGS="-DUSE_CUDA=ON" pip install -e .
#
#   Skip C++ extension (pure-Python / FIFO fallback only):
#       BARTORCH_SKIP_EXT=1 pip install -e .

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, src_dir: str = ""):
        super().__init__(name, sources=[])
        self.src_dir = Path(src_dir).resolve()


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        if os.environ.get("BARTORCH_SKIP_EXT") == "1":
            return

        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        import torch
        torch_cmake_prefix = torch.utils.cmake_prefix_path

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        extra = os.environ.get("CMAKE_ARGS", "")
        if extra:
            cmake_args += extra.split()

        build_args = ["--config", "Release", "--", "-j4"]

        subprocess.check_call(
            ["cmake", str(ext.src_dir)] + cmake_args,
            cwd=build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )


setup(
    name="bartorch",
    version="0.1.0.dev0",
    author="mrirecon",
    description=(
        "PyTorch-native interface to the Berkeley Advanced Reconstruction "
        "Toolbox (BART)"
    ),
    packages=["bartorch", "bartorch.core", "bartorch.ops",
              "bartorch.pipe", "bartorch.tools", "bartorch.utils",
              # Legacy
              "bartpy", "bartpy.utils", "bartpy.tools", "bartpy.wrapper"],
    ext_modules=[
        CMakeExtension("_bartorch_ext", src_dir="bartorch/csrc"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy>=1.21"],
    zip_safe=False,
)
