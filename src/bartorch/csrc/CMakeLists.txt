cmake_minimum_required(VERSION 3.18)
project(bartorch_ext LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA "Build with CUDA support" OFF)
option(BARTORCH_USE_FINUFFT "Replace BART's Kaiser-Bessel gridder with FINUFFT's optimised ES-kernel spreader" ON)
# ---------------------------------------------------------------------------
# Find PyTorch (libtorch)
# ---------------------------------------------------------------------------
# Auto-detect PyTorch's cmake prefix so that find_package(Torch) works without
# requiring an explicit -DCMAKE_PREFIX_PATH from the caller (e.g. setup.py or
# scikit-build-core).  This keeps the build self-contained: as long as torch is
# importable in the active Python interpreter, the build succeeds.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND AND NOT DEFINED TORCH_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_cmake_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _ret
    )
    if(_ret EQUAL 0 AND _torch_cmake_prefix)
        list(PREPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
        message(STATUS "Auto-detected PyTorch cmake prefix: ${_torch_cmake_prefix}")
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------------------------------------------------------------------------
# Find Python (for pybind11 / extension suffix)
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: use pybind11 bundled with PyTorch.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 REQUIRED CONFIG HINTS "${pybind11_DIR}")
endif()

# ---------------------------------------------------------------------------
# BART source tree (git submodule at <repo_root>/bart)
# ---------------------------------------------------------------------------
# Package layout:  src/bartorch/csrc/CMakeLists.txt
# CMAKE_SOURCE_DIR = <repo>/src/bartorch/csrc
# Three levels up reaches the repo root where bart/ lives.
set(BART_SRC_DIR "${CMAKE_SOURCE_DIR}/../../../bart/src")

# ---------------------------------------------------------------------------
# BART compile definitions — must be set BEFORE the MKL detection block
# so that list(APPEND BART_DEFS ...) in the detection block takes effect.
# ---------------------------------------------------------------------------
set(BART_DEFS
    BART_WITH_PYTHON         # suppress abort() → RuntimeError
    NO_MCU                   # skip microcontroller code
    NO_PNG                   # disable libpng dependency in misc/png.c
    # NOTE: MEMONLY_CFL is intentionally NOT set here.
    # At this BART commit, defining MEMONLY_CFL hides the declarations of
    # shared_cfl / private_cfl in mmio.h while mmio.c still references them
    # in FILE_TYPE_CFL branches, causing a compile error.
    # In-memory tensor registration uses names with the ".mem" suffix (e.g.
    # "input.mem"), which BART routes through the memory CFL subsystem at
    # runtime regardless of the MEMONLY_CFL compile flag.
)
if(USE_CUDA)
    list(APPEND BART_DEFS USE_CUDA)
    enable_language(CUDA)

    # ---------------------------------------------------------------------------
    # CUDA toolkit search — works with system CUDA, CUDA-enabled torch wheels,
    # and pip nvidia-* packages (nvidia-cuda-runtime-cu12, etc.).
    #
    # Strategy: derive candidate CUDA library roots from the location of torch,
    # exactly as we do for MKL (see below).  pip CUDA wheels install their .so
    # files under  site-packages/nvidia/<pkg>/lib/ ; we add those dirs to
    # CMAKE_LIBRARY_PATH so find_package(CUDAToolkit) picks them up without
    # needing a system-level CUDA installation.
    # ---------------------------------------------------------------------------
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "
import os, site, sys
dirs = []
try:
    import torch
    sp = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    nvidia = os.path.join(sp, 'nvidia')
    if os.path.isdir(nvidia):
        for pkg in sorted(os.listdir(nvidia)):
            for sub in ('lib', 'bin', 'include'):
                d = os.path.join(nvidia, pkg, sub)
                if os.path.isdir(d):
                    dirs.append(d)
except ImportError:
    pass
print(';'.join(dirs))
        "
        OUTPUT_VARIABLE _nvidia_pkg_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_nvidia_pkg_dirs)
        # Make find_package(CUDAToolkit) find libs from pip nvidia-* packages.
        list(PREPEND CMAKE_LIBRARY_PATH ${_nvidia_pkg_dirs})
        list(PREPEND CMAKE_INCLUDE_PATH ${_nvidia_pkg_dirs})
        message(STATUS "CUDA: added pip nvidia-* dirs to library search path")
    endif()

    find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------
# BLAS + FFT via MKL — required, no fallback.
#
# MKL is a declared bartorch dependency (pyproject.toml) and provides both
# CBLAS and an FFTW3 compatibility layer (libmkl_rt) used by BART and FINUFFT.
#
# Detection strategy (environment-agnostic):
#
#  mkl and mkl-include install into the *Python prefix*, not inside
#  site-packages itself.  The prefix varies across installers:
#    pip --user  → ~/.local
#    venv        → <venv root>
#    conda       → $CONDA_PREFIX
#    system      → /usr  or  /usr/local
#  sys.prefix alone does not reliably find these; instead we derive candidate
#  search roots from the location of torch (always co-installed with MKL):
#
#    torch.__file__  → .../site-packages/torch/__init__.py
#                                         ↑ site-packages (sp)
#                               3 levels up = Python prefix
#                                             (e.g. ~/.local  /usr  $CONDA_PREFIX)
#
#  We also include sys.prefix and all site.getsitepackages() parents to cover
#  every common environment type.
#
#  Header sub-dirs searched per root: include/  include/mkl/  Library/include/
#  Library sub-dirs searched per root: lib/  lib64/  Library/lib/  Library/bin/
#  Library file names: libmkl_rt.so.2  libmkl_rt.so  libmkl_rt.dylib
#                      mkl_rt.2.dll  mkl_rt.dll  (Windows / conda)
# ---------------------------------------------------------------------------

# ── Build candidate search roots via Python ──────────────────────────────────
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
    "
import os, site, sys
roots = set()
try:
    import torch
    sp = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    roots.add(os.path.normpath(os.path.join(sp, '..', '..', '..')))
except ImportError:
    pass
try:
    for sp in site.getsitepackages():
        roots.add(os.path.normpath(os.path.join(sp, '..', '..', '..')))
except AttributeError:
    pass
try:
    up = site.getusersitepackages()
    roots.add(os.path.normpath(os.path.join(up, '..', '..', '..')))
except AttributeError:
    pass
roots.add(sys.prefix)
print(';'.join(r for r in sorted(roots) if r))
    "
    OUTPUT_VARIABLE _py_roots
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ── Search for MKL headers ───────────────────────────────────────────────────
set(_mkl_include_dir "")
foreach(_root ${_py_roots})
    foreach(_rel "include" "include/mkl" "Library/include")
        if(EXISTS "${_root}/${_rel}/mkl_cblas.h")
            set(_mkl_include_dir "${_root}/${_rel}")
            break()
        endif()
    endforeach()
    if(_mkl_include_dir)
        break()
    endif()
endforeach()

# ── Search for MKL runtime library ──────────────────────────────────────────
set(_mkl_rt_lib "")
foreach(_root ${_py_roots})
    foreach(_rel "lib" "lib64" "Library/lib" "Library/bin")
        foreach(_name
            "libmkl_rt.so.2" "libmkl_rt.so" "libmkl_rt.dylib"
            "mkl_rt.2.dll"   "mkl_rt.dll")
            if(EXISTS "${_root}/${_rel}/${_name}")
                set(_mkl_rt_lib "${_root}/${_rel}/${_name}")
                break()
            endif()
        endforeach()
        if(_mkl_rt_lib)
            break()
        endif()
    endforeach()
    if(_mkl_rt_lib)
        break()
    endif()
endforeach()

# ── MKL found — use it ───────────────────────────────────────────────────────
if(_mkl_include_dir AND _mkl_rt_lib)
    set(CBLAS_INCLUDE_DIR  "${_mkl_include_dir}")
    set(CBLAS_LIBRARY      "${_mkl_rt_lib}")
    # MKL ships a drop-in FFTW3 compatibility layer; headers live in fftw/
    set(FFTW3F_INCLUDE_DIR "${_mkl_include_dir}/fftw")
    set(FFTW3F_LIBRARY     "${_mkl_rt_lib}")
    list(APPEND BART_DEFS USE_MKL)
    message(STATUS "BLAS+FFT: MKL")
    message(STATUS "  headers : ${_mkl_include_dir}")
    message(STATUS "  runtime : ${_mkl_rt_lib}")
else()
    # MKL is required — it provides both CBLAS and the FFTW3 compatibility
    # layer used by BART's FFT routines and FINUFFT.  There is no system
    # BLAS or FFTW3 fallback: MKL is the only supported backend.
    #
    # MKL is co-installed with torch in any standard Python environment.
    # If this error fires, ensure mkl-include is installed alongside torch:
    #   pip install mkl mkl-include
    message(FATAL_ERROR
        "MKL not found in any Python root (searched: ${_py_roots}).\n"
        "MKL must be installed in the same Python environment as torch:\n"
        "  pip install mkl mkl-include\n"
        "(On a standard `pip install torch` CPU wheel these are pulled in "
        "automatically as dependencies.)\n"
    )
endif()

# ---------------------------------------------------------------------------
# FINUFFT — optimised multithreaded ES-kernel spreader/interpolator
#
# When BARTORCH_USE_FINUFFT is ON (the default), FINUFFT is built as a static
# library from the finufft/ submodule and the final _bartorch_ext.so is linked
# with  -Wl,--wrap,grid2  -Wl,--wrap,grid2H .
#
# FINUFFT needs an FFTW3 backend.  We tell it to use whatever FFTW3 library
# was found above (FFTW3F_LIBRARY) — either libmkl_rt.so (MKL path) or the
# system libfftw3f (fallback path) — by setting FINUFFT_FFTW_LIBRARIES.
# ---------------------------------------------------------------------------
if(BARTORCH_USE_FINUFFT)
    set(_finufft_dir "${CMAKE_SOURCE_DIR}/../../finufft")
    if(NOT IS_DIRECTORY "${_finufft_dir}")
        message(WARNING
            "BARTORCH_USE_FINUFFT=ON but finufft/ submodule not found at "
            "${_finufft_dir}.\n"
            "Run:  git submodule update --init --recursive\n"
            "Disabling BARTORCH_USE_FINUFFT.")
        set(BARTORCH_USE_FINUFFT OFF)
    endif()
endif()

if(BARTORCH_USE_FINUFFT)
    # ── FINUFFT build options ────────────────────────────────────────────────
    # Disable everything we don't need to keep the build fast and slim.
    set(FINUFFT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_FORTRAN  OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_MATLAB   OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_PYTHON   OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_DEVEL    OFF CACHE BOOL "" FORCE)
    set(FINUFFT_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(FINUFFT_USE_CPU    ON  CACHE BOOL "" FORCE)
    set(FINUFFT_USE_CUDA   OFF CACHE BOOL "" FORCE)   # cuFINUFFT handled separately
    set(FINUFFT_STATIC_LINKING ON  CACHE BOOL "" FORCE)
    set(FINUFFT_USE_DUCC0  OFF CACHE BOOL "" FORCE)   # use FFTW backend, not DUCC0
    set(FINUFFT_USE_OPENMP ON  CACHE BOOL "" FORCE)   # multi-threaded spreading

    # Point FINUFFT at the FFTW3 library found above (MKL or system fftw3f).
    # Any non-DEFAULT / non-DOWNLOAD value makes setupFFTW.cmake skip its own
    # CPM/find_package search and use our library directly.
    set(FINUFFT_FFTW_LIBRARIES "${FFTW3F_LIBRARY}" CACHE STRING
        "FFTW library used by FINUFFT" FORCE)
    set(FINUFFT_FFTW_SUFFIX "" CACHE STRING "" FORCE)   # no _threads/_omp suffix

    # ── Build FINUFFT as a subproject ────────────────────────────────────────
    add_subdirectory("${_finufft_dir}" finufft_build EXCLUDE_FROM_ALL)

    # finufft_fftlibs is an INTERFACE library created by setupFFTW.cmake.
    # Inject the FFTW3 include dir so all FINUFFT compilation units find fftw3.h.
    # This must come AFTER add_subdirectory so the target already exists.
    target_include_directories(finufft_fftlibs INTERFACE "${FFTW3F_INCLUDE_DIR}")

    message(STATUS "FINUFFT: enabled — FFTW backend: ${FFTW3F_LIBRARY}")
endif()

# ---------------------------------------------------------------------------
# Generate version.inc — BART's Makefile normally creates this via
# `git describe`.  We generate it here so it's available when building
# only the subset of BART sources needed for the extension.
# ---------------------------------------------------------------------------
set(BART_VERSION_INC "${BART_SRC_DIR}/misc/version.inc")
if(NOT EXISTS "${BART_VERSION_INC}")
    # Try git describe first; fall back to the pinned tag.
    execute_process(
        COMMAND git -C "${BART_SRC_DIR}/.." describe --tags --always --dirty
        OUTPUT_VARIABLE _bart_git_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _bart_git_ret
        ERROR_QUIET
    )
    if(_bart_git_ret EQUAL 0 AND _bart_git_version)
        set(_bart_version_str "\"${_bart_git_version}\"")
    else()
        set(_bart_version_str "\"v1.0.00-bartorch\"")
    endif()
    file(WRITE "${BART_VERSION_INC}" "${_bart_version_str}\n")
    message(STATUS "Generated ${BART_VERSION_INC}: ${_bart_version_str}")
endif()

# ---------------------------------------------------------------------------
# Generate mainlist.inc — BART's Makefile creates this by expanding all
# tool targets.  We run the minimal make target so bart.c can compile.
# ---------------------------------------------------------------------------
set(BART_MAINLIST_INC "${BART_SRC_DIR}/mainlist.inc")
if(NOT EXISTS "${BART_MAINLIST_INC}")
    find_program(MAKE_EXECUTABLE NAMES make gmake)
    if(MAKE_EXECUTABLE)
        # Run `make src/mainlist.inc` from the BART root.  Suppress the
        # "CONFIGURATION MODIFIED" rebuild noise by ensuring a clean run.
        execute_process(
            COMMAND ${MAKE_EXECUTABLE} src/mainlist.inc
            WORKING_DIRECTORY "${BART_SRC_DIR}/.."
            RESULT_VARIABLE _make_ret
            OUTPUT_QUIET
            ERROR_QUIET
        )
    endif()
    # Verify that make generated all required MAIN_* groups.
    # If any are missing (can happen when make skips module Makefiles in a
    # restricted CI environment), fall back to CMake-level file scanning.
    set(_required_groups MAIN_CALIB MAIN_MRI MAIN_SIM MAIN_NN MAIN_MOTION)
    set(_mainlist_ok TRUE)
    if(EXISTS "${BART_MAINLIST_INC}")
        foreach(_grp ${_required_groups})
            file(READ "${BART_MAINLIST_INC}" _mainlist_content)
            string(FIND "${_mainlist_content}" "#define ${_grp}" _pos)
            if(_pos EQUAL -1)
                set(_mainlist_ok FALSE)
                break()
            endif()
        endforeach()
    else()
        set(_mainlist_ok FALSE)
    endif()

    if(NOT _mainlist_ok)
        # Fallback: scan all top-level .c files in bart/src — each one is a
        # tool with a main_xxx entry point.  Build one flat MAIN_LIST that
        # bart.c can compile.  The category sub-lists are satisfied with
        # empty definitions so the dispatch_table still compiles.
        message(STATUS "Generating ${BART_MAINLIST_INC} via CMake file scanning "
                       "(make src/mainlist.inc was incomplete)")
        file(GLOB _bart_tool_srcs "${BART_SRC_DIR}/*.c")
        set(_tool_names "")
        foreach(_src ${_bart_tool_srcs})
            get_filename_component(_name "${_src}" NAME_WE)
            # Exclude non-tool files: bart.c, main.c, bbox.c
            if(NOT _name STREQUAL "bart" AND
               NOT _name STREQUAL "main" AND
               NOT _name STREQUAL "bbox")
                list(APPEND _tool_names "${_name}")
            endif()
        endforeach()
        list(SORT _tool_names)
        string(REPLACE ";" ", " _tool_list "${_tool_names}")
        set(_mainlist_lines
            "#define MAIN_LIST ${_tool_list}, ()"
            "#define MAIN_BASE ()"
            "#define MAIN_FLP ()"
            "#define MAIN_NUM ()"
            "#define MAIN_IO ()"
            "#define MAIN_RECO ()"
            "#define MAIN_CALIB ()"
            "#define MAIN_MRI ()"
            "#define MAIN_SIM ()"
            "#define MAIN_NN ()"
            "#define MAIN_MOTION ()"
        )
        string(JOIN "\n" _mainlist_content ${_mainlist_lines})
        file(WRITE "${BART_MAINLIST_INC}" "${_mainlist_content}\n")
        message(STATUS "Generated ${BART_MAINLIST_INC}: ${_tool_list}")
    else()
        message(STATUS "Generated ${BART_MAINLIST_INC} via BART Makefile")
    endif()
endif()

# ---------------------------------------------------------------------------
# BART static library target
#
# Source selection mirrors BART's own Makefile, which uses:
#   $(wildcard $(srcdir)/$(module)/*.c)
# i.e. ALL .c files in each required module directory.
#
# Doing the same here with file(GLOB) avoids the class of bugs caused by
# hand-picking files and missing transitive dependencies (e.g. num/optimize.c
# defines `num_chunk_size`, a data symbol referenced by num/init.c; missing it
# causes an undefined-symbol ImportError at Python load time).
#
# Optional-feature files are made safe by compile definitions:
#   NO_PNG     — misc/png.c: skips libpng I/O behind #ifndef NO_PNG
#   USE_MKL    — num/lapack.c: selects <mkl.h> LAPACK header (set by MKL block above)
#   (no USE_CUDA)  — num/gpuops.c, fft-cuda.c, etc. are entirely inside
#                     #ifdef USE_CUDA so they compile cleanly as no-ops.
#   (no USE_MPI)   — num/mpi_ops.c guards all symbols with #ifdef USE_MPI.
# ---------------------------------------------------------------------------
file(GLOB _bart_misc_srcs   "${BART_SRC_DIR}/misc/*.c")
file(GLOB _bart_num_srcs    "${BART_SRC_DIR}/num/*.c")
file(GLOB _bart_simu_srcs   "${BART_SRC_DIR}/simu/*.c")
file(GLOB _bart_linops_srcs "${BART_SRC_DIR}/linops/*.c")
file(GLOB _bart_wavelet_srcs "${BART_SRC_DIR}/wavelet/*.c")
file(GLOB _bart_iter_srcs   "${BART_SRC_DIR}/iter/*.c")
file(GLOB _bart_nlops_srcs  "${BART_SRC_DIR}/nlops/*.c")
file(GLOB _bart_noncart_srcs "${BART_SRC_DIR}/noncart/*.c")

set(BART_SOURCES
    ${_bart_misc_srcs}
    ${_bart_num_srcs}
    ${_bart_simu_srcs}
    ${_bart_linops_srcs}
    ${_bart_wavelet_srcs}
    ${_bart_iter_srcs}
    ${_bart_nlops_srcs}
    ${_bart_noncart_srcs}
    # embed API + main dispatcher (top-level, not in a module directory)
    ${BART_SRC_DIR}/main.c
    ${BART_SRC_DIR}/bart.c
)

add_library(bart_static STATIC ${BART_SOURCES})
target_compile_definitions(bart_static PUBLIC ${BART_DEFS})
target_include_directories(bart_static PUBLIC
    ${BART_SRC_DIR}
    ${Python3_INCLUDE_DIRS}
    ${FFTW3F_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIR}
)
target_compile_options(bart_static PRIVATE -fPIC -O2 -std=gnu11)

# Link order: PyTorch (tensor ops) → MKL or system CBLAS+FFTW3.
# When MKL path is active, CBLAS_LIBRARY and FFTW3F_LIBRARY both point to
# libmkl_rt.so — a single lib that covers both BLAS and FFT.
target_link_libraries(bart_static PUBLIC
    ${TORCH_LIBRARIES}
    ${CBLAS_LIBRARY}
    ${FFTW3F_LIBRARY}
)

if(USE_CUDA)
    target_link_libraries(bart_static PUBLIC CUDA::cufft CUDA::cudart CUDA::cublas)
endif()

# ---------------------------------------------------------------------------
# _bartorch_ext Python extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_bartorch_ext
    bartorch_ext.cpp
    bart_ops.cpp
    torch_prior.cpp
    torch_prior_nlop.c
)
target_include_directories(_bartorch_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BART_SRC_DIR}
)
target_link_libraries(_bartorch_ext PRIVATE bart_static ${TORCH_LIBRARIES})
target_compile_definitions(_bartorch_ext PRIVATE ${BART_DEFS})

if(BARTORCH_USE_FINUFFT)
    # finufft_grid.cpp provides __wrap_grid2 / __wrap_grid2H.
    # The FINUFFT include directory is propagated transitively through the
    # finufft target, so no extra target_include_directories is needed here.
    target_sources(_bartorch_ext PRIVATE finufft_grid.cpp)
    target_compile_features(_bartorch_ext PRIVATE cxx_std_17)
    target_link_libraries(_bartorch_ext PRIVATE finufft)
    target_compile_definitions(_bartorch_ext PRIVATE BARTORCH_USE_FINUFFT)

    # Redirect grid2/grid2H and rolloff_correction* to finufft_grid.cpp implementations.
    if(APPLE)
        target_link_options(_bartorch_ext PRIVATE
            -Wl,-wrap,_grid2
            -Wl,-wrap,_grid2H
            -Wl,-wrap,_rolloff_correction
            -Wl,-wrap,_apply_rolloff_correction
            -Wl,-wrap,_apply_rolloff_correction2
        )
    else()
        target_link_options(_bartorch_ext PRIVATE
            -Wl,--wrap,grid2
            -Wl,--wrap,grid2H
            -Wl,--wrap,rolloff_correction
            -Wl,--wrap,apply_rolloff_correction
            -Wl,--wrap,apply_rolloff_correction2
        )
    endif()
    message(STATUS "FINUFFT grid replacement: --wrap,grid2 --wrap,grid2H --wrap,rolloff_correction --wrap,apply_rolloff_correction --wrap,apply_rolloff_correction2 active")
endif()

# ---------------------------------------------------------------------------
# torch_prior.cpp + torch_prior_nlop.c — --wrap nlop_tf_create
#
# torch_prior_nlop.c (pure C99) contains all BART nlop types and `complex float`
# callbacks.  torch_prior.cpp (C++) contains the registry, pybind11 bindings,
# and PyTorch tensor calls.  The split ensures no C99-only types (`complex float`,
# `_Complex`) appear in C++ translation units, making the code portable to
# GCC, Clang, and MSVC.
#
# Both files are already listed in pybind11_add_module above.
# We always add --wrap nlop_tf_create so __wrap_nlop_tf_create (in
# torch_prior_nlop.c) intercepts every call to nlop_tf_create() inside the
# binary.  Paths that do NOT start with "bartorch://" fall through to
# __real_nlop_tf_create (the original BART TF implementation), which will
# error gracefully if TF support is absent.
# ---------------------------------------------------------------------------
if(APPLE)
    target_link_options(_bartorch_ext PRIVATE
        -Wl,-wrap,_nlop_tf_create
    )
else()
    target_link_options(_bartorch_ext PRIVATE
        -Wl,--wrap,nlop_tf_create
    )
endif()
message(STATUS "Torch prior: --wrap,nlop_tf_create active")

if(USE_CUDA)
    target_sources(_bartorch_ext PRIVATE cuda/cuda_bridge.hpp)
    target_compile_definitions(_bartorch_ext PRIVATE USE_CUDA)
endif()

# Ensure the .so lands directly in the bartorch package directory so that
# `from bartorch import _bartorch_ext` works for both editable and installed builds.
# CMAKE_SOURCE_DIR = src/bartorch/csrc  →  ../ = src/bartorch/  (package root).
set_target_properties(_bartorch_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
