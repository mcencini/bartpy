cmake_minimum_required(VERSION 3.18)
project(bartorch_ext LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA "Build with CUDA support" OFF)
option(BARTORCH_SKIP_EXT "Skip building the C++ extension (pure-Python fallback)" OFF)

if(DEFINED ENV{BARTORCH_SKIP_EXT} AND "$ENV{BARTORCH_SKIP_EXT}" STREQUAL "1")
    set(BARTORCH_SKIP_EXT ON)
endif()

if(BARTORCH_SKIP_EXT)
    message(STATUS "BARTORCH_SKIP_EXT=1 — skipping C++ extension build")
    return()
endif()

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
# We build only the static libraries we need, not the bart binary.
set(BART_SRC_DIR "${CMAKE_SOURCE_DIR}/../../bart/src")

# ---------------------------------------------------------------------------
# FFTW3 (single-precision) — dynamic link to keep the wheel small.
# BART's fft_plan.c uses fftwf_* (float-precision FFTW3).
# The .so is NOT bundled into the wheel; it is a runtime dependency.
# TODO (Phase 1): replace with MKL FFT plan when USE_MKL is wired up.
# ---------------------------------------------------------------------------
find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED
    DOC "FFTW3 single-precision library")
find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h REQUIRED
    DOC "FFTW3 include directory")
message(STATUS "Found FFTW3f: ${FFTW3F_LIBRARY}  (headers: ${FFTW3F_INCLUDE_DIR})")

# ---------------------------------------------------------------------------
# CBLAS — dynamic link; headers from system (OpenBLAS or generic).
# BART's blas.c calls cblas_cgemm / cblas_cgemv / cblas_caxpy etc.
# PyTorch's libtorch_cpu.so only exports MKL-style batch variants, so we
# use the system BLAS shared library as the runtime provider.
# TODO (Phase 1): when linking a full MKL-enabled PyTorch wheel, set
#   USE_MKL and route BLAS through ${TORCH_LIBRARIES} instead.
# ---------------------------------------------------------------------------
find_library(CBLAS_LIBRARY NAMES openblas blas REQUIRED
    DOC "CBLAS/BLAS library")
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
    PATH_SUFFIXES openblas
    DOC "CBLAS include directory")
message(STATUS "Found CBLAS: ${CBLAS_LIBRARY}  (headers: ${CBLAS_INCLUDE_DIR})")

# BART compile definitions
set(BART_DEFS
    BART_WITH_PYTHON         # suppress abort() → RuntimeError
    NO_MCU                   # skip microcontroller code
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
    find_package(CUDAToolkit REQUIRED)
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
# The minimal set of BART source files needed for the initial op set.
# This list will grow as more ops are wrapped.
# ---------------------------------------------------------------------------
# Note: a full build would use BART's own Makefile; for the extension we
# compile only the files we need to keep the wheel small and build fast.
# The authoritative list is maintained in agents.md §§ Build System.
set(BART_SOURCES
    # misc
    ${BART_SRC_DIR}/misc/misc.c
    ${BART_SRC_DIR}/misc/debug.c
    ${BART_SRC_DIR}/misc/io.c
    ${BART_SRC_DIR}/misc/mmio.c
    ${BART_SRC_DIR}/misc/memcfl.c
    ${BART_SRC_DIR}/misc/stream.c
    ${BART_SRC_DIR}/misc/stream_protocol.c
    ${BART_SRC_DIR}/misc/lock.c
    ${BART_SRC_DIR}/misc/shrdptr.c
    ${BART_SRC_DIR}/misc/list.c
    ${BART_SRC_DIR}/misc/opts.c
    ${BART_SRC_DIR}/misc/utils.c
    ${BART_SRC_DIR}/misc/version.c
    # num
    ${BART_SRC_DIR}/num/multind.c
    ${BART_SRC_DIR}/num/flpmath.c
    ${BART_SRC_DIR}/num/fft.c
    ${BART_SRC_DIR}/num/fft_plan.c
    ${BART_SRC_DIR}/num/blas.c
    ${BART_SRC_DIR}/num/vecops.c
    ${BART_SRC_DIR}/num/rand.c
    ${BART_SRC_DIR}/num/init.c
    ${BART_SRC_DIR}/num/vptr.c
    ${BART_SRC_DIR}/num/delayed.c
    ${BART_SRC_DIR}/num/mpi_ops.c
    # simu
    ${BART_SRC_DIR}/simu/phantom.c
    ${BART_SRC_DIR}/simu/shape.c
    # linops
    ${BART_SRC_DIR}/linops/linop.c
    ${BART_SRC_DIR}/linops/someops.c
    ${BART_SRC_DIR}/linops/fmac.c
    # iter
    ${BART_SRC_DIR}/iter/iter.c
    ${BART_SRC_DIR}/iter/iter2.c
    ${BART_SRC_DIR}/iter/prox.c
    ${BART_SRC_DIR}/iter/thresh.c
    # embed API + main dispatcher
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

# Link order: PyTorch (tensor ops + its own bundled BLAS symbols) → system CBLAS
# → FFTW3f.  Both system libs are shared (.so) — not bundled in the wheel.
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
)
target_include_directories(_bartorch_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BART_SRC_DIR}
)
target_link_libraries(_bartorch_ext PRIVATE bart_static ${TORCH_LIBRARIES})
target_compile_definitions(_bartorch_ext PRIVATE ${BART_DEFS})

if(USE_CUDA)
    target_sources(_bartorch_ext PRIVATE cuda/cuda_bridge.hpp)
    target_compile_definitions(_bartorch_ext PRIVATE USE_CUDA)
endif()

# Ensure the .so lands directly in the bartorch package directory so that
# `from bartorch import _bartorch_ext` works for both editable and installed builds.
# CMAKE_SOURCE_DIR = bartorch/csrc  →  ../ = bartorch/  (the Python package root).
set_target_properties(_bartorch_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
