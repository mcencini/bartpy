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
# BLAS + FFT via MKL — required, no system-library fallback.
#
# MKL is always available when torch is installed via PyPI:
#   - `pip install torch` (CPU wheel) pulls in `mkl` (libmkl_rt.so) and
#     `intel-openmp` as runtime dependencies.
#   - `pip install mkl-include` provides the compile-time headers
#     (mkl_cblas.h, fftw/fftw3.h, …); add this to your build environment.
#
# All three packages install into the active Python environment's prefix so
# we locate them by inspecting sys.prefix — no pip metadata required.
#
# libmkl_rt.so is a single shared library that implements:
#   * CBLAS  (cblas_cgemm etc.)  via `USE_MKL` in BART's blas.c
#   * FFTW3  (fftwf_execute etc.) via MKL's FFTW3 compatibility layer
# Wheel size stays small: the .so lives in the user's Python environment
# and is NOT bundled into the bartorch wheel.
# ---------------------------------------------------------------------------

# Locate the Python environment prefix.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE _py_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find MKL CBLAS headers (mkl_cblas.h) — from `pip install mkl-include`.
# Both the standalone package and conda install them at <prefix>/include.
set(_mkl_include_dir "")
foreach(_d "${_py_prefix}/include" "${_py_prefix}/include/mkl")
    if(EXISTS "${_d}/mkl_cblas.h")
        set(_mkl_include_dir "${_d}")
        break()
    endif()
endforeach()

# Find libmkl_rt — from `pip install mkl` (a torch dependency).
set(_mkl_rt_lib "")
foreach(_lib "libmkl_rt.so.2" "libmkl_rt.so" "libmkl_rt.dylib")
    if(EXISTS "${_py_prefix}/lib/${_lib}")
        set(_mkl_rt_lib "${_py_prefix}/lib/${_lib}")
        break()
    endif()
endforeach()

if(NOT _mkl_include_dir OR NOT _mkl_rt_lib)
    message(FATAL_ERROR
        "MKL not found in Python environment '${_py_prefix}'.\n"
        "Install the build dependencies:\n"
        "  pip install mkl mkl-include intel-openmp\n"
        "(These are installed automatically alongside `pip install torch` on "
        "standard PyPI CPU wheels.)"
    )
endif()

set(CBLAS_INCLUDE_DIR  "${_mkl_include_dir}")
set(CBLAS_LIBRARY      "${_mkl_rt_lib}")
# MKL ships a drop-in FFTW3 compatibility layer; headers live in fftw/
set(FFTW3F_INCLUDE_DIR "${_mkl_include_dir}/fftw")
set(FFTW3F_LIBRARY     "${_mkl_rt_lib}")
# Tell BART's blas.c to #include <mkl_cblas.h> instead of <cblas.h>
list(APPEND BART_DEFS USE_MKL)
message(STATUS "BLAS+FFT: MKL in Python environment '${_py_prefix}'")
message(STATUS "  headers : ${_mkl_include_dir}")
message(STATUS "  runtime : ${_mkl_rt_lib}")

# ---------------------------------------------------------------------------
# FINUFFT — optimised multithreaded ES-kernel spreader/interpolator
#
# When BARTORCH_USE_FINUFFT is ON (the default), FINUFFT is built as a static
# library from the finufft/ submodule and the final _bartorch_ext.so is linked
# with  -Wl,--wrap,grid2  -Wl,--wrap,grid2H .  The linker then redirects every
# call to grid2()/grid2H() (including those buried in noncart/nufft.c) to the
# __wrap_grid2/__wrap_grid2H implementations in finufft_grid.cpp.
#
# MKL's FFTW3 compatibility layer (libmkl_rt.so) is used so that FINUFFT does
# not need to download or compile its own FFTW.  MKL's fftw3.h lives in the
# fftw/ subdirectory of the MKL include path (already found above).
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

    # Tell FINUFFT to use MKL's FFTW3 compatibility library instead of downloading
    # or probing the system for FFTW.  setupFFTW.cmake skips CPM/find_package(FFTW)
    # for any value of FINUFFT_FFTW_LIBRARIES other than "DEFAULT" or "DOWNLOAD".
    set(FINUFFT_FFTW_LIBRARIES "${_mkl_rt_lib}" CACHE STRING
        "FFTW library used by FINUFFT (MKL FFTW3 compatibility layer)" FORCE)
    set(FINUFFT_FFTW_SUFFIX "" CACHE STRING "" FORCE)   # no _threads/_omp suffix

    # ── Build FINUFFT as a subproject ────────────────────────────────────────
    add_subdirectory("${_finufft_dir}" finufft_build EXCLUDE_FROM_ALL)

    # finufft_fftlibs is an INTERFACE library created by setupFFTW.cmake.
    # Inject MKL's fftw/ include directory so that all FINUFFT compilation
    # units that #include <fftw3.h> find MKL's compatibility header.
    # This must come AFTER add_subdirectory so the target already exists.
    target_include_directories(finufft_fftlibs INTERFACE "${FFTW3F_INCLUDE_DIR}")

    message(STATUS "FINUFFT: enabled — MKL FFTW backend: ${_mkl_rt_lib}")
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
    # num — additional modules needed by noncart/grid.c and noncart/nufft.c
    ${BART_SRC_DIR}/num/specfun.c
    ${BART_SRC_DIR}/num/multiplace.c
    ${BART_SRC_DIR}/num/vptr_fun.c
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
    # noncart — KB gridding helpers (grid2/grid2H replaced by finufft_grid.cpp via --wrap)
    ${BART_SRC_DIR}/noncart/grid.c
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
# torch_prior.cpp — --wrap nlop_tf_create for Python denoiser priors.
#
# torch_prior.cpp is always compiled (already listed in pybind11_add_module).
# We always add --wrap nlop_tf_create so __wrap_nlop_tf_create intercepts
# every call to nlop_tf_create() inside the binary.  Paths that do NOT start
# with "bartorch://" fall through to __real_nlop_tf_create (the original BART
# TF implementation), which will error gracefully if TF support is absent.
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
