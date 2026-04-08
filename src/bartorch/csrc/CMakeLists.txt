cmake_minimum_required(VERSION 3.18)
project(bartorch_ext LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA "Build with CUDA support" OFF)
option(BARTORCH_USE_FINUFFT "Replace BART's Kaiser-Bessel gridder with FINUFFT's optimised ES-kernel spreader" ON)
# On macOS, use Apple Accelerate (vDSP + CBLAS) instead of requiring Homebrew
# fftw and openblas.  Accelerate is part of the macOS SDK and is always available.
# Set to OFF to fall back to the OpenBLAS + real FFTW3f path (still requires
# brew install fftw openblas).
if(APPLE)
    option(BARTORCH_USE_ACCELERATE "Use Apple Accelerate (vDSP) for BLAS+FFT on macOS (eliminates Homebrew fftw/openblas requirement)" ON)
else()
    option(BARTORCH_USE_ACCELERATE "Use Apple Accelerate (vDSP) for BLAS+FFT on macOS (eliminates Homebrew fftw/openblas requirement)" OFF)
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

# Disable LTO before finding pybind11 so pybind11_add_module never adds
# pybind11::lto.  GNU ld's --wrap resolves symbols AFTER LTO inlines them,
# so --wrap,fftwf_*/grid2/rolloff would silently have no effect with LTO on.
# Setting CMAKE_INTERPROCEDURAL_OPTIMIZATION here is the only reliable way:
# pybind11 3.x checks `if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)`
# and skips its -flto injection only when the variable is already defined.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL
    "Disable LTO so GNU --wrap symbols are applied correctly at link time")

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
    # NOTE: NOLAPACKE is intentionally NOT set. With USE_MKL (set below when
    # MKL is detected), num/lapack.c includes <mkl.h> via "#elif USE_MKL".
    # NOLAPACKE would override that branch and pull in bart/src/lapacke/*.h
    # which in turn needs system lapack.h — not available without LAPACK dev headers.
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

    # Auto-detect CUDA architecture when not specified by the user.
    # We first try nvidia-smi; if absent (build machine ≠ run machine) we fall
    # back to "native" (NVCC autodetects at build time from the local GPU) and
    # finally to sm_75 (Turing — safe for T4, RTX 20xx, A100, H100 all support it).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        find_program(_NVCC_EXEC NAMES nvcc)
        find_program(_NSMI_EXEC NAMES nvidia-smi)
        if(_NSMI_EXEC)
            execute_process(
                COMMAND "${_NSMI_EXEC}" --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE _smi_caps
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_smi_caps)
                string(REPLACE "." "" _smi_arch "${_smi_caps}")
                string(STRIP "${_smi_arch}" _smi_arch)
                set(CMAKE_CUDA_ARCHITECTURES "${_smi_arch}" CACHE STRING "CUDA architecture" FORCE)
                message(STATUS "CUDA: detected architecture ${_smi_arch} from nvidia-smi")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architecture" FORCE)
                message(STATUS "CUDA: using 'native' architecture (nvidia-smi gave no output)")
            endif()
        elseif(_NVCC_EXEC)
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architecture" FORCE)
            message(STATUS "CUDA: using 'native' architecture (nvidia-smi not found)")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architecture" FORCE)
            message(STATUS "CUDA: defaulting to sm_75 (no nvidia-smi or nvcc found)")
        endif()
    endif()
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

# Track which FFT backend is active.
#   _use_mkl_shim         ON → MKL DFTI shim (mkl_fftw_shim.c)
#   _use_accelerate_shim  ON → Apple Accelerate vDSP shim (accelerate_fftw_shim.c)
set(_use_mkl_shim        OFF)
set(_use_accelerate_shim OFF)

# ── MKL found — use it for CBLAS + FFTW3 (via mkl_fftw_shim) ────────────────
if(_mkl_include_dir AND _mkl_rt_lib)
    set(CBLAS_INCLUDE_DIR  "${_mkl_include_dir}")
    set(CBLAS_LIBRARY      "${_mkl_rt_lib}")
    # MKL ships a drop-in FFTW3 compatibility layer; headers live in fftw/.
    # The actual fftwf_plan_guru64_dft symbol is provided at runtime by MKL's
    # FFTW3 compat layer, but we intercept it with our mkl_fftw_shim.c (see
    # below) so that BART's multi-dimensional FFTs work correctly.
    set(FFTW3F_INCLUDE_DIR "${_mkl_include_dir}/fftw")
    set(FFTW3F_LIBRARY     "${_mkl_rt_lib}")
    list(APPEND BART_DEFS USE_MKL)
    set(_use_mkl_shim ON)
    message(STATUS "BLAS+FFT: MKL (fftwf shim via MKL DFTI)")
    message(STATUS "  headers : ${_mkl_include_dir}")
    message(STATUS "  runtime : ${_mkl_rt_lib}")
elseif(APPLE AND BARTORCH_USE_ACCELERATE)
    # ── Apple Accelerate path (macOS — no Homebrew dependencies required) ────
    #
    # Accelerate.framework ships with every macOS installation and provides:
    #   • CBLAS / LAPACK (via vecLib — always available in the SDK)
    #   • vDSP_DFT_zop_CreateSetup / vDSP_DFT_Execute (single-precision FFT)
    #   • vDSP_DFT_zop_CreateSetupD / vDSP_DFT_ExecuteD (double-precision FFT)
    #
    # BART's fftwf_plan_guru64_dft and FINUFFT's fftwf_plan_many_dft /
    # fftw_plan_many_dft calls are intercepted by accelerate_fftw_shim.c via
    # the linker --wrap mechanism and implemented on top of these vDSP APIs.
    #
    # The fftw3_stub/fftw3.h header provides the minimal FFTW3 declarations
    # required for compilation without linking against a real libfftw3f.

    # Locate cblas.h inside the active SDK (should always succeed on macOS).
    find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
        HINTS "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers"
              "${CMAKE_OSX_SYSROOT}/usr/include"
        NO_DEFAULT_PATH
    )
    if(NOT CBLAS_INCLUDE_DIR)
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h)
    endif()

    # Locate Accelerate.framework itself (for linking).
    find_library(CBLAS_LIBRARY NAMES Accelerate)

    if(CBLAS_INCLUDE_DIR AND CBLAS_LIBRARY)
        # FFTW3 include: point to our stub header (no real fftw3.h needed).
        set(FFTW3F_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fftw3_stub")
        # No external FFTW library; all symbols satisfied by the shim via --wrap.
        set(FFTW3F_LIBRARY     "")
        set(_use_accelerate_shim ON)
        message(STATUS "BLAS+FFT: Apple Accelerate (vDSP shim replaces FFTW3)")
        message(STATUS "  CBLAS headers  : ${CBLAS_INCLUDE_DIR}")
        message(STATUS "  Accelerate lib : ${CBLAS_LIBRARY}")
        message(STATUS "  FFTW3 stub dir : ${FFTW3F_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "BARTORCH_USE_ACCELERATE=ON but Accelerate.framework or cblas.h "
            "not found in the macOS SDK.  This should never happen on a "
            "standard macOS installation.  Try:\n"
            "  xcode-select --install\n"
            "or disable Accelerate and install fftw+openblas via Homebrew:\n"
            "  cmake -DBARTORCH_USE_ACCELERATE=OFF ...\n"
            "  brew install fftw openblas\n"
        )
    endif()
else()
    # ── Fallback: OpenBLAS (CBLAS + LAPACKE) + real FFTW3f ──────────────────
    #
    # Used on macOS (no ARM64 MKL wheels on PyPI) or any system where MKL is
    # not installed.
    #
    # Real FFTW3f handles BART's multi-dimensional howmany (batched) FFTs
    # correctly without any shim, unlike MKL's partial FFTW3 compat layer.
    #
    # NOTE: Windows is currently not supported.  bartorch's shim architecture
    # (mkl_fftw_shim, finufft_grid, torch_prior) relies on the GNU linker
    # --wrap mechanism, which is unavailable on Windows (link.exe, lld-link).
    #
    # Detection order for each library:
    #   1. Explicit CMake hint: FFTW_ROOT / OPENBLAS_ROOT (from CMAKE_ARGS)
    #   2. Homebrew auto-detection on macOS (brew --prefix fftw|openblas)
    #   3. CMake default system search (standard /usr, /usr/local, pkg-config)

    # ── Collect candidate prefix paths ──────────────────────────────────────
    set(_fallback_prefixes "")

    # User-supplied hints (highest priority)
    if(DEFINED FFTW_ROOT)
        list(APPEND _fallback_prefixes "${FFTW_ROOT}")
    endif()
    if(DEFINED OPENBLAS_ROOT)
        list(APPEND _fallback_prefixes "${OPENBLAS_ROOT}")
    endif()

    # Homebrew auto-detection (macOS and Linux-with-Homebrew)
    foreach(_brew_pkg "fftw" "openblas")
        execute_process(
            COMMAND brew --prefix "${_brew_pkg}"
            OUTPUT_VARIABLE _brew_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _brew_ret
            ERROR_QUIET
        )
        if(_brew_ret EQUAL 0 AND _brew_prefix)
            list(APPEND _fallback_prefixes "${_brew_prefix}")
        endif()
    endforeach()

    # Common system prefixes
    list(APPEND _fallback_prefixes "/usr/local" "/usr" "/opt/local")

    # ── Find FFTW3f (single-precision) library and header ───────────────────
    # Priority 1: explicit prefix search
    find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h
        HINTS ${_fallback_prefixes}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    # Priority 2: CMake default system search
    if(NOT FFTW3F_INCLUDE_DIR)
        find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h)
    endif()

    find_library(FFTW3F_LIBRARY
        NAMES fftw3f libfftw3f fftw3f-3
        HINTS ${_fallback_prefixes}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
    )
    if(NOT FFTW3F_LIBRARY)
        find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f fftw3f-3)
    endif()

    # ── Find OpenBLAS (cblas.h + lapacke.h + libopenblas) ───────────────────
    # cblas.h may live under include/ or include/openblas/ depending on the
    # package.  We search both and set CBLAS_INCLUDE_DIR to the directory that
    # directly contains cblas.h so that #include <cblas.h> resolves.
    find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
        HINTS ${_fallback_prefixes}
        PATH_SUFFIXES include include/openblas
        NO_DEFAULT_PATH
    )
    if(NOT CBLAS_INCLUDE_DIR)
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
            PATH_SUFFIXES include include/openblas
        )
    endif()

    find_library(CBLAS_LIBRARY
        NAMES openblas blas
        HINTS ${_fallback_prefixes}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
    )
    if(NOT CBLAS_LIBRARY)
        find_library(CBLAS_LIBRARY NAMES openblas blas)
    endif()

    # ── Validate and report ──────────────────────────────────────────────────
    if(FFTW3F_INCLUDE_DIR AND FFTW3F_LIBRARY AND CBLAS_INCLUDE_DIR AND CBLAS_LIBRARY)
        message(STATUS "BLAS+FFT: OpenBLAS + real FFTW3f (no MKL shim needed)")
        message(STATUS "  FFTW3f headers : ${FFTW3F_INCLUDE_DIR}")
        message(STATUS "  FFTW3f library : ${FFTW3F_LIBRARY}")
        message(STATUS "  CBLAS headers  : ${CBLAS_INCLUDE_DIR}")
        message(STATUS "  CBLAS library  : ${CBLAS_LIBRARY}")
        # _use_mkl_shim remains OFF: link real libfftw3f, no --wrap needed.
    else()
        message(FATAL_ERROR
            "Neither MKL nor OpenBLAS+FFTW3f found.\n"
            "Install the required libraries for your platform:\n"
            "  macOS  : brew install fftw openblas\n"
            "  Ubuntu : apt-get install libfftw3-dev libopenblas-dev\n"
            "Or install MKL: pip install mkl mkl-include\n"
            "\n"
            "For non-standard install locations, pass prefix hints to CMake:\n"
            "  CMAKE_ARGS='-DFFTW_ROOT=/path/to/fftw -DOPENBLAS_ROOT=/path/to/openblas' "
            "pip install -e . --no-build-isolation\n"
        )
    endif()
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
    set(_finufft_dir "${CMAKE_SOURCE_DIR}/../../../finufft")
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
    # cuFINUFFT: build when USE_CUDA is ON.  cuFINUFFT uses its own FFTW-free
    # cuFFT backend (no FFTW3 dependency on the GPU path), and CCCL (Thrust/CUB)
    # is fetched automatically via CPM inside FINUFFT's cuda_setup.cmake.
    if(USE_CUDA)
        set(FINUFFT_USE_CUDA   ON  CACHE BOOL "" FORCE)
        list(APPEND BART_DEFS BARTORCH_USE_CUFINUFFT)
        message(STATUS "cuFINUFFT: enabled (cuFINUFFT GPU gridder shim active)")
    else()
        set(FINUFFT_USE_CUDA   OFF CACHE BOOL "" FORCE)
    endif()
    set(FINUFFT_STATIC_LINKING ON  CACHE BOOL "" FORCE)
    set(FINUFFT_USE_DUCC0  OFF CACHE BOOL "" FORCE)   # use FFTW backend, not DUCC0
    set(FINUFFT_USE_OPENMP ON  CACHE BOOL "" FORCE)   # multi-threaded spreading

    # Point FINUFFT at the FFTW3 library found above (MKL or system fftw3f).
    # Any non-DEFAULT / non-DOWNLOAD value makes setupFFTW.cmake skip its own
    # CPM/find_package search and use our library directly.
    #
    # Accelerate path: FFTW3F_LIBRARY is empty — all fftwf_*/fftw_* symbols
    # will be satisfied by accelerate_fftw_shim.c via --wrap at final link time.
    # Passing an empty string (not "DEFAULT" or "DOWNLOAD") makes FINUFFT skip
    # its own FFTW3 search; finufft_fftlibs is created with no linked libraries.
    # We inject the Accelerate framework into finufft_fftlibs after add_subdirectory.
    set(FINUFFT_FFTW_LIBRARIES "${FFTW3F_LIBRARY}" CACHE STRING
        "FFTW library used by FINUFFT" FORCE)
    set(FINUFFT_FFTW_SUFFIX "" CACHE STRING "" FORCE)   # no _threads/_omp suffix

    # ── Apple Clang OpenMP setup (must come before add_subdirectory) ─────────
    # Apple's Clang does not include OpenMP; Homebrew installs libomp into a
    # keg-only path that CMake's FindOpenMP cannot auto-detect.  We query the
    # prefix with `brew --prefix libomp` and inject the necessary flags so
    # FINUFFT's find_package(OpenMP REQUIRED) succeeds.  If libomp is absent
    # (e.g. a minimal CI image), we gracefully disable FINUFFT OpenMP instead
    # of failing the build — FINUFFT will then run single-threaded.
    if(APPLE)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE _libomp_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _brew_libomp_ret
            ERROR_QUIET
        )
        if(_brew_libomp_ret EQUAL 0 AND EXISTS "${_libomp_prefix}")
            message(STATUS "Apple OpenMP: libomp found at ${_libomp_prefix}")
            set(OpenMP_C_FLAGS
                "-Xpreprocessor -fopenmp -I${_libomp_prefix}/include"
                CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS
                "-Xpreprocessor -fopenmp -I${_libomp_prefix}/include"
                CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES   "omp" CACHE STRING   "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING   "" FORCE)
            set(OpenMP_omp_LIBRARY
                "${_libomp_prefix}/lib/libomp.dylib"
                CACHE FILEPATH "" FORCE)
        else()
            message(STATUS "Apple OpenMP: libomp not found via brew — FINUFFT will run single-threaded")
            set(FINUFFT_USE_OPENMP OFF CACHE BOOL "" FORCE)
        endif()
    endif()

    # ── Build FINUFFT as a subproject ────────────────────────────────────────
    add_subdirectory("${_finufft_dir}" finufft_build EXCLUDE_FROM_ALL)

    # finufft_fftlibs is an INTERFACE library created by setupFFTW.cmake.
    # Inject the FFTW3 include dir so all FINUFFT compilation units find fftw3.h.
    # This must come AFTER add_subdirectory so the target already exists.
    target_include_directories(finufft_fftlibs INTERFACE "${FFTW3F_INCLUDE_DIR}")

    # Accelerate path: inject Accelerate framework so FINUFFT gets CBLAS/vDSP
    # at link time (the fftw symbols are intercepted by --wrap, not from Accelerate).
    if(_use_accelerate_shim)
        target_link_libraries(finufft_fftlibs INTERFACE "${CBLAS_LIBRARY}")
    endif()

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
file(GLOB _bart_misc_srcs    "${BART_SRC_DIR}/misc/*.c")
file(GLOB _bart_num_srcs     "${BART_SRC_DIR}/num/*.c")
file(GLOB _bart_simu_srcs    "${BART_SRC_DIR}/simu/*.c")
file(GLOB _bart_linops_srcs  "${BART_SRC_DIR}/linops/*.c")
file(GLOB _bart_wavelet_srcs "${BART_SRC_DIR}/wavelet/*.c")
file(GLOB _bart_iter_srcs    "${BART_SRC_DIR}/iter/*.c")
file(GLOB _bart_nlops_srcs   "${BART_SRC_DIR}/nlops/*.c")
file(GLOB _bart_noncart_srcs "${BART_SRC_DIR}/noncart/*.c")
file(GLOB _bart_calib_srcs   "${BART_SRC_DIR}/calib/*.c")
file(GLOB _bart_grecon_srcs  "${BART_SRC_DIR}/grecon/*.c")
file(GLOB _bart_lowrank_srcs "${BART_SRC_DIR}/lowrank/*.c")
file(GLOB _bart_networks_srcs "${BART_SRC_DIR}/networks/*.c")
file(GLOB _bart_nn_srcs      "${BART_SRC_DIR}/nn/*.c")
file(GLOB _bart_noir_srcs    "${BART_SRC_DIR}/noir/*.c")
file(GLOB _bart_sense_srcs   "${BART_SRC_DIR}/sense/*.c")
file(GLOB _bart_motion_srcs  "${BART_SRC_DIR}/motion/*.c")
file(GLOB _bart_moba_srcs    "${BART_SRC_DIR}/moba/*.c")
file(GLOB _bart_sake_srcs    "${BART_SRC_DIR}/sake/*.c")
file(GLOB _bart_geom_srcs    "${BART_SRC_DIR}/geom/*.c")
file(GLOB _bart_seq_srcs     "${BART_SRC_DIR}/seq/*.c")
file(GLOB _bart_stl_srcs     "${BART_SRC_DIR}/stl/*.c")
# Top-level tool files: each bart/src/<tool>.c provides main_<tool>()
# as well as the main dispatcher (bart.c) and entry-point wrapper (main.c).
file(GLOB _bart_tools_srcs   "${BART_SRC_DIR}/*.c")

# CUDA device-code sources — only included when USE_CUDA=ON.
# These .cu files are compiled to PTX/SASS device code by nvcc.
# BART guards them with #ifdef USE_CUDA so they compile cleanly when
# the definition is set.
if(USE_CUDA)
    file(GLOB _bart_num_cu_srcs      "${BART_SRC_DIR}/num/*.cu")
    file(GLOB _bart_wavelet_cu_srcs  "${BART_SRC_DIR}/wavelet/*.cu")
    file(GLOB _bart_noncart_cu_srcs  "${BART_SRC_DIR}/noncart/*.cu")
else()
    set(_bart_num_cu_srcs "")
    set(_bart_wavelet_cu_srcs "")
    set(_bart_noncart_cu_srcs "")
endif()

# Exclude files that require external optional libraries without #ifdef guards:
#
# ismrmrd.c       — includes ismrm/read.h → ismrmrd/ismrmrd.h (libismrmrd).
#                   BART's Makefile only adds this when ISMRM_BASE is set.
# ismrm/read.c    — same reason (libismrmrd).
# win/*.c         — Windows POSIX compatibility shims (mmap/shm_open,
#                   fmemopen).  Absent on Linux/macOS; added below for WIN32.
# lapacke/*.c     — BART's bundled LAPACKE shim: not needed when LAPACKE is
#                   provided by MKL (USE_MKL=1) or OpenBLAS.
list(REMOVE_ITEM _bart_tools_srcs "${BART_SRC_DIR}/ismrmrd.c")

set(BART_SOURCES
    ${_bart_misc_srcs}
    ${_bart_num_srcs}
    ${_bart_num_cu_srcs}
    ${_bart_simu_srcs}
    ${_bart_linops_srcs}
    ${_bart_wavelet_srcs}
    ${_bart_wavelet_cu_srcs}
    ${_bart_iter_srcs}
    ${_bart_nlops_srcs}
    ${_bart_noncart_srcs}
    ${_bart_noncart_cu_srcs}
    ${_bart_calib_srcs}
    ${_bart_grecon_srcs}
    ${_bart_lowrank_srcs}
    ${_bart_networks_srcs}
    ${_bart_nn_srcs}
    ${_bart_noir_srcs}
    ${_bart_sense_srcs}
    ${_bart_motion_srcs}
    ${_bart_moba_srcs}
    ${_bart_sake_srcs}
    ${_bart_geom_srcs}
    ${_bart_seq_srcs}
    ${_bart_stl_srcs}
    ${_bart_tools_srcs}
    # ismrm/read.c intentionally excluded (requires libismrmrd).
    # win/*.c added below for WIN32.
    # lapacke/*.c not needed (LAPACKE provided by MKL or OpenBLAS).
)

# On Windows (if ever revisited), include POSIX compatibility shims that
# provide mmap/munmap/shm_open (mman.c) and fmemopen (fmemopen.c).  These
# are used by BART's I/O layer but absent from the Win32 API.
# NOTE: Windows is not currently supported due to --wrap linker limitations.
#       These sources are included here to avoid compile errors when someone
#       attempts a Windows build and to ease a future port.
if(WIN32)
    list(APPEND BART_SOURCES
        "${BART_SRC_DIR}/win/mman.c"
        "${BART_SRC_DIR}/win/fmemopen.c"
    )
    message(STATUS "Windows: adding win/mman.c and win/fmemopen.c (POSIX shims)")
endif()

add_library(bart_static STATIC ${BART_SOURCES})
target_compile_definitions(bart_static PUBLIC ${BART_DEFS})
target_include_directories(bart_static PUBLIC
    ${BART_SRC_DIR}
    ${Python3_INCLUDE_DIRS}
    ${FFTW3F_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIR}
)
target_compile_options(bart_static PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-fPIC -O2 -std=gnu11>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC -O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --generate-code arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}>
)

# Link order: PyTorch → math libraries (CBLAS + FFTW3).
# With MKL: FFTW3F_LIBRARY == CBLAS_LIBRARY == libmkl_rt; both link the same file.
# With OpenBLAS: CBLAS_LIBRARY = libopenblas, FFTW3F_LIBRARY = libfftw3f.
target_link_libraries(bart_static PUBLIC
    ${TORCH_LIBRARIES}
    ${CBLAS_LIBRARY}
    ${FFTW3F_LIBRARY}
)

if(USE_CUDA)
    target_include_directories(bart_static PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(bart_static PUBLIC
        CUDA::cufft CUDA::cudart CUDA::cublas CUDA::cusolver)
    set_target_properties(bart_static PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ---------------------------------------------------------------------------
# _bartorch_ext Python extension module
# ---------------------------------------------------------------------------

# Find libtorch_python.so — it contains pybind11 type_caster specialisations
# for at::Tensor.  find_package(Torch) does NOT add it to TORCH_LIBRARIES, but
# any extension that converts Python objects ↔ torch.Tensor must link it.
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH
)
if(NOT TORCH_PYTHON_LIBRARY)
    # Fallback: search inside the torch package directory discovered during
    # MKL detection (same parent as site-packages/torch/lib/).
    find_library(TORCH_PYTHON_LIBRARY
        NAMES torch_python
        PATHS "${_mkl_site_packages}/torch/lib"
        NO_DEFAULT_PATH
    )
endif()
if(TORCH_PYTHON_LIBRARY)
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "libtorch_python not found — torch.Tensor pybind11 "
                    "type_casters may be unresolved at import time")
endif()

# ---------------------------------------------------------------------------
# _bartorch_ext Python extension module
# ---------------------------------------------------------------------------
# Core extension sources (compiled on all platforms)
set(_bartorch_ext_srcs
    bartorch_ext.cpp
    bart_ops.cpp
    torch_prior.cpp
    torch_prior_nlop.c
    lib_shim.c
    lib_ops.cpp
)
# MKL FFTW3 shim: only compiled when MKL is the BLAS/FFT backend.
# When using real FFTW3 (macOS without MKL, MSYS2, etc.) BART's fftwf_* calls
# resolve directly against libfftw3f — no interception required.
if(_use_mkl_shim)
    list(APPEND _bartorch_ext_srcs mkl_fftw_shim.c)
endif()
# Accelerate FFTW3 shim: only compiled when Accelerate is the BLAS/FFT backend.
# Intercepts all fftwf_* and fftw_* calls from BART and FINUFFT and routes them
# through Apple's vDSP_DFT_Execute / vDSP_DFT_ExecuteD APIs.
if(_use_accelerate_shim)
    list(APPEND _bartorch_ext_srcs accelerate_fftw_shim.c)
endif()

pybind11_add_module(_bartorch_ext ${_bartorch_ext_srcs})
target_include_directories(_bartorch_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BART_SRC_DIR}
    ${CBLAS_INCLUDE_DIR}
)
target_link_libraries(_bartorch_ext PRIVATE
    bart_static
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)
target_compile_definitions(_bartorch_ext PRIVATE ${BART_DEFS})

# --wrap the fftwf_* symbols so the active FFT shim intercepts FFTW3 calls.
#
# MKL path (_use_mkl_shim):
#   mkl_fftw_shim.c intercepts BART's fftwf_plan_guru64_dft calls and routes
#   them through MKL's DFTI API, working around MKL's broken FFTW3 compat layer.
#   Only BART's symbols are wrapped (FINUFFT resolves fftwf_plan_many_dft
#   directly against MKL's FFTW3 compat layer, which handles plan_many correctly).
#
# Accelerate path (_use_accelerate_shim):
#   accelerate_fftw_shim.c intercepts ALL fftwf_* and fftw_* calls from both
#   BART and FINUFFT and routes them through vDSP_DFT_Execute / vDSP_DFT_ExecuteD.
#   No real libfftw3f is linked.
#
# Real FFTW3 path (neither shim):
#   BART's and FINUFFT's calls resolve directly against libfftw3f — no wrap needed.
if(_use_mkl_shim)
    if(APPLE)
        target_link_options(_bartorch_ext PRIVATE
            -Wl,-wrap,_fftwf_plan_guru64_dft
            -Wl,-wrap,_fftwf_execute_dft
            -Wl,-wrap,_fftwf_destroy_plan
            -Wl,-wrap,_fftwf_export_wisdom_to_filename
            -Wl,-wrap,_fftwf_import_wisdom_from_filename
            -Wl,-wrap,_fftwf_init_threads
            -Wl,-wrap,_fftwf_plan_with_nthreads
            -Wl,-wrap,_fftwf_cleanup_threads
        )
    else()
        target_link_options(_bartorch_ext PRIVATE
            -Wl,--wrap,fftwf_plan_guru64_dft
            -Wl,--wrap,fftwf_execute_dft
            -Wl,--wrap,fftwf_destroy_plan
            -Wl,--wrap,fftwf_export_wisdom_to_filename
            -Wl,--wrap,fftwf_import_wisdom_from_filename
            -Wl,--wrap,fftwf_init_threads
            -Wl,--wrap,fftwf_plan_with_nthreads
            -Wl,--wrap,fftwf_cleanup_threads
        )
    endif()
    message(STATUS "FFTW shim: --wrap fftwf_plan_guru64_dft active (MKL DFTI backend)")
elseif(_use_accelerate_shim)
    # Apple ld: Mach-O symbols are prefixed with '_' in the --wrap argument.
    # Wrap float precision (BART + FINUFFT float path):
    target_link_options(_bartorch_ext PRIVATE
        -Wl,-wrap,_fftwf_plan_guru64_dft
        -Wl,-wrap,_fftwf_plan_many_dft
        -Wl,-wrap,_fftwf_execute_dft
        -Wl,-wrap,_fftwf_destroy_plan
        -Wl,-wrap,_fftwf_export_wisdom_to_filename
        -Wl,-wrap,_fftwf_import_wisdom_from_filename
        -Wl,-wrap,_fftwf_forget_wisdom
        -Wl,-wrap,_fftwf_cleanup
        -Wl,-wrap,_fftwf_init_threads
        -Wl,-wrap,_fftwf_plan_with_nthreads
        -Wl,-wrap,_fftwf_cleanup_threads
    )
    # Wrap double precision (FINUFFT double cleanup path):
    target_link_options(_bartorch_ext PRIVATE
        -Wl,-wrap,_fftw_plan_many_dft
        -Wl,-wrap,_fftw_execute_dft
        -Wl,-wrap,_fftw_destroy_plan
        -Wl,-wrap,_fftw_forget_wisdom
        -Wl,-wrap,_fftw_cleanup
        -Wl,-wrap,_fftw_init_threads
        -Wl,-wrap,_fftw_plan_with_nthreads
        -Wl,-wrap,_fftw_cleanup_threads
    )
    # Link Accelerate framework (provides CBLAS/vecLib; vDSP used in shim).
    target_link_libraries(_bartorch_ext PRIVATE "${CBLAS_LIBRARY}")
    message(STATUS "FFTW shim: --wrap fftwf_*/fftw_* active (Apple Accelerate vDSP backend)")
else()
    message(STATUS "FFTW: real libfftw3f linked directly (no shim required)")
endif()

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

    # ── cuFINUFFT GPU gridder shim ─────────────────────────────────────────
    # When both FINUFFT and USE_CUDA are active, also compile the GPU gridder
    # shim that wraps cuda_grid/cuda_gridH/cuda_apply_rolloff_correction2 using
    # cuFINUFFT's type-1/2 spread-only kernels.
    if(USE_CUDA)
        target_sources(_bartorch_ext PRIVATE cuda/cufinufft_grid.cu)
        target_link_libraries(_bartorch_ext PRIVATE cufinufft)
        target_compile_definitions(_bartorch_ext PRIVATE BARTORCH_USE_CUFINUFFT)

        target_link_options(_bartorch_ext PRIVATE
            -Wl,--wrap,cuda_grid
            -Wl,--wrap,cuda_gridH
            -Wl,--wrap,cuda_apply_rolloff_correction2
        )
        message(STATUS "cuFINUFFT GPU gridder: --wrap,cuda_grid --wrap,cuda_gridH --wrap,cuda_apply_rolloff_correction2 active")
    endif()
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
    target_sources(_bartorch_ext PRIVATE
        cuda/cuda_bridge.hpp
        cuda/bartorch_cuda_dispatch.cpp
    )
    target_compile_definitions(_bartorch_ext PRIVATE USE_CUDA)
    target_include_directories(_bartorch_ext PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(_bartorch_ext PRIVATE
        CUDA::cudart CUDA::cufft CUDA::cublas CUDA::cusolver)

    # --wrap,memcfl_create: GPU dispatch mode intercepts CFL output allocation
    # (cuda/bartorch_cuda_dispatch.cpp provides __wrap_memcfl_create).
    target_link_options(_bartorch_ext PRIVATE
        -Wl,--wrap,memcfl_create
    )
    message(STATUS "CUDA dispatch: --wrap,memcfl_create active")
endif()

# Ensure the .so lands directly in the bartorch package directory so that
# `from bartorch import _bartorch_ext` works for both editable and installed builds.
# CMAKE_SOURCE_DIR = src/bartorch/csrc  →  ../ = src/bartorch/  (package root).
set_target_properties(_bartorch_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
