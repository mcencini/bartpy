cmake_minimum_required(VERSION 3.18)
project(bartorch_ext LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA "Build with CUDA support" OFF)
option(BART_MEMONLY_CFL "Build BART with MEMONLY_CFL (all I/O in-memory)" ON)
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
set(BART_SRC_DIR "${CMAKE_SOURCE_DIR}/../bart/src")

# BART compile definitions
set(BART_DEFS
    BART_WITH_PYTHON         # suppress abort() → RuntimeError
    NO_MCU                   # skip microcontroller code
)
if(BART_MEMONLY_CFL)
    list(APPEND BART_DEFS MEMONLY_CFL)
endif()
if(USE_CUDA)
    list(APPEND BART_DEFS USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
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
    # num
    ${BART_SRC_DIR}/num/multind.c
    ${BART_SRC_DIR}/num/flpmath.c
    ${BART_SRC_DIR}/num/fft.c
    ${BART_SRC_DIR}/num/blas.c
    ${BART_SRC_DIR}/num/vecops.c
    ${BART_SRC_DIR}/num/rand.c
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
    ${BART_SRC_DIR}/bart_embed_api.c
    ${BART_SRC_DIR}/main.c
    ${BART_SRC_DIR}/bart.c
)

add_library(bart_static STATIC ${BART_SOURCES})
target_compile_definitions(bart_static PUBLIC ${BART_DEFS})
target_include_directories(bart_static PUBLIC ${BART_SRC_DIR})
target_compile_options(bart_static PRIVATE -fPIC -O2 -std=gnu11)

# Link BART against PyTorch's bundled BLAS + FFT libraries.
# This avoids shipping duplicate FFTW/OpenBLAS in the wheel.
target_link_libraries(bart_static PUBLIC ${TORCH_LIBRARIES})

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

# Ensure the .so lands in the bartorch package directory.
set_target_properties(_bartorch_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bartorch"
)
