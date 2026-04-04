/*
 * bartorch_ext.cpp — PyTorch C++ extension entry point.
 *
 * This file is the sole compiled entry point for the ``_bartorch_ext``
 * Python extension module.  It exposes BART functionality through three
 * layers:
 *
 *  1. ``run(op_name, inputs, output_dims, kwargs)``
 *     Generic hot-path dispatcher: registers input BartTensor data_ptr()
 *     buffers in BART's in-memory CFL registry (via
 *     register_mem_cfl_non_managed), allocates output BartTensors with
 *     Fortran strides, then calls bart_command() in-process.
 *
 *  2. Named op bindings (fft, phantom, pics, ecalib, …)
 *     Thin wrappers around run() that construct the argv array from typed
 *     Python arguments, mirroring the SWIG-based API from the old bartpy.
 *
 *  3. BartTensor factory helpers (bart_empty, bart_zeros, bart_from_numpy)
 *     exposed to Python for use in scripts/notebooks.
 *
 * Build requirements
 * ------------------
 *  - libtorch (from the user's PyTorch install, found via find_package(Torch))
 *  - BART static libraries compiled with -DMEMONLY_CFL
 *    (all I/O routed through the in-memory CFL registry)
 *  - optionally: CUDA toolkit when building with USE_CUDA=ON
 *
 * See CMakeLists.txt for the full link list.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// BART embed API — declared in bart/src/bart_embed_api.h
// The include path is set by CMakeLists.txt.
extern "C" {
#include "bart_embed_api.h"
#include "misc/memcfl.h"
#include "misc/misc.h"
}

#include "tensor_bridge.hpp"

#ifdef USE_CUDA
#include "cuda/cuda_bridge.hpp"
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Generic hot-path dispatcher
// ---------------------------------------------------------------------------

/**
 * run() — zero-copy in-process BART command execution.
 *
 * For every tensor input:
 *   - asserts dtype == complex64 and Fortran-order strides
 *   - registers data_ptr() under a unique *.mem name via
 *     register_mem_cfl_non_managed()
 *
 * Allocates output tensors (Fortran strides, complex64) using output_dims
 * (or dims inferred from a dry-run when output_dims is None), registers them,
 * then calls bart_command().
 *
 * On return, unlinks all *.mem names and returns the output BartTensor(s).
 */
static py::object run(
    const std::string& op_name,
    py::list inputs,
    py::object output_dims_py,
    py::dict kwargs)
{
    // TODO: implement once CMake/BART build is wired up.
    throw std::runtime_error(
        "_bartorch_ext.run() is not yet implemented — "
        "please build the C++ extension first.");
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_bartorch_ext, m) {
    m.doc() = "bartorch C++ extension — zero-copy PyTorch↔BART bridge";

    m.def("run", &run,
          py::arg("op_name"),
          py::arg("inputs"),
          py::arg("output_dims"),
          py::arg("kwargs"),
          "Generic zero-copy BART command dispatcher (hot path).");

    // Named op bindings will be added here as the implementation matures.
    // Example (not yet active):
    //   m.def("fft",  &bart_fft,  ...);
    //   m.def("pics", &bart_pics, ...);
}
