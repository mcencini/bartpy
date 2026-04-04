/*
 * tensor_bridge.hpp — Zero-copy bridge between torch::Tensor and BART CFL.
 *
 * Design
 * ------
 * BART's in-memory CFL registry stores named ``complex float*`` buffers.
 * A ``torch::Tensor`` of dtype ``complex64`` in column-major (Fortran) order
 * has the identical memory layout.  Therefore we can:
 *
 *   1. Register an existing tensor's data_ptr() as a non-owned CFL entry
 *      (BART does NOT free the memory; PyTorch retains ownership).
 *   2. Allocate a new tensor with Fortran strides, register it as an output
 *      CFL entry, and let bart_command() write directly into the tensor.
 *
 * No copies are performed in either direction for CPU tensors.
 * For CUDA tensors, see cuda/cuda_bridge.hpp — the same approach applies
 * when BART is compiled with USE_CUDA: BART's vptr system recognises device
 * pointers and dispatches GPU kernels transparently.
 *
 * Utility functions
 * -----------------
 *   fortran_strides(dims)   — compute column-major stride vector
 *   is_bart_compatible(t)   — true iff t is complex64 + Fortran strides
 *   make_bart_tensor(dims)  — allocate output tensor with Fortran strides
 *   register_input(name, t) — call register_mem_cfl_non_managed()
 *   register_output(name,t) — same, used for pre-allocated outputs
 *   unlink_cfl(name)        — memcfl_unlink() after bart_command() returns
 */

#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>

extern "C" {
#include "misc/memcfl.h"
}

namespace bartorch {

// ---------------------------------------------------------------------------
// Stride helpers
// ---------------------------------------------------------------------------

inline std::vector<int64_t> fortran_strides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size());
    int64_t s = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        strides[i] = s;
        s *= dims[i];
    }
    return strides;
}

inline bool is_bart_compatible(const torch::Tensor& t)
{
    if (t.dtype() != torch::kComplexFloat)
        return false;
    auto dims = t.sizes().vec();
    auto expected = fortran_strides(dims);
    auto actual   = t.strides().vec();
    return expected == actual;
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/**
 * Allocate an uninitialised complex64 tensor with Fortran (column-major)
 * strides on the requested device.
 *
 * The tensor is the canonical "output slot" that bart_command() writes into.
 */
inline torch::Tensor make_bart_tensor(
    const std::vector<int64_t>& dims,
    torch::Device device = torch::kCPU)
{
    auto opts = torch::TensorOptions()
                    .dtype(torch::kComplexFloat)
                    .device(device);

    // Allocate storage with reversed dims (Fortran trick), then view back.
    std::vector<int64_t> rdims(dims.rbegin(), dims.rend());
    auto storage = torch::empty(rdims, opts).contiguous();

    return storage.as_strided(dims, fortran_strides(dims));
}

// ---------------------------------------------------------------------------
// CFL registry operations
// ---------------------------------------------------------------------------

/**
 * Register a tensor's data buffer in BART's in-memory CFL registry without
 * transferring ownership.  The tensor must remain alive for the duration of
 * the bart_command() call.
 *
 * @param name   Unique *.mem name (e.g. "_bt_abc123.mem").
 * @param t      BartTensor (must satisfy is_bart_compatible()).
 */
inline void register_input(const std::string& name, const torch::Tensor& t)
{
    TORCH_CHECK(is_bart_compatible(t),
                "register_input: tensor must be complex64 with Fortran strides");

    auto dims_i64 = t.sizes().vec();
    std::vector<long> dims(dims_i64.begin(), dims_i64.end());

    memcfl_register(
        name.c_str(),
        static_cast<int>(dims.size()),
        dims.data(),
        reinterpret_cast<std::complex<float>*>(t.data_ptr()),
        /* managed = */ false);
}

/**
 * Register a pre-allocated output tensor (same semantics as register_input).
 */
inline void register_output(const std::string& name, torch::Tensor& t)
{
    register_input(name, t);
}

/**
 * Remove a name from the in-memory CFL registry after bart_command() returns.
 * Since the tensor is non-managed, BART will NOT free the underlying memory.
 */
inline void unlink_cfl(const std::string& name)
{
    memcfl_unlink(name.c_str());
}

} // namespace bartorch
