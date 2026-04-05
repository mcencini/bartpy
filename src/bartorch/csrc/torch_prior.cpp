/*
 * torch_prior.cpp — Plug-and-play PyTorch denoiser prior via --wrap nlop_tf_create.
 *
 * Strategy (identical to finufft_grid.cpp for grid2/grid2H):
 *
 *   By linking with
 *     Linux : -Wl,--wrap,nlop_tf_create
 *     macOS : -Wl,-wrap,_nlop_tf_create
 *   every call to nlop_tf_create() inside the linked binary is redirected to
 *   __wrap_nlop_tf_create().  When the path argument starts with
 *   "bartorch://", the function looks up a registered Python callable in a
 *   global C++ map and builds a custom BART nlop whose forward/adjoint steps
 *   invoke the denoiser through the CPython GIL.  All other paths fall through
 *   to __real_nlop_tf_create() — the original BART TF implementation (which
 *   will raise an error unless BART was compiled with TensorFlow support).
 *
 * Usage from Python (via pics() in _commands.py):
 *
 *   1. Python calls _ext.register_torch_prior(name, fn, bart_img_dims).
 *   2. Python appends R='TF:{bartorch://name}:lambda' to the BART pics flags.
 *   3. BART's optreg.c TENFL case calls nlop_tf_create("bartorch://name").
 *   4. __wrap_nlop_tf_create intercepts, looks up the entry, returns our nlop.
 *   5. BART's prox_nlgrad_create wraps the nlop in a proximal operator.
 *   6. BART's own ADMM/IST/FISTA loop calls the proximal operator each
 *      iteration, which calls nlop_apply / nlop_adjoint on our nlop, which
 *      in turn calls back into Python to run the denoiser.
 *
 * Denoiser convention (no sigma, pure PnP):
 *
 *   The Python callable has signature:  fn(x: torch.Tensor) -> torch.Tensor
 *   x is passed as a flat complex64 tensor with numel = prod(spatial_dims).
 *   The denoiser must return a tensor of the same shape and dtype.
 *
 * nlop semantics (grad_nlop=false, one gradient step):
 *
 *   forward(x)  → scalar:  ||x − D(x)||² / 2  (debug; also caches residual)
 *   adjoint(x, grad_scalar=1) → x − D(x)       (cached noise residual)
 *
 *   prox_nlgrad_create(nlop, steps=1, stepsize=1.0, lambda, false) computes:
 *     prox(z, mu) = z − mu·lambda · (z − D(z))
 *
 *   With lambda = 1 and mu = 1 (ADMM default rho = 1):
 *     prox(z) = D(z)   ✓  (standard PnP proximal denoiser)
 *
 *   Users control the effective denoising strength via torch_prior_lambda in
 *   bt.pics(..., torch_prior_lambda=<value>).
 */

#include <Python.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

extern "C" {
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"       /* DIMS */
#include "num/multind.h"    /* md_calc_size, md_clear, md_alloc, md_free */
#include "nlops/nlop.h"     /* nlop_create, nlop_data_t, … */
}

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Global prior registry
// ---------------------------------------------------------------------------

struct TorchPriorEntry {
    PyObject*         fn_obj; // borrowed-ref incremented on register
    std::vector<long> dims;   // BART Fortran img_dims (DIMS=16 elements)
};

static std::unordered_map<std::string, TorchPriorEntry> g_prior_registry;

/* Called from Python (pybind11 binding in bartorch_ext.cpp) before pics(). */
void bartorch_register_torch_prior(const std::string& name,
                                   py::object        fn,
                                   std::vector<long> dims)
{
    PyObject* obj = fn.ptr();
    Py_INCREF(obj);
    g_prior_registry[name] = { obj, std::move(dims) };
}

/* Called from Python (pybind11 binding) after pics() returns / on error. */
void bartorch_unregister_torch_prior(const std::string& name)
{
    auto it = g_prior_registry.find(name);
    if (it != g_prior_registry.end()) {
        py::gil_scoped_acquire gil;
        Py_DECREF(it->second.fn_obj);
        g_prior_registry.erase(it);
    }
}

// ---------------------------------------------------------------------------
// BART nlop data structure
// ---------------------------------------------------------------------------

struct torch_prior_nlop_s {
    nlop_data_t  super;    /* MUST be first — CAST_UP/CAST_DOWN rely on this */
    PyObject*    fn_obj;   /* Python denoiser callable (reference owned here) */
    int          N;        /* DIMS = 16                                        */
    long         dims[16]; /* img_dims in BART Fortran order                   */
    complex float* res;    /* cached residual x − D(x) from last forward call  */
};

DEF_TYPEID(torch_prior_nlop_s);

/*
 * Forward: image → scalar
 * Calls D(x), stores residual x−D(x), returns ‖x−D(x)‖²/2.
 */
static void torch_prior_fwd(const nlop_data_t* _data,
                             complex float*       dst,  /* scalar (1 elem)  */
                             const complex float* src)  /* image  (N elems) */
{
    auto* d = CAST_DOWN(torch_prior_nlop_s, _data);
    const long numel = md_calc_size(d->N, d->dims);

    if (nullptr == d->res)
        d->res = (complex float*)md_alloc(d->N, d->dims, CFL_SIZE);

    /* Wrap src as a flat complex64 torch::Tensor (zero-copy view). */
    const auto opts = torch::TensorOptions().dtype(torch::kComplexFloat);
    auto x_t = torch::from_blob(const_cast<complex float*>(src),
                                 { numel }, opts);

    /* Call Python denoiser — GIL must be held for pybind11 / CPython calls. */
    torch::Tensor dx_t;
    {
        py::gil_scoped_acquire gil;
        py::object py_fn = py::reinterpret_borrow<py::object>(d->fn_obj);
        dx_t = py_fn(x_t).cast<torch::Tensor>();
    }

    dx_t = dx_t.contiguous().to(torch::kComplexFloat).cpu();

    /* residual = x − D(x)  (stored for adjoint) */
    const auto* dx_ptr =
        reinterpret_cast<const complex float*>(
            dx_t.data_ptr<c10::complex<float>>());
    for (long i = 0; i < numel; ++i)
        d->res[i] = src[i] - dx_ptr[i];

    /* Scalar output = ‖residual‖²/2 — only used for BART debug logging. */
    double acc = 0.0;
    const auto* rp = reinterpret_cast<const float*>(d->res);
    for (long i = 0; i < 2 * numel; ++i)
        acc += (double)rp[i] * rp[i];
    dst[0] = (complex float)(acc * 0.5);
}

/*
 * Forward derivative (linearised, domain → codomain).
 * Not used by prox_nlgrad with grad_nlop=false; set to zero.
 */
static void torch_prior_der(const nlop_data_t* _data,
                              int  /*o*/, int /*i*/,
                              complex float*       dst,  /* scalar (codomain) */
                              const complex float* /*src*/)
{
    (void)_data;
    dst[0] = 0.f;
}

/*
 * Adjoint (codomain → domain): returns scale * cached residual x − D(x).
 * In prox_nlgrad, src[0] = {1.0}, so dst = residual = x − D(x).
 */
static void torch_prior_adj(const nlop_data_t* _data,
                              int  /*o*/, int /*i*/,
                              complex float*       dst,  /* image  (domain)   */
                              const complex float* src)  /* scalar (codomain) */
{
    auto* d = CAST_DOWN(torch_prior_nlop_s, _data);
    const long numel = md_calc_size(d->N, d->dims);

    if (nullptr == d->res) {
        md_clear(d->N, d->dims, dst, CFL_SIZE);
        return;
    }

    const complex float scale = src[0];
    for (long i = 0; i < numel; ++i)
        dst[i] = scale * d->res[i];
}

/* Destructor */
static void torch_prior_del(const nlop_data_t* _data)
{
    auto* d = CAST_DOWN(torch_prior_nlop_s, _data);
    {
        py::gil_scoped_acquire gil;
        Py_DECREF(d->fn_obj);
    }
    md_free(d->res);
    xfree(d);
}

/*
 * Factory: allocate and return a BART nlop backed by a Python denoiser.
 */
static const struct nlop_s*
nlop_torch_prior_create(PyObject* fn_obj, const std::vector<long>& dims)
{
    PTR_ALLOC(struct torch_prior_nlop_s, data);
    SET_TYPEID(torch_prior_nlop_s, data);

    {
        py::gil_scoped_acquire gil;
        Py_INCREF(fn_obj);
    }
    data->fn_obj = fn_obj;
    data->N      = DIMS;
    for (int i = 0; i < DIMS; ++i)
        data->dims[i] = (i < (int)dims.size()) ? dims[i] : 1L;
    data->res = nullptr;

    /* Codomain: scalar (1 complex float). */
    const long scalar_dims[1] = { 1L };

    return nlop_create(
        /*OD*/ 1,    scalar_dims,
        /*ID*/ DIMS, data->dims,
        CAST_UP(PTR_PASS(data)),
        torch_prior_fwd,
        torch_prior_der,
        torch_prior_adj,
        /*normal*/   nullptr,
        /*norm_inv*/ nullptr,
        torch_prior_del
    );
}

// ---------------------------------------------------------------------------
// --wrap nlop_tf_create
// ---------------------------------------------------------------------------

/*
 * Declare the original symbol.  The linker, when given --wrap,nlop_tf_create
 * (Linux) or -wrap,_nlop_tf_create (macOS), renames the original function
 * body to __real_nlop_tf_create so we can fall through to it for non-bartorch
 * paths.
 */
extern "C" const struct nlop_s* __real_nlop_tf_create(const char* path);

extern "C" const struct nlop_s* __wrap_nlop_tf_create(const char* path)
{
    static constexpr char     kPrefix[]  = "bartorch://";
    static constexpr ptrdiff_t kPrefixLen = sizeof(kPrefix) - 1;

    if (strncmp(path, kPrefix, (size_t)kPrefixLen) == 0) {
        const std::string name(path + kPrefixLen);
        auto it = g_prior_registry.find(name);
        if (it == g_prior_registry.end())
            error("bartorch: torch_prior '%s' not found in registry — "
                  "was bt.pics() called with torch_prior= set?\n",
                  name.c_str());
        return nlop_torch_prior_create(it->second.fn_obj, it->second.dims);
    }

    /* Fall through to BART's real TF implementation. */
    return __real_nlop_tf_create(path);
}
