/*
 * mkl_fftw_shim.c — intercept BART's fftwf_* calls and route them through MKL DFTI.
 *
 * MKL ships a partial FFTW3 compatibility layer in libmkl_rt, but its
 * fftwf_plan_guru64_dft implementation rejects any call with l > 1 howmany
 * dimensions.  BART uses l = D − k (up to 15 for DIMS=16 arrays), so every
 * FFT-based BART command silently fails with the MKL FFTW3 compat wrapper.
 *
 * This shim intercepts fftwf_plan_guru64_dft / fftwf_execute_dft /
 * fftwf_destroy_plan (and the no-op wisdom + thread helpers) via the GNU
 * linker --wrap mechanism and re-implements them on top of MKL's native DFTI
 * interface, which has no such restriction.
 *
 * Link with:
 *   -Wl,--wrap,fftwf_plan_guru64_dft
 *   -Wl,--wrap,fftwf_execute_dft
 *   -Wl,--wrap,fftwf_destroy_plan
 *   -Wl,--wrap,fftwf_export_wisdom_to_filename
 *   -Wl,--wrap,fftwf_import_wisdom_from_filename
 *   -Wl,--wrap,fftwf_init_threads
 *   -Wl,--wrap,fftwf_plan_with_nthreads
 *   -Wl,--wrap,fftwf_cleanup_threads
 *
 * MKL DFTI handles threading natively; no per-plan thread count is needed.
 *
 * Strides convention:
 *   FFTW guru64: is/os are element strides (complex float = 1 element).
 *   DFTI strides: strides[0] = first-element offset, strides[j] = stride
 *                 of j-th dimension in elements.
 *
 * For a batched 1-D FFT with k=1, l=1 (e.g. 4x4 non-contiguous):
 *   FFTW: dims[0]={n=4, is=1, os=1}, hmdims[0]={n=4, is=4, os=4}
 *   DFTI: lengths[0]=4, istrides[0]=0,istrides[1]=1, ostrides same,
 *         number_of_transforms=4, idist=4, odist=4
 *
 * For arbitrary mixed k/l layout we flatten the l howmany dims into a single
 * number_of_transforms product, and keep the k transform dims as the
 * multi-dimensional DFTI descriptor.  When all howmany strides are
 * compatible with a single dist (last-dim stride × last-dim size = next
 * stride), this is exact.  For BART's standard contiguous layout with unit
 * trailing dims (all hmdims[i].n == 1), number_of_transforms = 1 and no
 * distance is needed — which covers the overwhelmingly common case of BART's
 * 16-dimensional arrays where only a few leading dims are non-unit.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <complex.h>

#include <mkl_dfti.h>

/* ─── FFTW3 type stubs (we don't link against real FFTW3) ──────────────── */

typedef struct {
    MKL_LONG          n;
    MKL_LONG          is;
    MKL_LONG          os;
} _bt_iodim64;

/* Internal plan struct that wraps a DFTI descriptor */
typedef struct {
    DFTI_DESCRIPTOR_HANDLE  desc;
    int                     sign;     /* FFTW_FORWARD=-1, FFTW_BACKWARD=+1 */
} _bt_plan;

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static inline void _bt_check(MKL_LONG status, const char *where)
{
    if (status != DFTI_NO_ERROR) {
        /* BART's assert() macro calls error() and longjmps on failure;
         * here we just abort with a message since we're in C not in BART's
         * error-handling scope. */
        fprintf(stderr, "bartorch mkl_fftw_shim: DFTI error in %s: %s\n",
                where, DftiErrorMessage(status));
        abort();
    }
}

/* ─── Main plan creation shim ──────────────────────────────────────────── */

/*
 * Signature mirrors FFTW3's fftwf_plan_guru64_dft.
 * The __real_ prefix is the GNU linker --wrap symbol for the original.
 * We implement __wrap_ which is called instead.
 */
void *__wrap_fftwf_plan_guru64_dft(
    int                    k,        /* number of transform dimensions */
    const _bt_iodim64     *dims,     /* transform dims: n, is, os per dim */
    int                    l,        /* number of howmany (batch) dims */
    const _bt_iodim64     *hmdims,   /* batch dims: n, is, os per dim */
    float _Complex        *in,       /* input buffer (planning) — ignored for DFTI */
    float _Complex        *out,      /* output buffer (planning) — ignored for DFTI */
    int                    sign,     /* -1 = forward, +1 = backward */
    unsigned               flags)    /* FFTW_ESTIMATE etc. — ignored */
{
    (void)in; (void)out; (void)flags;

    if (k <= 0) return NULL;

    /* ── Compute number_of_transforms and batch distances ──────────────── */
    MKL_LONG ntrans = 1;
    for (int i = 0; i < l; i++)
        ntrans *= (MKL_LONG)hmdims[i].n;

    /* Determine idist/odist: stride of the outermost batch dimension.
     * For BART's standard case (all l batch dims have n == 1) ntrans == 1 and
     * dist is irrelevant.  For the general case we use the is/os of the first
     * howmany dim (the one with the largest stride) as the inter-transform
     * distance — valid when batch dims are contiguous in memory, which holds
     * for BART's md_alloc layout. */
    MKL_LONG idist = (l > 0) ? (MKL_LONG)hmdims[0].is : 1;
    MKL_LONG odist = (l > 0) ? (MKL_LONG)hmdims[0].os : 1;

    /* ── Build dimension arrays for DFTI ───────────────────────────────── */
    MKL_LONG *lengths    = (MKL_LONG *)malloc((size_t)k * sizeof(MKL_LONG));
    MKL_LONG *istrides   = (MKL_LONG *)malloc((size_t)(k + 1) * sizeof(MKL_LONG));
    MKL_LONG *ostrides   = (MKL_LONG *)malloc((size_t)(k + 1) * sizeof(MKL_LONG));

    if (!lengths || !istrides || !ostrides) {
        free(lengths); free(istrides); free(ostrides);
        return NULL;
    }

    /* DFTI strides: strides[0] = first-element offset (always 0), strides[j]
     * = element stride of j-th dimension (1-indexed, outermost first). */
    istrides[0] = 0;
    ostrides[0] = 0;
    for (int i = 0; i < k; i++) {
        lengths[i]       = (MKL_LONG)dims[i].n;
        istrides[i + 1]  = (MKL_LONG)dims[i].is;
        ostrides[i + 1]  = (MKL_LONG)dims[i].os;
    }

    /* ── Create DFTI descriptor ─────────────────────────────────────────── */
    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG status;

    if (k == 1) {
        status = DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX,
                                      (MKL_LONG)1, lengths[0]);
    } else {
        status = DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX,
                                      (MKL_LONG)k, lengths);
    }

    free(lengths);

    if (status != DFTI_NO_ERROR) {
        free(istrides); free(ostrides);
        return NULL;
    }

    /* Set strides */
    status = DftiSetValue(desc, DFTI_INPUT_STRIDES, istrides);
    if (status != DFTI_NO_ERROR) {
        free(istrides); free(ostrides);
        DftiFreeDescriptor(&desc);
        return NULL;
    }
    status = DftiSetValue(desc, DFTI_OUTPUT_STRIDES, ostrides);
    free(istrides); free(ostrides);
    if (status != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* Set number of transforms (batch) */
    if (ntrans > 1) {
        status = DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, ntrans);
        if (status != DFTI_NO_ERROR) { DftiFreeDescriptor(&desc); return NULL; }

        status = DftiSetValue(desc, DFTI_INPUT_DISTANCE, idist);
        if (status != DFTI_NO_ERROR) { DftiFreeDescriptor(&desc); return NULL; }

        status = DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, odist);
        if (status != DFTI_NO_ERROR) { DftiFreeDescriptor(&desc); return NULL; }
    }

    /* Out-of-place transform */
    status = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status != DFTI_NO_ERROR) { DftiFreeDescriptor(&desc); return NULL; }

    /* Commit */
    status = DftiCommitDescriptor(desc);
    if (status != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* ── Wrap in our plan struct ────────────────────────────────────────── */
    _bt_plan *plan = (_bt_plan *)malloc(sizeof(_bt_plan));
    if (!plan) {
        DftiFreeDescriptor(&desc);
        return NULL;
    }
    plan->desc = desc;
    plan->sign = sign;
    return (void *)plan;
}

/* ─── Execute shim ─────────────────────────────────────────────────────── */

void __wrap_fftwf_execute_dft(void *_plan, float _Complex *in, float _Complex *out)
{
    _bt_plan *plan = (_bt_plan *)_plan;
    assert(plan && plan->desc);

    MKL_LONG status;
    if (plan->sign == -1)
        status = DftiComputeForward(plan->desc, (void *)in, (void *)out);
    else
        status = DftiComputeBackward(plan->desc, (void *)in, (void *)out);

    _bt_check(status, "fftwf_execute_dft");
}

/* ─── Destroy shim ─────────────────────────────────────────────────────── */

void __wrap_fftwf_destroy_plan(void *_plan)
{
    if (!_plan) return;
    _bt_plan *plan = (_bt_plan *)_plan;
    DftiFreeDescriptor(&plan->desc);
    free(plan);
}

/* ─── No-op stubs for wisdom and thread helpers ────────────────────────── */

int __wrap_fftwf_export_wisdom_to_filename(const char *filename)
{
    (void)filename;
    return 1; /* FFTW_SUCCESS */
}

int __wrap_fftwf_import_wisdom_from_filename(const char *filename)
{
    (void)filename;
    return 0; /* FFTW_FAILURE — no wisdom, BART ignores this */
}

int __wrap_fftwf_init_threads(void)
{
    return 1; /* success */
}

void __wrap_fftwf_plan_with_nthreads(int n)
{
    (void)n; /* MKL DFTI uses its own threading */
}

void __wrap_fftwf_cleanup_threads(void)
{
    /* no-op */
}
