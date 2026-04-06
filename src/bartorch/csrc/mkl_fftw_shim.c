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
 * Strides convention
 * ------------------
 * FFTW guru64: is/os are element strides (complex float = 1 element).
 *   dims[0] = innermost transform dim (stride = 1 for contiguous arrays)
 *   dims[k-1] = outermost transform dim (largest stride)
 *
 * DFTI strides: strides[0] = first-element offset, strides[j] = element
 *               stride of j-th dimension (1-indexed, outermost first).
 *
 * Key difference: DFTI's strides[1] is the OUTERMOST dimension, while
 * FFTW's dims[0] is the INNERMOST dimension.  Since MKL DFTI requires
 * strides[j] >= lengths[j+1] * strides[j+1] (no overlapping rows), we
 * keep the FFTW/BART ordering (strides[0] maps to BART's innermost dim)
 * which satisfies this constraint for BART's standard Fortran-order layout:
 *   lengths = {dims[0].n, dims[1].n, ...}  (innermost first)
 *   strides = {0, dims[0].is, dims[1].is, ...}
 *   → strides[j+1] = dims[j].is = prod(dims[0..j-1].n) >= dims[j].n * strides[j]
 *     is TRUE for standard contiguous Fortran layout.
 *
 * Batched transforms (ntrans > 1)
 * --------------------------------
 * MKL DFTI's DFTI_NUMBER_OF_TRANSFORMS with non-rectangular strides fails the
 * consistency check (DftiCommitDescriptor returns
 * DFTI_INCONSISTENT_CONFIGURATION).  To avoid this, we store ntrans, idist,
 * odist in the plan and execute individual transforms in a loop at call time.
 * Each individual transform uses a single-transform DFTI descriptor (ntrans=1
 * in DFTI terms), so the consistency check always passes.
 *
 * For a batched transform with n transforms:
 *   transform i: input  at in  + i * idist
 *                output at out + i * odist
 *
 * In-place transforms (in == out)
 * ---------------------------------
 * MKL DFTI with DFTI_NOT_INPLACE does NOT permit aliased input/output.
 * BART's fftmod calls fftwf_execute_dft(plan, ptr, ptr) (in == out).
 * We handle this by copying the entire input to a temp buffer first,
 * then computing out-of-place from the temp to the original buffer.
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
    size_t                  span;     /* element count covering ONE input transform
                                       * (max_offset + 1 over transform dims only).
                                       * Used to size temp buffer for in-place execute. */
    MKL_LONG                ntrans;   /* number of transforms (batch size) */
    MKL_LONG                idist;    /* element stride between input transforms */
    MKL_LONG                odist;    /* element stride between output transforms */
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
    /*
     * ntrans = product of all batch dim sizes.
     * idist/odist = element stride between consecutive transforms.
     *
     * For BART's standard Fortran-order layout, transforms are contiguous
     * and separated by the stride of the first batch dim with n > 1.
     * If all batch dims have n == 1 (single transform), ntrans = 1.
     */
    MKL_LONG ntrans = 1;
    for (int i = 0; i < l; i++)
        ntrans *= (MKL_LONG)hmdims[i].n;

    /*
     * Inter-transform distance: use the .is/.os of the first batch dimension
     * that has n > 1.  For all-trivial batch dims (ntrans == 1) the distance
     * is irrelevant; we leave it as 0 (will not be used).
     */
    MKL_LONG idist = 0;
    MKL_LONG odist = 0;
    if (ntrans > 1) {
        for (int i = 0; i < l; i++) {
            if (hmdims[i].n > 1) {
                idist = (MKL_LONG)hmdims[i].is;
                odist = (MKL_LONG)hmdims[i].os;
                break;
            }
        }
    }

    /* ── Build dimension arrays for DFTI ───────────────────────────────── */
    /*
     * DFTI strides: strides[0] = first-element offset (always 0),
     *   strides[j+1] = element stride of dims[j] (innermost first).
     *
     * We keep BART/FFTW3's dim ordering (dims[0] = innermost = smallest
     * stride).  MKL DFTI accepts this for single-transform descriptors
     * (DFTI_NUMBER_OF_TRANSFORMS = 1, the default), because the consistency
     * check only becomes strict when batch parameters are set.  Since we
     * handle batching ourselves in __wrap_fftwf_execute_dft, the descriptor
     * always describes exactly one transform and the check always passes.
     */
    MKL_LONG *lengths    = (MKL_LONG *)malloc((size_t)k * sizeof(MKL_LONG));
    MKL_LONG *istrides   = (MKL_LONG *)malloc((size_t)(k + 1) * sizeof(MKL_LONG));
    MKL_LONG *ostrides   = (MKL_LONG *)malloc((size_t)(k + 1) * sizeof(MKL_LONG));

    if (!lengths || !istrides || !ostrides) {
        free(lengths); free(istrides); free(ostrides);
        return NULL;
    }

    istrides[0] = 0;
    ostrides[0] = 0;
    for (int i = 0; i < k; i++) {
        lengths[i]       = (MKL_LONG)dims[i].n;
        istrides[i + 1]  = (MKL_LONG)dims[i].is;
        ostrides[i + 1]  = (MKL_LONG)dims[i].os;
    }

    /* ── Create DFTI descriptor (single transform — no batch params) ───── */
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

    /* Out-of-place transform (we copy to temp when in == out at execute time) */
    status = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status != DFTI_NO_ERROR) { DftiFreeDescriptor(&desc); return NULL; }

    /* Commit */
    status = DftiCommitDescriptor(desc);
    if (status != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* ── Compute span of ONE transform (for in-place temp buffer) ──────── */
    /*
     * span = max element offset reachable within a single transform + 1.
     * span = 1 + sum_d((dims[d].n - 1) * dims[d].is)
     *
     * This covers the input buffer of one transform.  When in == out (global
     * in-place), we copy span elements before executing, so this is the
     * minimum temp buffer size per transform.
     */
    MKL_LONG max_off = 0;
    for (int i = 0; i < k; i++)
        max_off += (MKL_LONG)(dims[i].n - 1) * dims[i].is;
    /* Guard against negative max_off (degenerate all-size-1 plans). */
    size_t span = (max_off < 0) ? 1 : (size_t)(max_off + 1);

    /* ── Wrap in our plan struct ────────────────────────────────────────── */
    _bt_plan *plan = (_bt_plan *)malloc(sizeof(_bt_plan));
    if (!plan) {
        DftiFreeDescriptor(&desc);
        return NULL;
    }
    plan->desc   = desc;
    plan->sign   = sign;
    plan->span   = span;
    plan->ntrans = ntrans;
    plan->idist  = idist;
    plan->odist  = odist;
    return (void *)plan;
}

/* ─── Execute shim ─────────────────────────────────────────────────────── */

void __wrap_fftwf_execute_dft(void *_plan, float _Complex *in, float _Complex *out)
{
    _bt_plan *plan = (_bt_plan *)_plan;
    assert(plan && plan->desc);

    MKL_LONG status = DFTI_NO_ERROR;

    /*
     * MKL DFTI with DFTI_NOT_INPLACE does NOT permit aliased input/output.
     * BART's fftmod frequently calls fftwf_execute_dft(plan, ptr, ptr).
     *
     * We handle this globally: if in == out, copy the entire multi-transform
     * input to a temporary buffer (size = ntrans * idist or span for ntrans=1),
     * then execute each sub-transform out-of-place from temp into the original.
     *
     * For ntrans > 1 we loop over individual transforms explicitly, advancing
     * the input/output pointers by idist/odist.  This avoids the DFTI batch
     * API (DFTI_NUMBER_OF_TRANSFORMS + DFTI_INPUT_DISTANCE) which fails the
     * consistency check for BART's non-rectangular Fortran-order strides.
     */

    if (in == out) {
        /*
         * Compute total size needed to buffer all transforms.
         * For ntrans==1: size = span.
         * For ntrans>1:  the last transform ends at (ntrans-1)*idist + span,
         *               but we only need to buffer the full input range.
         * Use: total_span = (ntrans - 1) * idist + span  (last transform's end)
         */
        size_t total;
        if (plan->ntrans <= 1 || plan->idist <= 0) {
            total = plan->span;
        } else {
            total = (size_t)((plan->ntrans - 1) * plan->idist) + plan->span;
        }

        float _Complex *tmp = (float _Complex *)malloc(total * sizeof(float _Complex));
        if (!tmp) {
            fprintf(stderr,
                    "bartorch mkl_fftw_shim: failed to allocate %zu-element temp buffer "
                    "for in-place FFT\n", total);
            abort();
        }
        memcpy(tmp, in, total * sizeof(float _Complex));

        for (MKL_LONG tr = 0; tr < plan->ntrans; tr++) {
            float _Complex *tmp_tr = tmp + tr * plan->idist;
            float _Complex *out_tr = out + tr * plan->odist;
            if (plan->sign == -1)
                status = DftiComputeForward(plan->desc, (void *)tmp_tr, (void *)out_tr);
            else
                status = DftiComputeBackward(plan->desc, (void *)tmp_tr, (void *)out_tr);
            if (status != DFTI_NO_ERROR) break;
        }
        free(tmp);
    } else {
        /* Standard out-of-place path: loop over individual transforms. */
        for (MKL_LONG tr = 0; tr < plan->ntrans; tr++) {
            float _Complex *in_tr  = in  + tr * plan->idist;
            float _Complex *out_tr = out + tr * plan->odist;
            if (plan->sign == -1)
                status = DftiComputeForward(plan->desc, (void *)in_tr, (void *)out_tr);
            else
                status = DftiComputeBackward(plan->desc, (void *)in_tr, (void *)out_tr);
            if (status != DFTI_NO_ERROR) break;
        }
    }

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
