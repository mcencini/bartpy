/*
 * accelerate_fftw_shim.c -- intercept BART's and FINUFFT's fftwf_XXX/fftw_XXX
 * calls and route them through Apple's Accelerate vDSP framework.
 *
 * This shim replaces the real libfftw3f (which is not bundled with PyTorch on
 * macOS) with an implementation based on vDSP_DFT_zop_CreateSetup /
 * vDSP_DFT_Execute (single precision) and the analogous _D variants (double
 * precision).  It is used when BARTORCH_USE_ACCELERATE=ON (the default on macOS).
 *
 * Wrapped symbols (single precision -- BART + FINUFFT):
 *   fftwf_plan_guru64_dft         (BART's multi-dim planner)
 *   fftwf_plan_many_dft           (FINUFFT's planner)
 *   fftwf_execute_dft
 *   fftwf_destroy_plan
 *   fftwf_export_wisdom_to_filename  - no-op stubs
 *   fftwf_import_wisdom_from_filename
 *   fftwf_forget_wisdom
 *   fftwf_cleanup
 *   fftwf_init_threads
 *   fftwf_plan_with_nthreads
 *   fftwf_cleanup_threads
 *
 * Wrapped symbols (double precision -- FINUFFT's cleanup path):
 *   fftw_plan_many_dft
 *   fftw_execute_dft
 *   fftw_destroy_plan
 *   fftw_forget_wisdom  - no-op stubs
 *   fftw_cleanup
 *   fftw_init_threads
 *   fftw_plan_with_nthreads
 *   fftw_cleanup_threads
 *
 * Apple ld --wrap syntax (symbol names are prefixed with '_' in Mach-O):
 *   -Wl,-wrap,_fftwf_plan_guru64_dft  etc.
 *
 * Algorithm: gather -> deinterleave -> k-pass row-column FFT (ping-pong in
 * split-complex) -> reinterleave -> scatter.
 *
 * The contiguous working buffer uses Fortran-major (first-index-fastest) order.
 * A "gather" step copies the input from its original strided layout into this
 * buffer; a symmetric "scatter" step copies the result back.  The row-column
 * passes operate purely on the Fortran-major buffer using contig_strides.
 *
 * vDSP notes:
 *   - vDSP_DFT_Execute requires DISTINCT input and output pointers.  We always
 *     use ping-pong buffers for each 1D pass so this is never violated.
 *   - vDSP scaling: no normalisation applied (consistent with FFTW's convention).
 *   - Size support: vDSP_DFT_zop_CreateSetup accepts N = f * 2^n where
 *     f in {1, 3, 5, 15}.  For sizes outside this set the shim aborts with an
 *     informative message.  Typical MRI sizes (powers of 2, multiples of 3/5)
 *     are always supported.
 *
 * Thread safety: all working buffers are allocated per execute() call from the
 * heap.  Multiple OpenMP threads may call execute() concurrently on the same plan
 * without races.
 */

#include <Accelerate/Accelerate.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

/* --- Internal type stubs (we do not link the real fftw3) ------------------- */

typedef struct {
    ptrdiff_t n;   /* transform dimension size           */
    ptrdiff_t is;  /* input  stride (in complex elements) */
    ptrdiff_t os;  /* output stride (in complex elements) */
} _bt_iodim64;

typedef struct {
    long n;    /* howmany size   */
    long is;   /* input  stride  */
    long os;   /* output stride  */
} _bt_hmdim;

/* --- Plan type tag -------------------------------------------------------- */

#define PLAN_FLOAT  0
#define PLAN_DOUBLE 1

/* --- Unified plan struct (float or double, distinguished by tag) ----------- */

typedef struct {
    uint8_t  precision;    /* PLAN_FLOAT or PLAN_DOUBLE */

    /* Transform geometry */
    int      k;            /* number of transform dimensions */
    long    *lengths;      /* [k] sizes                     */
    long    *contig_strides; /* [k] strides in Fortran-major contig buffer */
    long    *istrides;     /* [k] original input strides    */
    long    *ostrides;     /* [k] original output strides   */
    size_t   nel;          /* product of lengths (elements per transform) */
    long     max_dim;      /* max(lengths[d])               */

    /* vDSP setups -- cast to vDSP_DFT_Setup* or vDSP_DFT_SetupD* via tag */
    void   **setups;       /* [k] opaque DFT setup pointers */

    /* Direction */
    vDSP_DFT_Direction dir; /* vDSP_DFT_FORWARD or vDSP_DFT_INVERSE */

    /* Howmany (batch) dimensions */
    int      l;            /* number of howmany dims */
    _bt_hmdim *hmdims;     /* [l]                    */
    long     ntrans;       /* product of hmdims[i].n */

    /* Contiguity flags for the initial gather / final scatter */
    int      need_gather;
    int      need_scatter;
} _bt_plan;

/* ===========================================================================
 * Helpers
 * ========================================================================= */

/*
 * Compute the input and output base offsets for howmany transform index `tr`.
 */
static void _compute_offsets(long tr, int l, const _bt_hmdim *hm,
                              long *in_off, long *out_off)
{
    long ioff = 0, ooff = 0;
    long rem = tr;
    for (int d = 0; d < l; d++) {
        long coord = rem % hm[d].n;
        rem /= hm[d].n;
        ioff += coord * hm[d].is;
        ooff += coord * hm[d].os;
    }
    *in_off  = ioff;
    *out_off = ooff;
}

/*
 * Gather from strided interleaved-complex input to Fortran-major contiguous
 * interleaved-complex buffer.
 *
 * The contiguous element (i0, i1, ..., i_{k-1}) maps to flat index
 *   contig[i0 + i1*n0 + i2*n0*n1 + ...] = src[i0*is0 + i1*is1 + ...].
 *
 * We enumerate all `nel` elements via a flat contig index and decompose.
 */
static void _gather_f(float _Complex *dst, const float _Complex *src,
                      int k, const long *lengths, const long *istrides, size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        long   offset = 0;
        for (int d = 0; d < k; d++) {
            long coord = (long)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * istrides[d];
        }
        dst[idx] = src[offset];
    }
}

static void _gather_d(double _Complex *dst, const double _Complex *src,
                      int k, const long *lengths, const long *istrides, size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        long   offset = 0;
        for (int d = 0; d < k; d++) {
            long coord = (long)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * istrides[d];
        }
        dst[idx] = src[offset];
    }
}

/*
 * Scatter from Fortran-major contiguous to strided interleaved-complex output.
 */
static void _scatter_f(float _Complex *dst, const float _Complex *src,
                       int k, const long *lengths, const long *ostrides, size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        long   offset = 0;
        for (int d = 0; d < k; d++) {
            long coord = (long)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * ostrides[d];
        }
        dst[offset] = src[idx];
    }
}

static void _scatter_d(double _Complex *dst, const double _Complex *src,
                       int k, const long *lengths, const long *ostrides, size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        long   offset = 0;
        for (int d = 0; d < k; d++) {
            long coord = (long)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * ostrides[d];
        }
        dst[offset] = src[idx];
    }
}

/*
 * Compute the base index (when i_d = 0) for fiber `fo` of dimension `skip_d`
 * in the Fortran-major contig buffer.
 *
 * Fibers of dimension d are indexed by all other index combinations.
 * We enumerate them with a flat other-index `fo` and decompose.
 */
static long _fiber_base(long fo, int skip_d, int k,
                        const long *lengths, const long *strides)
{
    long base = 0;
    long rem  = fo;
    for (int j = 0; j < k; j++) {
        if (j == skip_d) continue;
        long coord = rem % lengths[j];
        rem /= lengths[j];
        base += coord * strides[j];
    }
    return base;
}

/* --- Float row-column FFT on Fortran-major split-complex buffer ----------- */
/*
 * Apply k 1-D FFT passes (one per dimension) on a split-complex buffer of
 * `nel` elements.  The buffer is updated in-place via ping-pong between
 * `(a_r, a_i)` and `(b_r, b_i)`.
 *
 * On entry:  data is in (a_r, a_i).
 * On exit:   data is in (*src_r, *src_i) -- the caller swaps back if needed.
 * After k passes, if k is odd the result is in (b_r, b_i); if even it's in
 * (a_r, a_i).  We return the active buffer via the src_r/src_i pointers.
 *
 * tmp_in_r/i and tmp_out_r/i are scratch arrays of length plan->max_dim each.
 */
static void _rowcol_fft_f(const _bt_plan *plan,
                           float *a_r, float *a_i,
                           float *b_r, float *b_i,
                           float *tmp_in_r,  float *tmp_in_i,
                           float *tmp_out_r, float *tmp_out_i,
                           float **src_r, float **src_i)
{
    float *cur_r = a_r, *cur_i = a_i;
    float *nxt_r = b_r, *nxt_i = b_i;

    for (int d = 0; d < plan->k; d++) {
        long nd        = plan->lengths[d];
        long sd        = plan->contig_strides[d];
        long n_fibers  = (long)(plan->nel / (size_t)nd);

        vDSP_DFT_Setup setup = (vDSP_DFT_Setup)plan->setups[d];

        for (long fo = 0; fo < n_fibers; fo++) {
            long base = _fiber_base(fo, d, plan->k,
                                    plan->lengths, plan->contig_strides);

            /* Gather fiber from cur buffer into tmp_in */
            for (long i = 0; i < nd; i++) {
                tmp_in_r[i] = cur_r[base + i * sd];
                tmp_in_i[i] = cur_i[base + i * sd];
            }

            /* FFT via vDSP (tmp_in -> tmp_out, distinct buffers required) */
            vDSP_DFT_Execute(setup, tmp_in_r, tmp_in_i, tmp_out_r, tmp_out_i);

            /* Scatter fiber from tmp_out into nxt buffer */
            for (long i = 0; i < nd; i++) {
                nxt_r[base + i * sd] = tmp_out_r[i];
                nxt_i[base + i * sd] = tmp_out_i[i];
            }
        }

        /* Ping-pong: nxt becomes cur for next pass */
        float *swap_r = cur_r; cur_r = nxt_r; nxt_r = swap_r;
        float *swap_i = cur_i; cur_i = nxt_i; nxt_i = swap_i;
    }

    *src_r = cur_r;
    *src_i = cur_i;
}

/* --- Double row-column FFT ------------------------------------------------ */

static void _rowcol_fft_d(const _bt_plan *plan,
                           double *a_r, double *a_i,
                           double *b_r, double *b_i,
                           double *tmp_in_r,  double *tmp_in_i,
                           double *tmp_out_r, double *tmp_out_i,
                           double **src_r, double **src_i)
{
    double *cur_r = a_r, *cur_i = a_i;
    double *nxt_r = b_r, *nxt_i = b_i;

    for (int d = 0; d < plan->k; d++) {
        long nd        = plan->lengths[d];
        long sd        = plan->contig_strides[d];
        long n_fibers  = (long)(plan->nel / (size_t)nd);

        vDSP_DFT_SetupD setup = (vDSP_DFT_SetupD)plan->setups[d];

        for (long fo = 0; fo < n_fibers; fo++) {
            long base = _fiber_base(fo, d, plan->k,
                                    plan->lengths, plan->contig_strides);

            for (long i = 0; i < nd; i++) {
                tmp_in_r[i] = cur_r[base + i * sd];
                tmp_in_i[i] = cur_i[base + i * sd];
            }

            vDSP_DFT_ExecuteD(setup, tmp_in_r, tmp_in_i, tmp_out_r, tmp_out_i);

            for (long i = 0; i < nd; i++) {
                nxt_r[base + i * sd] = tmp_out_r[i];
                nxt_i[base + i * sd] = tmp_out_i[i];
            }
        }

        double *swap_r = cur_r; cur_r = nxt_r; nxt_r = swap_r;
        double *swap_i = cur_i; cur_i = nxt_i; nxt_i = swap_i;
    }

    *src_r = cur_r;
    *src_i = cur_i;
}

/* ===========================================================================
 * Plan creation helpers
 * ========================================================================= */

/*
 * Allocate and initialise the common plan fields from the guru64-style
 * dimension arrays.  On success returns a heap-allocated _bt_plan.
 * On failure (e.g. vDSP unsupported size) prints to stderr and returns NULL.
 */
static _bt_plan *_plan_create_common(uint8_t precision,
                                      int k,
                                      const long *lengths_in,
                                      const long *istrides_in,
                                      const long *ostrides_in,
                                      int l,
                                      const _bt_hmdim *hmdims_in,
                                      vDSP_DFT_Direction dir)
{
    if (k <= 0) return NULL;

    /* Allocate plan */
    _bt_plan *plan = (_bt_plan *)calloc(1, sizeof(_bt_plan));
    if (!plan) return NULL;

    plan->precision = precision;
    plan->k         = k;
    plan->dir       = dir;
    plan->l         = l;

    /* Copy lengths, strides */
    plan->lengths        = (long *)malloc((size_t)k * sizeof(long));
    plan->contig_strides = (long *)malloc((size_t)k * sizeof(long));
    plan->istrides       = (long *)malloc((size_t)k * sizeof(long));
    plan->ostrides       = (long *)malloc((size_t)k * sizeof(long));
    plan->setups         = (void **)calloc((size_t)k, sizeof(void *));

    if (!plan->lengths || !plan->contig_strides || !plan->istrides ||
        !plan->ostrides || !plan->setups) goto fail;

    /* Compute nel and max_dim */
    size_t nel    = 1;
    long   maxdim = 0;
    for (int d = 0; d < k; d++) {
        plan->lengths[d]  = lengths_in[d];
        plan->istrides[d] = istrides_in[d];
        plan->ostrides[d] = ostrides_in[d];
        nel *= (size_t)lengths_in[d];
        if (lengths_in[d] > maxdim) maxdim = lengths_in[d];
    }
    plan->nel     = nel;
    plan->max_dim = maxdim;

    /* Fortran-major (first-index-fastest) contig strides */
    long stride = 1;
    for (int d = 0; d < k; d++) {
        plan->contig_strides[d] = stride;
        stride *= plan->lengths[d];
    }

    /* need_gather / need_scatter */
    plan->need_gather  = 0;
    plan->need_scatter = 0;
    for (int d = 0; d < k; d++) {
        if (plan->istrides[d] != plan->contig_strides[d]) plan->need_gather  = 1;
        if (plan->ostrides[d] != plan->contig_strides[d]) plan->need_scatter = 1;
    }

    /* Create per-dimension vDSP DFT setups */
    for (int d = 0; d < k; d++) {
        vDSP_Length N = (vDSP_Length)plan->lengths[d];
        if (precision == PLAN_FLOAT) {
            vDSP_DFT_Setup s = vDSP_DFT_zop_CreateSetup(NULL, N, dir);
            if (!s) {
                fprintf(stderr,
                    "bartorch accelerate_fftw_shim: vDSP_DFT_zop_CreateSetup "
                    "failed for N=%lu (must be f*2^n, fin{1,3,5,15}).\n",
                    (unsigned long)N);
                goto fail;
            }
            plan->setups[d] = (void *)s;
        } else {
            vDSP_DFT_SetupD s = vDSP_DFT_zop_CreateSetupD(NULL, N, dir);
            if (!s) {
                fprintf(stderr,
                    "bartorch accelerate_fftw_shim: vDSP_DFT_zop_CreateSetupD "
                    "failed for N=%lu (must be f*2^n, fin{1,3,5,15}).\n",
                    (unsigned long)N);
                goto fail;
            }
            plan->setups[d] = (void *)s;
        }
    }

    /* Copy howmany dims */
    if (l > 0) {
        plan->hmdims = (_bt_hmdim *)malloc((size_t)l * sizeof(_bt_hmdim));
        if (!plan->hmdims) goto fail;
        long ntrans = 1;
        for (int i = 0; i < l; i++) {
            plan->hmdims[i] = hmdims_in[i];
            ntrans *= hmdims_in[i].n;
        }
        plan->ntrans = ntrans;
    } else {
        plan->hmdims = NULL;
        plan->ntrans = 1;
    }

    return plan;

fail:
    /* Clean up partial allocation */
    if (plan->setups) {
        for (int d = 0; d < k; d++) {
            if (!plan->setups[d]) continue;
            if (precision == PLAN_FLOAT)
                vDSP_DFT_DestroySetup((vDSP_DFT_Setup)plan->setups[d]);
            else
                vDSP_DFT_DestroySetupD((vDSP_DFT_SetupD)plan->setups[d]);
        }
        free(plan->setups);
    }
    free(plan->lengths);
    free(plan->contig_strides);
    free(plan->istrides);
    free(plan->ostrides);
    free(plan->hmdims);
    free(plan);
    return NULL;
}

/* ===========================================================================
 * Execute helpers
 * ========================================================================= */

static void _execute_f(_bt_plan *plan, float _Complex *in, float _Complex *out)
{
    int inplace = (in == out);
    size_t nel  = plan->nel;

    /*
     * In-place safety: pre-copy the full input span so that later scatter
     * calls (which write to `out`) cannot corrupt unread input data.
     */
    float _Complex *in_copy = NULL;
    float _Complex *in_safe = in;
    if (inplace && plan->ntrans > 1) {
        long max_hm_off = 0;
        for (int d = 0; d < plan->l; d++) {
            if (plan->hmdims[d].n > 1)
                max_hm_off += (plan->hmdims[d].n - 1)
                              * (long)(plan->hmdims[d].is > plan->hmdims[d].os
                                       ? plan->hmdims[d].is : plan->hmdims[d].os);
        }
        long max_xf_off = 0;
        for (int d = 0; d < plan->k; d++) {
            long s = plan->istrides[d] > plan->ostrides[d]
                     ? plan->istrides[d] : plan->ostrides[d];
            max_xf_off += (plan->lengths[d] - 1) * s;
        }
        size_t total_span = (size_t)(max_hm_off + max_xf_off) + 1;
        in_copy = (float _Complex *)malloc(total_span * sizeof(float _Complex));
        if (!in_copy) {
            fprintf(stderr, "bartorch accelerate_fftw_shim: alloc failed "
                    "(in_copy float, %zu elems)\n", total_span);
            abort();
        }
        memcpy(in_copy, in, total_span * sizeof(float _Complex));
        in_safe = in_copy;
    }

    /* Working split-complex buffers (two ping-pong pairs) */
    float *a_r = (float *)malloc(nel * sizeof(float));
    float *a_i = (float *)malloc(nel * sizeof(float));
    float *b_r = (float *)malloc(nel * sizeof(float));
    float *b_i = (float *)malloc(nel * sizeof(float));
    /* Temp gather/scatter scratch (size = max single-dim length) */
    float *tin_r  = (float *)malloc((size_t)plan->max_dim * sizeof(float));
    float *tin_i  = (float *)malloc((size_t)plan->max_dim * sizeof(float));
    float *tout_r = (float *)malloc((size_t)plan->max_dim * sizeof(float));
    float *tout_i = (float *)malloc((size_t)plan->max_dim * sizeof(float));

    if (!a_r || !a_i || !b_r || !b_i || !tin_r || !tin_i || !tout_r || !tout_i) {
        fprintf(stderr, "bartorch accelerate_fftw_shim: alloc failed "
                "(working buffers float, nel=%zu)\n", nel);
        abort();
    }

    /* Interleaved-complex -> DSPSplitComplex conversion helpers */
    DSPSplitComplex sp_a = { a_r, a_i };

    for (long tr = 0; tr < plan->ntrans; tr++) {
        long in_off = 0, out_off = 0;
        _compute_offsets(tr, plan->l, plan->hmdims, &in_off, &out_off);

        const float _Complex *in_ptr  = in_safe + in_off;
        float _Complex       *out_ptr = out      + out_off;

        /* Step 1: gather strided input to contiguous interleaved temp, then
         * deinterleave to split-complex working buffer `a`.               */
        if (plan->need_gather) {
            float _Complex *ctig = (float _Complex *)malloc(nel * sizeof(float _Complex));
            if (!ctig) { fprintf(stderr, "bartorch: alloc failed (ctig)\n"); abort(); }
            _gather_f(ctig, in_ptr, plan->k, plan->lengths, plan->istrides, nel);
            vDSP_ctoz((const DSPComplex *)ctig, 2, &sp_a, 1, nel);
            free(ctig);
        } else {
            vDSP_ctoz((const DSPComplex *)in_ptr, 2, &sp_a, 1, nel);
        }

        /* Step 2: row-column k-D FFT via ping-pong split-complex buffers */
        float *res_r, *res_i;
        _rowcol_fft_f(plan, a_r, a_i, b_r, b_i, tin_r, tin_i, tout_r, tout_i,
                      &res_r, &res_i);

        /* Step 3: reinterleave result and scatter to strided output */
        DSPSplitComplex sp_res = { res_r, res_i };
        if (plan->need_scatter) {
            float _Complex *ctig = (float _Complex *)malloc(nel * sizeof(float _Complex));
            if (!ctig) { fprintf(stderr, "bartorch: alloc failed (ctig out)\n"); abort(); }
            vDSP_ztoc(&sp_res, 1, (DSPComplex *)ctig, 2, nel);
            _scatter_f(out_ptr, ctig, plan->k, plan->lengths, plan->ostrides, nel);
            free(ctig);
        } else {
            vDSP_ztoc(&sp_res, 1, (DSPComplex *)out_ptr, 2, nel);
        }
    }

    free(a_r); free(a_i); free(b_r); free(b_i);
    free(tin_r); free(tin_i); free(tout_r); free(tout_i);
    free(in_copy);
}

static void _execute_d(_bt_plan *plan, double _Complex *in, double _Complex *out)
{
    int inplace = (in == out);
    size_t nel  = plan->nel;

    double _Complex *in_copy = NULL;
    double _Complex *in_safe = in;
    if (inplace && plan->ntrans > 1) {
        long max_hm_off = 0;
        for (int d = 0; d < plan->l; d++) {
            if (plan->hmdims[d].n > 1)
                max_hm_off += (plan->hmdims[d].n - 1)
                              * (long)(plan->hmdims[d].is > plan->hmdims[d].os
                                       ? plan->hmdims[d].is : plan->hmdims[d].os);
        }
        long max_xf_off = 0;
        for (int d = 0; d < plan->k; d++) {
            long s = plan->istrides[d] > plan->ostrides[d]
                     ? plan->istrides[d] : plan->ostrides[d];
            max_xf_off += (plan->lengths[d] - 1) * s;
        }
        size_t total_span = (size_t)(max_hm_off + max_xf_off) + 1;
        in_copy = (double _Complex *)malloc(total_span * sizeof(double _Complex));
        if (!in_copy) {
            fprintf(stderr, "bartorch accelerate_fftw_shim: alloc failed "
                    "(in_copy double, %zu elems)\n", total_span);
            abort();
        }
        memcpy(in_copy, in, total_span * sizeof(double _Complex));
        in_safe = in_copy;
    }

    double *a_r = (double *)malloc(nel * sizeof(double));
    double *a_i = (double *)malloc(nel * sizeof(double));
    double *b_r = (double *)malloc(nel * sizeof(double));
    double *b_i = (double *)malloc(nel * sizeof(double));
    double *tin_r  = (double *)malloc((size_t)plan->max_dim * sizeof(double));
    double *tin_i  = (double *)malloc((size_t)plan->max_dim * sizeof(double));
    double *tout_r = (double *)malloc((size_t)plan->max_dim * sizeof(double));
    double *tout_i = (double *)malloc((size_t)plan->max_dim * sizeof(double));

    if (!a_r || !a_i || !b_r || !b_i || !tin_r || !tin_i || !tout_r || !tout_i) {
        fprintf(stderr, "bartorch accelerate_fftw_shim: alloc failed "
                "(working buffers double, nel=%zu)\n", nel);
        abort();
    }

    DSPDoubleSplitComplex sp_a = { a_r, a_i };

    for (long tr = 0; tr < plan->ntrans; tr++) {
        long in_off = 0, out_off = 0;
        _compute_offsets(tr, plan->l, plan->hmdims, &in_off, &out_off);

        const double _Complex *in_ptr  = in_safe + in_off;
        double _Complex       *out_ptr = out      + out_off;

        if (plan->need_gather) {
            double _Complex *ctig = (double _Complex *)malloc(nel * sizeof(double _Complex));
            if (!ctig) { fprintf(stderr, "bartorch: alloc failed (ctig d)\n"); abort(); }
            _gather_d(ctig, in_ptr, plan->k, plan->lengths, plan->istrides, nel);
            vDSP_ctozD((const DSPDoubleComplex *)ctig, 2, &sp_a, 1, nel);
            free(ctig);
        } else {
            vDSP_ctozD((const DSPDoubleComplex *)in_ptr, 2, &sp_a, 1, nel);
        }

        double *res_r, *res_i;
        _rowcol_fft_d(plan, a_r, a_i, b_r, b_i, tin_r, tin_i, tout_r, tout_i,
                      &res_r, &res_i);

        DSPDoubleSplitComplex sp_res = { res_r, res_i };
        if (plan->need_scatter) {
            double _Complex *ctig = (double _Complex *)malloc(nel * sizeof(double _Complex));
            if (!ctig) { fprintf(stderr, "bartorch: alloc failed (ctig out d)\n"); abort(); }
            vDSP_ztocD(&sp_res, 1, (DSPDoubleComplex *)ctig, 2, nel);
            _scatter_d(out_ptr, ctig, plan->k, plan->lengths, plan->ostrides, nel);
            free(ctig);
        } else {
            vDSP_ztocD(&sp_res, 1, (DSPDoubleComplex *)out_ptr, 2, nel);
        }
    }

    free(a_r); free(a_i); free(b_r); free(b_i);
    free(tin_r); free(tin_i); free(tout_r); free(tout_i);
    free(in_copy);
}

/* ===========================================================================
 * Destroy helper
 * ========================================================================= */

static void _plan_destroy(_bt_plan *plan)
{
    if (!plan) return;
    for (int d = 0; d < plan->k; d++) {
        if (!plan->setups[d]) continue;
        if (plan->precision == PLAN_FLOAT)
            vDSP_DFT_DestroySetup((vDSP_DFT_Setup)plan->setups[d]);
        else
            vDSP_DFT_DestroySetupD((vDSP_DFT_SetupD)plan->setups[d]);
    }
    free(plan->setups);
    free(plan->lengths);
    free(plan->contig_strides);
    free(plan->istrides);
    free(plan->ostrides);
    free(plan->hmdims);
    free(plan);
}

/* ===========================================================================
 * Single-precision (fftwf_*) wrapped functions
 * ========================================================================= */

/*
 * __wrap_fftwf_plan_guru64_dft -- intercepted BART planner.
 *
 * Converts FFTW's guru64 iodim arrays to our internal plan format.
 * The `in` and `out` arguments to FFTW's planner are only used for alignment
 * hinting; we ignore them (we create plans independent of data buffers).
 */
void *__wrap_fftwf_plan_guru64_dft(
    int                k,
    const _bt_iodim64 *dims,
    int                l,
    const _bt_iodim64 *hmdims,
    float _Complex    *in,
    float _Complex    *out,
    int                sign,
    unsigned           flags)
{
    (void)in; (void)out; (void)flags;

    if (k <= 0) return NULL;

    /* Extract per-dimension info */
    long lengths[k], istrides[k], ostrides[k];
    for (int d = 0; d < k; d++) {
        lengths[d]  = (long)dims[d].n;
        istrides[d] = (long)dims[d].is;
        ostrides[d] = (long)dims[d].os;
    }

    _bt_hmdim hm[l > 0 ? l : 1];
    for (int i = 0; i < l; i++) {
        hm[i].n  = (long)hmdims[i].n;
        hm[i].is = (long)hmdims[i].is;
        hm[i].os = (long)hmdims[i].os;
    }

    vDSP_DFT_Direction dir = (sign == -1) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;

    return (void *)_plan_create_common(PLAN_FLOAT, k,
                                       lengths, istrides, ostrides,
                                       l, hm, dir);
}

/*
 * __wrap_fftwf_plan_many_dft -- intercepted FINUFFT planner.
 *
 * Converts plan_many's "C-order many" layout to the internal format:
 *   - transform dims stored in order (d=0 is outermost -> smallest contig stride
 *     in C-order; we always gather to Fortran-major so order doesn't matter)
 *   - howmany represented as a single _bt_hmdim with n=howmany, is=idist, os=odist
 *
 * When inembed==NULL and onembed==NULL (FINUFFT's usage), istrides and ostrides
 * for the transform dims follow the standard C-row-major pattern:
 *   istrides[d] = istride * product(n[d+1..rank-1])
 *   (innermost dim has stride istride, outermost has stride product * istride)
 */
void *__wrap_fftwf_plan_many_dft(
    int             rank,
    const int      *n,
    int             howmany,
    float _Complex *in,
    const int      *inembed,
    int             istride,
    int             idist,
    float _Complex *out,
    const int      *onembed,
    int             ostride,
    int             odist,
    int             sign,
    unsigned        flags)
{
    (void)in; (void)out; (void)flags;
    (void)inembed; (void)onembed; /* only NULL case supported */

    if (rank <= 0) return NULL;

    /* Compute C-order strides for transform dims */
    long lengths[rank], istrides_a[rank], ostrides_a[rank];
    long istr = (long)istride, ostr = (long)ostride;
    /* istr for innermost (last) dim; outer dims multiply */
    long isuffix = (long)istride;
    long osuffix = (long)ostride;
    for (int d = rank - 1; d >= 0; d--) {
        lengths[d]    = (long)n[d];
        istrides_a[d] = isuffix;
        ostrides_a[d] = osuffix;
        if (d > 0) {
            isuffix *= (long)n[d];
            osuffix *= (long)n[d];
        }
    }
    (void)istr; (void)ostr;

    _bt_hmdim hm[1];
    hm[0].n  = (long)howmany;
    hm[0].is = (long)idist;
    hm[0].os = (long)odist;
    /* When howmany == 1 there is no howmany loop; suppress by setting l=0. */
    int l = (howmany > 1) ? 1 : 0;

    vDSP_DFT_Direction dir = (sign == -1) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;

    return (void *)_plan_create_common(PLAN_FLOAT, rank,
                                       lengths, istrides_a, ostrides_a,
                                       l, hm, dir);
}

void __wrap_fftwf_execute_dft(void *_plan, float _Complex *in, float _Complex *out)
{
    _bt_plan *plan = (_bt_plan *)_plan;
    assert(plan && plan->precision == PLAN_FLOAT);
    _execute_f(plan, in, out);
}

void __wrap_fftwf_destroy_plan(void *_plan)
{
    _plan_destroy((_bt_plan *)_plan);
}

/* --- Single-precision no-op stubs ---------------------------------------- */

int __wrap_fftwf_export_wisdom_to_filename(const char *filename)
{
    (void)filename;
    return 1; /* FFTW_SUCCESS */
}

int __wrap_fftwf_import_wisdom_from_filename(const char *filename)
{
    (void)filename;
    return 0; /* FFTW_FAILURE -- no wisdom; BART ignores this */
}

void __wrap_fftwf_forget_wisdom(void) { /* no-op */ }
void __wrap_fftwf_cleanup(void)       { /* no-op */ }

int __wrap_fftwf_init_threads(void)
{
    return 1; /* success */
}

void __wrap_fftwf_plan_with_nthreads(int n)
{
    (void)n; /* Accelerate uses its own threading model */
}

void __wrap_fftwf_cleanup_threads(void) { /* no-op */ }

/* ===========================================================================
 * Double-precision (fftw_*) wrapped functions
 * ========================================================================= */

void *__wrap_fftw_plan_many_dft(
    int              rank,
    const int       *n,
    int              howmany,
    double _Complex *in,
    const int       *inembed,
    int              istride,
    int              idist,
    double _Complex *out,
    const int       *onembed,
    int              ostride,
    int              odist,
    int              sign,
    unsigned         flags)
{
    (void)in; (void)out; (void)flags;
    (void)inembed; (void)onembed;

    if (rank <= 0) return NULL;

    long lengths[rank], istrides_a[rank], ostrides_a[rank];
    long isuffix = (long)istride;
    long osuffix = (long)ostride;
    for (int d = rank - 1; d >= 0; d--) {
        lengths[d]    = (long)n[d];
        istrides_a[d] = isuffix;
        ostrides_a[d] = osuffix;
        if (d > 0) {
            isuffix *= (long)n[d];
            osuffix *= (long)n[d];
        }
    }

    _bt_hmdim hm[1];
    hm[0].n  = (long)howmany;
    hm[0].is = (long)idist;
    hm[0].os = (long)odist;
    int l = (howmany > 1) ? 1 : 0;

    vDSP_DFT_Direction dir = (sign == -1) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;

    return (void *)_plan_create_common(PLAN_DOUBLE, rank,
                                       lengths, istrides_a, ostrides_a,
                                       l, hm, dir);
}

void __wrap_fftw_execute_dft(void *_plan, double _Complex *in, double _Complex *out)
{
    _bt_plan *plan = (_bt_plan *)_plan;
    assert(plan && plan->precision == PLAN_DOUBLE);
    _execute_d(plan, in, out);
}

void __wrap_fftw_destroy_plan(void *_plan)
{
    _plan_destroy((_bt_plan *)_plan);
}

/* --- Double-precision no-op stubs ---------------------------------------- */

void __wrap_fftw_forget_wisdom(void)       { /* no-op */ }
void __wrap_fftw_cleanup(void)             { /* no-op */ }

int  __wrap_fftw_init_threads(void)        { return 1; }
void __wrap_fftw_plan_with_nthreads(int n) { (void)n; }
void __wrap_fftw_cleanup_threads(void)     { /* no-op */ }
