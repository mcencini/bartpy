/*
 * fftw3_stub/fftw3.h -- minimal FFTW3 declarations for the Accelerate build path.
 *
 * When bartorch is built on macOS with Apple Accelerate (BARTORCH_USE_ACCELERATE=ON),
 * we do NOT link against a real libfftw3f.  Instead, all fftwf_XXX / fftw_XXX symbols are
 * intercepted by accelerate_fftw_shim.c via the GNU/Apple ld --wrap mechanism and
 * implemented on top of vDSP.
 *
 * This header provides just enough declarations so that BART's fft_plan.c and
 * FINUFFT's fft.cpp compile cleanly without the real fftw3.h.
 *
 * It is intentionally NOT a full FFTW3 compatibility header -- only the types and
 * functions actually referenced by BART and FINUFFT are declared here.
 */

#pragma once

#include <stddef.h>   /* ptrdiff_t, size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* --- Complex types -------------------------------------------------------- */

/* Single-precision interleaved complex: [real, imag] */
typedef float  fftwf_complex[2];
/* Double-precision interleaved complex: [real, imag] */
typedef double fftw_complex[2];

/* --- Opaque plan types ---------------------------------------------------- */

typedef void *fftwf_plan;
typedef void *fftw_plan;

/* --- Guru 64-bit dimension descriptor ------------------------------------ */

typedef struct {
    ptrdiff_t n;   /* transform dimension size           */
    ptrdiff_t is;  /* input  stride (in complex elements) */
    ptrdiff_t os;  /* output stride (in complex elements) */
} fftwf_iodim64;

typedef fftwf_iodim64 fftw_iodim64;

/* --- Direction and flag constants ----------------------------------------- */

#define FFTW_FORWARD   (-1)
#define FFTW_BACKWARD  (+1)
#define FFTW_MEASURE   (0U)
#define FFTW_ESTIMATE  (1U << 6)
#define FFTW_DESTROY_INPUT (1U << 0)

/* --- Single-precision (fftwf_*) function declarations --------------------- */

/* Guru 64-bit planner (used by BART) */
fftwf_plan fftwf_plan_guru64_dft(
    int rank, const fftwf_iodim64 *dims,
    int howmany_rank, const fftwf_iodim64 *howmany_dims,
    fftwf_complex *in, fftwf_complex *out,
    int sign, unsigned flags);

/* Many-FFT planner (used by FINUFFT) */
fftwf_plan fftwf_plan_many_dft(
    int rank, const int *n, int howmany,
    fftwf_complex *in,  const int *inembed, int istride, int idist,
    fftwf_complex *out, const int *onembed, int ostride, int odist,
    int sign, unsigned flags);

/* Execution and teardown */
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in, fftwf_complex *out);
void fftwf_destroy_plan(fftwf_plan p);

/* Wisdom -- no-op stubs */
int  fftwf_export_wisdom_to_filename(const char *filename);
int  fftwf_import_wisdom_from_filename(const char *filename);
void fftwf_forget_wisdom(void);
void fftwf_cleanup(void);

/* Threading -- no-op stubs (Accelerate handles its own threading) */
int  fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int nthreads);
void fftwf_cleanup_threads(void);

/* --- Double-precision (fftw_*) function declarations ---------------------- */
/* Used by FINUFFT_FFT_plan<double> (cleanup/forget_wisdom path) and for   */
/* double-precision FINUFFT plans (if ever requested).                       */

fftw_plan fftw_plan_many_dft(
    int rank, const int *n, int howmany,
    fftw_complex *in,  const int *inembed, int istride, int idist,
    fftw_complex *out, const int *onembed, int ostride, int odist,
    int sign, unsigned flags);

void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out);
void fftw_destroy_plan(fftw_plan p);

/* Wisdom -- no-op stubs */
void fftw_forget_wisdom(void);
void fftw_cleanup(void);

/* Threading -- no-op stubs */
int  fftw_init_threads(void);
void fftw_plan_with_nthreads(int nthreads);
void fftw_cleanup_threads(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
