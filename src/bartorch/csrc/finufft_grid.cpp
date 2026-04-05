/*
 * finufft_grid.cpp — FINUFFT-backed spread/interpolate replacements for
 *                    BART's grid2() and grid2H().
 *
 * Linked via  -Wl,--wrap,grid2  -Wl,--wrap,grid2H
 *             -Wl,--wrap,rolloff_correction
 *             -Wl,--wrap,apply_rolloff_correction
 *             -Wl,--wrap,apply_rolloff_correction2
 * so that ALL BART CLI commands that reach the NUFFT codepath (nufft, pics,
 * moba, nlinv, …) automatically use FINUFFT's optimised multi-threaded
 * ES-kernel spreader in place of BART's single-threaded Kaiser-Bessel gridder,
 * AND use a matching ES-kernel rolloff correction instead of BART's KB one.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Semantics
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  grid2  (adjoint, nonuniform k-space → oversampled Cartesian grid):
 *      FINUFFT type-1, spreadinterponly=1
 *      f_k  = Σ_j  c_j · φ( (k/N) − x_j/(2π) )
 *      where φ is FINUFFT's piecewise-polynomial ES spreading kernel.
 *
 *  grid2H (forward, oversampled Cartesian grid → nonuniform k-space):
 *      FINUFFT type-2, spreadinterponly=1
 *      c_j  = Σ_k  f_k · φ( (k/N) − x_j/(2π) )
 *
 * Both operations are purely the spread/interpolation step.  The FFT step
 * remains in BART's nufft.c unchanged.  The rolloff (deconvolution) step is
 * also replaced — see __wrap_rolloff_correction and friends below.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Coordinate mapping
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  BART trajectory element:
 *    real( traj[3*j + d] )  ∈ [-N_d/2, N_d/2]  (non-oversampled grid units)
 *
 *  BART's gridH() applies:  pos_d = conf->os × Re(traj[d]) + grid_dims[d]/2
 *    to get a 0-indexed position on the OVERSAMPLED grid of size grid_dims[d].
 *
 *  FINUFFT spread-only expects x_j ∈ [-π, π) where the positive π edge
 *  corresponds to grid position N_fine (wraps to 0).  Mapping:
 *
 *    x_finufft_d = conf->os × Re(traj[3j+d]) × 2π / grid_dims[d]
 *
 *  Derivation: pos_d / grid_dims[d] × 2π − π
 *     = (conf->os × Re(traj) + grid_dims[d]/2) / grid_dims[d] × 2π − π
 *     = conf->os × Re(traj) × 2π / grid_dims[d] + π − π
 *     = conf->os × Re(traj) × 2π / grid_dims[d]                     ✓
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Rolloff (deconvolution) correction — ES-kernel FT
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  After spreading (grid2) and FFT, BART calls apply_rolloff_correction to
 *  deconvolve the spreading kernel from image space.  The correct deconvolution
 *  factor for FINUFFT's ES ("exponential of semicircle") kernel is:
 *
 *    w(p, d) = 1 / hat_phi_ES( (p - n_d/2) / (n_d * os) )
 *
 *  where hat_phi_ES(xi) = 2 * ∫₀^{J/2} exp(β·√(1−(2x/J)²)) · cos(2π·xi·x) dx
 *  is the continuous FT of the ES kernel at cycles-per-sample frequency xi.
 *
 *  This replaces BART's KB correction (1 / ftkb(beta, xi*width) / width).
 *  Both corrections are derived from the respective kernel FTs; using matching
 *  spreading kernel and rolloff correction gives exact deconvolution.
 *
 *  Kernel parameters for FINUFFT_GRID_TOL=1e-6f and upsampfac=2.0
 *  (from FINUFFT's setup_spreader, src/spreadinterp.cpp):
 *    nspread = ceil(-log10(tol/10)) = 7
 *    betaovns = 2.30  (default for nspread > 4, upsampfac=2.0)
 *    ES_beta  = betaovns * nspread = 16.1
 *    ES_hw    = nspread / 2       = 3.5
 *
 *  The kernel FT integral is evaluated with a 256-point midpoint rule on
 *  [0, ES_hw], which gives machine-precision accuracy (the integrand is
 *  smooth and decays to exp(-ES_beta) ≈ 1e-7 at the integration boundary).
 *
 *  The __wrap symbols intercept:
 *    rolloff_correction         — fills a 3-D correction weight array
 *    apply_rolloff_correction2  — applies correction with strides + batch
 *    apply_rolloff_correction   — contiguous wrapper → apply_rolloff_correction2
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Batch / coil handling
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  FINUFFT's n_transf parameter is set to  ∏ ksp_dims[3..D-1]  so that coils
 *  (and any higher batch dims) are processed in a single finufftf_execute()
 *  call, exploiting FINUFFT's OpenMP-parallel batch loop.
 *
 *  Memory layout compatibility:
 *    BART ksp_dims = [1, M1, M2, Ncoils, ...]  (dim-0 fastest)
 *    → src[j + M*coil + M*Ncoils*frame + ...]  (M = M1*M2)
 *    FINUFFT batch: c[i*M + j] for transform i and sample j
 *    These are the same layout                                        ✓
 *
 *    BART grid_dims = [Nx, Ny, Nz, Ncoils, ...]
 *    → dst[n + N_spatial*coil + ...]  (N_spatial = Nx*Ny*Nz)
 *    FINUFFT batch: fk[i*N_spatial + n]
 *    These are the same layout                                        ✓
 */

#include <complex>
#include <cstring>
#include <vector>
#include <cmath>

#include <finufft.h>   // single + double precision public API

// BART's grid_conf_s definition — must be included with C linkage.
// grid.h uses _Bool (C99 built-in) which is not a keyword in strict C++ mode
// (it would be __Bool or just bool).  We provide a C++ typedef alias before
// including the header.  GCC/Clang accept this as a harmless no-op on recent
// toolchains where _Bool is already defined as a built-in.
extern "C" {
#ifndef _Bool
typedef bool _Bool;  // C99 ↔ C++ compatibility alias for BART's grid.h
#endif
#include "noncart/grid.h"
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr float FINUFFT_GRID_TOL = 1e-6f;  // float32 NUFFT accuracy

using FC = std::complex<float>;  // alias for readability

/// Deduce spatial dimensionality from grid_dims[0..2].
static int spatial_ndim(const long grid_dims[])
{
    if (grid_dims[2] > 1) return 3;
    if (grid_dims[1] > 1) return 2;
    return 1;
}

/// Extract BART trajectory into FINUFFT coordinate arrays (in [-π, π)).
///
/// BART layout: traj[3*j + d] is complex float; real part holds the
/// coordinate in non-oversampled grid units [-N_d/2, N_d/2].
/// conf->os scales to the oversampled grid; grid_dims[d] is the oversampled
/// size.  Mapping:  x_finufft = conf->os * Re(traj[d]) * 2π / grid_dims[d]
static void extract_coords(
    long                         M,
    const long                   trj_dims[],
    const FC*                    traj,
    const long                   grid_dims[],
    float                        os,
    int                          ndim,
    std::vector<float>&          xj,
    std::vector<float>&          yj,
    std::vector<float>&          zj)
{
    static constexpr float TWO_PI = 2.0f * static_cast<float>(M_PI);

    xj.resize(M);
    if (ndim >= 2) yj.resize(M);
    if (ndim >= 3) zj.resize(M);

    const float sx = os * TWO_PI / static_cast<float>(grid_dims[0]);
    const float sy = (ndim >= 2)
                   ? os * TWO_PI / static_cast<float>(grid_dims[1]) : 0.f;
    const float sz = (ndim >= 3)
                   ? os * TWO_PI / static_cast<float>(grid_dims[2]) : 0.f;

    for (long j = 0; j < M; ++j) {
        xj[j] = traj[3*j + 0].real() * sx;
        if (ndim >= 2) yj[j] = traj[3*j + 1].real() * sy;
        if (ndim >= 3) zj[j] = traj[3*j + 2].real() * sz;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_grid2:  adjoint gridding — nonuniform k-space → Cartesian grid
//
// BART signature (noncart/grid.h):
//   void grid2(const struct grid_conf_s* conf,
//              int D, const long trj_dims[D], const complex float* traj,
//              const long grid_dims[D], complex float* dst,
//              const long ksp_dims[D],  const complex float* src)
//
// FINUFFT call: type-1, spreadinterponly=1
//   execute(plan, c=src (NU input), fk=dst (uniform output))
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_grid2(
    const struct grid_conf_s* conf,
    int                       D,
    const long                trj_dims[],
    const _Complex float*     traj_c,
    const long                grid_dims[],
    _Complex float*           dst_c,
    const long                ksp_dims[],
    const _Complex float*     src_c)
{
    // Reinterpret C complex ↔ C++ complex (same layout, guaranteed by standard)
    const FC* traj = reinterpret_cast<const FC*>(traj_c);
    const FC* src  = reinterpret_cast<const FC*>(src_c);
    FC*       dst  = reinterpret_cast<FC*>(dst_c);

    const int  ndim     = spatial_ndim(grid_dims);
    const long Nx       = grid_dims[0];
    const long Ny       = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz       = (ndim >= 3) ? grid_dims[2] : 1L;
    const long N_spatial = Nx * Ny * Nz;

    // Total k-space samples M = ∏ trj_dims[1..D-1]  (trj_dims[0] = 3, trj_dims[3] = 1)
    long M = 1;
    for (int d = 1; d < D; ++d) M *= trj_dims[d];

    // Batch count = ∏ ksp_dims[3..D-1]  (coils + higher dims)
    long n_transf = 1;
    for (int d = 3; d < D; ++d) n_transf *= ksp_dims[d];

    if (M == 0 || n_transf == 0) return;

    // Extract and scale trajectory
    std::vector<float> xj, yj, zj;
    extract_coords(M, trj_dims, traj, grid_dims, conf->os, ndim, xj, yj, zj);

    // Zero the output: BART's caller may not clear dst before calling grid2
    std::memset(dst, 0,
        static_cast<std::size_t>(n_transf * N_spatial) * sizeof(FC));

    // Configure FINUFFT
    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.spreadinterponly = 1;    // spreading only — no FFT, no deconvolution
    // upsampfac controls the ES kernel width (nspread), NOT the actual grid
    // oversampling.  In spread-only mode FINUFFT writes directly to the
    // n_modes-sized grid with no internal upsampling.  Setting 2.0 matches
    // BART's default os=2.0 regime and yields ~10 spreading points for
    // tol=1e-6.  conf->os is used separately in coordinate scaling (see
    // extract_coords); the two parameters are independent.
    opts.upsampfac        = 2.0;
    opts.debug            = 0;
    opts.showwarn         = 0;

    // Make plan: type 1 (NU → U), ndim-D, n_transf batches
    finufftf_plan plan = nullptr;
    int64_t n_modes[3] = { Nx, Ny, Nz };
    int ier = finufftf_makeplan(
        1, ndim, n_modes,
        +1,                          // sign (irrelevant for spread-only)
        static_cast<int>(n_transf),
        FINUFFT_GRID_TOL, &plan, &opts);
    if (ier != 0 || plan == nullptr) return;

    ier = finufftf_setpts(plan, M, xj.data(),
        (ndim >= 2) ? yj.data() : nullptr,
        (ndim >= 3) ? zj.data() : nullptr,
        0, nullptr, nullptr, nullptr);
    if (ier != 0) { finufftf_destroy(plan); return; }

    // type-1 execute: weights = src (NU k-space input), result = dst (grid output)
    // FINUFFT's C API takes non-const pointers even for read-only operands;
    // the const_cast is safe because for type-1 the weights array is only read.
    finufftf_execute(plan,
        const_cast<FC*>(src),
        dst);

    finufftf_destroy(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_grid2H:  forward gridding — Cartesian grid → nonuniform k-space
//
// BART signature (noncart/grid.h):
//   void grid2H(const struct grid_conf_s* conf,
//               int D, const long trj_dims[D], const complex float* traj,
//               const long ksp_dims[D],  complex float* dst,
//               const long grid_dims[D], const complex float* src)
//
// FINUFFT call: type-2, spreadinterponly=1
//   execute(plan, c=dst (NU output), fk=src (uniform input))
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_grid2H(
    const struct grid_conf_s* conf,
    int                       D,
    const long                trj_dims[],
    const _Complex float*     traj_c,
    const long                ksp_dims[],
    _Complex float*           dst_c,
    const long                grid_dims[],
    const _Complex float*     src_c)
{
    const FC* traj = reinterpret_cast<const FC*>(traj_c);
    FC*       dst  = reinterpret_cast<FC*>(dst_c);
    const FC* src  = reinterpret_cast<const FC*>(src_c);

    const int  ndim = spatial_ndim(grid_dims);
    const long Nx   = grid_dims[0];
    const long Ny   = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz   = (ndim >= 3) ? grid_dims[2] : 1L;

    long M = 1;
    for (int d = 1; d < D; ++d) M *= trj_dims[d];

    long n_transf = 1;
    for (int d = 3; d < D; ++d) n_transf *= ksp_dims[d];

    if (M == 0 || n_transf == 0) return;

    std::vector<float> xj, yj, zj;
    extract_coords(M, trj_dims, traj, grid_dims, conf->os, ndim, xj, yj, zj);

    // Zero k-space output
    std::memset(dst, 0, static_cast<std::size_t>(n_transf * M) * sizeof(FC));

    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.spreadinterponly = 1;    // interpolation step only — no FFT, no deconvolution
    opts.upsampfac        = 2.0;  // same kernel design as grid2 (see comment there)
    opts.debug            = 0;
    opts.showwarn         = 0;

    finufftf_plan plan = nullptr;
    int64_t n_modes[3] = { Nx, Ny, Nz };
    int ier = finufftf_makeplan(
        2, ndim, n_modes,
        +1,
        static_cast<int>(n_transf),
        FINUFFT_GRID_TOL, &plan, &opts);
    if (ier != 0 || plan == nullptr) return;

    ier = finufftf_setpts(plan, M, xj.data(),
        (ndim >= 2) ? yj.data() : nullptr,
        (ndim >= 3) ? zj.data() : nullptr,
        0, nullptr, nullptr, nullptr);
    if (ier != 0) { finufftf_destroy(plan); return; }

    // type-2 execute: weights = dst (NU output, will be filled),
    //                 result  = src (uniform grid input, only read)
    // const_cast on src: FINUFFT's C API takes non-const even for read-only operands.
    finufftf_execute(plan,
        dst,
        const_cast<FC*>(src));

    finufftf_destroy(plan);
}

// =============================================================================
// ES-kernel rolloff (deconvolution) correction
// =============================================================================
//
// Replaces BART's Kaiser-Bessel rolloff with a correction that is consistent
// with the FINUFFT ES-kernel used in __wrap_grid2 / __wrap_grid2H.
//
// ES kernel parameters — must match FINUFFT_GRID_TOL=1e-6f and upsampfac=2.0.
// From FINUFFT setup_spreader (src/spreadinterp.cpp, upsampfac==2.0 branch):
//   nspread  = ceil(-log10(tol/10))     = 7
//   betaovns = 2.30  (default, ns > 4)
//   ES_beta  = betaovns * nspread       = 16.1
//   ES_hw    = nspread / 2.0            = 3.5
// ES kernel: phi(x) = exp(ES_beta * sqrt(1 - (x/ES_hw)^2))  for |x| <= ES_hw

static constexpr double ESRO_BETA  = 2.30 * 7.0;  // 16.1
static constexpr double ESRO_HW    = 7.0 / 2.0;   // 3.5  (ES_halfwidth)
static constexpr int    ESRO_NQUAD = 256;          // midpoint quadrature points

/// ES kernel phi(x) at x ∈ [0, ESRO_HW].
static inline double esro_phi(double x) {
    const double t = x / ESRO_HW;
    const double arg = 1.0 - t * t;
    if (arg < 0.0) return 0.0;
    return std::exp(ESRO_BETA * std::sqrt(arg));
}

/// Continuous FT of the ES kernel at cycles-per-sample frequency xi_cps:
///   hat_phi(xi) = 2 * ∫₀^{ES_hw}  phi(x) · cos(2π·xi·x) dx
/// Evaluated via midpoint rule on [0, ES_hw] with ESRO_NQUAD points.
/// The integrand decays to exp(-ESRO_BETA) ≈ 1e-7 at x=ES_hw, so the
/// midpoint rule converges rapidly even with a relatively small node count.
static double esro_hat_phi(double xi_cps) {
    const double h   = ESRO_HW / ESRO_NQUAD;
    const double tpi = 2.0 * M_PI;
    double sum = 0.0;
    for (int i = 0; i < ESRO_NQUAD; ++i) {
        const double x = (i + 0.5) * h;
        sum += esro_phi(x) * std::cos(tpi * xi_cps * x);
    }
    return 2.0 * h * sum;    // factor 2: symmetry over [-ES_hw, ES_hw]
}

/// Rolloff correction weight for pixel p in a dimension of size n with
/// oversampling factor os.  Matches BART's pos() convention exactly:
///   xi = (p - n/2.0) / (n * os)   [floating-point midpoint, same as BART]
/// Returns (float)(1 / hat_phi_ES(xi)), clamped away from zero.
static float esro_weight(long p, long n, double os) {
    if (n <= 1) return 1.0f;
    const double xi = ((double)p - (double)n * 0.5) / ((double)n * os);
    const double hp = esro_hat_phi(xi);
    if (hp == 0.0) return 1.0f;
    return static_cast<float>(1.0 / hp);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_rolloff_correction
//
// BART v1.0.00 signature (noncart/grid.h):
//   void rolloff_correction(float os, float width, float beta,
//                           const long dim[3], complex float* dst)
//
// Fills dst[x + dim[0]*(y + z*dim[1])] with the product of 1-D ES rolloff
// weights for each active spatial dimension.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_rolloff_correction(
    float           os,
    float           /*width*/,
    float           /*beta*/,
    const long      dim[3],
    _Complex float* dst_c)
{
    FC* dst = reinterpret_cast<FC*>(dst_c);
    const double os_d = static_cast<double>(os);

    // Pre-compute per-dimension weight vectors (avoids repeated integral eval).
    std::vector<float> wx(dim[0]), wy(dim[1]), wz(dim[2]);
    for (long x = 0; x < dim[0]; ++x) wx[x] = esro_weight(x, dim[0], os_d);
    for (long y = 0; y < dim[1]; ++y) wy[y] = esro_weight(y, dim[1], os_d);
    for (long z = 0; z < dim[2]; ++z) wz[z] = esro_weight(z, dim[2], os_d);

    for (long z = 0; z < dim[2]; ++z)
        for (long y = 0; y < dim[1]; ++y)
            for (long x = 0; x < dim[0]; ++x)
                dst[x + dim[0] * (y + z * dim[1])] =
                    static_cast<FC>(wx[x] * wy[y] * wz[z]);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_apply_rolloff_correction2
//
// BART v1.0.00 signature:
//   void apply_rolloff_correction2(float os, float width, float beta,
//          int N, const long dims[N], const long ostrs[N], complex float* dst,
//          const long istrs[N], const complex float* src)
//
// Applies the ES rolloff correction with full stride support and optional
// batch dimensions (dims[3..N-1]).
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_apply_rolloff_correction2(
    float                os,
    float                /*width*/,
    float                /*beta*/,
    int                  N,
    const long           dims[],
    const long           ostrs[],
    _Complex float*      dst_c,
    const long           istrs[],
    const _Complex float* src_c)
{
    FC*       dst = reinterpret_cast<FC*>(dst_c);
    const FC* src = reinterpret_cast<const FC*>(src_c);
    const double os_d = static_cast<double>(os);

    const long d0 = dims[0], d1 = dims[1], d2 = dims[2];

    // Pre-compute per-dimension 1-D weight vectors.
    std::vector<float> wx(d0), wy(d1), wz(d2);
    for (long x = 0; x < d0; ++x) wx[x] = esro_weight(x, d0, os_d);
    for (long y = 0; y < d1; ++y) wy[y] = esro_weight(y, d1, os_d);
    for (long z = 0; z < d2; ++z) wz[z] = esro_weight(z, d2, os_d);

    // Accumulate batch size and strides from dims[3..N-1].
    long size_bat = 1;
    long obstr = -1, ibstr = -1;
    for (int i = 3; i < N; ++i) {
        if (1 == dims[i]) continue;
        if (-1 == obstr) { obstr = ostrs[i]; ibstr = istrs[i]; }
        size_bat *= dims[i];
    }
    // Convert byte strides to element strides (CFL_SIZE = sizeof(complex float) = 8).
    const long cfl = static_cast<long>(sizeof(FC));
    const long os0 = ostrs[0] / cfl;
    const long os1 = ostrs[1] / cfl;
    const long os2 = ostrs[2] / cfl;
    const long is0 = istrs[0] / cfl;
    const long is1 = istrs[1] / cfl;
    const long is2 = istrs[2] / cfl;
    const long obs = (obstr == -1) ? 0L : obstr / cfl;
    const long ibs = (ibstr == -1) ? 0L : ibstr / cfl;

#pragma omp parallel for collapse(3)
    for (long z = 0; z < d2; ++z) {
        for (long y = 0; y < d1; ++y) {
            for (long x = 0; x < d0; ++x) {
                const long oidx = x * os0 + y * os1 + z * os2;
                const long iidx = x * is0 + y * is1 + z * is2;
                const float val  = wx[x] * wy[y] * wz[z];
                for (long i = 0; i < size_bat; ++i)
                    dst[oidx + i * obs] =
                        static_cast<FC>(val) * src[iidx + i * ibs];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_apply_rolloff_correction
//
// BART v1.0.00 signature:
//   void apply_rolloff_correction(float os, float width, float beta,
//          int N, const long dimensions[N],
//          complex float* dst, const complex float* src)
//
// Delegates to __wrap_apply_rolloff_correction2 with contiguous strides,
// matching BART's own implementation which calls:
//   apply_rolloff_correction2(..., MD_STRIDES(N,dims,CFL_SIZE), dst,
//                                  MD_STRIDES(N,dims,CFL_SIZE), src)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_apply_rolloff_correction(
    float                os,
    float                width,
    float                beta,
    int                  N,
    const long           dims[],
    _Complex float*      dst_c,
    const _Complex float* src_c)
{
    // Build contiguous strides: strs[0] = CFL_SIZE, strs[i] = strs[i-1]*dims[i-1].
    std::vector<long> strs(N);
    long s = static_cast<long>(sizeof(FC));  // CFL_SIZE = 8
    for (int i = 0; i < N; ++i) { strs[i] = s; s *= dims[i]; }

    __wrap_apply_rolloff_correction2(os, width, beta, N, dims,
        strs.data(), dst_c, strs.data(), src_c);
}
