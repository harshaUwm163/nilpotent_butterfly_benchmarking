// Tile-config-swept standalone block-diagonal matmul (BDMM).
//
// Same op / layout as bdmm_only/kernel.cu (batched CUTLASS GemmUniversal + SwapZXBatchSwizzle,
// output (nb*N, M) feature-major), but the threadblock/warp/stage TILE is a compile-time template
// so a Python sweep can pick the best tile per block shape (K,N). The block dims (K,N,nb,stride)
// stay runtime (from W / stride). Used to tune each Family-B block toward compute-bound and to feed
// the race-through experiment with the best-tiled BDMM.
#include <iostream>
#include <cstdio>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <c10/cuda/CUDAStream.h>

namespace cutlass { namespace gemm { namespace threadblock {

// batch fastest-varying grid dim -> adjacent blocks' K-overlap stays L2-resident (see kernel.cu).
struct SwapZXBatchSwizzle {
    CUTLASS_HOST_DEVICE SwapZXBatchSwizzle() {}
    CUTLASS_HOST_DEVICE
    static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int batch_count) {
        return GemmCoord(
            (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
            (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
            batch_count % (1 << 16));
    }
    CUTLASS_HOST_DEVICE static int get_log_tile(GemmCoord) { return 0; }
    CUTLASS_HOST_DEVICE static dim3 get_grid_shape(GemmCoord t) {
        return dim3(t.k(), t.n(), t.m());
    }
    CUTLASS_DEVICE static GemmCoord get_tile_offset(int) {
        return GemmCoord{RematerializeBlockIdxZ(), RematerializeBlockIdxY(), RematerializeBlockIdxX()};
    }
    CUTLASS_DEVICE static GemmCoord get_tile_offset(GemmCoord) {
        return GemmCoord{RematerializeBlockIdxZ(), RematerializeBlockIdxY(), RematerializeBlockIdxX()};
    }
};

}}}  // namespace

using EA  = cutlass::half_t;
using EB  = cutlass::half_t;
using EC  = cutlass::half_t;
using Acc = float;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;
using IS  = cutlass::gemm::GemmShape<16, 8, 16>;
using Epi = cutlass::epilogue::thread::LinearCombination<EC, 8, Acc, Acc>;
using Swz = cutlass::gemm::threadblock::SwapZXBatchSwizzle;

// Compile-time tile bundle. C' = Wᵀ·Xᵀ, output (n_out, M) RowMajor; A'=W ColM, B'=X ColM.
template <int TBM_, int TBN_, int TBK_, int WM_, int WN_, int WK_, int Stages_>
struct BDMMCfg {
    static constexpr int TBM = TBM_, TBN = TBN_, TBK = TBK_;
    static constexpr int WM = WM_, WN = WN_, WK = WK_, Stages = Stages_;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WP = cutlass::gemm::GemmShape<WM,  WN,  WK>;
    using Gemm = cutlass::gemm::device::GemmUniversal<
        EA, ColM, EB, ColM, EC, RowM, Acc,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TB, WP, IS, Epi, Swz, Stages>;
};

// Run one config (batched main + tail). Returns 0 on success, nonzero if the tile can't run the
// problem (bad alignment / smem over device limit / launch error) so the sweep can skip it.
template <typename Cfg>
static int run_cfg(const void* rX, const void* rW, void* rY,
                   int M, int N, int K, int nb, int stride, int Ktot)
{
    using Gemm = typename Cfg::Gemm;
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    auto stm  = at::cuda::getCurrentCUDAStream();

    typename Gemm::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, K}, nmain, {alpha, beta},
        dW, dX, dY, dY,
        (int64_t)K * N, (int64_t)stride, (int64_t)N * M, (int64_t)N * M,
        N, Ktot, M, M);
    Gemm g;
    if (g.can_implement(args_main) != cutlass::Status::kSuccess) return 20;
    if (g.initialize(args_main) != cutlass::Status::kSuccess)    return 21;
    if (g(stm) != cutlass::Status::kSuccess)                     return 22;

    if (Ktail > 0) {
        typename Gemm::Arguments args_tail(
            cutlass::gemm::GemmUniversalMode::kBatched,
            {N, M, Ktail}, 1, {alpha, beta},
            dW + (int64_t)nmain * K * N,
            dX + (int64_t)nmain * stride,
            dY + (int64_t)nmain * N * M,
            dY + (int64_t)nmain * N * M,
            0, 0, 0, 0,
            N, Ktot, M, M);
        if (g.initialize(args_tail) != cutlass::Status::kSuccess) return 23;
        if (g(stm) != cutlass::Status::kSuccess)                  return 24;
    }
    if (cudaGetLastError() != cudaSuccess) { return 25; }
    return 0;
}

template <typename Cfg>
static int occ_cfg() { return Cfg::Gemm::maximum_active_blocks(); }

template <typename Cfg>
static int threads_cfg() { return (int)Cfg::Gemm::GemmKernel::kThreadCount; }

// ---------------------------------------------------------------------------
// Tile config table.  X(id, TBM, TBN, TBK, WM, WN, WK, Stages)
//   swz layout: cutlass-M = N (output width), cutlass-N = M (tokens), cutlass-K = K (block depth).
//   TBM tiles the output width; TBN tiles the (huge) token dim; warp 64x64 unless noted.
// ---------------------------------------------------------------------------
#define BDMM_TILE_LIST \
    X( 0,  64, 128, 32,  32, 64, 32, 4) \
    X( 1,  64, 256, 32,  64, 64, 32, 4)  /* current swz tile */ \
    X( 2, 128, 128, 32,  64, 64, 32, 3) \
    X( 3, 128, 128, 32,  64, 64, 32, 4) \
    X( 4, 128, 256, 32,  64, 64, 32, 3) \
    X( 5, 256, 128, 32,  64, 64, 32, 3) \
    X( 6, 128, 128, 64,  64, 64, 64, 3) \
    X( 7, 256, 128, 64,  64, 64, 64, 2) \
    X( 8,  64,  64, 32,  32, 32, 32, 4) \
    X( 9, 128,  64, 32,  64, 32, 32, 4) \
    X(10, 256, 256, 32,  64, 64, 32, 2) \
    X(11,  64, 128, 64,  32, 64, 64, 3)

int run_bdmm_cfg(int cfg,
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M, int N)
{
    switch (cfg) {
    #define X(id, tbm, tbn, tbk, wm, wn, wk, st) \
        case id: return run_cfg<BDMMCfg<tbm, tbn, tbk, wm, wn, wk, st>>( \
            X, W, Y, M, N, K, nb, stride, Ktot);
        BDMM_TILE_LIST
    #undef X
        default: return -1;
    }
}

int bdmm_cfg_count() {
    int n = 0;
    #define X(id, tbm, tbn, tbk, wm, wn, wk, st) n = (id + 1 > n ? id + 1 : n);
    BDMM_TILE_LIST
    #undef X
    return n;
}

// Fills `out` with tile description + achieved occupancy (blocks/SM). Returns kThreadCount, or -1.
int bdmm_cfg_desc(int cfg, char* out, int out_len) {
    switch (cfg) {
    #define X(id, tbm, tbn, tbk, wm, wn, wk, st) \
        case id: { \
            using C = BDMMCfg<tbm, tbn, tbk, wm, wn, wk, st>; \
            snprintf(out, out_len, \
                "TB=%dx%dx%d WP=%dx%dx%d S=%d thr=%d occ=%d", \
                tbm, tbn, tbk, wm, wn, wk, st, threads_cfg<C>(), occ_cfg<C>()); \
            return threads_cfg<C>(); }
        BDMM_TILE_LIST
    #undef X
        default: snprintf(out, out_len, "<invalid>"); return -1;
    }
}
