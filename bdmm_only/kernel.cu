// Standalone strided block-diagonal matmul (BDMM) — the swz_s1 producer, isolated.
//
// Reference op: X (M, Ktot) fp16 row-major @ block-diagonal W (nb, K, N), stride `stride`
//   -> Y (n_out = nb*N, M) fp16 row-major (feature-major). Block b = output rows
//   [b*N : (b+1)*N] = W[b]^T @ X[:, b*stride : b*stride+K]. Adjacent blocks share a
//   (K - stride)-column overlap of X.
//
// This is a verbatim copy of readonce_swizzle's run_swz_s1 GEMM path (batched CUTLASS
// GemmUniversal + the custom SwapZXBatchSwizzle that puts the block index in the fastest-
// varying grid dim so adjacent blocks' overlap stays L2-resident). Pulled out on its own so
// the block-diagonal GEMM can be profiled / retuned (e.g. larger K, N to raise arithmetic
// intensity) without the rotation or the two-stage layer around it.
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <c10/cuda/CUDAStream.h>

namespace cutlass { namespace gemm { namespace threadblock {

// Custom swizzle: identical math to GemmIdentityThreadblockSwizzle<1> (no n-tiling, log_tile=0),
// but lays the grid out as dim3(batch, n_tiles, m_tiles) so batch is the fastest-varying grid
// dim in CUDA block-launch order -> all blocks of one M-region issue consecutively -> the
// K-column overlap batch b touched is still L2-resident when batch b+1 reads it.
struct SwapZXBatchSwizzle {

    CUTLASS_HOST_DEVICE
    SwapZXBatchSwizzle() {}

    CUTLASS_HOST_DEVICE
    static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int batch_count) {
        return GemmCoord(
            (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
            (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
            batch_count % (1 << 16));
    }

    CUTLASS_HOST_DEVICE
    static int get_log_tile(GemmCoord /*tiled_shape*/) { return 0; }

    CUTLASS_HOST_DEVICE
    static dim3 get_grid_shape(GemmCoord tiled_shape) {
        return dim3(tiled_shape.k(), tiled_shape.n(), tiled_shape.m());
    }

    CUTLASS_DEVICE
    static GemmCoord get_tile_offset(int /*log_tile*/) {
        return GemmCoord{
            RematerializeBlockIdxZ(),   // m_tile
            RematerializeBlockIdxY(),   // n_tile (M-region)
            RematerializeBlockIdxX()    // batch
        };
    }

    CUTLASS_DEVICE
    static GemmCoord get_tile_offset(GemmCoord /*tiled_shape*/) {
        return GemmCoord{
            RematerializeBlockIdxZ(),
            RematerializeBlockIdxY(),
            RematerializeBlockIdxX()
        };
    }
};

}}}  // namespace cutlass::gemm::threadblock

using EA  = cutlass::half_t;
using EB  = cutlass::half_t;
using EC  = cutlass::half_t;
using Acc = float;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

using TB  = cutlass::gemm::GemmShape<64, 256, 32>;   // tuned tile for the {N=64, M, K=128} problem
using WP  = cutlass::gemm::GemmShape<64,  64, 32>;
using IS  = cutlass::gemm::GemmShape<16,   8, 16>;
using Epi = cutlass::epilogue::thread::LinearCombination<EC, 8, Acc, Acc>;
using Swz = cutlass::gemm::threadblock::SwapZXBatchSwizzle;

// Pipeline depth. 4 tuned best on A100 for this latency-bound GEMM. Override with -DBDMM_STAGES=N.
#ifndef BDMM_STAGES
#define BDMM_STAGES 4
#endif

// C' = Wᵀ·Xᵀ, output (n_out, M) RowMajor.  A' = W ColMajor, B' = X ColMajor.
using GemmBDMM = cutlass::gemm::device::GemmUniversal<
    EA, ColM,
    EB, ColM,
    EC, RowM,
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TB, WP, IS, Epi, Swz, BDMM_STAGES>;

// X: (M, Ktot) row-major.  W: (nb, K, N).  Y: (n_out = nb*N, M) row-major (feature-major).
void run_bdmm(const void* rX, const void* rW, void* rY,
              int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    auto stm  = at::cuda::getCurrentCUDAStream();

    // Main blocks [0, nmain): full-width K slices.
    typename GemmBDMM::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, K}, nmain, {alpha, beta},
        dW, dX, dY, dY,
        (int64_t)K * N,     // batch_stride_A (one W block)
        (int64_t)stride,    // batch_stride_B (slide the K-window along X's columns)
        (int64_t)N * M,     // batch_stride_C (next N output rows)
        (int64_t)N * M,     // batch_stride_D
        N, Ktot, M, M);     // lda, ldb, ldc, ldd
    GemmBDMM g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "bdmm main initialize failed\n"; return;
    }
    g(stm);

    // Tail block [nmain]: narrower K slice (Ktail cols).
    if (Ktail > 0) {
        typename GemmBDMM::Arguments args_tail(
            cutlass::gemm::GemmUniversalMode::kBatched,
            {N, M, Ktail}, 1, {alpha, beta},
            dW + (int64_t)nmain * K * N,
            dX + (int64_t)nmain * stride,
            dY + (int64_t)nmain * N * M,
            dY + (int64_t)nmain * N * M,
            0, 0, 0, 0,     // single batch
            N, Ktot, M, M);
        if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
            std::cerr << "bdmm tail initialize failed\n"; return;
        }
        g(stm);
    }
}
