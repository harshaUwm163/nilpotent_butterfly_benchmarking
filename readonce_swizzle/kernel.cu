// Read-once strided block-diagonal matmul via a CUSTOM CUTLASS threadblock swizzle.
//
// Reference op: col_s1(x, W, 64). x (M,2048) fp16 row-major @ block-diagonal W (32,128,64),
// stride 64 -> Y (2048, M) fp16 row-major. Block b = output rows [b*64:(b+1)*64] =
// W[b]^T @ x[:, b*64 : b*64+128]. Adjacent blocks share a 64-col overlap of x.
//
// MECHANISM: keep col_s1's HIGH-OCCUPANCY batched GemmUniversal structure (one tiny 64x64
// accumulator tile per batch -> register-light -> high occupancy -> ~76% DRAM throughput).
// The ONLY change vs col_s1 is the threadblock swizzle: the default GemmIdentityThreadblockSwizzle
// puts the batch (k) in grid.z (slowest-varying launch order), so all M-tiles of batch b launch
// before any of batch b+1 -> by the time b+1 reads the shared 64 cols, they're evicted from L2 ->
// x is re-read 1.94x. Our SwapZX swizzle puts the batch in grid.x (FASTEST-varying) and the
// M-region (n-tile) in grid.y, so for a fixed M-region all 32 batches schedule back-to-back and
// the 64-col overlap that batch b just touched is still L2-resident when batch b+1 reads it.
//
// For S1 the GemmUniversal problem is {N=64, M, K=128}, tile {64,256,32}:
//   tiled_shape.m() = ceil(64/64)   = 1            (the W output-feature dim)
//   tiled_shape.n() = ceil(M/256)   = M/256        (the M-region partition -> grid.y)
//   tiled_shape.k() = batch_count   = nb-1 = 31    (the block index -> grid.x)
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <c10/cuda/CUDAStream.h>

namespace cutlass { namespace gemm { namespace threadblock {

// Custom swizzle: identical math to GemmIdentityThreadblockSwizzle<1> (no n-tiling, log_tile=0),
// but lays the grid out as dim3(batch, n_tiles, m_tiles) so that batch is the fastest-varying
// grid dimension in CUDA block-launch order. get_tile_offset decodes it back to {m, n, batch}.
struct SwapZXBatchSwizzle {

    CUTLASS_HOST_DEVICE
    SwapZXBatchSwizzle() {}

    // problem in units of logical tiles. .k() carries the batch_count (kBatched mode).
    CUTLASS_HOST_DEVICE
    static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int batch_count) {
        return GemmCoord(
            (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
            (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
            batch_count % (1 << 16));
    }

    CUTLASS_HOST_DEVICE
    static int get_log_tile(GemmCoord /*tiled_shape*/) { return 0; }

    // Launch grid: x = batch (fastest), y = n-tiles (M-region), z = m-tiles.
    // CUDA schedules blocks roughly x-fastest then y then z, so all batches of one M-region
    // are issued consecutively -> overlap stays L2-resident across adjacent batches.
    CUTLASS_HOST_DEVICE
    static dim3 get_grid_shape(GemmCoord tiled_shape) {
        return dim3(tiled_shape.k(), tiled_shape.n(), tiled_shape.m());
    }

    // Decode blockIdx -> {m_tile, n_tile, batch}. .k() is consumed as the batch pointer offset.
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

using TB  = cutlass::gemm::GemmShape<64, 256, 32>;   // same tuned tile as col_s1
using WP  = cutlass::gemm::GemmShape<64,  64, 32>;
using IS  = cutlass::gemm::GemmShape<16,   8, 16>;
using Epi = cutlass::epilogue::thread::LinearCombination<EC, 8, Acc, Acc>;
using Swz    = cutlass::gemm::threadblock::SwapZXBatchSwizzle;
using NoSwz  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Pipeline depth. Tuned on A100: 4 beats 3, and beats 5/6 (which spill shared memory and drop
// occupancy). This GEMM is latency-bound, not occupancy-bound: shrinking the warp tile to raise
// occupancy made it slower (8.9 vs 7.5 ms), while one extra pipeline stage helped (esp. swz_s2's
// strided read, 7.88 -> 7.46 ms). Set -DSWZ_STAGES=N to re-tune; 4 is the shipped default.
#ifndef SWZ_STAGES
#define SWZ_STAGES 4
#endif

// Stage 1: C' = Wᵀ·Xᵀ, output (n_out, M) RowMajor.  A'=W ColMajor, B'=X ColMajor.
using GemmS1 = cutlass::gemm::device::GemmUniversal<
    EA, ColM,
    EB, ColM,
    EC, RowM,
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TB, WP, IS, Epi, Swz, SWZ_STAGES>;

// Ablation: identical to GemmS1 but with the default identity swizzle (batch in grid.z,
// slowest-varying) so adjacent batches' 64-col overlap is evicted from L2 before reuse.
using GemmS1_NoSwz = cutlass::gemm::device::GemmUniversal<
    EA, ColM,
    EB, ColM,
    EC, RowM,
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TB, WP, IS, Epi, NoSwz, 3>;

// X: (M, Ktot) unpadded. Output Y: (n_out, M) row-major, block-major (no permute). Identical
// argument wiring to run_col_s1 in block_nopad_col/kernel.cu; only Swz differs.
void run_swz_s1(const void* rX, const void* rW, void* rY,
                int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    auto stm  = at::cuda::getCurrentCUDAStream();

    // Main batches [0, nmain): all full-width K slices.
    typename GemmS1::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, K}, nmain, {alpha, beta},
        dW, dX, dY, dY,
        (int64_t)K * N,     // batch_stride_A
        (int64_t)stride,    // batch_stride_B
        (int64_t)N * M,     // batch_stride_C
        (int64_t)N * M,     // batch_stride_D
        N, Ktot, M, M);     // lda, ldb, ldc, ldd
    GemmS1 g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "swz s1 main\n"; return;
    }
    g(stm);

    // Tail batch [nmain]: narrower K slice (Ktail cols).
    typename GemmS1::Arguments args_tail(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, Ktail}, 1, {alpha, beta},
        dW + (int64_t)nmain * K * N,
        dX + (int64_t)nmain * stride,
        dY + (int64_t)nmain * N * M,
        dY + (int64_t)nmain * N * M,
        0, 0, 0, 0,         // batch strides (single batch)
        N, Ktot, M, M);
    if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
        std::cerr << "swz s1 tail\n"; return;
    }
    g(stm);
}

// Ablation: same as run_swz_s1 with identity swizzle instead of SwapZXBatchSwizzle.
void run_noswz_s1(const void* rX, const void* rW, void* rY,
                  int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    auto stm  = at::cuda::getCurrentCUDAStream();

    typename GemmS1_NoSwz::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, K}, nmain, {alpha, beta},
        dW, dX, dY, dY,
        (int64_t)K * N,
        (int64_t)stride,
        (int64_t)N * M,
        (int64_t)N * M,
        N, Ktot, M, M);
    GemmS1_NoSwz g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "noswz s1 main\n"; return;
    }
    g(stm);

    typename GemmS1_NoSwz::Arguments args_tail(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, Ktail}, 1, {alpha, beta},
        dW + (int64_t)nmain * K * N,
        dX + (int64_t)nmain * stride,
        dY + (int64_t)nmain * N * M,
        dY + (int64_t)nmain * N * M,
        0, 0, 0, 0,
        N, Ktot, M, M);
    if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
        std::cerr << "noswz s1 tail\n"; return;
    }
    g(stm);
}

// RowMajor stage 1: Y(M, N_out) = X(M,K) @ W(K,N), both inputs RowMajor, no transpose tricks.
// Problem {M, N, K}: m-tiles = M/256, n-tiles = N/64 = 1.  Tile is transposed vs GemmS1
// ({256,64,32} vs {64,256,32}) so the large M-dimension maps to m (many tiles) and the small
// N=64 maps to n (one exact tile).  No swizzle needed: the overlap is in X's column direction,
// which consecutive m-tiles don't share, so there's no L2-reuse opportunity to exploit.
using TBt = cutlass::gemm::GemmShape<256, 64, 32>;   // transposed tile for {M, N=64, K} orientation

using GemmS1_RowOut = cutlass::gemm::device::GemmUniversal<
    EA, RowM,   // A = X, (M,K) RowMajor
    EB, RowM,   // B = W, (K,N) RowMajor
    EC, RowM,   // C = output, (M,N_out) RowMajor
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TBt, WP, IS, Epi, NoSwz, 3>;

// Same as GemmS1_RowOut but with SwapZXBatchSwizzle.  For problem {M, N=64, K} with tile
// {256,64,32}: grid = dim3(nb-1, 1, M/256), so for each fixed grid.z (M-region) all batches
// run consecutively in grid.x — same L2-reuse guarantee as the ColMajor swz kernel.
using GemmS1_RowOutSwz = cutlass::gemm::device::GemmUniversal<
    EA, RowM,
    EB, RowM,
    EC, RowM,
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TBt, WP, IS, Epi, Swz, 3>;

// X: (M, Ktot). W: (nb, K, N). Output Y: (M, nb*N) RowMajor, block columns side by side.
// Each batch b computes X[:, b*stride : b*stride+K] @ W[b] into Y[:, b*N : (b+1)*N].
void run_rowout_s1(const void* rX, const void* rW, void* rY,
                   int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    int N_out = nb * N;
    auto stm  = at::cuda::getCurrentCUDAStream();

    typename GemmS1_RowOut::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, K}, nmain, {alpha, beta},
        dX, dW, dY, dY,
        (int64_t)stride,    // batch_stride_A: slide K-wide window along X's columns
        (int64_t)K * N,     // batch_stride_B: one W block per batch
        (int64_t)N,         // batch_stride_C: each batch writes the next N columns of Y
        (int64_t)N,         // batch_stride_D
        Ktot, N, N_out, N_out);   // lda, ldb, ldc, ldd
    GemmS1_RowOut g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "rowout s1 main\n"; return;
    }
    g(stm);

    typename GemmS1_RowOut::Arguments args_tail(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, Ktail}, 1, {alpha, beta},
        dX + (int64_t)nmain * stride,
        dW + (int64_t)nmain * K * N,
        dY + (int64_t)nmain * N,
        dY + (int64_t)nmain * N,
        0, 0, 0, 0,
        Ktot, N, N_out, N_out);
    if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
        std::cerr << "rowout s1 tail\n"; return;
    }
    g(stm);
}

void run_rowout_swz_s1(const void* rX, const void* rW, void* rY,
                       int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dX = (const cutlass::half_t*)rX;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    int N_out = nb * N;
    auto stm  = at::cuda::getCurrentCUDAStream();

    typename GemmS1_RowOutSwz::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, K}, nmain, {alpha, beta},
        dX, dW, dY, dY,
        (int64_t)stride,
        (int64_t)K * N,
        (int64_t)N,
        (int64_t)N,
        Ktot, N, N_out, N_out);
    GemmS1_RowOutSwz g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "rowout_swz s1 main\n"; return;
    }
    g(stm);

    typename GemmS1_RowOutSwz::Arguments args_tail(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, Ktail}, 1, {alpha, beta},
        dX + (int64_t)nmain * stride,
        dW + (int64_t)nmain * K * N,
        dY + (int64_t)nmain * N,
        dY + (int64_t)nmain * N,
        0, 0, 0, 0,
        Ktot, N, N_out, N_out);
    if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
        std::cerr << "rowout_swz s1 tail\n"; return;
    }
    g(stm);
}

// Stage 2: input I is the (n_out, M) row-major intermediate read as B RowMajor (ldb=M,
// batch_stride=stride*M). Same SwapZXBatchSwizzle so adjacent blocks' 64-row overlap is
// an L2 hit. Identical wiring to run_col_s2.
using GemmS2 = cutlass::gemm::device::GemmUniversal<
    EA, ColM,
    EB, RowM,
    EC, RowM,
    Acc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TB, WP, IS, Epi, Swz, SWZ_STAGES>;

void run_swz_s2(const void* rI, const void* rW, void* rY,
                int M, int N, int K, int nb, int stride, int Ktot)
{
    auto dI = (const cutlass::half_t*)rI;
    auto dW = (const cutlass::half_t*)rW;
    auto dY = (cutlass::half_t*)rY;

    Acc alpha = 1, beta = 0;
    int nmain = nb - 1;
    int Ktail = Ktot - nmain * stride;
    auto stm  = at::cuda::getCurrentCUDAStream();

    // Main batches [0, nmain).
    typename GemmS2::Arguments args_main(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, K}, nmain, {alpha, beta},
        dW, dI, dY, dY,
        (int64_t)K * N,         // batch_stride_A
        (int64_t)stride * M,    // batch_stride_B
        (int64_t)N * M,         // batch_stride_C
        (int64_t)N * M,         // batch_stride_D
        N, M, M, M);            // lda, ldb, ldc, ldd
    GemmS2 g;
    if (g.initialize(args_main) != cutlass::Status::kSuccess) {
        std::cerr << "swz s2 main\n"; return;
    }
    g(stm);

    // Tail batch [nmain].
    typename GemmS2::Arguments args_tail(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {N, M, Ktail}, 1, {alpha, beta},
        dW + (int64_t)nmain * K * N,
        dI + (int64_t)nmain * stride * M,
        dY + (int64_t)nmain * N * M,
        dY + (int64_t)nmain * N * M,
        0, 0, 0, 0,             // batch strides (single batch)
        N, M, M, M);
    if (g.initialize(args_tail) != cutlass::Status::kSuccess) {
        std::cerr << "swz s2 tail\n"; return;
    }
    g(stm);
}
