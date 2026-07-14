// Config-swept variant of the FULL-STAGE fused kernel (all-block block-diagonal matmul + two
// Givens rotation layers in shared memory, no (2048, M) round-trip). Same op as kernel.cu, but
// TBM / WarpShape-M / TBK / Stages / min-blocks / buffer-precision are template arguments so a
// Python sweep can measure how far raising occupancy closes the gap to the free-read ceiling.
//
// The occupancy wall is inherent: the epilogue rotation needs the WHOLE 2048-wide row, so the
// smem buffer is TBM * N_PAD * sizeof(BufElem) with N_PAD = 2049. At fp16 that is 131 KB for
// TBM=32 (1 block/SM) and 65 KB for TBM=16 (2 blocks/SM); TBM must stay >= WM (>= 16), so 2
// blocks/SM is about the ceiling. The row-parallel rotation also has only TBM-way parallelism,
// so shrinking TBM to raise CTA occupancy trades away rotation parallelism per CTA -- the sweep
// quantifies that tension.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/arch.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <type_traits>
#include <cstdio>

using ElementA   = cutlass::half_t;
using ElementB   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::RowMajor;
using LayoutC    = cutlass::layout::RowMajor;

static constexpr int kAlign = 8;   // 8 fp16 = 128-bit loads

// ---------------------------------------------------------------------------
// Compile-time config bundle. N_OUT / InstShape / WarpShape-N / FEAT are fixed.
// ---------------------------------------------------------------------------
template <int TBM_, int WM_, int TBK_, int Stages_, int MinBlocks_, int BufHalf_>
struct FSCfg {
    static constexpr int TBM       = TBM_;
    static constexpr int N_OUT     = 64;
    static constexpr int TBK       = TBK_;
    static constexpr int WM        = WM_;
    static constexpr int WN        = 64;
    static constexpr int Stages    = Stages_;
    static constexpr int MinBlocks = MinBlocks_;
    static constexpr int FEAT      = 2048;
    static constexpr int N_PAD     = FEAT + 1;    // 2049: bank(r,f)=(r+f)%32 -> conflict-free
    static constexpr bool kBufHalf = BufHalf_ != 0;
    using BufElem = typename std::conditional<kBufHalf, cutlass::half_t, float>::type;

    using TBShape   = cutlass::gemm::GemmShape<TBM, N_OUT, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM,  WN,    TBK>;
    using InstShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, kAlign, ElementAcc, ElementAcc>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA, kAlign,
        ElementB, LayoutB, kAlign,
        ElementC, LayoutC,
        ElementAcc,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        TBShape, WarpShape, InstShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages, /*SplitKSerial=*/false, cutlass::arch::OpMultiplyAdd>::GemmKernel;

    using Mma           = typename GemmKernel::Mma;
    using AccumIterator = typename Mma::Operator::IteratorC;
    using HalfAccumIter = cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        typename AccumIterator::Shape, cutlass::half_t, cutlass::layout::RowMajor,
        typename AccumIterator::InstructionShape, typename AccumIterator::OpDelta>;

    static constexpr int kWarpsM  = TBShape::kM / WarpShape::kM;
    static constexpr int kWarpsN  = TBShape::kN / WarpShape::kN;   // = 1
    static constexpr int kThreads = kWarpsM * kWarpsN * 32;
    // The accumulator store maps warp w to rows [warp_m*WM, ...) with warp_m = w % kWarpsM; that
    // matches CUTLASS's internal warp layout only when kWarpsM is a power of two.
    static_assert((kWarpsM & (kWarpsM - 1)) == 0,
                  "kWarpsM (= TBM/WM) must be a power of two for the manual warp->row store");

    static constexpr size_t mma_smem_bytes =
        (sizeof(typename Mma::SharedStorage) + 15) & ~size_t(15);
    static constexpr size_t smem_bytes =
        mma_smem_bytes + (size_t)TBM * N_PAD * sizeof(BufElem);
};

// ---------------------------------------------------------------------------
// Templated kernel (body identical to kernel.cu, parameterized by Cfg).
// ---------------------------------------------------------------------------
template <typename Cfg>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::MinBlocks)
fused_stage_kernel_cfg(
    const ElementA* __restrict__ X,  int Ktot,
    const ElementB* __restrict__ W,  int K, int nb, int stride,
    ElementC* __restrict__ Y,        int M,
    const int*   __restrict__ I1, const int*   __restrict__ J1,
    const float* __restrict__ C1, const float* __restrict__ S1, int P1,
    const int*   __restrict__ I2, const int*   __restrict__ J2,
    const float* __restrict__ C2, const float* __restrict__ S2, int P2)
{
    using Mma     = typename Cfg::Mma;
    using BufElem = typename Cfg::BufElem;
    constexpr int TBM   = Cfg::TBM;
    constexpr int N_OUT = Cfg::N_OUT;
    constexpr int N_PAD = Cfg::N_PAD;
    constexpr int FEAT  = Cfg::FEAT;

    extern __shared__ char smem[];
    auto& mma_smem = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
    BufElem* osh = reinterpret_cast<BufElem*>(smem + Cfg::mma_smem_bytes);

    const int thread_idx = threadIdx.x;
    const int warp_idx   = cutlass::canonical_warp_idx_sync();
    const int lane_idx   = thread_idx % 32;
    const int row_tile   = blockIdx.x;

    const int warp_m = warp_idx % Cfg::kWarpsM;
    const int warp_n = (warp_idx / Cfg::kWarpsM) % Cfg::kWarpsN;

    cutlass::NumericArrayConverter<
        cutlass::half_t, float, Mma::FragmentC::kElements> cvt;

    // ---- GEMM: loop over all nb blocks, each into its own 64-wide feature window ----
    const int nmain = nb - 1;
    const int Ktail = Ktot - nmain * stride;
    for (int b = 0; b < nb; ++b) {
        const int   Kb    = (b < nmain) ? K : Ktail;
        const int   x_off = b * stride;

        cutlass::MatrixCoord off_A(row_tile * TBM, 0);
        cutlass::MatrixCoord extent_A(M, Kb);
        typename Mma::IteratorA::Params params_A{LayoutA(Ktot)};
        typename Mma::IteratorA iterator_A(
            params_A, const_cast<ElementA*>(X + x_off), extent_A, thread_idx, off_A);

        cutlass::MatrixCoord off_B(0, 0);
        cutlass::MatrixCoord extent_B(Kb, N_OUT);
        typename Mma::IteratorB::Params params_B{LayoutB(N_OUT)};
        typename Mma::IteratorB iterator_B(
            params_B, const_cast<ElementB*>(W + (long)b * K * N_OUT), extent_B, thread_idx, off_B);

        Mma mma(mma_smem, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accum;
        accum.clear();
        const int k_iters = (Kb + Mma::Shape::kK - 1) / Mma::Shape::kK;
        mma(k_iters, accum, iterator_A, iterator_B, accum);

        BufElem* warp_ptr = osh
            + (long)(warp_m * Cfg::WarpShape::kM) * N_PAD
            + (long)(warp_n * Cfg::WarpShape::kN)
            + (long)b * N_OUT;
        if constexpr (Cfg::kBufHalf) {
            using HAI = typename Cfg::HalfAccumIter;
            typename HAI::Fragment h = cvt(accum);
            typename HAI::TensorRef ref(warp_ptr, cutlass::layout::RowMajor(N_PAD));
            HAI iter_C(ref, lane_idx);
            iter_C.store(h);
        } else {
            using AI = typename Cfg::AccumIterator;
            typename AI::TensorRef ref(warp_ptr, LayoutC(N_PAD));
            AI iter_C(ref, lane_idx);
            iter_C.store(accum);
        }
        __syncthreads();   // Mma re-stages A/B through mma_smem each block; guard reuse
    }

    // ---- Givens rotations: full-width, row-parallel, fp32 math, zero bank conflicts ----
    for (int r = thread_idx; r < TBM; r += blockDim.x) {
        BufElem* row = osh + (long)r * N_PAD;
        for (int p = 0; p < P1; p++) {
            const float a = static_cast<float>(row[I1[p]]);
            const float b = static_cast<float>(row[J1[p]]);
            row[I1[p]] = static_cast<BufElem>(C1[p] * a - S1[p] * b);
            row[J1[p]] = static_cast<BufElem>(S1[p] * a + C1[p] * b);
        }
        for (int p = 0; p < P2; p++) {
            const float a = static_cast<float>(row[I2[p]]);
            const float b = static_cast<float>(row[J2[p]]);
            row[I2[p]] = static_cast<BufElem>(C2[p] * a - S2[p] * b);
            row[J2[p]] = static_cast<BufElem>(S2[p] * a + C2[p] * b);
        }
    }
    __syncthreads();

    // ---- Write fp16 output Y (M, FEAT) row-major, f-varies-fastest ----
    const int row_base = row_tile * TBM;
    for (int idx = thread_idx; idx < TBM * FEAT; idx += blockDim.x) {
        const int r = idx / FEAT, f = idx % FEAT;
        const int grow = row_base + r;
        if (grow < M)
            Y[(long)grow * FEAT + f] =
                static_cast<ElementC>(static_cast<float>(osh[(long)r * N_PAD + f]));
    }
}

// ---------------------------------------------------------------------------
// Per-config launcher. Returns 0 on success, nonzero if the config can't run.
// ---------------------------------------------------------------------------
template <typename Cfg>
static int launch_cfg(
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2)
{
    const int smem_size = (int)Cfg::smem_bytes;

    int dev = 0; cudaGetDevice(&dev);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (smem_size > max_optin) return 10;

    auto kern = fused_stage_kernel_cfg<Cfg>;
    cudaError_t e = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != cudaSuccess) { cudaGetLastError(); return 11; }

    const int grid    = (M + Cfg::TBM - 1) / Cfg::TBM;
    const int threads = Cfg::kThreads;

    kern<<<grid, threads, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        (const ElementA*)X, Ktot,
        (const ElementB*)W, K, nb, stride,
        (ElementC*)Y, M,
        (const int*)I1, (const int*)J1, (const float*)C1, (const float*)S1, P1,
        (const int*)I2, (const int*)J2, (const float*)C2, (const float*)S2, P2);

    e = cudaGetLastError();
    if (e != cudaSuccess) return 12;
    return 0;
}

// Actual achieved blocks/SM for a config (occupancy), or 0 if it can't run.
template <typename Cfg>
static int occ_cfg() {
    const int smem_size = (int)Cfg::smem_bytes;
    auto kern = fused_stage_kernel_cfg<Cfg>;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
        != cudaSuccess) { cudaGetLastError(); return 0; }
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, Cfg::kThreads, smem_size);
    return occ;
}

// ---------------------------------------------------------------------------
// Config table.  X(id, TBM, WM, TBK, Stages, MinBlocks, BufHalf)
//   threads = (TBM/WM)*32 warps/block.  fp16 buffer: TBM=32 -> 131KB (1 blk/SM),
//   TBM=16 -> 65KB (2 blk/SM). TBM >= WM >= 16.
// ---------------------------------------------------------------------------
#define FS_CFG_LIST \
    X( 0, 32, 32, 32, 2, 1, 1)  /* baseline == kernel.cu: 1 warp, 1 blk/SM, fp16 */ \
    X( 1, 32, 32, 32, 3, 1, 1)  /* +stages */ \
    X( 2, 32, 16, 32, 2, 1, 1)  /* 2 warps/blk (more GEMM+rot threads), 1 blk/SM */ \
    X( 3, 32, 16, 32, 3, 1, 1) \
    X( 4, 16, 16, 32, 2, 2, 1)  /* TBM=16 -> target 2 blk/SM, 1 warp/blk */ \
    X( 5, 16, 16, 32, 3, 2, 1) \
    X( 6, 16, 16, 32, 2, 1, 1)  /* TBM=16, let occupancy be found */ \
    X( 7, 16, 16, 64, 2, 2, 1)  /* wider TBK */ \
    X( 8, 32, 16, 64, 2, 1, 1)  /* 2 warps + wider TBK */ \
    X( 9, 16, 16, 32, 2, 2, 0)  /* fp32 buffer at TBM=16 (131KB, 1 blk/SM, accurate) */

int run_fused_stage_cfg(int cfg,
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2)
{
    switch (cfg) {
    #define X(id, tbm, wm, tbk, st, mb, bh) \
        case id: return launch_cfg<FSCfg<tbm, wm, tbk, st, mb, bh>>( \
            X, Ktot, W, K, nb, stride, Y, M, I1, J1, C1, S1, P1, I2, J2, C2, S2, P2);
        FS_CFG_LIST
    #undef X
        default: return -1;
    }
}

int fused_stage_cfg_count() {
    int n = 0;
    #define X(id, tbm, wm, tbk, st, mb, bh) n = (id + 1 > n ? id + 1 : n);
    FS_CFG_LIST
    #undef X
    return n;
}

// Fills `out` with a short description + achieved occupancy; returns kThreads or -1.
int fused_stage_cfg_desc(int cfg, char* out, int out_len) {
    switch (cfg) {
    #define X(id, tbm, wm, tbk, st, mb, bh) \
        case id: { \
            using C = FSCfg<tbm, wm, tbk, st, mb, bh>; \
            snprintf(out, out_len, \
                "TBM=%d WM=%d TBK=%d S=%d minB=%d buf=%s thr=%d smem=%dKB occ=%d", \
                tbm, wm, tbk, st, mb, (bh ? "fp16" : "fp32"), (tbm/wm)*32, \
                (int)(C::smem_bytes / 1024), occ_cfg<C>()); \
            return (tbm/wm)*32; }
        FS_CFG_LIST
    #undef X
        default: snprintf(out, out_len, "<invalid>"); return -1;
    }
}
