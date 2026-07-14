// Fused FULL stage: all-block strided block-diagonal matmul + two Givens rotation layers,
// entirely in shared memory — NO (2048, M) intermediate round-trip through HBM.
//
// One CTA owns a strip of TBM rows. It computes ALL nb blocks of the block-diagonal GEMM for
// those rows (looping the CUTLASS Mma over blocks, storing each block's 64 output features into
// a full-width fp16 smem buffer osh[TBM][2049]), then applies the full-width tau+rho Givens
// rotations to the 2048 features of each row — in shared memory — and writes Y (M, 2048).
//
// This is fused_block_gemm_rot generalized from 1 block / 64 features to nb blocks / 2048
// features. The rotation is the SAME row-parallel loop; only the buffer is full-width and the
// pair indices I*/J* range over [0, 2048).
//
// Reference op (matches block_gemm_nopad + fused_two_givens, i.e. two_stage_row's stage 1):
//   i1 = X @ block-diagonal W  (block b: X[:, b*stride : b*stride+Kb] @ W[b]),  Y (M, 2048)
//   g1 = tau-rotate then rho-rotate i1's 2048 features per row.
//
// Shared memory (A100, ~164 KB optin): fp16 buffer 32*2049*2 = 131 KB + Mma A/B staging ~12 KB.
// The fp32 buffer would only fit at TBM=16 (disallowed by WM>=32), so the buffer is fp16 with
// fp32-register rotation math (same kBufHalf trick as kernel_cfg.cu). N_PAD = 2048+1 = 2049;
// 2049 % 32 == 1 makes the row-parallel rotation bank-conflict-free (32 consecutive rows land on
// 32 distinct banks), exactly the single-block kernel's N_PAD=65 reasoning at full width.
//
// Occupancy is the inherent limiter: the full-row buffer caps this at 1 block/SM, and TBM=32 /
// WM=32 gives a single warp (32 threads) per CTA.

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/arch.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

using ElementA   = cutlass::half_t;
using ElementB   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::RowMajor;
using LayoutC    = cutlass::layout::RowMajor;

using TBShape   = cutlass::gemm::GemmShape<32, 64, 32>;   // TBM=32, N_OUT=64, TBK=32
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;    // WM=32 -> kWarpsM=1
using InstShape = cutlass::gemm::GemmShape<16,  8, 16>;

static constexpr int kStages = 2;
static constexpr int kAlign  = 8;               // 8 fp16 = 128-bit loads
static constexpr int TBM     = TBShape::kM;     // 32
static constexpr int N_OUT   = 64;              // per-block output width (== TBShape::kN)
static constexpr int FEAT    = 2048;            // total features across all blocks (nb * N_OUT)
static constexpr int N_PAD   = FEAT + 1;        // 2049: bank(r,f) = (r + f) % 32 -> conflict-free

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
    kStages, /*SplitKSerial=*/false, cutlass::arch::OpMultiplyAdd>::GemmKernel;

using Mma           = typename GemmKernel::Mma;
using AccumIterator = typename Mma::Operator::IteratorC;
// Same warp accumulator tile iterator but storing half_t (fp16 smem buffer).
using HalfAccumIter = cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
    typename AccumIterator::Shape, cutlass::half_t, cutlass::layout::RowMajor,
    typename AccumIterator::InstructionShape, typename AccumIterator::OpDelta>;

static constexpr int kWarpsM  = TBShape::kM / WarpShape::kM;   // 1
static constexpr int kWarpsN  = TBShape::kN / WarpShape::kN;   // 1
static constexpr int kThreads = kWarpsM * kWarpsN * 32;        // 32

static constexpr size_t mma_smem_bytes =
    (sizeof(typename Mma::SharedStorage) + 15) & ~size_t(15);

__global__ void __launch_bounds__(kThreads)
fused_stage_kernel(
    const ElementA* __restrict__ X,  int Ktot,
    const ElementB* __restrict__ W,  int K, int nb, int stride,
    ElementC* __restrict__ Y,        int M,
    const int*   __restrict__ I1, const int*   __restrict__ J1,
    const float* __restrict__ C1, const float* __restrict__ S1, int P1,
    const int*   __restrict__ I2, const int*   __restrict__ J2,
    const float* __restrict__ C2, const float* __restrict__ S2, int P2)
{
    extern __shared__ char smem[];
    auto& mma_smem = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
    // Full-width fp16 buffer: element (row r, feat f) at osh[r * N_PAD + f].
    cutlass::half_t* osh = reinterpret_cast<cutlass::half_t*>(smem + mma_smem_bytes);

    const int thread_idx = threadIdx.x;
    const int warp_idx   = cutlass::canonical_warp_idx_sync();
    const int lane_idx   = thread_idx % 32;
    const int row_tile   = blockIdx.x;

    const int warp_m = warp_idx % kWarpsM;   // always 0 (kWarpsM==1)
    const int warp_n = (warp_idx / kWarpsM) % kWarpsN;

    cutlass::NumericArrayConverter<
        cutlass::half_t, float, Mma::FragmentC::kElements> cvt;

    // ---- GEMM: loop over all nb blocks, each into its own 64-wide feature window ----
    // Block b: A = X[:, b*stride : b*stride+Kb] (Kb=K for main blocks, truncated for the tail so
    // it never reads past Ktot), B = W[b] (Kb, N_OUT).  Store into osh columns [b*N_OUT, ...).
    const int nmain = nb - 1;
    const int Ktail = Ktot - nmain * stride;
    for (int b = 0; b < nb; ++b) {
        const int   Kb    = (b < nmain) ? K : Ktail;
        const int   x_off = b * stride;

        cutlass::MatrixCoord off_A(row_tile * TBM, 0);
        cutlass::MatrixCoord extent_A(M, Kb);
        typename Mma::IteratorA::Params params_A{LayoutA(Ktot)};
        typename Mma::IteratorA iterator_A(
            params_A,
            const_cast<ElementA*>(X + x_off),
            extent_A, thread_idx, off_A);

        cutlass::MatrixCoord off_B(0, 0);
        cutlass::MatrixCoord extent_B(Kb, N_OUT);
        typename Mma::IteratorB::Params params_B{LayoutB(N_OUT)};
        typename Mma::IteratorB iterator_B(
            params_B,
            const_cast<ElementB*>(W + (long)b * K * N_OUT),
            extent_B, thread_idx, off_B);

        Mma mma(mma_smem, thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accum;
        accum.clear();
        const int k_iters = (Kb + Mma::Shape::kK - 1) / Mma::Shape::kK;
        mma(k_iters, accum, iterator_A, iterator_B, accum);

        // Store this block's accumulator into osh at feature offset b*N_OUT (row-major, stride N_PAD).
        cutlass::half_t* warp_ptr = osh
            + (long)(warp_m * WarpShape::kM) * N_PAD
            + (long)(warp_n * WarpShape::kN)
            + (long)b * N_OUT;
        typename HalfAccumIter::Fragment h = cvt(accum);
        typename HalfAccumIter::TensorRef ref(warp_ptr, cutlass::layout::RowMajor(N_PAD));
        HalfAccumIter iter_C(ref, lane_idx);
        iter_C.store(h);
        // Mma re-stages A/B through mma_smem each iteration; guard reuse of that storage.
        __syncthreads();
    }

    // ---- Givens rotations: full-width, row-parallel, fp32 math, zero bank conflicts ----
    // Thread t owns rows t, t+blockDim, ... and applies both layers over all 2048 features.
    for (int r = thread_idx; r < TBM; r += blockDim.x) {
        cutlass::half_t* row = osh + (long)r * N_PAD;
        for (int p = 0; p < P1; p++) {
            const float a = static_cast<float>(row[I1[p]]);
            const float b = static_cast<float>(row[J1[p]]);
            row[I1[p]] = static_cast<cutlass::half_t>(C1[p] * a - S1[p] * b);
            row[J1[p]] = static_cast<cutlass::half_t>(S1[p] * a + C1[p] * b);
        }
        for (int p = 0; p < P2; p++) {
            const float a = static_cast<float>(row[I2[p]]);
            const float b = static_cast<float>(row[J2[p]]);
            row[I2[p]] = static_cast<cutlass::half_t>(C2[p] * a - S2[p] * b);
            row[J2[p]] = static_cast<cutlass::half_t>(S2[p] * a + C2[p] * b);
        }
    }
    __syncthreads();   // output write reads rows owned by other threads

    // ---- Write fp16 output Y (M, FEAT) row-major, f-varies-fastest ----
    const int row_base = row_tile * TBM;
    for (int idx = thread_idx; idx < TBM * FEAT; idx += blockDim.x) {
        const int r = idx / FEAT, f = idx % FEAT;
        const int grow = row_base + r;
        if (grow < M)
            Y[(long)grow * FEAT + f] = osh[(long)r * N_PAD + f];
    }
}

void launch_fused_stage(
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2)
{
    const size_t smem_size = mma_smem_bytes + (size_t)TBM * N_PAD * sizeof(cutlass::half_t);

    cudaFuncSetAttribute(fused_stage_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_size);

    const int grid    = (M + TBM - 1) / TBM;
    const int threads = kThreads;   // 32

    fused_stage_kernel<<<grid, threads, smem_size,
                         at::cuda::getCurrentCUDAStream()>>>(
        (const ElementA*)X, Ktot,
        (const ElementB*)W, K, nb, stride,
        (ElementC*)Y, M,
        (const int*)I1, (const int*)J1, (const float*)C1, (const float*)S1, P1,
        (const int*)I2, (const int*)J2, (const float*)C2, (const float*)S2, P2);
}
