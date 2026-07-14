// Column-major Givens rotation on data (n_out, M) row-major: feature f's M values are contiguous
// at data[f*M : (f+1)*M], so combining a pair of features is coalesced.
//
//  - rot_layer_cm  : one layer = one coalesced pass (two launches for tau,rho). Simple, ~2x floor.
//  - rot_cycle_cm  : per-cycle ONE pass. A thread owns column m of one cycle, loads the cycle's L
//                    features (coalesced), applies BOTH layers' rotations in registers, writes L.
//                    Each feature touched once -> ~memory floor.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rot_layer_cm(__half* __restrict__ data, int M,
                             const int* __restrict__ I, const int* __restrict__ J,
                             const float* __restrict__ C, const float* __restrict__ S) {
    int p = blockIdx.y;
    long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int i = I[p], j = J[p]; float c = C[p], s = S[p];
    long bi = (long)i * M + m, bj = (long)j * M + m;
    float a = __half2float(data[bi]), b = __half2float(data[bj]);
    data[bi] = __float2half(c * a - s * b);
    data[bj] = __float2half(s * a + c * b);
}

// cyc_feat:(C,L) global feature index per (cycle, local position). swap_* :(C,Ns), first half are
// layer-1 (tau) rotations, second half layer-2 (rho); each rotation acts on local positions (lo,hi)
// with out_lo = cos*lo - sin*hi, out_hi = sin*lo + cos*hi.
__global__ void rot_cycle_cm(__half* __restrict__ data, int M, int L, int Ns,
                             const int* __restrict__ cyc_feat,
                             const int* __restrict__ swap_lo, const int* __restrict__ swap_hi,
                             const float* __restrict__ swap_cos, const float* __restrict__ swap_sin) {
    int c = blockIdx.y;
    long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const int* feat = cyc_feat + (long)c * L;
    float v[16];                                    // L <= 16
    #pragma unroll
    for (int k = 0; k < L; ++k) v[k] = __half2float(data[(long)feat[k] * M + m]);
    const int* lo = swap_lo + c * Ns; const int* hi = swap_hi + c * Ns;
    const float* cs = swap_cos + c * Ns; const float* sn = swap_sin + c * Ns;
    for (int s = 0; s < Ns; ++s) {
        int a = lo[s], b = hi[s]; float cc = cs[s], ss = sn[s];
        float va = v[a], vb = v[b];
        v[a] = cc * va - ss * vb; v[b] = ss * va + cc * vb;
    }
    #pragma unroll
    for (int k = 0; k < L; ++k) data[(long)feat[k] * M + m] = __float2half(v[k]);
}

// Vectorized per-cycle: each thread does 8 columns via float4 (8 fp16) loads -> ~8x fewer memory
// instructions, and 8 independent rotation chains to hide latency. Cycle swap-table cached in shared.
#define LMAX 11
#define NSMAX 10
__global__ void rot_cycle_cm_vec(__half* __restrict__ data, int M, int L, int Ns,
                                 const int* __restrict__ cyc_feat,
                                 const int* __restrict__ swap_lo, const int* __restrict__ swap_hi,
                                 const float* __restrict__ swap_cos, const float* __restrict__ swap_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ int slo[NSMAX], shi[NSMAX];
    __shared__ float scs[NSMAX], ssn[NSMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    if (threadIdx.x < Ns) {
        int o = c * Ns + threadIdx.x;
        slo[threadIdx.x] = swap_lo[o]; shi[threadIdx.x] = swap_hi[o];
        scs[threadIdx.x] = swap_cos[o]; ssn[threadIdx.x] = swap_sin[o];
    }
    __syncthreads();

    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (m_base >= M) return;

    // L,Ns are LMAX,NSMAX at compile time here so these loops unroll and fv[] stays in registers
    // (not local memory). Valid for the L=11 cycle case.
    float4 fv[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k)
        fv[k] = *reinterpret_cast<const float4*>(data + (long)sfeat[k] * M + m_base);

    for (int col = 0; col < 8; ++col) {                 // rolled: val[] reused, no register blowup
        float val[LMAX];
        #pragma unroll
        for (int k = 0; k < LMAX; ++k) val[k] = __half2float(reinterpret_cast<__half*>(&fv[k])[col]);
        #pragma unroll
        for (int s = 0; s < NSMAX; ++s) {
            int a = slo[s], b = shi[s]; float cc = scs[s], ss = ssn[s];
            float va = val[a], vb = val[b];
            val[a] = cc * va - ss * vb; val[b] = ss * va + cc * vb;
        }
        #pragma unroll
        for (int k = 0; k < LMAX; ++k) reinterpret_cast<__half*>(&fv[k])[col] = __float2half(val[k]);
    }
    #pragma unroll
    for (int k = 0; k < LMAX; ++k)
        *reinterpret_cast<float4*>(data + (long)sfeat[k] * M + m_base) = fv[k];
}

void launch_rot_cycle_vec(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * 8 - 1) / (256 * 8), n_cyc);
    rot_cycle_cm_vec<<<grid, block>>>((__half*)data, M, L, Ns,
        (const int*)cyc_feat, (const int*)swap_lo, (const int*)swap_hi, (const float*)swap_cos, (const float*)swap_sin);
}

// Parameterized columns-per-thread variant. COLS fp16 packed in VEC (VEC = COLS*2 bytes):
//   COLS=2 -> float (1 word), COLS=4 -> float2, COLS=8 -> float4 (== rot_cycle_cm_vec).
// Fewer COLS  -> fewer live VEC regs (fv[LMAX]) -> higher occupancy, but fewer independent
// in-flight loads per thread. Sweep to find the occupancy/MLP sweet spot for our memory-bound case.
template<int COLS, typename VEC>
__global__ void rot_cycle_cm_vecT(__half* __restrict__ data, int M, int L, int Ns,
                                  const int* __restrict__ cyc_feat,
                                  const int* __restrict__ swap_lo, const int* __restrict__ swap_hi,
                                  const float* __restrict__ swap_cos, const float* __restrict__ swap_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ int slo[NSMAX], shi[NSMAX];
    __shared__ float scs[NSMAX], ssn[NSMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    if (threadIdx.x < Ns) {
        int o = c * Ns + threadIdx.x;
        slo[threadIdx.x] = swap_lo[o]; shi[threadIdx.x] = swap_hi[o];
        scs[threadIdx.x] = swap_cos[o]; ssn[threadIdx.x] = swap_sin[o];
    }
    __syncthreads();

    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * COLS;
    if (m_base >= M) return;

    VEC fv[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k)
        fv[k] = *reinterpret_cast<const VEC*>(data + (long)sfeat[k] * M + m_base);

    for (int col = 0; col < COLS; ++col) {
        float val[LMAX];
        #pragma unroll
        for (int k = 0; k < LMAX; ++k) val[k] = __half2float(reinterpret_cast<__half*>(&fv[k])[col]);
        #pragma unroll
        for (int s = 0; s < NSMAX; ++s) {
            int a = slo[s], b = shi[s]; float cc = scs[s], ss = ssn[s];
            float va = val[a], vb = val[b];
            val[a] = cc * va - ss * vb; val[b] = ss * va + cc * vb;
        }
        #pragma unroll
        for (int k = 0; k < LMAX; ++k) reinterpret_cast<__half*>(&fv[k])[col] = __float2half(val[k]);
    }
    #pragma unroll
    for (int k = 0; k < LMAX; ++k)
        *reinterpret_cast<VEC*>(data + (long)sfeat[k] * M + m_base) = fv[k];
}

void launch_rot_cycle_vec_n(void* data, int M, int n_cyc, int L, int Ns, int cols,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * cols - 1) / (256 * cols), n_cyc);
    auto cf = (const int*)cyc_feat; auto lo = (const int*)swap_lo; auto hi = (const int*)swap_hi;
    auto co = (const float*)swap_cos; auto si = (const float*)swap_sin; auto d = (__half*)data;
    auto stm = at::cuda::getCurrentCUDAStream();
    if (cols == 2)      rot_cycle_cm_vecT<2, float ><<<grid, block, 0, stm>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else if (cols == 4) rot_cycle_cm_vecT<4, float2><<<grid, block, 0, stm>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else                rot_cycle_cm_vecT<8, float4><<<grid, block, 0, stm>>>(d, M, L, Ns, cf, lo, hi, co, si);
}

// COLS=4 via two __half2 per feature (no address-of-register; uses __low2float/__high2float
// intrinsics so it compiles correctly under __launch_bounds__). MINB = min blocks/SM target.
template<int MINB>
__global__ void __launch_bounds__(256, MINB) rot_cycle_cm_h2(__half* __restrict__ data, int M, int L, int Ns,
                                 const int* __restrict__ cyc_feat,
                                 const int* __restrict__ swap_lo, const int* __restrict__ swap_hi,
                                 const float* __restrict__ swap_cos, const float* __restrict__ swap_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ int slo[NSMAX], shi[NSMAX];
    __shared__ float scs[NSMAX], ssn[NSMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    if (threadIdx.x < Ns) {
        int o = c * Ns + threadIdx.x;
        slo[threadIdx.x] = swap_lo[o]; shi[threadIdx.x] = swap_hi[o];
        scs[threadIdx.x] = swap_cos[o]; ssn[threadIdx.x] = swap_sin[o];
    }
    __syncthreads();

    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;

    __half2 a[LMAX], b[LMAX];                       // a = cols (0,1), b = cols (2,3)
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
        a[k] = p[0]; b[k] = p[1];
    }
    float v0[LMAX], v1[LMAX], v2[LMAX], v3[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        v0[k] = __low2float(a[k]);  v1[k] = __high2float(a[k]);
        v2[k] = __low2float(b[k]);  v3[k] = __high2float(b[k]);
    }
    #pragma unroll
    for (int s = 0; s < NSMAX; ++s) {
        int lo = slo[s], hi = shi[s]; float cc = scs[s], ss = ssn[s];
        float x, y;
        x = v0[lo]; y = v0[hi]; v0[lo] = cc*x - ss*y; v0[hi] = ss*x + cc*y;
        x = v1[lo]; y = v1[hi]; v1[lo] = cc*x - ss*y; v1[hi] = ss*x + cc*y;
        x = v2[lo]; y = v2[hi]; v2[lo] = cc*x - ss*y; v2[hi] = ss*x + cc*y;
        x = v3[lo]; y = v3[hi]; v3[lo] = cc*x - ss*y; v3[hi] = ss*x + cc*y;
    }
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        a[k] = __floats2half2_rn(v0[k], v1[k]);
        b[k] = __floats2half2_rn(v2[k], v3[k]);
    }
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = a[k]; p[1] = b[k];
    }
}

// REGISTER-RESIDENT half2 rotation, COLS=4. Data stays as __half2[11] in registers; each of the 4
// columns is unpacked/computed/repacked with __low2float/__high2float/__halves2half2 intrinsics —
// NO address-of-register, so fv never spills to local memory (the bug that gave the cols=4 kernel
// 3.3x L1/L2 traffic). One val[11] reused across the 4 columns -> ~register-light, high occupancy.
__global__ void __launch_bounds__(256) rot_cycle_h2roll(__half* __restrict__ data, int M, int L, int Ns,
                                 const int* __restrict__ cyc_feat,
                                 const int* __restrict__ swap_lo, const int* __restrict__ swap_hi,
                                 const float* __restrict__ swap_cos, const float* __restrict__ swap_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ int slo[NSMAX], shi[NSMAX];
    __shared__ float scs[NSMAX], ssn[NSMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    if (threadIdx.x < Ns) {
        int o = c * Ns + threadIdx.x;
        slo[threadIdx.x] = swap_lo[o]; shi[threadIdx.x] = swap_hi[o];
        scs[threadIdx.x] = swap_cos[o]; ssn[threadIdx.x] = swap_sin[o];
    }
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;

    __half2 a[LMAX], b[LMAX];                        // a = cols(0,1), b = cols(2,3)
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
        a[k] = p[0]; b[k] = p[1];
    }
    // each column: unpack -> rotate -> repack, all via register intrinsics (no &fv[k])
    #define ROT_COL(EXTRACT, INSERT) {                                                  \
        float val[LMAX];                                                               \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) val[k] = (EXTRACT);           \
        _Pragma("unroll") for (int s = 0; s < NSMAX; ++s) {                            \
            int la = slo[s], lb = shi[s]; float cc = scs[s], ss = ssn[s];              \
            float va = val[la], vb = val[lb];                                          \
            val[la] = cc * va - ss * vb; val[lb] = ss * va + cc * vb;                  \
        }                                                                              \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) { INSERT; }                   \
    }
    ROT_COL(__low2float(a[k]),  a[k] = __halves2half2(__float2half(val[k]), __high2half(a[k])))
    ROT_COL(__high2float(a[k]), a[k] = __halves2half2(__low2half(a[k]),  __float2half(val[k])))
    ROT_COL(__low2float(b[k]),  b[k] = __halves2half2(__float2half(val[k]), __high2half(b[k])))
    ROT_COL(__high2float(b[k]), b[k] = __halves2half2(__low2half(b[k]),  __float2half(val[k])))
    #undef ROT_COL

    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = a[k]; p[1] = b[k];
    }
}

void launch_rot_h2roll(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * 4 - 1) / (256 * 4), n_cyc);
    rot_cycle_h2roll<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((__half*)data, M, L, Ns,
        (const int*)cyc_feat, (const int*)swap_lo, (const int*)swap_hi, (const float*)swap_cos, (const float*)swap_sin);
}

// FIXED-POSITION rotation: the 10 swap pairs are compile-time constants (the involution structure of
// the 11-cycle is identical for every cycle), so val[] is statically indexed and stays in REGISTERS
// (no local-memory spill -> no L1/L2 amplification). Only cos[10] and SIGNED sin[10] vary per cycle
// (the lo/hi direction is folded into sin's sign at setup). COLS=4 via two __half2 in registers.
//   tau pairs: (1,10)(2,9)(3,8)(4,7)(5,6) ; rho pairs: (0,1)(2,10)(3,9)(4,8)(5,7)
__global__ void __launch_bounds__(256) rot_cycle_fixed(__half* __restrict__ data, int M,
                                 const int* __restrict__ cyc_feat,
                                 const float* __restrict__ cyc_cos, const float* __restrict__ cyc_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ float scs[10], ssn[10];
    if (threadIdx.x < LMAX) sfeat[threadIdx.x] = cyc_feat[(long)c * LMAX + threadIdx.x];
    if (threadIdx.x < 10) { scs[threadIdx.x] = cyc_cos[c * 10 + threadIdx.x]; ssn[threadIdx.x] = cyc_sin[c * 10 + threadIdx.x]; }
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;

    __half2 a[LMAX], b[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
        a[k] = p[0]; b[k] = p[1];
    }
    // LITERAL pair indices -> val[] statically indexed -> stays in registers (no local-mem spill).
    #define RP(LO, HI, I) { float va = val[LO], vb = val[HI]; float cc = scs[I], ss = ssn[I]; \
                            val[LO] = cc * va - ss * vb; val[HI] = ss * va + cc * vb; }
    #define ROT_COLF(EXTRACT, INSERT) {                                                \
        float val[LMAX];                                                               \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) val[k] = (EXTRACT);           \
        RP(1,10,0) RP(2,9,1) RP(3,8,2) RP(4,7,3) RP(5,6,4)                             \
        RP(0,1,5) RP(2,10,6) RP(3,9,7) RP(4,8,8) RP(5,7,9)                             \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) { INSERT; }                   \
    }
    ROT_COLF(__low2float(a[k]),  a[k] = __halves2half2(__float2half(val[k]), __high2half(a[k])))
    ROT_COLF(__high2float(a[k]), a[k] = __halves2half2(__low2half(a[k]),  __float2half(val[k])))
    ROT_COLF(__low2float(b[k]),  b[k] = __halves2half2(__float2half(val[k]), __high2half(b[k])))
    ROT_COLF(__high2float(b[k]), b[k] = __halves2half2(__low2half(b[k]),  __float2half(val[k])))
    #undef ROT_COLF
    #undef RP

    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = a[k]; p[1] = b[k];
    }
}

void launch_rot_fixed(void* data, int M, int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * 4 - 1) / (256 * 4), n_cyc);
    rot_cycle_fixed<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((__half*)data, M,
        (const int*)cyc_feat, (const float*)cyc_cos, (const float*)cyc_sin);
}

// FREE-READ upper bound: identical to rot_cycle_fixed but every thread READS from m=0 (the same few
// cache lines -> L2-resident, ~no HBM read) while doing full compute and writing to its real m_base.
// Measures the ceiling of removing the intermediate's HBM read (what perfect fusion would buy).
__global__ void __launch_bounds__(256) rot_cycle_fixed_fr(__half* __restrict__ data, int M,
                                 const int* __restrict__ cyc_feat,
                                 const float* __restrict__ cyc_cos, const float* __restrict__ cyc_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ float scs[10], ssn[10];
    if (threadIdx.x < LMAX) sfeat[threadIdx.x] = cyc_feat[(long)c * LMAX + threadIdx.x];
    if (threadIdx.x < 10) { scs[threadIdx.x] = cyc_cos[c * 10 + threadIdx.x]; ssn[threadIdx.x] = cyc_sin[c * 10 + threadIdx.x]; }
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;
    __half2 a[LMAX], b[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {                                 // READ from m=0 (cached), not m_base
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + 0);
        a[k] = p[0]; b[k] = p[1];
    }
    #define RP(LO, HI, I) { float va = val[LO], vb = val[HI]; float cc = scs[I], ss = ssn[I]; \
                            val[LO] = cc * va - ss * vb; val[HI] = ss * va + cc * vb; }
    #define ROT_COLF(EXTRACT, INSERT) {                                                \
        float val[LMAX];                                                               \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) val[k] = (EXTRACT);           \
        RP(1,10,0) RP(2,9,1) RP(3,8,2) RP(4,7,3) RP(5,6,4)                             \
        RP(0,1,5) RP(2,10,6) RP(3,9,7) RP(4,8,8) RP(5,7,9)                             \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) { INSERT; }                   \
    }
    ROT_COLF(__low2float(a[k]),  a[k] = __halves2half2(__float2half(val[k]), __high2half(a[k])))
    ROT_COLF(__high2float(a[k]), a[k] = __halves2half2(__low2half(a[k]),  __float2half(val[k])))
    ROT_COLF(__low2float(b[k]),  b[k] = __halves2half2(__float2half(val[k]), __high2half(b[k])))
    ROT_COLF(__high2float(b[k]), b[k] = __halves2half2(__low2half(b[k]),  __float2half(val[k])))
    #undef ROT_COLF
    #undef RP
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {                                 // WRITE to real m_base
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = a[k]; p[1] = b[k];
    }
}
void launch_rot_fixed_fr(void* data, int M, int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * 4 - 1) / (256 * 4), n_cyc);
    rot_cycle_fixed_fr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((__half*)data, M,
        (const int*)cyc_feat, (const float*)cyc_cos, (const float*)cyc_sin);
}

// Strided-IO fixed rotation: READ feature f's column m from in[f*ld_in + m] (m in [0,Mcols)), WRITE the
// rotated result to out[f*ld_out + off + m]. Lets the final rotation write straight into a strided
// slab of the (n_out, M_full) output (ld_out=M_full, off=col_start) — NO assembly copy.
__global__ void __launch_bounds__(256) rot_cycle_fixed_io(const __half* __restrict__ in, __half* __restrict__ out,
                                 int Mcols, long ld_in, long ld_out, long off,
                                 const int* __restrict__ cyc_feat,
                                 const float* __restrict__ cyc_cos, const float* __restrict__ cyc_sin) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX]; __shared__ float scs[10], ssn[10];
    if (threadIdx.x < LMAX) sfeat[threadIdx.x] = cyc_feat[(long)c * LMAX + threadIdx.x];
    if (threadIdx.x < 10) { scs[threadIdx.x] = cyc_cos[c * 10 + threadIdx.x]; ssn[threadIdx.x] = cyc_sin[c * 10 + threadIdx.x]; }
    __syncthreads();
    long m = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m >= Mcols) return;
    __half2 a[LMAX], b[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(in + (long)sfeat[k] * ld_in + m);
        a[k] = p[0]; b[k] = p[1];
    }
    #define RP(LO, HI, I) { float va = val[LO], vb = val[HI]; float cc = scs[I], ss = ssn[I]; \
                            val[LO] = cc * va - ss * vb; val[HI] = ss * va + cc * vb; }
    #define ROT_COLF(EXTRACT, INSERT) {                                                \
        float val[LMAX];                                                               \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) val[k] = (EXTRACT);           \
        RP(1,10,0) RP(2,9,1) RP(3,8,2) RP(4,7,3) RP(5,6,4)                             \
        RP(0,1,5) RP(2,10,6) RP(3,9,7) RP(4,8,8) RP(5,7,9)                             \
        _Pragma("unroll") for (int k = 0; k < LMAX; ++k) { INSERT; }                   \
    }
    ROT_COLF(__low2float(a[k]),  a[k] = __halves2half2(__float2half(val[k]), __high2half(a[k])))
    ROT_COLF(__high2float(a[k]), a[k] = __halves2half2(__low2half(a[k]),  __float2half(val[k])))
    ROT_COLF(__low2float(b[k]),  b[k] = __halves2half2(__float2half(val[k]), __high2half(b[k])))
    ROT_COLF(__high2float(b[k]), b[k] = __halves2half2(__low2half(b[k]),  __float2half(val[k])))
    #undef ROT_COLF
    #undef RP
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(out + (long)sfeat[k] * ld_out + off + m);
        p[0] = a[k]; p[1] = b[k];
    }
}
void launch_rot_fixed_io(const void* in, void* out, int Mcols, long ld_in, long ld_out, long off,
                         int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin) {
    dim3 block(256);
    dim3 grid((Mcols + 256 * 4 - 1) / (256 * 4), n_cyc);
    rot_cycle_fixed_io<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)in, (__half*)out,
        Mcols, ld_in, ld_out, off, (const int*)cyc_feat, (const float*)cyc_cos, (const float*)cyc_sin);
}

void launch_rot_cycle_h2(void* data, int M, int n_cyc, int L, int Ns, int minb,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin) {
    dim3 block(256);
    dim3 grid((M + 256 * 4 - 1) / (256 * 4), n_cyc);
    auto cf = (const int*)cyc_feat; auto lo = (const int*)swap_lo; auto hi = (const int*)swap_hi;
    auto co = (const float*)swap_cos; auto si = (const float*)swap_sin; auto d = (__half*)data;
    if (minb == 3)      rot_cycle_cm_h2<3><<<grid, block>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else if (minb == 4) rot_cycle_cm_h2<4><<<grid, block>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else if (minb == 6) rot_cycle_cm_h2<6><<<grid, block>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else if (minb == 8) rot_cycle_cm_h2<8><<<grid, block>>>(d, M, L, Ns, cf, lo, hi, co, si);
    else                rot_cycle_cm_h2<1><<<grid, block>>>(d, M, L, Ns, cf, lo, hi, co, si);
}

// Decompose the rotation's memory round-trip: same per-cycle access pattern (11 features, COLS=4
// via 2 half2), but isolate read / write / read+write to attribute where the time goes.
__global__ void rot_loadonly(__half* __restrict__ data, int M, int L,
                             const int* __restrict__ cyc_feat, float sentinel) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
        __half2 a = p[0], b = p[1];
        acc += __low2float(a) + __high2float(a) + __low2float(b) + __high2float(b);
    }
    if (acc == sentinel) data[m_base] = __float2half(acc);   // anti-DCE, ~never taken
}

__global__ void rot_storeonly(__half* __restrict__ data, int M, int L,
                              const int* __restrict__ cyc_feat) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;
    __half2 v = __floats2half2_rn(1.f, 1.f);
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = v; p[1] = v;
    }
}

__global__ void rot_rw(__half* __restrict__ data, int M, int L,
                       const int* __restrict__ cyc_feat) {
    int c = blockIdx.y;
    __shared__ int sfeat[LMAX];
    if (threadIdx.x < L) sfeat[threadIdx.x] = cyc_feat[(long)c * L + threadIdx.x];
    __syncthreads();
    long m_base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (m_base >= M) return;
    __half2 scale = __floats2half2_rn(1.00001f, 1.00001f);   // runtime-ish, cheap, blocks DCE
    __half2 a[LMAX], b[LMAX];
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
        a[k] = __hmul2(p[0], scale); b[k] = __hmul2(p[1], scale);
    }
    #pragma unroll
    for (int k = 0; k < LMAX; ++k) {
        __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
        p[0] = a[k]; p[1] = b[k];
    }
}

void launch_rot_probe(void* data, int M, int n_cyc, int L, int mode, const void* cyc_feat) {
    dim3 block(256);
    dim3 grid((M + 256 * 4 - 1) / (256 * 4), n_cyc);
    auto d = (__half*)data; auto cf = (const int*)cyc_feat;
    if (mode == 0)      rot_loadonly<<<grid, block>>>(d, M, L, cf, -123456.0f);
    else if (mode == 1) rot_storeonly<<<grid, block>>>(d, M, L, cf);
    else                rot_rw<<<grid, block>>>(d, M, L, cf);
}

void launch_rot_colmajor(void* data, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2) {
    dim3 block(256);
    rot_layer_cm<<<dim3((M + 255) / 256, P1), block>>>((__half*)data, M, (const int*)I1, (const int*)J1, (const float*)C1, (const float*)S1);
    rot_layer_cm<<<dim3((M + 255) / 256, P2), block>>>((__half*)data, M, (const int*)I2, (const int*)J2, (const float*)C2, (const float*)S2);
}

void launch_rot_cycle(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin) {
    dim3 block(256);
    rot_cycle_cm<<<dim3((M + 255) / 256, n_cyc), block>>>((__half*)data, M, L, Ns,
        (const int*)cyc_feat, (const int*)swap_lo, (const int*)swap_hi, (const float*)swap_cos, (const float*)swap_sin);
}
