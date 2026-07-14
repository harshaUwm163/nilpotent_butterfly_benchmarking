// Persistent-CTA Givens rotation, co-resident with a BDMM GEMM.
//
// The stock rot_cycle_fixed launches ~n_cyc * M/1024 CTAs. Against a BDMM grid of comparable size
// the work distributor never dispatches it until BDMM drains, so two-stream "racing" only overlaps
// the tail. Here the grid is FIXED (typically 1 CTA/SM) and launched first, so the CTAs are resident
// from t=0 and BDMM backfills around them. The kernel then self-throttles to leftover bandwidth.
//
// Register footprint is the binding constraint on co-residency: to sit alongside a BDMM CTA that
// holds R regs, this CTA must fit in (65536 - R). THREADS/VEC/MINB expose that knob.
//   VEC   = __half2 lanes per thread (COLS = 2*VEC); VEC=2 matches rot_cycle_fixed.
//   MINB  = __launch_bounds__ min-CTAs-per-SM; caps regs/thread at 65536/(MINB*THREADS).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define LMAX 11

// STREAM=1: evict-first (ld.global.cs / st.global.cs). The rotation streams the whole intermediate
// through L2 exactly once and never reuses a line, but the default policy still allocates it --
// evicting the X tiles that the co-resident GEMM re-reads across its column-blocks. Marking these
// accesses streaming keeps the GEMM's working set in L2.
union h2u { __half2 h; unsigned int u; };

__device__ __forceinline__ __half2 ld_h2(const __half2* p, int stream) {
    h2u r;
    if (stream) r.u = __ldcs(reinterpret_cast<const unsigned int*>(p));
    else        r.h = *p;
    return r.h;
}
__device__ __forceinline__ void st_h2(__half2* p, __half2 v, int stream) {
    h2u s; s.h = v;
    if (stream) __stcs(reinterpret_cast<unsigned int*>(p), s.u);
    else        *p = v;
}

template <int THREADS, int VEC, int MINB, int STREAM>
__global__ void __launch_bounds__(THREADS, MINB) rot_persist(
    __half* __restrict__ data, int M, int n_cyc,
    const int* __restrict__ cyc_feat,
    const float* __restrict__ cyc_cos,
    const float* __restrict__ cyc_sin)
{
    __shared__ int   sfeat[LMAX];
    __shared__ float scs[10], ssn[10];

    const long cols_per_tile = (long)THREADS * 2 * VEC;
    const long tiles_per_cyc = (M + cols_per_tile - 1) / cols_per_tile;
    const long total         = (long)n_cyc * tiles_per_cyc;

    int cur_c = -1;
    // t is uniform across the CTA, so the __syncthreads() below is never divergent.
    for (long t = blockIdx.x; t < total; t += gridDim.x) {
        const int  c  = (int)(t / tiles_per_cyc);
        const long bx = t - (long)c * tiles_per_cyc;
        if (c != cur_c) {
            __syncthreads();
            if (threadIdx.x < LMAX) sfeat[threadIdx.x] = cyc_feat[(long)c * LMAX + threadIdx.x];
            if (threadIdx.x < 10) {
                scs[threadIdx.x] = cyc_cos[c * 10 + threadIdx.x];
                ssn[threadIdx.x] = cyc_sin[c * 10 + threadIdx.x];
            }
            __syncthreads();
            cur_c = c;
        }
        const long m_base = (bx * THREADS + threadIdx.x) * (2 * VEC);
        if (m_base >= M) continue;   // M % (2*VEC) == 0, so no partial lane

        __half2 v[VEC][LMAX];
        #pragma unroll
        for (int k = 0; k < LMAX; ++k) {
            const __half2* p = reinterpret_cast<const __half2*>(data + (long)sfeat[k] * M + m_base);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) v[j][k] = ld_h2(p + j, STREAM);
        }

        // Literal pair indices -> val[] statically indexed -> stays in registers (no local spill).
        #define RP(LO, HI, I) { float va = val[LO], vb = val[HI]; float cc = scs[I], ss = ssn[I]; \
                                val[LO] = cc * va - ss * vb; val[HI] = ss * va + cc * vb; }
        #define ROTL(EXTRACT, INSERT) {                                          \
            float val[LMAX];                                                     \
            _Pragma("unroll") for (int k = 0; k < LMAX; ++k) val[k] = (EXTRACT); \
            RP(1,10,0) RP(2,9,1) RP(3,8,2) RP(4,7,3) RP(5,6,4)                   \
            RP(0,1,5) RP(2,10,6) RP(3,9,7) RP(4,8,8) RP(5,7,9)                   \
            _Pragma("unroll") for (int k = 0; k < LMAX; ++k) { INSERT; } }
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            ROTL(__low2float(v[j][k]),  v[j][k] = __halves2half2(__float2half(val[k]), __high2half(v[j][k])))
            ROTL(__high2float(v[j][k]), v[j][k] = __halves2half2(__low2half(v[j][k]),  __float2half(val[k])))
        }
        #undef ROTL
        #undef RP

        #pragma unroll
        for (int k = 0; k < LMAX; ++k) {
            __half2* p = reinterpret_cast<__half2*>(data + (long)sfeat[k] * M + m_base);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) st_h2(p + j, v[j][k], STREAM);
        }
    }
}

// (threads, vec, minb) x stream. Keep the list small: these are the combos that fit some BDMM cfg.
#define ROT_VARIANTS(F) \
    F(128, 1, 1) F(128, 1, 4) F(128, 1, 6) F(128, 1, 8) \
    F(128, 2, 1) F(128, 2, 6) \
    F(256, 1, 1) F(256, 1, 4) \
    F(256, 2, 1) F(256, 2, 3) F(256, 2, 4)

template <int THREADS, int VEC, int MINB, int STREAM>
static void run(void* data, int M, int n_cyc, const void* cf, const void* cc, const void* sn, int ctas) {
    rot_persist<THREADS, VEC, MINB, STREAM><<<ctas, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (__half*)data, M, n_cyc, (const int*)cf, (const float*)cc, (const float*)sn);
}

// Returns 0 on success, -1 if the (threads,vec,minb,stream) combo is not instantiated.
int launch_rot_persist(void* data, int M, int n_cyc, const void* cf, const void* cc, const void* sn,
                       int threads, int vec, int minb, int stream, int ctas)
{
#define INST(T, V, B) \
    if (threads == T && vec == V && minb == B) { \
        if (stream) run<T, V, B, 1>(data, M, n_cyc, cf, cc, sn, ctas); \
        else        run<T, V, B, 0>(data, M, n_cyc, cf, cc, sn, ctas); \
        return 0; }
    ROT_VARIANTS(INST)
#undef INST
    return -1;
}

int rot_persist_occupancy(int threads, int vec, int minb, int stream) {
    int occ = -1;
#define OCC(T, V, B) \
    if (threads == T && vec == V && minb == B) { \
        if (stream) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, rot_persist<T, V, B, 1>, T, 0); \
        else        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, rot_persist<T, V, B, 0>, T, 0); }
    ROT_VARIANTS(OCC)
#undef OCC
    return occ;
}

int rot_persist_regs(int threads, int vec, int minb, int stream) {
    cudaFuncAttributes a{}; a.numRegs = -1;
#define REG(T, V, B) \
    if (threads == T && vec == V && minb == B) { \
        if (stream) cudaFuncGetAttributes(&a, rot_persist<T, V, B, 1>); \
        else        cudaFuncGetAttributes(&a, rot_persist<T, V, B, 0>); }
    ROT_VARIANTS(REG)
#undef REG
    return a.numRegs;
}
