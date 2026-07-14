#include <torch/extension.h>
#include <string>

void launch_fused_stage(
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2);

// config-swept variant (kernel_cfg.cu)
int run_fused_stage_cfg(int cfg,
    const void* X, int Ktot,
    const void* W, int K, int nb, int stride,
    void* Y, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2);
int  fused_stage_cfg_count();
int  fused_stage_cfg_desc(int cfg, char* out, int out_len);

// Fused FULL stage: all-block block-diagonal matmul + two Givens rotation layers, in shared memory.
//
// X     : (M, Ktot=2048)  fp16 — full input matrix
// W     : (nb=32, K=128, 64) fp16 — per-block weights (block_w=64 == N_OUT)
// stride: int — column stride between adjacent blocks (= 64); block b reads X[:, b*stride : ...]
// I1,J1 : (P1,) int32 — tau rotation pair indices over [0, 2048)
// C1,S1 : (P1,) float32
// I2,J2 : (P2,) int32 — rho rotation pair indices over [0, 2048)
// C2,S2 : (P2,) float32
// Returns Y : (M, 2048) fp16  (== fused_two_givens(block_gemm_nopad(X, W, stride), ...))
torch::Tensor fused_stage_gemm_rot(
    torch::Tensor X,
    torch::Tensor W,
    int64_t       stride,
    torch::Tensor I1, torch::Tensor J1,
    torch::Tensor C1, torch::Tensor S1,
    torch::Tensor I2, torch::Tensor J2,
    torch::Tensor C2, torch::Tensor S2)
{
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(W.dim() == 3, "W must be (nb, K, 64)");
    X = X.contiguous(); W = W.contiguous();
    const int M     = X.size(0);
    const int Ktot  = X.size(1);
    const int nb    = W.size(0);
    const int K     = W.size(1);
    const int N     = W.size(2);
    TORCH_CHECK(N == 64, "W block width must be 64 (N_OUT is compile-time 64)");
    TORCH_CHECK(nb * N == 2048, "kernel is specialized for 2048 total features (nb*64)");
    TORCH_CHECK(Ktot == 2048, "kernel is specialized for Ktot=2048");
    auto Y = torch::empty({M, nb * N}, X.options());
    launch_fused_stage(
        X.data_ptr<at::Half>(), Ktot,
        W.data_ptr<at::Half>(), K, nb, (int)stride,
        Y.data_ptr<at::Half>(), M,
        I1.data_ptr<int>(), J1.data_ptr<int>(),
        C1.data_ptr<float>(), S1.data_ptr<float>(), (int)I1.size(0),
        I2.data_ptr<int>(), J2.data_ptr<int>(),
        C2.data_ptr<float>(), S2.data_ptr<float>(), (int)I2.size(0));
    return Y;
}

// Config-swept dispatch: same op as fused_stage_gemm_rot but pick the compile-time tile/warp/
// stage/min-block/buffer config by integer `cfg`. Raises on a config that can't run so the
// sweep reports it as skipped.
torch::Tensor fused_stage_gemm_rot_cfg(
    int64_t       cfg,
    torch::Tensor X,
    torch::Tensor W,
    int64_t       stride,
    torch::Tensor I1, torch::Tensor J1,
    torch::Tensor C1, torch::Tensor S1,
    torch::Tensor I2, torch::Tensor J2,
    torch::Tensor C2, torch::Tensor S2)
{
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(W.dim() == 3, "W must be (nb, K, 64)");
    X = X.contiguous(); W = W.contiguous();
    const int M     = X.size(0);
    const int Ktot  = X.size(1);
    const int nb    = W.size(0);
    const int K     = W.size(1);
    const int N     = W.size(2);
    TORCH_CHECK(N == 64, "W block width must be 64 (N_OUT is compile-time 64)");
    TORCH_CHECK(nb * N == 2048, "kernel is specialized for 2048 total features (nb*64)");
    TORCH_CHECK(Ktot == 2048, "kernel is specialized for Ktot=2048");
    auto Y = torch::empty({M, nb * N}, X.options());
    int rc = run_fused_stage_cfg((int)cfg,
        X.data_ptr<at::Half>(), Ktot,
        W.data_ptr<at::Half>(), K, nb, (int)stride,
        Y.data_ptr<at::Half>(), M,
        I1.data_ptr<int>(), J1.data_ptr<int>(),
        C1.data_ptr<float>(), S1.data_ptr<float>(), (int)I1.size(0),
        I2.data_ptr<int>(), J2.data_ptr<int>(),
        C2.data_ptr<float>(), S2.data_ptr<float>(), (int)I2.size(0));
    TORCH_CHECK(rc == 0, "fused_stage_gemm_rot_cfg: config ", cfg, " failed (rc=", rc, ")");
    return Y;
}

std::string fused_stage_cfg_desc_str(int64_t cfg) {
    char buf[256];
    fused_stage_cfg_desc((int)cfg, buf, sizeof(buf));
    return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_stage_gemm_rot", &fused_stage_gemm_rot,
          "Fused all-block (M,2048) block-diagonal matmul + two full-width Givens layers -> (M,2048)");
    m.def("fused_stage_gemm_rot_cfg", &fused_stage_gemm_rot_cfg,
          "Config-swept fused stage (cfg selects compile-time tile/warp/stage/buffer config)");
    m.def("fused_stage_cfg_count", &fused_stage_cfg_count, "number of sweep configs");
    m.def("fused_stage_cfg_desc",  &fused_stage_cfg_desc_str, "human-readable cfg + achieved occupancy");
}
