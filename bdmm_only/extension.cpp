#include <torch/extension.h>
#include <string>

void run_bdmm(const void* rX, const void* rW, void* rY,
              int M, int N, int K, int nb, int stride, int Ktot);

// tile-config-swept variant (kernel_cfg.cu)
int run_bdmm_cfg(int cfg, const void* X, int Ktot, const void* W, int K, int nb, int stride,
                 void* Y, int M, int N);
int bdmm_cfg_count();
int bdmm_cfg_desc(int cfg, char* out, int out_len);

// Standalone strided block-diagonal matmul (the swz_s1 producer, isolated).
//
// X : (M, Ktot)   fp16 row-major — input
// W : (nb, K, N)  fp16           — per-block weights; block b reads X[:, b*stride : b*stride+K]
// stride : int    — column stride between adjacent blocks
// Returns Y : (nb*N, M) fp16 row-major (feature-major), same layout as readonce_swz_ext.swz_s1.
torch::Tensor bdmm(torch::Tensor X, torch::Tensor W, int64_t stride) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(W.dim() == 3, "W must be (nb, K, N)");
    X = X.contiguous(); W = W.contiguous();
    int M = X.size(0), Ktot = X.size(1);
    int nb = W.size(0), K = W.size(1), N = W.size(2);
    auto Y = torch::empty({nb * N, M}, X.options());
    run_bdmm(X.data_ptr<at::Half>(), W.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
             M, N, K, nb, (int)stride, Ktot);
    return Y;
}

// Same op, writing into a caller-provided (nb*N, M) buffer (for reuse / profiling harnesses).
void bdmm_into(torch::Tensor X, torch::Tensor W, torch::Tensor Y, int64_t stride) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kFloat16, "Y must be CUDA fp16");
    X = X.contiguous(); W = W.contiguous();
    int M = X.size(0), Ktot = X.size(1);
    int nb = W.size(0), K = W.size(1), N = W.size(2);
    TORCH_CHECK(Y.numel() == (int64_t)nb * N * M, "Y must be (nb*N, M)");
    run_bdmm(X.data_ptr<at::Half>(), W.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
             M, N, K, nb, (int)stride, Ktot);
}

// --- tile-config-swept variants: `cfg` picks the compile-time CUTLASS tile ---
torch::Tensor bdmm_cfg(int64_t cfg, torch::Tensor X, torch::Tensor W, int64_t stride) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(W.dim() == 3, "W must be (nb, K, N)");
    X = X.contiguous(); W = W.contiguous();
    int M = X.size(0), Ktot = X.size(1);
    int nb = W.size(0), K = W.size(1), N = W.size(2);
    auto Y = torch::empty({nb * N, M}, X.options());
    int rc = run_bdmm_cfg((int)cfg, X.data_ptr<at::Half>(), Ktot,
                          W.data_ptr<at::Half>(), K, nb, (int)stride, Y.data_ptr<at::Half>(), M, N);
    TORCH_CHECK(rc == 0, "bdmm_cfg: cfg ", cfg, " failed (rc=", rc, ")");
    return Y;
}

void bdmm_cfg_into(int64_t cfg, torch::Tensor X, torch::Tensor W, torch::Tensor Y, int64_t stride) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16, "X must be CUDA fp16");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat16, "W must be CUDA fp16");
    TORCH_CHECK(Y.is_cuda() && Y.dtype() == torch::kFloat16, "Y must be CUDA fp16");
    X = X.contiguous(); W = W.contiguous();
    int M = X.size(0), Ktot = X.size(1);
    int nb = W.size(0), K = W.size(1), N = W.size(2);
    TORCH_CHECK(Y.numel() == (int64_t)nb * N * M, "Y must be (nb*N, M)");
    int rc = run_bdmm_cfg((int)cfg, X.data_ptr<at::Half>(), Ktot,
                          W.data_ptr<at::Half>(), K, nb, (int)stride, Y.data_ptr<at::Half>(), M, N);
    TORCH_CHECK(rc == 0, "bdmm_cfg_into: cfg ", cfg, " failed (rc=", rc, ")");
}

std::string bdmm_cfg_desc_str(int64_t cfg) {
    char buf[160]; bdmm_cfg_desc((int)cfg, buf, sizeof(buf)); return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bdmm", &bdmm,
          "Strided block-diagonal matmul: X(M,Ktot) @ blockdiag W(nb,K,N) -> Y(nb*N, M)");
    m.def("bdmm_into", &bdmm_into, "bdmm into a caller-provided (nb*N, M) buffer");
    m.def("bdmm_cfg", &bdmm_cfg, "bdmm with a swept compile-time tile (cfg selects the tile)");
    m.def("bdmm_cfg_into", &bdmm_cfg_into, "bdmm_cfg into a caller-provided (nb*N, M) buffer");
    m.def("bdmm_cfg_count", &bdmm_cfg_count, "number of tile configs");
    m.def("bdmm_cfg_desc", &bdmm_cfg_desc_str, "tile description + achieved occupancy (blocks/SM)");
}
