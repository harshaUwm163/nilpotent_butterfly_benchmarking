#include <torch/extension.h>

int launch_rot_persist(void* data, int M, int n_cyc, const void* cf, const void* cc, const void* sn,
                       int threads, int vec, int minb, int stream, int ctas);
int rot_persist_occupancy(int threads, int vec, int minb, int stream);
int rot_persist_regs(int threads, int vec, int minb, int stream);

void rot_persist(torch::Tensor y, torch::Tensor cyc_feat, torch::Tensor cyc_cos, torch::Tensor cyc_sin,
                 int64_t threads, int64_t vec, int64_t minb, int64_t stream, int64_t ctas) {
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    const int M = (int)y.size(1);
    TORCH_CHECK(M % (2 * (int)vec) == 0, "M must be divisible by 2*vec");
    const int rc = launch_rot_persist(y.data_ptr<at::Half>(), M, (int)cyc_feat.size(0),
        cyc_feat.data_ptr<int>(), cyc_cos.data_ptr<float>(), cyc_sin.data_ptr<float>(),
        (int)threads, (int)vec, (int)minb, (int)stream, (int)ctas);
    TORCH_CHECK(rc == 0, "rot_persist: (threads=", threads, ", vec=", vec, ", minb=", minb, ") not instantiated");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rot_persist", &rot_persist, "persistent-CTA Givens rotation (fixed grid)");
    m.def("occupancy", &rot_persist_occupancy, "max active CTAs/SM");
    m.def("regs", &rot_persist_regs, "registers per thread");
}
