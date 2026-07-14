#include <torch/extension.h>
void launch_rot_colmajor(void* data, int M,
    const void* I1, const void* J1, const void* C1, const void* S1, int P1,
    const void* I2, const void* J2, const void* C2, const void* S2, int P2);
void launch_rot_cycle_vec(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin);
void launch_rot_cycle_vec_n(void* data, int M, int n_cyc, int L, int Ns, int cols,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin);
void launch_rot_cycle_h2(void* data, int M, int n_cyc, int L, int Ns, int minb,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin);
void launch_rot_h2roll(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin);
void launch_rot_fixed(void* data, int M, int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin);
void launch_rot_fixed_fr(void* data, int M, int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin);
void launch_rot_fixed_io(const void* in, void* out, int Mcols, long ld_in, long ld_out, long off, int n_cyc, const void* cyc_feat, const void* cyc_cos, const void* cyc_sin);
void launch_rot_probe(void* data, int M, int n_cyc, int L, int mode, const void* cyc_feat);
void launch_rot_cycle(void* data, int M, int n_cyc, int L, int Ns,
    const void* cyc_feat, const void* swap_lo, const void* swap_hi, const void* swap_cos, const void* swap_sin);

torch::Tensor rot_colmajor(torch::Tensor y,
    torch::Tensor I1, torch::Tensor J1, torch::Tensor C1, torch::Tensor S1,
    torch::Tensor I2, torch::Tensor J2, torch::Tensor C2, torch::Tensor S2) {
    y = y.contiguous(); int M = y.size(1);
    launch_rot_colmajor(y.data_ptr<at::Half>(), M,
        I1.data_ptr<int>(),J1.data_ptr<int>(),C1.data_ptr<float>(),S1.data_ptr<float>(),(int)I1.size(0),
        I2.data_ptr<int>(),J2.data_ptr<int>(),C2.data_ptr<float>(),S2.data_ptr<float>(),(int)I2.size(0));
    return y;
}
torch::Tensor rot_cycle(torch::Tensor y, torch::Tensor cyc_feat,
    torch::Tensor swap_lo, torch::Tensor swap_hi, torch::Tensor swap_cos, torch::Tensor swap_sin) {
    y = y.contiguous(); int M = y.size(1);
    int n_cyc = cyc_feat.size(0), L = cyc_feat.size(1), Ns = swap_lo.size(1);
    launch_rot_cycle(y.data_ptr<at::Half>(), M, n_cyc, L, Ns,
        cyc_feat.data_ptr<int>(), swap_lo.data_ptr<int>(), swap_hi.data_ptr<int>(),
        swap_cos.data_ptr<float>(), swap_sin.data_ptr<float>());
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("rot_colmajor",&rot_colmajor,"per-pair 2-pass");
    m.def("rot_cycle",&rot_cycle,"per-cycle one pass");
    m.def("rot_cycle_vec",[](torch::Tensor y, torch::Tensor cf, torch::Tensor lo, torch::Tensor hi, torch::Tensor co, torch::Tensor si){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0),L=cf.size(1),Ns=lo.size(1);
        launch_rot_cycle_vec(y.data_ptr<at::Half>(),M,nc,L,Ns,cf.data_ptr<int>(),lo.data_ptr<int>(),hi.data_ptr<int>(),co.data_ptr<float>(),si.data_ptr<float>());
        return y;}, "vectorized per-cycle");
    m.def("rot_cycle_vec_n",[](torch::Tensor y, torch::Tensor cf, torch::Tensor lo, torch::Tensor hi, torch::Tensor co, torch::Tensor si, int cols){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0),L=cf.size(1),Ns=lo.size(1);
        launch_rot_cycle_vec_n(y.data_ptr<at::Half>(),M,nc,L,Ns,cols,cf.data_ptr<int>(),lo.data_ptr<int>(),hi.data_ptr<int>(),co.data_ptr<float>(),si.data_ptr<float>());
        return y;}, "vectorized per-cycle, configurable cols/thread");
    m.def("rot_cycle_h2",[](torch::Tensor y, torch::Tensor cf, torch::Tensor lo, torch::Tensor hi, torch::Tensor co, torch::Tensor si, int minb){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0),L=cf.size(1),Ns=lo.size(1);
        launch_rot_cycle_h2(y.data_ptr<at::Half>(),M,nc,L,Ns,minb,cf.data_ptr<int>(),lo.data_ptr<int>(),hi.data_ptr<int>(),co.data_ptr<float>(),si.data_ptr<float>());
        return y;}, "half2 COLS=4 with launch_bounds min-blocks");
    m.def("rot_probe",[](torch::Tensor y, torch::Tensor cf, int mode){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0),L=cf.size(1);
        launch_rot_probe(y.data_ptr<at::Half>(),M,nc,L,mode,cf.data_ptr<int>());
        return y;}, "mode 0=load-only 1=store-only 2=read+write, same access pattern");
    m.def("rot_h2roll",[](torch::Tensor y, torch::Tensor cf, torch::Tensor lo, torch::Tensor hi, torch::Tensor co, torch::Tensor si){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0),L=cf.size(1),Ns=lo.size(1);
        launch_rot_h2roll(y.data_ptr<at::Half>(),M,nc,L,Ns,cf.data_ptr<int>(),lo.data_ptr<int>(),hi.data_ptr<int>(),co.data_ptr<float>(),si.data_ptr<float>());
        return y;}, "register-resident half2 rolled COLS=4");
    m.def("rot_fixed",[](torch::Tensor y, torch::Tensor cf, torch::Tensor cc, torch::Tensor sn){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0);
        launch_rot_fixed(y.data_ptr<at::Half>(),M,nc,cf.data_ptr<int>(),cc.data_ptr<float>(),sn.data_ptr<float>());
        return y;}, "fixed-position compile-time-pairs rotation COLS=4 (cos[nc,10], signed sin[nc,10])");
    m.def("rot_fixed_fr",[](torch::Tensor y, torch::Tensor cf, torch::Tensor cc, torch::Tensor sn){
        y=y.contiguous(); int M=y.size(1); int nc=cf.size(0);
        launch_rot_fixed_fr(y.data_ptr<at::Half>(),M,nc,cf.data_ptr<int>(),cc.data_ptr<float>(),sn.data_ptr<float>());
        return y;}, "free-read upper bound of rot_fixed");
    m.def("rot_fixed_io",[](torch::Tensor in, torch::Tensor out, int64_t Mcols, int64_t ld_in, int64_t ld_out, int64_t off, torch::Tensor cf, torch::Tensor cc, torch::Tensor sn){
        int nc=cf.size(0);
        launch_rot_fixed_io(in.data_ptr<at::Half>(), out.data_ptr<at::Half>(), (int)Mcols, ld_in, ld_out, off, nc, cf.data_ptr<int>(), cc.data_ptr<float>(), sn.data_ptr<float>());
    }, "strided-IO fixed rotation: read contiguous, write strided slab");
}
