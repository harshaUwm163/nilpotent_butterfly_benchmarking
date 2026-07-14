#include <torch/extension.h>
void run_swz_s1   (const void*, const void*, void*, int,int,int,int,int,int);
void run_swz_s2   (const void*, const void*, void*, int,int,int,int,int,int);
void run_noswz_s1 (const void*, const void*, void*, int,int,int,int,int,int);
void run_rowout_s1    (const void*, const void*, void*, int,int,int,int,int,int);
void run_rowout_swz_s1(const void*, const void*, void*, int,int,int,int,int,int);
torch::Tensor swz_s1(torch::Tensor X, torch::Tensor W, int64_t stride){
    X=X.contiguous();W=W.contiguous();
    int M=X.size(0),Ktot=X.size(1),nb=W.size(0),K=W.size(1),N=W.size(2),n_out=nb*N;
    auto Y=torch::empty({n_out,M},X.options());
    run_swz_s1(X.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,Ktot);
    return Y;
}
torch::Tensor swz_s2(torch::Tensor I, torch::Tensor W, int64_t stride){
    I=I.contiguous();W=W.contiguous();
    int n_out=I.size(0),M=I.size(1),nb=W.size(0),K=W.size(1),N=W.size(2);
    auto Y=torch::empty({n_out,M},I.options());
    run_swz_s2(I.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,n_out);
    return Y;
}
// write into a caller-provided output buffer (for the M-tiled L2-resident pipeline / CUDA graphs)
void swz_s1_into(torch::Tensor X, torch::Tensor W, torch::Tensor Y, int64_t stride){
    X=X.contiguous();W=W.contiguous();
    int M=X.size(0),Ktot=X.size(1),nb=W.size(0),K=W.size(1),N=W.size(2);
    run_swz_s1(X.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,Ktot);
}
void swz_s2_into(torch::Tensor I, torch::Tensor W, torch::Tensor Y, int64_t stride){
    I=I.contiguous();W=W.contiguous();
    int n_out=I.size(0),M=I.size(1),nb=W.size(0),K=W.size(1),N=W.size(2);
    run_swz_s2(I.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,n_out);
}
torch::Tensor rowout_swz_s1(torch::Tensor X, torch::Tensor W, int64_t stride){
    X=X.contiguous();W=W.contiguous();
    int M=X.size(0),Ktot=X.size(1),nb=W.size(0),K=W.size(1),N=W.size(2),n_out=nb*N;
    auto Y=torch::empty({M,n_out},X.options());
    run_rowout_swz_s1(X.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,Ktot);
    return Y;
}
torch::Tensor rowout_s1(torch::Tensor X, torch::Tensor W, int64_t stride){
    X=X.contiguous();W=W.contiguous();
    int M=X.size(0),Ktot=X.size(1),nb=W.size(0),K=W.size(1),N=W.size(2),n_out=nb*N;
    auto Y=torch::empty({M,n_out},X.options());
    run_rowout_s1(X.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,Ktot);
    return Y;
}
torch::Tensor noswz_s1(torch::Tensor X, torch::Tensor W, int64_t stride){
    X=X.contiguous();W=W.contiguous();
    int M=X.size(0),Ktot=X.size(1),nb=W.size(0),K=W.size(1),N=W.size(2),n_out=nb*N;
    auto Y=torch::empty({n_out,M},X.options());
    run_noswz_s1(X.data_ptr<at::Half>(),W.data_ptr<at::Half>(),Y.data_ptr<at::Half>(),M,N,K,nb,(int)stride,Ktot);
    return Y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("swz_s1",   &swz_s1,    "stage1 block GEMM with SwapZXBatchSwizzle (batch-fastest, L2 overlap reuse)");
    m.def("swz_s2",   &swz_s2,    "stage2 block GEMM with SwapZXBatchSwizzle");
    m.def("swz_s1_into", &swz_s1_into, "swz_s1 into provided buffer");
    m.def("swz_s2_into", &swz_s2_into, "swz_s2 into provided buffer");
    m.def("noswz_s1",  &noswz_s1,  "stage1 block GEMM with GemmIdentityThreadblockSwizzle (ablation)");
    m.def("rowout_s1",     &rowout_s1,     "stage1 X@W, both RowMajor, output (M,N_out), no swizzle");
    m.def("rowout_swz_s1", &rowout_swz_s1, "stage1 X@W, both RowMajor, output (M,N_out), SwapZXBatchSwizzle");
}
