from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='givens_colmajor_ext', ext_modules=[CUDAExtension(name='givens_colmajor_ext',
    sources=['extension.cpp','kernel.cu'],
    extra_compile_args={'cxx':['-O3'],'nvcc':['-O3','-gencode=arch=compute_80,code=sm_80','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__']})],
    cmdclass={'build_ext':BuildExtension})
