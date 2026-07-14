from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
C=os.environ.get("CUTLASS_DIR","/workspace/nilpotent_butterfly/ComposingLinearLayers/cutlass")
NAME=os.environ.get("EXTNAME","readonce_swz_ext")          # module name (lets us build 3- and 4-stage twins)
nvcc=['-O3','-gencode=arch=compute_80,code=sm_80','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__']
S=os.environ.get("SWZ_STAGES")                             # pipeline depth override; unset -> kernel default (4)
if S: nvcc.append(f'-DSWZ_STAGES={S}')
setup(name=NAME,ext_modules=[CUDAExtension(name=NAME,
    sources=['extension.cpp','kernel.cu'],include_dirs=[os.path.join(C,'include'),os.path.join(C,'tools/util/include')],
    extra_compile_args={'cxx':['-O3'],'nvcc':nvcc})],
    cmdclass={'build_ext':BuildExtension})
