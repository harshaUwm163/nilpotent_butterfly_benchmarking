from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_DIR = os.environ.get("CUTLASS_DIR", os.path.join(_HERE, "..", "cutlass"))

nvcc = ['-O3', '-gencode=arch=compute_80,code=sm_80',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__']
S = os.environ.get("BDMM_STAGES")            # pipeline depth override; unset -> kernel default (4)
if S:
    nvcc.append(f'-DBDMM_STAGES={S}')

setup(
    name='bdmm_ext',
    ext_modules=[CUDAExtension(
        name='bdmm_ext',
        sources=['extension.cpp', 'kernel.cu', 'kernel_cfg.cu'],
        include_dirs=[
            os.path.join(CUTLASS_DIR, 'include'),
            os.path.join(CUTLASS_DIR, 'tools/util/include'),
        ],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': nvcc},
    )],
    cmdclass={'build_ext': BuildExtension},
)
