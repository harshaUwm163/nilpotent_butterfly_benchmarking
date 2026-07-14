from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_DIR = os.environ.get(
    "CUTLASS_DIR",
    os.path.join(_HERE, "..", "cutlass"))

setup(
    name='fused_stage_gemm_rot_ext',
    ext_modules=[CUDAExtension(
        name='fused_stage_gemm_rot_ext',
        sources=['extension.cpp', 'kernel.cu', 'kernel_cfg.cu'],
        include_dirs=[
            os.path.join(CUTLASS_DIR, 'include'),
            os.path.join(CUTLASS_DIR, 'tools/util/include'),
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-gencode=arch=compute_80,code=sm_80',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
            ],
        },
    )],
    cmdclass={'build_ext': BuildExtension},
)
