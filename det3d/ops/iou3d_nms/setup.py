from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d_nms',
    ext_modules=[
        CUDAExtension('iou3d_nms_cuda', [
            'src/iou3d_nms.cpp',
            'src/iou3d_nms_kernel.cu',
        ],
    include_dirs= ['C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include'],
    extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2'],
                                                        }
    #     extra_compile_args = {'include_dirs': ['C:\\Program Files (x86)\\Microsoft\ Visual\ Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include'],
    # 'cxx': [],
    # 'nvcc': [
    #     '-D__CUDA_NO_HALF_OPERATORS__',
    #     '-D__CUDA_NO_HALF_CONVERSIONS__',
    #     '-D__CUDA_NO_HALF2_OPERATORS__',
    # ]}
                      )
    ],
    cmdclass={'build_ext': BuildExtension}
)
