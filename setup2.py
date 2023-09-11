
from setuptools import setup
import os
from torch.utils import cpp_extension

cl_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"
os.environ["PATH"] = os.environ["PATH"] + ";" + cl_path
print(os.environ["PATH"])



setup(
    name='multicrop',
    ext_modules=[
        cpp_extension.CUDAExtension('multicrop',
                                    ['src/gpu_ops.cpp',
            'src/extract_glimpses_cuda.cu',
            'src/extract_glimpses.cpp'
                                     ]),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)}
)
