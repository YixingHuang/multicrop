
from setuptools import setup
import os
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension

cl_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"
os.environ["PATH"] = os.environ["PATH"] + ";" + cl_path
print(os.environ["PATH"])

ext_modules = []

#https://stackoverflow.com/questions/45600866/add-c-function-to-existing-python-module-with-pybind11

if torch.cuda.is_available():
    extension = CUDAExtension(
        name='multicrop',
        sources = [
            'src/gpu_ops.cpp',
            'src/extract_glimpses_cuda.cu',
            'src/extract_glimpses.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp'],
                            'nvcc': ['-O2']
                            },
    )
else:
    extension = CppExtension(
        name='multicrop',
        sources = [
            'src/cpu_ops.cpp',
            'src/extract_glimpses.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp']}
    )

ext_modules.append(extension)


setup(
    name='multicrop',
    version='1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)})
