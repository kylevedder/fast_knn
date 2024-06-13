from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_knn_pkg",
    ext_modules=[
        CUDAExtension(
            name="custom_knn_ext",
            sources=[
                "custom_knn/knn.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
