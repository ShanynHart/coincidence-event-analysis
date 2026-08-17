import os
import sys
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from setuptools import setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "CLF_LaBr3POLARIS_utils",
        sources=["CLF_LaBr3POLARIS_utils.pyx"],
        include_dirs=[np.get_include()],  # Include NumPy headers
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
