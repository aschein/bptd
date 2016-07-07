#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os
import sys

if sys.platform == 'darwin':
    os.environ["CC"] = "g++-5"
    os.environ["CXX"] = "g++-5"

ext = [Extension( "cy_test", sources=["cy_test.pyx"] )]

setup(
   cmdclass={'build_ext' : build_ext}, 
   include_dirs = [np.get_include(), 'gslrandom'],   
   ext_modules=cythonize('**/*.pyx', include_path=['gslrandom']),
   )
