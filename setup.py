#setup.py
#python setup.py build_ext --inplace

import os
import shutil
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


# -------------------- Remove old cython files and folders -------------------------------------------------
''' Delete files: 'cosmos_cy.c' and 'cosmos_cy.so'
https://www.w3schools.com/python/python_file_remove.asp '''
if os.path.exists("cosmos_cy.c"):
    os.remove("cosmos_cy.c")
else:
    print "cosmos_cy.c file does not exist."


if os.path.exists("cosmos_cy.so"):
    os.remove("cosmos_cy.so")
else:
    print "cosmos_cy.so file does not exist."


''' Check if folder 'build' exists :
https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python '''

''' Remove folder:
 https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python '''

if( not os.path.isdir("build") ): 
    print "build directory does not exist." 

if( os.path.isdir("build") ): 
    shutil.rmtree('build') 


# -------------------- Cython setup ------------------------------------------------------------------------
ext_modules = [Extension(
    name="cosmos_cy",
    sources=["cosmos_cy.pyx", "potential.c"],
    # extra_objects=["fc.o"],  # if you compile fc.cpp separately
    include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
    language="c",
    # libraries=
    extra_compile_args = ['-O3'],
    # extra_link_args = "...".)
    )]

setup(
    name = 'cosmos_cy',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,)
