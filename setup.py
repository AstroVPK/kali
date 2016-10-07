import numpy as np
import os
import platform
import shlex
import subprocess
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pdb


def which(program):
    """
    Mimic functionality of unix which command
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

if which('icpc') is not None:
    os.environ['CC'] = 'icpc'
    os.environ['CXX'] = 'icpc'
    VERFLAGS = ['-gxx-name=g++-4.8', '-std=c++11']
    CPPFLAGS = ['-O3', '-xHost', '-ip', '-parallel', '-funroll-loops', '-fno-alias', '-fno-fnalias',
                '-fargument-noalias', '-fstrict-aliasing', '-ansi-alias', '-fno-stack-protector-all', '-Wall']
    OMPFLAGS = ['-qopenmp', '-qopenmp-simd', '-qopt-report=5', '-qopt-report-phase=all']
    OMPFLAGS = ['-qopenmp', '-qopenmp-simd', '-qopt-report=5', '-qopt-report-phase=all']
    OMPLIBS = ['-liomp5']
    ALIGHFLAGS = ['-falign-functions']
else:
    os.environ['CC'] = 'g++'
    os.environ['CXX'] = 'g++'
    VERFLAGS = ['-std=c++11']
    CPPFLAGS = ['-O3']
    OMPFLAGS = ['-fopenmp', '-fopenmp-simd']
    OMPLIBS = []
    ALIGHFLAGS = []

INCLUDE = os.path.join(os.environ['PWD'], 'include')

MKLFLAGS = ['-I$MKLROOT/include', '-limf']

NLOPTLIBS = ['-lnlopt']

System = platform.system()
if System == 'Linux':
    MKLLIBS = ['-L$MKLROOT/lib/intel64', '-lmkl_rt', '-lpthread', '-lm', '-ldl']
elif System == 'Darwin':
    VERFLAGS += ['-stdlib=libc++']
    MKLLIBS = ['-L$MKLROOT/lib', '-Wl,-rpath,$MKLROOT/lib', '-lmkl_rt', '-lpthread', '-lm', '-ldl']
else:
    MKLLIBS = []

MKLDIR = MKLLIBS[0][2:-1]

rand_sourceList = ['rand.pyx', 'rdrand.cpp']
rand_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in rand_sourceList]

rand_ext = Extension(
    name='rand', sources=rand_List, language='c++',
    extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS,
    include_dirs=[INCLUDE, np.get_include()], extra_link_args=MKLLIBS + OMPLIBS + NLOPTLIBS,
    library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

CARMATask_sourceList = ['rdrand.cpp', 'Constants.cpp',
                        'LC.cpp', 'MCMC.cpp', 'CARMA.cpp', 'CARMATask.cpp', 'CARMATask_cython.pyx']
CARMATask_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in CARMATask_sourceList]

CARMATask_ext = Extension(
    name='CARMATask_cython', sources=CARMATask_List, language='c++',
    extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS,
    include_dirs=[INCLUDE, np.get_include()], extra_link_args=OMPLIBS + MKLLIBS + NLOPTLIBS,
    library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

MBHBTask_sourceList = ['rdrand.cpp', 'Constants.cpp', 'LC.cpp',
                       'MCMC.cpp', 'MBHB.cpp', 'MBHBTask.cpp', 'MBHBTask_cython.pyx']
MBHBTask_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in MBHBTask_sourceList]

MBHBTask_ext = Extension(
    name='MBHBTask_cython', sources=MBHBTask_List, language='c++',
    extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS,
    include_dirs=[INCLUDE, np.get_include()], extra_link_args=OMPLIBS + MKLLIBS + NLOPTLIBS,
    library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

setup(
    name='kali',
    version='2.0.0',
    author='Vishal Pramod Kasliwal',
    author_email='vishal.kasliwal@gmail.com',
    maintainer='Vishal Pramod Kasliwal',
    maintainer_email='vishal.kasliwal@gmail.com',
    url='https://github.com/AstroVPK/kali',
    description='Tools to study stochastic light curves',
    long_description='Tools to model stochastic light curves using various stochastic process. Tools also \
    include components to model MBHBs with relativistic beaming.',
    download_url='https://github.com/AstroVPK/kali',
    classifiers=['AGN', 'C-ARMA', 'stochastic', 'MBHBs'],
    platforms=['Linux', 'Mac OSX'],
    license='GNU GENERAL PUBLIC LICENSE, Version 2, June 1991',
    ext_modules=cythonize([rand_ext, CARMATask_ext, MBHBTask_ext])
)
