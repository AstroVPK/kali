import numpy as np
import os
import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pdb

os.environ['CC'] = 'icpc'
os.environ['CXX'] = 'icpc'

INCLUDE = os.path.join(os.environ['PWD'], 'include')

VERFLAGS = ['-gxx-name=g++-4.8', '-std=c++11']

CPPFLAGS = ['-O3', '-xHost', '-ip', '-parallel', '-funroll-loops', '-fno-alias', '-fno-fnalias',
            '-fargument-noalias', '-fstrict-aliasing', '-ansi-alias', '-fno-stack-protector-all', '-Wall']

ALIGHFLAGS = ['-falign-functions']

MKLFLAGS = ['-I$MKLROOT/include', '-limf']

OMPFLAGS = ['-qopenmp', '-qopenmp-simd', '-qopt-report=5', '-qopt-report-phase=all']

OMPLIBS = ['-liomp5']

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

'''bSMBH_sourceList = ['bSMBH.pyx', 'MBHB.cpp', 'Constants.cpp']
bSMBH_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in bSMBH_sourceList]

bSMBH_ext = Extension(name='bSMBH', sources=bSMBH_List, language='c++', extra_compile_args = CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS, include_dirs=[INCLUDE, np.get_include()], extra_link_args = MKLLIBS + OMPLIBS + NLOPTLIBS, library_dirs = [MKLDIR], runtime_library_dirs = [MKLDIR])'''

rand_sourceList = ['rand.pyx', 'rdrand.cpp']
rand_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in rand_sourceList]

rand_ext = Extension(
    name='rand', sources=rand_List, language='c++', extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS,
                     include_dirs=[INCLUDE, np.get_include()], extra_link_args=MKLLIBS + OMPLIBS + NLOPTLIBS, library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

CARMATask_sourceList = ['rdrand.cpp', 'Constants.cpp',
                        'LC.cpp', 'MCMC.cpp', 'CARMA.cpp', 'Task.cpp', 'CARMATask.pyx']
CARMATask_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in CARMATask_sourceList]

CARMATask_ext = Extension(name='CARMATask', sources=CARMATask_List, language='c++', extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS +
                          OMPFLAGS, include_dirs=[INCLUDE, np.get_include()], extra_link_args=OMPLIBS + MKLLIBS + NLOPTLIBS, library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

MBHBTask_sourceList = ['rdrand.cpp', 'Constants.cpp', 'LC.cpp',
                        'MCMC.cpp', 'MBHB.cpp', 'MBHBTask.cpp', 'MBHBTask_cython.pyx']
MBHBTask_List = [os.path.join(os.environ['PWD'], 'src', srcFile) for srcFile in MBHBTask_sourceList]

MBHBTask_ext = Extension(name='MBHBTask_cython', sources=MBHBTask_List, language='c++', extra_compile_args=CPPFLAGS + VERFLAGS + ALIGHFLAGS + MKLFLAGS +
                          OMPFLAGS, include_dirs=[INCLUDE, np.get_include()], extra_link_args=OMPLIBS + MKLLIBS + NLOPTLIBS, library_dirs=[MKLDIR], runtime_library_dirs=[MKLDIR])

setup(
    name='kali',
        version='1.0.0',
        author='Vishal Pramod Kasliwal',
        author_email='vishal.kasliwal@gmail.com',
        maintainer='Vishal Pramod Kasliwal',
        maintainer_email='vishal.kasliwal@gmail.com',
        url='https://github.com/AstroVPK/kali',
        description='Tools to study stochastic light curves',
        long_description='Tools to model stochastic light curves using various stochastic process. Tools also include components to model MBHBs with relativistic beaming.',
        download_url='https://github.com/AstroVPK/kali',
        classifiers=['AGN', 'C-ARMA', 'stochastic', 'MBHBs'],
        platforms=['Linux', 'Mac OSX'],
        license='GNU GENERAL PUBLIC LICENSE, Version 2, June 1991',
        # ext_modules = cythonize([bSMBH_ext, rand_ext, CARMATask_ext, bSMBHTask_ext])
        ext_modules=cythonize([rand_ext, CARMATask_ext, MBHBTask_ext])
)
