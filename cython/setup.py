import os
import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

os.environ['CC'] = 'icpc'
os.environ['CXX'] = 'icpc'

VERFLAGS = ['-gxx-name=g++-4.8', '-std=c++11']

CPPFLAGS = ['-O3', '-xHost', '-ip', '-parallel', '-funroll-loops', '-fno-alias', '-fno-fnalias', '-fargument-noalias', '-fstrict-aliasing', '-ansi-alias', '-fno-stack-protector-all', '-Wall']

ALIGHFLAGS = ['-falign-functions']

MKLFLAGS = ['-qopenmp', '-I$MKLROOT/include', '-limf']

OMPFLAGS = ['-qopenmp', '-qopenmp-simd']

OMPLIBS = ['-liomp5']

NLOPTLIBS = ['-lnlopt']

System = platform.system()
if System == 'Linux':
	MKLLIBS = ['-L$MKLROOT/lib/intel64', '-lmkl_rt', '-lpthread', '-lm', '-ldl']
elif System == 'Darwin':
	MKLLIBS = ['-L$MKLROOT/lib', '-Wl,-rpath,$MKLROOT/lib' , '-lmkl_rt', '-lpthread', '-lm', '-ldl']
else:
	MKLLIBS = []

bSMBH_sourceList = ['bSMBH.pyx', 'binarySMBH.cpp', 'Constants.cpp']

bSMBH_ext = Extension(name='bSMBH', sources=bSMBH_sourceList, language='c++', extra_compile_args = VERFLAGS + CPPFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS, include_dirs=['/home/vish/code/trunk/cpp/libcarma/cython'], extra_link_args = MKLLIBS + NLOPTLIBS, library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'], runtime_library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'])

rand_sourceList = ['rand.pyx', 'rdrand.cpp']

rand_ext = Extension(name='rand', sources=rand_sourceList, language='c++', extra_compile_args = VERFLAGS + CPPFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS, include_dirs=['/home/vish/code/trunk/cpp/libcarma/cython'], extra_link_args = MKLLIBS + NLOPTLIBS, library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'], runtime_library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'])

CARMATask_sourceList = ['CARMATask.pyx', 'Task.cpp', 'CARMA.cpp', 'LC.cpp', 'Constants.cpp']

CARMATask_ext = Extension(name='CARMATask', sources=CARMATask_sourceList, language='c++', extra_compile_args = VERFLAGS + CPPFLAGS + ALIGHFLAGS + MKLFLAGS + OMPFLAGS, include_dirs=['/home/vish/code/trunk/cpp/libcarma/cython'], extra_link_args = OMPLIBS + MKLLIBS + NLOPTLIBS, library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'], runtime_library_dirs = ['/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64'])

setup(
	ext_modules = cythonize([bSMBH_ext, rand_ext, CARMATask_ext])
)
