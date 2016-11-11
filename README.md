# `Kālī`
`Kālī` is a software library to model time series data using various stochastic processes such as
Continuous-time ARMA (C-ARMA) processes. The name of the library is taken from the Hindu goddess Kālī, who is
the goddess of time and change. It also stands for KArma (C-ARMA) LIbrary because the library began as a tool
to model C-ARMA light curves. `Kālī` is written in `c++` and is exposed to `python` using `cython`.


Version: 2.0.0


Install
-------
Install instructions are provided for Linux & Mac OSX machines. The following OSs have been tested


1. Ubuntu 14.04 LTS Trusty Tahr

2. Red Hat Enterprise Linux Server release 6.5 (Santiago)

3. Mac OS X 10.10.5 Yosemite

4. Fedora 23

5. Mac OS X 10.11.2 El Capitan

6. Ubuntu 16.04 LTS Xenial Xerus

If you are working on Mac OSX, please be sure to install the latest XCode. You will need to have Anaconda
Python, the Intel C++ Compiler XE or the GNU C++ Compiler, Intel MKL, NLOpt, `cython` , the `future`,
`fitsio`, `py.test`, \& `gatspy` packages, & Brandon Kelly's `carma_pack` (optional) installed. At the moment,
Anaconda Python, Intel MKL, `cython`, `future`, `fitsio`, `py.test`, `gatspy`, & NLOpt are required. Either of
the Intel C++ Compiler or the GNU C++ Compiler are required though the plan is to evetually support the
clang++ Compiler as well. Brandon Kelly's `carma_pack` is not required but is recommended.

You may encounter the following error when running `bash-prompt$ python setup.py build_ext` -
`icpc: error #10001: could not find directory in which g++-x.x resides`. This error occurs when the `g++`
compiler is installed in a non-standard location and can be resolved by setting the environment variable
`GXX_ROOT` to the installation location of the `g++` compiler. To determine the installation location of the
`g++` compiler, execute the command

`bash-prompt$ g++ --print-search-dirs`

and set `GXX_ROOT` to the location indicated by the install field from the output of the above command.

1. Anaconda Python


  Anaconda Python is a free suite of Python development tools including a Python2/3 interpreter along with
most of the common Python scientific computing packages such as `numpy`, `scipy`, and `matplotlib`. Anaconda
Python may be obtained from


  [Download Anaconda Python](https://www.continuum.io/downloads)


  This package is written for Python2, though it may eventually get updated to Python3. Download the 64-bit
Python2 installer for your platform and follow the provided install instructions. Keep your installation up
to date by periodically doing


  `bash-prompt$ conda update conda` to update the package manager.


  `bash-prompt$ conda update --all` to update all the installed packages.


  This software has been tested with


  1. Python 2.7.11 |Anaconda 2.4.1 (64-bit)

  2. Python 2.7.11 |Anaconda 4.0.0 (64-bit)


2. Intel C++ Compiler XE


  The Intel C++ compiler is included in the Intel® Parallel Studio XE Composer Edition (and higher editions).


  [Intel® C++ Compiler Overview](https://software.intel.com/en-us/c-compilers/ipsxe)


  [Try & Buy Intel C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe/try-buy#buynow)


  Students are eligible for a free license. Greatly discounted pricing is available for academic users. Use
of the Intel compiler results in the largest performance gains on Intel CPUs (but not on AMD machines!) To
install the Intel compiler, download the Parallel Studio XE edition of your choice by following the
instructions at


  [Student + free Compiler & MKL](https://software.intel.com/en-us/qualify-for-free-software/student)


  [Academic Resercher + free MKL](https://software.intel.com/en-us/qualify-for-free-software/academicresearcher)


  Unpack the tarball as usual and then change to the top-level directory of the un-tarred folder. In the
terminal type


  `<Intel C & C++ Compiler dir>$ sudo install_GUI.sh`


  to start the install. Add the following line to your `.bashrc` to setup the necessary environment variables
required by the compiler.


  `source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64`


  This software has been tested with


  1. Intel® Parallel Studio XE 2016 Cluster Edition Initial Release 16.0.1.111 (icpc version 16.0.0)
(gcc version 4.9.0 compatibility)

  2. Intel® Parallel Studio XE 2016 Cluster Edition Update 1 16.0.1.150 (icpc version 16.0.1)
(gcc version 4.8.0 compatibility)

  3. Intel® Parallel Studio XE 2016 Cluster Edition Update 2 16.0.2.180 (icpc version 16.0.2)
(gcc version 4.8.0 compatibility)

  4. Intel® Parallel Studio XE 2016 Cluster Edition Update 3 16.0.3.210 (icpc version 16.0.3)
(gcc version 4.8.0 compatibility)


3. GNU C++ Compiler


  The GNU C++ compiler is free-ware and is available on most systems.


  [GNU C++ Compiler Overview](https://gcc.gnu.org/)


  This software has been tested with


  1. gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.2)


4. Intel MKL Library


  The Intel MKL library is a high performance math library that is used extensively in this package. Since the
library can be obtained free of cost by both students as well as academic researchers, there is no plan to
replace it with an alternative. Intel MKL may be obtained from


  [Student + free Compiler & MKL](https://software.intel.com/en-us/qualify-for-free-software/student)


  [Academic Resercher + free MKL](https://software.intel.com/en-us/qualify-for-free-software/academicresearcher)


  Add the following line to your `.bashrc` to setup the necessary environment variables
  required by the compiler.


  `source /opt/intel/mkl/bin/mklvars.sh intel64`


  This software has been tested with


  1. Intel® Math Kernel Library 11.3 11.3

  1. Intel® Math Kernel Library 11.3 Update 1 11.3.1

  3. Intel® Math Kernel Library 11.3 Update 2 11.3.2

  4. Intel® Math Kernel Library 11.3 Update 3 11.3.3


5. NLOpt


  NLOpt is a free/open-source non-liner optimization library written by the AbInitio Group at MIT.


  [Steven G. Johnson, The NLopt nonlinear-optimization package](http://ab-initio.mit.edu/nlopt)


  NLOpt can be downloaded from


  [Download NLOpt](http://ab-initio.mit.edu/wiki/index.php/NLopt)


  After un-tarring the tar file, NLOpt should be installed as follows


  `<nlopt dir>$ ./configure --enable-shared`


  `<nlopt dir>$ make`


  `<nlopt dir>$ sudo make install`

  Lastly, one must update the list of installed shared libraries by running

  `bash-prompt$ sudo ldconfig`

  This software has been tested with


  1. NLOpt Version 2.4.2


6. `cython`

  `cython` is used to wrap the `c++` parts of `Kālī` in Python. Make sure that you have the latest `cython`
build. You can get the most recent version using


  `bash-prompt$ conda update conda`


  `bash-prompt$ conda update --all`


  `bash-prompt$ conda install cython`


  This software has been tested with


  1. Cython Version 0.23.4

  2. Cython Version 0.24


7. `future`

  `future` is a Python 2 package that makes a number of useful Python 3 modules useable in Python 2. You can
  get `future` using


  `bash-prompt$ conda update conda`


  `bash-prompt$ conda update --all`


  `bash-prompt$ conda install future`


  This software has been tested with


  1. `future` Version 0.15.2

8. `fitsio`

  `fitsio` is a Python 2 package for read and writing FITS files. You can get `fitsio` using


  `bash-prompt$ pip install fitsio`


  This software has been tested with


  1. `fitsio` Version 0.9.10


8. `py.test`

  `py.test` is used for testing purposes. `py.test` can be installed into Anaconda using


  `bash-prompt$ conda update conda`


  `bash-prompt$ conda update --all`


  `bash-prompt$ conda install pytest`


  This software has been tested with


  1. `py.test` Version 2.9.2


9. `gatspy`


  General tools for Astronomical Time Series in Python `gatspy` are used to search for periods using a fast
  version of the Lomb-Scargle periodogram. `gatspy` can be installed into Anaconda using


  `bash-prompt$ pip install gatspy`


10. `carma_pack`


  Brandon Kelly's `carma_pack` is a C-ARMA analysis package written in C++ and Python. It may be obtained at


  [`carma_pack`](https://github.com/brandonckelly/carma_pack)


  Please install `carma_pack` using the instructions provided with the package. This library includes a
Python script `cffi/python/KellyAnalysis.py` to use `carma_pack` with the same interface as the rest of this
package. Please read


  `bash-prompt$ KellyAnalysis --help` and the python docstring for usage instructions.


To make `Kālī` after cloning the repository, simply run


`bash-prompt$ source ./bin/setup.sh`


followed by


`bash-prompt$ python setup.py build_ext`


This will compile all the `c++` source files in the folder `src/` with the headers in `include/` and put the
built object files in the directory `build/`. Then it will link the object files together into the library
which will be located in `lib/`. Python `__init__.py` files make the libary visible to the Python interface
files located in `python/`. You must re-run


`bash-prompt$ source ./bin/setup.sh`


in every new terminal that you use `Kālī` in. You may consider adding


`source <path to kali>/bin/setup.sh`


to your `.bashrc`. To clean the library, just delete the `build/` directory and any files inside  `lib/`
except, of course, the `__init__.py` file. You
may consider adding


`source <path to kali>/bin/setup.sh`


to your `.bashrc`. This covers installation. Please feel free to try the
package out by running


`bash-prompt$ source <path to kali>/bin/setup.sh`


and following through the user guide available at `<path to kali>/guide/Introduction.ipynb`. More example code
can be found in the folders `<path to kali>/examples` and `<path to kali>/tests`.
