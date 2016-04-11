# libcarma
A library to model a time series as a Continuous-time ARMA (C-ARMA) process. This library is written in C++ 
and is exposed to Python using `cython` and `cffi` (deprecated)


Version: 1.0.0


Install
-------
Install instructions are provided for Linux & Mac OSX machines. The following OSs have been tested


1. Ubuntu 14.04 LTS Trusty Tahr

2. Mac OS X 10.10.5 Yosemite

  NB: On Mac OS X, Anaconda's matplotlib.pyplot seems to have trouble understanding the tilde '~' character as 
  part of a path. So instead of

  `<libcarma dir>$ python python/DemoScript.py ~/<path to working dir> Config.ini`

  do

  `<libcarma dir>$ python python/DemoScript.py /Users/<user name>/<path to working dir> Config.ini`

You will need to have Anaconda Python, the Intel C++ Compiler XE, Intel MKL, NLOpt, cython , cffi (optional), 
& Brandon Kelly's `carma_pack` (optional) installed. At the moment, Anaconda Python, Intel C++ Compiler XE, 
Intel MKL, `cython` & NLOpt are required though the plan is to eventually allow the use of g++ etc... Brandon 
Kelly's `carma_pack` is not required but is recommended. `cffi` is only required if you wish to use the older 
depracted `cffi` interface to the `c++` code.

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


  1. Intel® Parallel Studio XE 2016 Cluster Edition Initial Release 16.0.1.111 / 20151020 (icpc version 16.0.0 
(gcc version 4.9.0 compatibility)

  2. Intel® Parallel Studio XE 2016 Cluster Edition Update 1 16.0.1.150 / 20151021 (icpc version 16.0.1 
(gcc version 4.8.0 compatibility)

  3. Intel® Parallel Studio XE 2016 Cluster Edition Update 2 16.0.1.181 / 20151021 (icpc version 16.0.2 
(gcc version 4.8.0 compatibility)

3. Intel MKL Library


  The Intel MKL library is a high performance math library that is used extensively in this package. Since the 
library can be obtained free of cost by both students as well as academic researchers, there is no plan to 
replace it with an alternative. Intel MKL may be obtained from


  [Student + free Compiler & MKL](https://software.intel.com/en-us/qualify-for-free-software/student)


  [Academic Resercher + free MKL](https://software.intel.com/en-us/qualify-for-free-software/academicresearcher)


  This software has been tested with


  1. Intel® Math Kernel Library 11.3 11.3

  1. Intel® Math Kernel Library 11.3 Update 1 11.3.1

  3. Intel® Math Kernel Library 11.3 Update 2 11.3.2

4. NLOpt


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

  `cython` is used to wrap the `c++` parts of libcarma in Python. Make sure that you have the latest `cython` 
build. You can get the most recent version using

  `bash-prompt$ conda update conda`


  `bash-prompt$ conda update --all`


  `bash-prompt$ conda install cython`

  This software has been tested with
  1. Cython Version 0.23.4

5. `cffi`


  The C Foreign Function Interface `cffi` is used to make the libcarma.so library calls from Python. It can 
be installed into Anaconda using


  `bash-prompt$ sudo apt-get install libffi6` (On linux only)


  `bash-prompt$ sudo apt-get install libffi6:i386` (On linux only)


  `bash-prompt$ sudo apt-get install libffi-dev` (On linux only)


  `bash-prompt$ conda update conda`


  `bash-prompt$ conda update --all`


  `bash-prompt$ conda install cffi`

7. `carma_pack`


  Brandon Kelly's `carma_pack` is a C-ARMA analysis package written in C++ and Python. It may be obtained at


  [`carma_pack`](https://github.com/brandonckelly/carma_pack)


  Please install `carma_pack` using the instructions provided with the package. This library includes a 
Python script `cffi/python/KellyAnalysis.py` to use `carma_pack` with the same interface as the rest of this 
package. Please read


  `bash-prompt$ KellyAnalysis --help` and the python docstring for usage instructions.


To make `libcarma` after cloning the repository, simply run


`bash-prompt$ source ./bin/setup.sh`


`bash-prompt$ python setup.py build_ext`


This will compile all the `c++` source files in the folder `src/` with the headers in `include/` and put the 
built object files in the directory `build/`. Then it will link the object files together into the library 
which will be located in `lib/`. Python `__init__.py` files make the libary visible to the Python interface 
files located in `python/`. You must re-run


`bash-prompt$ source ./bin/setup.sh`


in every new terminal that you use `libcarma` in. You may consider adding 


`source <path to libcarma>/bin/setup.sh`


to your `.bashrc`. To clean the library, just delete the `build/` directory and any files inside  `lib/`. You 
may consider adding 


`source <path to libcarma>/bin/setup.sh`


to your `.bashrc`. This covers installation. Please feel free to try the 
package out by running


`bash-prompt$ source <path to libcarma>/bin/setup.sh`


`cython` version: `bash-prompt$/usr/bin/time -p -v python cython/examples/CARMADemo.py`


`cffi` version: `bash-prompt$/usr/bin/time -p -v python cffi/scripts/DemoScript.py <path to libcarma>/cffi/examples/Demo01/ Config.ini | tee <path to libcarma>/cffi/examples/Demo01/timing.dat`


A preliminary user guide is now available at `<path to libcarma>/guide/Introduction.ipynb`.