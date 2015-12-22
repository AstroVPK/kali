# libcarma
A library to model a time series as a Continuous-time ARMA (C-ARMA) process.

Install
-------
Install instructions are provided for a

You will need to have the Intel C++ compiler installed along with Intel MKL and NLOpt . At the moment, all 
three are required though the plan is to eventually allow the use of g++ etc...

1. Intel C++ Compiler
The Intel C++ compiler is included in the Intel® Parallel Studio XE Composer Edition (and higher editions).
https://software.intel.com/en-us/c-compilers/ipsxe
https://software.intel.com/en-us/intel-parallel-studio-xe/try-buy#buynow
Students are eligible for a free license. Greatly discounted pricing is available for academic users. Use of 
the Intel compiler results in the largest performance gains on Intel architectures. To install the Intel 
compiler, download the Parallel Studio XE edition of your choice by following the instructions at 
https://software.intel.com/en-us/qualify-for-free-software/student (Student + free Compiler & MKL)
https://software.intel.com/en-us/qualify-for-free-software/academicresearcher (Academic Resercher + free MKL)
Unpack the tarball as usual and then change to the top-level directory of the un-tarred folder. In the 
terminal type 
bash-prompt$ sudo install_GUI.sh
to start the install.
Add the following line to your .bashrc to setup the necessary environment variables required by the compiler.
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

2. Intel MKL Library
The Intel MKL library is a high performance math library that is used extensively in this package. Since the 
library can be obtained free of cost by both students as well as academic researchers, there is no plan to 
replace it with an alternative. Intel MKL may be obtained from
https://software.intel.com/en-us/qualify-for-free-software/student (Student + free Compiler & MKL)
https://software.intel.com/en-us/qualify-for-free-software/academicresearcher (Academic Resercher + free MKL)

3. NLOpt
NLOpt is a free/open-source non-liner optimization library written by the AbInitio Group at MIT.
Steven G. Johnson, The NLopt nonlinear-optimization package, http://ab-initio.mit.edu/nlopt 
NLOpt can be downloaded from 
http://ab-initio.mit.edu/wiki/index.php/NLopt
After un-tarring the tar file, NLOpt should be installed as follows
bash-prompt$ ./configure --enable-shared
bash-prompt& make
bash-prompt$ sudo make install

To make this library after cloning the repository, simply run
bash-prompt$ make
Before using the library, you should run
bash-prompt$ source lib/setup.sh
To clean the directory, run 
bash-prompt$ make clean
