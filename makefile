CPPC = icpc

IDIR = include/
SRCDIR = src/
ODIR = src/obj/
LDIR = lib/
PDIR = python/
#BOOSTLINK = -Bstatic -lboost_system -lboost_filesystem -lboost_system
#BOOSTLIB = /usr/local/include/boost



VERFLAGS = -gxx-name=g++-4.8 -std=c++11

CPPFLAGS = -O3 -ip -parallel -funroll-loops -fno-alias -fno-fnalias -fargument-noalias -fstrict-aliasing -ansi-alias -fno-stack-protector-all -Wall
#-g
#-opt-streaming-stores always

OFFLOAD_FLAGS =
#OFFLOAD_FLAGS = -offload=optional

#MKL Flags.
MKLFLAGS = -qopenmp -I$(MKLROOT)/include -limf

#MKL link line.
#MKL_LIBS = -L$(MKLROOT)/lib/intel64  -lmkl_rt -lpthread -lm
# Dynamic Linking
#MKL_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm
# Static linking
MKL_LIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm

NLOPTLIBS = -lnlopt
OMPFLAGS = -openmp -openmp-simd

ALIGN_FLAGS = -falign-functions 
#-falign-jumps  -falign-loops -falign-labels  -freorder-blocks  -freorder-blocks-and-partition -fprefetch-loop-arrays

REPORTFLAG = -qopt-report-phase=vec -qopt-report-file=stdout -openmp-report=0
#-guide
# -opt-report-phase=offload

FPFLAGS = -fp-model strict -fp-model extended -fimf-arch-consistency=true -fimf-precision=high -no-fma 
# enable <name> floating point model variation
#     except[-]  - enable/disable floating point semantics
#     extended   - enables intermediates in 80-bit precision
#     fast       - allows value-unsafe optimizations
#     precise    - allows value-safe optimizations
#     source     - enables intermediates in source precision
#     strict     - enables -fp-model precise -fp-model except and disables floating point multiply add

_DEPENDENCIES = Constants.hpp CARMA.hpp MCMC.hpp Correlation.hpp DLAPACKE.hpp
#Constants.hpp Utilities.hpp Acquire.hpp Universe.hpp Spherical.hpp Obj.hpp Kepler.hpp CARMA.hpp MCMC.hpp DLAPACKE.hpp Correlation.hpp PRH.hpp
DEPENDENCIES = $(patsubst %,$(IDIR)/%,$(_DEPENDENCIES))

_OBJECTS = Constants.o CARMA.o MCMC.o Correlation.o Functions.o DLAPACKE.o
#Constants.o Utilities.o Acquire.o Universe.o Spherical.o Obj.o Kepler.o CARMA.o MCMC.o DLAPACKE.o Correlation.o Functions.o PRH.o
OBJECTS = $(patsubst %,$(ODIR)%,$(_OBJECTS))

#EXEC1 = testPoint
#EXEC2 = endToEndTest
#EXEC3 = plotCARMARegions
#EXEC4 = fitCARMA
#EXEC5 = writeKeplerLC
#EXEC6 = writeMockLC
#EXEC7 = computeCFs
#EXEC8 = recoveryTest
EXT = .cpp
EXTOBJ = .o
STATLIB_EXT = .a
DYLIB_EXT = .so
FUNCTIONS = Functions
LIBCARMA_DYN = libcarma.so.1.0.0
LIBCARMA_STAT = libcarma.a
LIBCARMA_LINKING = libcarma.so
PYTHON_BINDING = _libcarma.py

#all: $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC4) $(EXEC5) $(EXEC6) $(EXEC7) $(EXEC8) $(LIBCARMA)
all: $(LIBCARMA_STAT) $(LIBCARMA_DYN) $(LIBCARMA_LINKING) $(PYTHON_BINDING)

$(LIBCARMA_DYN): $(OBJECTS)
	$(CPPC) -shared -Wl,-soname,libcarma.so.1 -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) -o $(LDIR)/libcarma.so.1.0.0 $(ODIR)Functions.o $(ODIR)CARMA.o $(ODIR)MCMC.o $(ODIR)Constants.o $(ODIR)Correlation.o $(ODIR)DLAPACKE.o $(MKL_LIBS) $(NLOPTLIBS)

$(LIBCARMA_STAT): $(OBJECTS)
	ar r $(LDIR)libcarma.a $(ODIR)Functions.o $(ODIR)CARMA.o $(ODIR)MCMC.o $(ODIR)Constants.o $(ODIR)Correlation.o $(ODIR)DLAPACKE.o

$(LIBCARMA_LINKING): $(LIBCARMA)
	rm -rf $(LDIR)libcarma.so.1
	rm -rf $(LDIR)libcarma.so
	ln -s $(LDIR)libcarma.so.1.0.0 $(LDIR)libcarma.so.1
	ln -s $(LDIR)libcarma.so.1 $(LDIR)libcarma.so

$(PYTHON_BINDING): $(LIBCARMA_LINKING)
	python $(PDIR)build/libcarma_build.py

#$(EXEC1): $(OBJECTS) $(patsub %,$(EXEC1)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC1)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC2): $(OBJECTS) $(patsub %,$(EXEC2)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC2)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC3): $(OBJECTS) $(patsub %,$(EXEC3)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC3)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC4): $(OBJECTS) $(patsub %,$(EXEC4)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC4)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC5): $(OBJECTS) $(patsub %,$(EXEC5)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC5)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC6): $(OBJECTS) $(patsub %,$(EXEC6)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC6)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC7): $(OBJECTS) $(patsub %,$(EXEC7)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC7)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(EXEC8): $(OBJECTS) $(patsub %,$(EXEC8)%,$(EXT))
#	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC8)$(EXT) $(OMPFLAGS) $(MKL_LIBS) $(BOOSTLINK) $(NLOPTLIBS) -o $@

#$(ODIR)/Universe.o: $(SRCDIR)/Universe.cpp $(IDIR)/Universe.hpp
#	$(CPPC) -c $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAGS) $(MKLFLAGS) -I $(IDIR) -I $(BOOSTLIB) $< -o $@

#$(ODIR)/Spherical.o: $(SRCDIR)/Spherical.cpp $(IDIR)/Spherical.hpp
#	$(CPPC) -c $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAGS) $(MKLFLAGS) -I $(IDIR) -I $(BOOSTLIB) $< -o $@

$(ODIR)CARMA.o: $(SRCDIR)CARMA.cpp $(IDIR)CARMA.hpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)MCMC.o: $(SRCDIR)MCMC.cpp $(IDIR)MCMC.hpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)DLAPACKE.o: $(SRCDIR)/DLAPACKE.cpp $(IDIR)/DLAPACKE.hpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)Correlation.o: $(SRCDIR)Correlation.cpp $(IDIR)Correlation.hpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)Constants.o: $(SRCDIR)Constants.cpp $(IDIR)Constants.hpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)Functions.o: $(SRCDIR)Functions.cpp
	$(CPPC) -c -Wall -fpic $(VERFLAGS) -xHost $(CPPFLAGS) $(ALIGN_FLAGS) $(OMPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(REPORTFLAG) -I $(IDIR) $(NLOPTLIBS) $< -o $@ 

#$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPENDENCIES)
#	$(CPPC) -c $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAGS) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR) $< -o $@

.PHONY: clean
clean:
	rm -rf $(ODIR)/*.o $(LDIR)*.* *~ $(SRCDIR)/*~ $(IDIR)*~ $(LDIR)*~
