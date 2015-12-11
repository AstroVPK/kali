# file "libcarma_build.py"

# Note: this particular example fails before version 1.0.2
# because it combines variadic function and ABI level.

from cffi import FFI

ffi = FFI()
ffi.set_source("_libcarma", None)
ffi.cdef("""
    int cffi_makeMockLC(double dt, int numP, int numQ, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
    double cffi_computeLnLike(double dt, int p, int q, double *Theta, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr);
         """)

if __name__ == "__main__":
    ffi.compile()