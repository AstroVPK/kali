# file "libcarma_build.py"

# Note: this particular example fails before version 1.0.2
# because it combines variadic function and ABI level.

from cffi import FFI

ffi = FFI()
ffi.set_source("_libcarma", None)
#ffi.cdef("""
#    double _makeMockLC(double dt, int numP, int numQ, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
#    double _computeLnLike(double dt, int p, int q, double *Theta, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr);
#    void _fitCARMA(double dt, int p, int q, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike);
#    int _getIntrinsicLC(double dt, int p, int q, double *Theta, int IR, double maxT, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
#         """)
ffi.cdef("""
    int* _malloc_int(int length);
    void _free_int(int *mem);
    double* _malloc_double(int length);
    void _free_double(double *mem);
    int _makeIntrinsicLC(double dt, int p, int q, double *Theta, int IR, double tolIR, double maxT, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
    int _makeObservedLC(double dt, int p, int q, double *Theta, int IR, double tolIR, double maxT, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
    double _computeLnlike(double dt, int p, int q, double *Theta, int IR, double tolIR, double maxT, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr);
    int _fitCARMA(double dt, int p, int q, int IR, double tolIR, double maxT, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike);
         """)

if __name__ == "__main__":
    ffi.compile()