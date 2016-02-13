# file "libcarma_build.py"

# Note: this particular example fails before version 1.0.2
# because it combines variadic function and ABI level.

from cffi import FFI

ffi = FFI()
ffi.set_source("bin/_libcarma", None)
ffi.cdef("""
    int _getRandoms(int numRequested, unsigned int *Randoms);
    unsigned int* _malloc_uint(int length);
    void _free_uint(unsigned int *mem);
    int* _malloc_int(int length);
    void _free_int(int *mem);
    double* _malloc_double(int length);
    void _free_double(double *mem);
    int _testSystem(double dt, int p, int q, double *Theta);
    int _printSystem(double dt, int p, int q, double *Theta);
    int _makeIntrinsicLC(double dt, int p, int q, double *Theta, int IR, double tolIR, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *x);
    int _makeObservedLC(double dt, int p, int q, double *Theta, int IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr);
    double _computeLnLikelihood(double dt, int p, int q, double *Theta, int IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr);
    double _computeLnPosterior(double dt, int p, int q, double *Theta, int IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale);
    int _fitCARMA(double dt, int p, int q, int IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnLike);
         """)

if __name__ == "__main__":
    ffi.compile()
