# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
#DTYPE = np.uint32
#ctypedef np.uint32_t DTYPE_t

cdef extern from 'LC.hpp':
	cdef cppclass LCData:
		int numCadences
		double dt
		bool IR
		double tolIR
		double fracIntrinsicVar
		double fracSignalToNoise
		double maxSigma
		double minTimescale
		double maxTimescale
		double *t
		double *x
		double *y
		double *yerr
		double *mask
		LCData() except+

cdef extern from 'Functions.hpp':
	cdef int getRandomsCARMA(int numRequested, unsigned int *Randoms)

	cdef int testSystemCARMA(double dt, int p, int q, double *Theta)

	cdef int printSystemCARMA(double dt, int p, int q, double *Theta)

	cdef int makeIntrinsicLCCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *x)

	cdef double getMeanFluxCARMA(int p, int q, double *Theta, double fracIntrinsicVar)

	cdef int makeObservedLCCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr)

	cdef double computeLnLikelihoodCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr)

	cdef double computeLnPosteriorCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale)

	cdef int fitCARMACARMA(double dt, int p, int q, bool IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior)

cdef class lc:
	cdef LCData *thisptr

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracSignalToNoise = 0.001, maxSigma = 1.0e2, minTimescale = 5.0e-1, maxTimescale = 5.0):
		self.thisptr = new LCData()
		self.thisptr.numCadences = t.shape[0]
		self.thisptr.dt = dt
		self.thisptr.IR = IR
		self.thisptr.tolIR = tolIR
		self.thisptr.fracIntrinsicVar = fracIntrinsicVar
		self.thisptr.fracSignalToNoise = fracSignalToNoise
		self.thisptr.maxSigma = maxSigma
		self.thisptr.minTimescale = minTimescale
		self.thisptr.maxTimescale = maxTimescale
		self.thisptr.t = &t[0]
		self.thisptr.x = &x[0]
		self.thisptr.y = &y[0]
		self.thisptr.yerr = &yerr[0]
		self.thisptr.mask = &mask[0]

	property numCadences:
		def __get__(self): return self.thisptr.numCadences

	property dt:
		def __get__(self): return self.thisptr.dt

	property IR:
		def __get__(self): return self.thisptr.IR
		def __set__(self, IR): self.thisptr.IR = IR

	property tolIR:
		def __get__(self): return self.thisptr.tolIR
		def __set__(self, tolIR): self.thisptr.tolIR = tolIR

	property fracIntrinsicVar:
		def __get__(self): return self.thisptr.fracIntrinsicVar
		def __set__(self, fracIntrinsicVar): self.thisptr.fracIntrinsicVar = fracIntrinsicVar

	property fracSignalToNoise:
		def __get__(self): return self.thisptr.fracSignalToNoise
		def __set__(self, fracSignalToNoise): self.thisptr.fracSignalToNoise = fracSignalToNoise

	property maxSigma:
		def __get__(self): return self.thisptr.maxSigma
		def __set__(self, maxSigma): self.thisptr.maxSigma = maxSigma

	property minTimescale:
		def __get__(self): return self.thisptr.minTimescale
		def __set__(self, minTimescale): self.thisptr.minTimescale = minTimescale

	property maxTimescale:
		def __get__(self): return self.thisptr.maxTimescale
		def __set__(self, maxTimescale): self.thisptr.maxTimescale = maxTimescale

	def __len__(self):
		return self.thisptr.numCadences

	def __setitem__(self, cadence, value):
		self.thisptr.t[cadence] = value[0]
		self.thisptr.x[cadence] = value[1]
		self.thisptr.y[cadence] = value[2]
		self.thisptr.yerr[cadence] = value[3]
		self.thisptr.mask[cadence] = value[4]

	def __getitem__(self, cadence):
		return self.thisptr.t[cadence], self.thisptr.x[cadence], self.thisptr.y[cadence], self.thisptr.yerr[cadence], self.thisptr.mask[cadence]

@cython.boundscheck(False)
@cython.wraparound(False)
def getRandoms(np.ndarray[unsigned int, ndim=1, mode="c"] Randoms not None):
	cdef int numRand = Randoms.shape[0]
	success = getRandomsCARMA(numRand, &Randoms[0])
	if success != 1:
		Randoms = np.random.randint(0, 4294967295, numRand)
		#raise RuntimeError('Intel RDRAND failed with error code %d! Using numpy.random'%(success))
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def testSystem(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None):
	success = testSystemCARMA(dt, p, q, &Theta[0])
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def printSystem(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None):
	success = printSystemCARMA(dt, p, q, &Theta[0])
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def makeIntrinsicLC(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None, IR, tolIR, numBurn, numCadences, startCadence, burnSeed, distSeed, np.ndarray[int, ndim=1, mode="c"] cadence not None, np.ndarray[double, ndim=1, mode="c"] mask not None, np.ndarray[double, ndim=1, mode="c"] t not None, np.ndarray[double, ndim=1, mode="c"] x not None):
	success = makeIntrinsicLCCARMA(dt, p, q, &Theta[0], IR, tolIR, numBurn, numCadences, startCadence, burnSeed, distSeed, &cadence[0], &mask[0], &t[0], &x[0])
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def makeObservedLC(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, numBurn, numCadences, startCadence, burnSeed, distSeed, noiseSeed, np.ndarray[int, ndim=1, mode="c"] cadence not None, np.ndarray[double, ndim=1, mode="c"] mask not None, np.ndarray[double, ndim=1, mode="c"] t not None, np.ndarray[double, ndim=1, mode="c"] y not None, np.ndarray[double, ndim=1, mode="c"] yerr not None):
	success = makeObservedLCCARMA(dt, p, q, &Theta[0], IR, tolIR, fracIntrinsicVar, fracSignalToNoise, numBurn, numCadences, startCadence, burnSeed, distSeed, noiseSeed, &cadence[0], &mask[0], &t[0], &y[0], &yerr[0])
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def computeLnLikelihood(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None, IR, tolIR, numCadences, np.ndarray[int, ndim=1, mode="c"] cadence not None, np.ndarray[double, ndim=1, mode="c"] mask not None, np.ndarray[double, ndim=1, mode="c"] t not None, np.ndarray[double, ndim=1, mode="c"] y not None, np.ndarray[double, ndim=1, mode="c"] yerr not None):
	success = computeLnLikelihoodCARMA(dt, p, q, &Theta[0], IR, tolIR, numCadences, &cadence[0], &mask[0], &t[0], &y[0], &yerr[0])
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def computeLnPosterior(dt, p, q, np.ndarray[double, ndim=1, mode="c"] Theta not None, IR, tolIR, numCadences, np.ndarray[int, ndim=1, mode="c"] cadence not None, np.ndarray[double, ndim=1, mode="c"] mask not None, np.ndarray[double, ndim=1, mode="c"] t not None, np.ndarray[double, ndim=1, mode="c"] y not None, np.ndarray[double, ndim=1, mode="c"] yerr not None, maxSigma, minTimescale, maxTimescale):
	success = computeLnPosteriorCARMA(dt, p, q, &Theta[0], IR, tolIR, numCadences, &cadence[0], &mask[0], &t[0], &y[0], &yerr[0], maxSigma, minTimescale, maxTimescale)
	return success

@cython.boundscheck(False)
@cython.wraparound(False)
def fitCARMA(dt, p, q, IR, tolIR, scatterFactor, numCadences, np.ndarray[int, ndim=1, mode="c"] cadence not None, np.ndarray[double, ndim=1, mode="c"] mask not None, np.ndarray[double, ndim=1, mode="c"] t not None, np.ndarray[double, ndim=1, mode="c"] y not None, np.ndarray[double, ndim=1, mode="c"] yerr not None, maxSigma, minTimescale, maxTimescale, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode="c"] xStart not None, np.ndarray[double, ndim=1, mode="c"] Chain not None, np.ndarray[double, ndim=1, mode="c"] LnPosterior not None):
	success = fitCARMACARMA(dt, p, q, IR, tolIR, scatterFactor, numCadences, &cadence[0], &mask[0], &t[0], &y[0], &yerr[0], maxSigma, minTimescale, maxTimescale, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])
	return success