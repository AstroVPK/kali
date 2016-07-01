# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool

cdef double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
cdef double Parsec = 3.0857e16
cdef double Day = 86164.090530833
cdef double Year = 31557600.0
cdef double SolarMass = 1.98855e30

cdef double d2r(double degreeVal):
	return degreeVal*(pi/180.0)

cdef double r2d(double radianVal):
	return radianVal*(180.0/pi)

cdef extern from 'binarySMBHTask.hpp':
	cdef cppclass binarySMBHTask:
		binarySMBHTask(int numThreads) except+
		int check_Theta(double *Theta, int threadNum);
		void get_Theta(double *Theta, int threadNum);
		int set_System(double *Theta, int threadNum);
		int reset_System(double timeGiven, int threadNum);
		void get_setSystemsVec(int *setSystems);
		int print_System(int threadNum);
		double get_Period(int threadNum);

		int make_IntrinsicLC(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int add_ObservationNoise(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);
		double compute_LnPrior(int numCadences, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double compute_LnLikelihood(int numCadences, int cadenceNum, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int fit_BinarySMBHModel(int numCadences, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);

cdef class bSMBHTask:
	cdef binarySMBHTask *thisptr

	def __cinit__(self, numThreads = None):
		if numThreads == None:
			numThreads = int(psutil.cpu_count(logical = True))
		self.thisptr = new binarySMBHTask(numThreads)

	def __dealloc__(self):
		del self.thisptr

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def check_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.check_Theta(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.get_Theta(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_System(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.set_System(&Theta[0], threadNum)

	def reset_System(self, epoch, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.reset_System(epoch, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_setSystemsVec(self, np.ndarray[int, ndim=1, mode='c'] setSystems not None):
		self.thisptr.get_setSystemsVec(&setSystems[0])

	def print_System(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.print_System(threadNum)

	def get_Period(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Period(threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_IntrinsicLC(self, numCadences, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(numCadences, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def add_ObservationNoise(self, numCadences, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.add_ObservationNoise(numCadences, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPrior(self, numCadences, lowestFlux, highestFlux, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPrior(numCadences, lowestFlux, highestFlux, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnLikelihood(self, numCadences, cadenceNum, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnLikelihood(numCadences, cadenceNum, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fit_BinarySMBHModel(self, numCadences, lowestFlux, highestFlux, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPosterior not None):
		return self.thisptr.fit_BinarySMBHModel(numCadences, lowestFlux, highestFlux, &t[0], &x[0], &y[0], &yerr[0], &mask[0], nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])