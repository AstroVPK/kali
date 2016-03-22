# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool

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

cdef extern from 'Task.hpp':
	cdef cppclass Task:
		Task(int p, int q, int numThreads, int numBurn) except+
		int getNumBurn()
		void setNumBurn(int numBurn)
		int checkParams(double *Theta, int threadNum)
		void setDT(double dt, int threadNum)
		int printSystem(double dt, double *Theta, int threadNum)
		int getSigma(double dt, double *Theta, double *Sigma, int threadNum)
		int makeIntrinsicLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum)
		double getMeanFlux(double dt, double *Theta, double fracIntrinsicVar, int threadNum)
		int makeObservedLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum)
		double computeLnPrior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		double computeLnLikelihood(double dt, double *Theta, int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		double computeLnPosterior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		int fitCARMA(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double scatterFactor, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior)

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

cdef class CARMATask:
	cdef Task *thisptr

	def __cinit__(self, p, q, numThreads = None, numBurn = None):
		if numThreads == None:
			numThreads = int(psutil.cpu_count(logical = False))
		if numBurn == None:
			numBurn = 1000000
		self.thisptr = new Task(p, q, numThreads, numBurn)

	def __dealloc__(self):
		del self.thisptr

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def checkParams(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.checkParams(&Theta[0], threadNum)

	def setDT(self, dt, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.setDT(dt, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def printSystem(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.printSystem(dt, &Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def getSigma(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[double, ndim=1, mode='c'] Sigma not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.getSigma(dt, &Theta[0], &Sigma[0], threadNum);

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def makeIntrinsicLC(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.makeIntrinsicLC(dt, &Theta[0], numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, threadNum)

	'''@cython.boundscheck(False)
	@cython.wraparound(False)
	def makeIntrinsicLC2(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, workingLC, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.makeIntrinsicLC(workingLC.dt, &Theta[0], workingLC.numCadences, workingLC.IR, workingLC.tolIR, workingLC.fracIntrinsicVar, workingLC.fracSignalToNoise, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, threadNum)'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def getMeanFlux(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, fracIntrinsicVar, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.getMeanFlux(dt, &Theta[0], fracIntrinsicVar, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def makeObservedLC(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.makeObservedLC(dt, &Theta[0], numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeLnPrior(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.computeLnPrior(dt, &Theta[0], numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeLnLikelihood(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.computeLnLikelihood(dt, &Theta[0], numCadences, IR, tolIR, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeLnPosterior(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.computeLnPosterior(dt, &Theta[0], numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fitCARMA(self, dt, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, scatterFactor, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPosterior not None):
		return self.thisptr.fitCARMA(dt, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, scatterFactor, &t[0], &x[0], &y[0], &yerr[0], &mask[0], nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])