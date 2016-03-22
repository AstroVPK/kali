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
		int makeIntrinsicLC(double *Theta, int numCadences, double dt, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum)

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
	def makeIntrinsicLC(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, dt, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.makeIntrinsicLC(&Theta[0], numCadences, dt, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, threadNum)