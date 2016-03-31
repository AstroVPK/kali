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
		int reset_Task(int pGiven, int qGiven, int numBurn) except+
		int get_numBurn()
		void set_numBurn(int numBurn)
		int check_Theta(double *Theta, int threadNum)
		double get_dt(int threadNum)
		void get_ThetaVec(double *Theta, int threadNum)
		int set_System(double dt, double *Theta, int threadNum)
		void get_setSystemsVec(int *setSystems)
		int print_System(double dt, double *Theta, int threadNum)
		#int get_A(double dt, double *Theta, complex[double] *A, int threadNum)
		#int get_B(double dt, double *Theta, complex[double] *B, int threadNum)
		int get_Sigma(double dt, double *Theta, double *Sigma, int threadNum)
		int make_IntrinsicLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum)
		double get_meanFlux(double dt, double *Theta, double fracIntrinsicVar, int threadNum)
		int make_ObservedLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum)
		double compute_LnPrior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		double compute_LnLikelihood(double dt, double *Theta, int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		double compute_LnPosterior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		int fit_CARMAModel(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double scatterFactor, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior)

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

	def reset_Task(self, p, q, numBurn = None):
		if numBurn == None:
			numBurn = 1000000
		self.thisptr.reset_Task(p, q, numBurn)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def check_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.check_Theta(&Theta[0], threadNum)

	def get_dt(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_dt(threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_ThetaVec(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.get_ThetaVec(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_System(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.set_System(dt, &Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_setSystemsVec(self, np.ndarray[int, ndim=1, mode='c'] setSystems not None):
		self.thisptr.get_setSystemsVec(&setSystems[0])

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def print_System(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.print_System(dt, &Theta[0], threadNum)

	'''@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_A(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[np.complex128, ndim=1, mode='c'] A not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_A(dt, &Theta[0], &A[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_B(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[np.complex128, ndim=1, mode='c'] B not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_B(dt, &Theta[0], &B[0], threadNum)'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_Sigma(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[double, ndim=1, mode='c'] Sigma not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Sigma(dt, &Theta[0], &Sigma[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_IntrinsicLC(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(dt, &Theta[0], numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_meanFlux(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, fracIntrinsicVar, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_meanFlux(dt, &Theta[0], fracIntrinsicVar, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_ObservedLC(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_ObservedLC(dt, &Theta[0], numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPrior(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPrior(dt, &Theta[0], numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnLikelihood(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnLikelihood(dt, &Theta[0], numCadences, IR, tolIR, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPosterior(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPosterior(dt, &Theta[0], numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fit_CARMAModel(self, dt, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, scatterFactor, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPosterior not None):
		return self.thisptr.fit_CARMAModel(dt, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], scatterFactor, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])