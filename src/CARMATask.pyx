# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool

cdef extern from 'CARMA.hpp':
	void getSigma(int numP, int numQ, double *Theta, double *SigmaOut)

cdef extern from 'LC.hpp':
	cdef cppclass LCData:
		int numCadences
		double dt
		double meandt
		double mindt
		double maxdt
		double dtSmooth
		double tolIR
		double fracIntrinsicVar
		double fracNoiseToSignal
		double maxSigma
		double minTimescale
		double maxTimescale
		double *t
		double *x
		double *y
		double *yerr
		double *mask
		double *lcXSim
		double *lcPSim
		double *lcXComp
		double *lcPComp
		LCData() except+

		int acvf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *acvfVals, double *acvfErrvals, int threadNum)
		int sf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double*maskIn, double *lagVals, double *sfVals, double *sfErrVals, int threadNum)
		int dacf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, int numBins, double *lagVals, double *acvfVals, double *acvfErrVals, int threadNum)

cdef extern from 'Task.hpp':
	cdef cppclass Task:
		Task(int p, int q, int numThreads, int numBurn) except+
		int reset_Task(int pGiven, int qGiven, int numBurn) except+
		int get_numBurn()
		void set_numBurn(int numBurn)
		int check_Theta(double *Theta, int threadNum)
		double get_dt(int threadNum)
		void get_Theta(double *Theta, int threadNum)
		int set_System(double dt, double *Theta, int threadNum)
		int reset_System(int threadNum);
		void get_setSystemsVec(int *setSystems)
		int print_System(int threadNum)
		#int get_A(complex[double] *A, int threadNum)
		#int get_B(complex[double] *B, int threadNum)
		int get_Sigma(double *Sigma, int threadNum)
		int get_X(double *newX, int threadNum)
		int set_X(double *newX, int threadNum)
		int get_P(double *newP, int threadNum)
		int set_P(double *newP, int threadNum)

		int make_IntrinsicLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int burnSeed, unsigned int distSeed, int threadNum)
		double get_meanFlux(double fracIntrinsicVar, int threadNum)
		int extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum)
		int make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum)
		int add_ObservationNoise(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum)
		int extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);

		double compute_LnPrior(int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)
		double update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)

		double compute_LnLikelihood(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)
		double update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)

		double compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)
		double update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)

		void compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum)

		int fit_CARMAModel(double dt, int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior)

		int smooth_RTS(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, int threadNum)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_Sigma(pNum, qNum, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[double, ndim=1, mode='c'] Sigma not None):
	getSigma(pNum, qNum, &Theta[0], &Sigma[0])

cdef class lc:
	cdef LCData *thisptr

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcXSim not None, np.ndarray[double, ndim=1, mode='c'] lcPSim not None, np.ndarray[double, ndim=1, mode='c'] lcXComp not None, np.ndarray[double, ndim=1, mode='c'] lcPComp not None, dt = 1.0, meandt = 1.0, mindt = 1.0, maxdt = 1.0, dtSmooth = 1.0, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 2.0, maxTimescale = 0.5):
		self.thisptr = new LCData()
		self.thisptr.numCadences = t.shape[0]
		self.thisptr.dt = dt
		self.thisptr.mindt = mindt
		self.thisptr.maxdt = maxdt
		self.thisptr.meandt = meandt
		self.thisptr.dtSmooth = dtSmooth
		self.thisptr.tolIR = tolIR
		self.thisptr.fracIntrinsicVar = fracIntrinsicVar
		self.thisptr.fracNoiseToSignal = fracNoiseToSignal
		self.thisptr.maxSigma = maxSigma
		self.thisptr.minTimescale = minTimescale
		self.thisptr.maxTimescale = maxTimescale
		self.thisptr.t = &t[0]
		self.thisptr.x = &x[0]
		self.thisptr.y = &y[0]
		self.thisptr.yerr = &yerr[0]
		self.thisptr.mask = &mask[0]
		self.thisptr.lcXSim = &lcXSim[0]
		self.thisptr.lcPSim = &lcPSim[0]
		self.thisptr.lcXComp = &lcXComp[0]
		self.thisptr.lcPComp = &lcPComp[0]

	property numCadences:
		def __get__(self): return self.thisptr.numCadences
		def __set__(self, numCadences): self.thisptr.numCadences = numCadences

	property dt:
		def __get__(self): return self.thisptr.dt
		def __set__(self, dt): self.thisptr.dt = dt

	property meandt:
		def __get__(self): return self.thisptr.meandt
		def __set__(self, meandt): self.thisptr.meandt = meandt

	property mindt:
		def __get__(self): return self.thisptr.mindt
		def __set__(self, mindt): self.thisptr.mindt = mindt

	property maxdt:
		def __get__(self): return self.thisptr.maxdt
		def __set__(self, maxdt): self.thisptr.maxdt = maxdt

	property dtSmooth:
		def __get__(self): return self.thisptr.dtSmooth
		def __set__(self, dtSmooth): self.thisptr.dtSmooth = dtSmooth

	property tolIR:
		def __get__(self): return self.thisptr.tolIR
		def __set__(self, tolIR): self.thisptr.tolIR = tolIR

	property fracIntrinsicVar:
		def __get__(self): return self.thisptr.fracIntrinsicVar
		def __set__(self, fracIntrinsicVar): self.thisptr.fracIntrinsicVar = fracIntrinsicVar

	property fracNoiseToSignal:
		def __get__(self): return self.thisptr.fracNoiseToSignal
		def __set__(self, fracNoiseToSignal): self.thisptr.fracNoiseToSignal = fracNoiseToSignal

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
	def compute_ACVF(self, numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] acvfVals not None, np.ndarray[double, ndim=1, mode='c'] acvfErrVals not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.acvf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], &lagVals[0], &acvfVals[0], &acvfErrVals[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_SF(self, numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] sfVals not None, np.ndarray[double, ndim=1, mode='c'] sfErrVals not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.sf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], &lagVals[0], &sfVals[0], &sfErrVals[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_DACF(self, numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, numBins, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] dacfVals not None, np.ndarray[double, ndim=1, mode='c'] dacfErrVals not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.dacf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], numBins, &lagVals[0], &dacfVals[0], &dacfErrVals[0], threadNum)

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
	def get_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.get_Theta(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_System(self, dt, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.set_System(dt, &Theta[0], threadNum)

	def reset_System(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.reset_System(threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_setSystemsVec(self, np.ndarray[int, ndim=1, mode='c'] setSystems not None):
		self.thisptr.get_setSystemsVec(&setSystems[0])

	def print_System(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.print_System(threadNum)

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
	def get_Sigma(self, np.ndarray[double, ndim=1, mode='c'] Sigma not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Sigma(&Sigma[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_X(self, np.ndarray[double, ndim=1, mode='c'] newX not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_X(&newX[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_X(self, np.ndarray[double, ndim=1, mode='c'] newX not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.set_X(&newX[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_P(self, np.ndarray[double, ndim=1, mode='c'] newP not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_P(&newP[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_P(self, np.ndarray[double, ndim=1, mode='c'] newP not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.set_P(&newP[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_IntrinsicLC(self, numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcX not None, np.ndarray[double, ndim=1, mode='c'] lcP not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &lcX[0], &lcP[0], burnSeed, distSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def extend_IntrinsicLC(self, numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcX not None, np.ndarray[double, ndim=1, mode='c'] lcP not None, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.extend_IntrinsicLC(numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &lcX[0], &lcP[0], distSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_meanFlux(self, fracIntrinsicVar, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_meanFlux(fracIntrinsicVar, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_ObservedLC(self, numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_ObservedLC(numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def add_ObservationNoise(self, numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.add_ObservationNoise(numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def extend_ObservationNoise(self, numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.extend_ObservationNoise(numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPrior(self, numCadences, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPrior(numCadences, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def update_LnPrior(self, numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.update_LnPrior(numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnLikelihood(self, numCadences, cadenceNum, tolIR, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnLikelihood(numCadences, cadenceNum, tolIR, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def update_LnLikelihood(self, numCadences, cadenceNum, currentLnLikelihood, tolIR, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.update_LnLikelihood(numCadences, cadenceNum, currentLnLikelihood, tolIR, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPosterior(self, numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPosterior(numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def update_LnPosterior(self, numCadences, cadenceNum, currentLnLikelihood, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.update_LnPosterior(numCadences, cadenceNum, currentLnLikelihood, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_ACVF(self, numLags, np.ndarray[double, ndim=1, mode='c'] Lags not None, np.ndarray[double, ndim=1, mode='c'] ACVF not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.compute_ACVF(numLags, &Lags[0], &ACVF[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fit_CARMAModel(self, dt, numCadences, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPosterior not None):
		return self.thisptr.fit_CARMAModel(dt, numCadences, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def smooth_RTS(self, numCadences, cadenceNum, tolIR, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, np.ndarray[double, ndim=1, mode='c'] XSmooth not None, np.ndarray[double, ndim=1, mode='c'] PSmooth not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.smooth_RTS(numCadences, cadenceNum, tolIR, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], &XSmooth[0], &PSmooth[0], threadNum)