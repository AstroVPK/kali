# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool


cdef extern from 'MBHBCARMA.hpp' namespace "kali":
	void getSigma(int numR, int numP, int numQ, double *Theta, double *SigmaOut)


cdef extern from 'MBHBCARMATask.hpp' namespace "kali":
	cdef cppclass MBHBCARMATask:
		MBHBCARMATask(int p, int q, int numThreads, int numBurn) except+
		int reset_MBHBCARMATask(int pGiven, int qGiven, int numBurn) except+
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

		void set_Epoch(double tIn, int threadNum);
		double get_Epoch(int threadNum);
		double get_Period(int threadNum);
		double get_A1(int threadNum);
		double get_A2(int threadNum);
		double get_M1(int threadNum);
		double get_M2(int threadNum);
		double get_M12(int threadNum);
		double get_M2OverM1(int threadNum);
		double get_RPeribothron1(int threadNum);
		double get_RPeribothron2(int threadNum);
		double get_RApobothron1(int threadNum);
		double get_RApobothron2(int threadNum);
		double get_RPeribothronTot(int threadNum);
		double get_RApobothronTot(int threadNum);
		double get_RS1(int threadNum);
		double get_RS2(int threadNum);
		double get_Eccentricity(int threadNum);
		double get_Omega1(int threadNum);
		double get_Omega2(int threadNum);
		double get_Inclination(int threadNum);
		double get_Tau(int threadNum);
		double get_MeanAnomoly(int threadNum);
		double get_EccentricAnomoly(int threadNum);
		double get_TrueAnomoly(int threadNum);
		double get_R1(int threadNum);
		double get_R2(int threadNum);
		double get_Theta1(int threadNum);
		double get_Theta2(int threadNum);
		double get_Beta1(int threadNum);
		double get_Beta2(int threadNum);
		double get_RadialBeta1(int threadNum);
		double get_RadialBeta2(int threadNum);
		double get_DopplerFactor1(int threadNum);
		double get_DopplerFactor2(int threadNum);
		double get_BeamingFactor1(int threadNum);
		double get_BeamingFactor2(int threadNum);
		double get_aH(double sigmaStars, int threadNum);
		double get_aGW(double sigmaStars, double rhoStars, double H, int threadNum);
		double get_durationInHardState(double sigmaStars, double rhoStars, double H, int threadNum);
		double get_ejectedMass(double sigmaStars, double rhoStars, double H, int threadNum);

		int make_BeamedLC(int numCadences, double tolIR, double startT, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);
		int make_IntrinsicLC(int numCadences, double tolIR, double startT, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int burnSeed, unsigned int distSeed, int threadNum)
		double get_meanFlux(double fracIntrinsicVar, int threadNum)
		#int extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum)
		int make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum)
		int add_ObservationNoise(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum)
		#int extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);

		double compute_LnPrior(int numCadences, double meandt, double tolIR, double startT, double maxSigma, double minTimescale, double maxTimescale, double lowestFlux, double highestflux, double *t, double *x, double *y, double *yerr, double *mask, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth, int threadNum)
		#double update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum)

		double compute_LnLikelihood(int numCadences, int cadenceNum, double tolIR, double startT, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth, int threadNum)
		#double update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)

		#double compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)
		#double update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum)

		#void compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum)

		int fit_MBHBCARMAModel(double dt, int numCadences, double meandt, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double lowestFlux, double highestFlux, double startT, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPrior, double *LnLikelihood, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth);

		int smooth_RTS(int numCadences, int cadenceNum, double tolIR, double startT, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, double *xSmooth, double *xerrSmooth, int threadNum)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_Sigma(rNum, pNum, qNum, np.ndarray[double, ndim=1, mode='c'] Theta not None, np.ndarray[double, ndim=1, mode='c'] Sigma not None):
	getSigma(rNum, pNum, qNum, &Theta[0], &Sigma[0])


cdef class MBHBCARMATask_cython:
	cdef MBHBCARMATask *thisptr

	def __cinit__(self, p, q, numThreads = None, numBurn = None):
		if numThreads == None:
			numThreads = int(psutil.cpu_count(logical = False))
		if numBurn == None:
			numBurn = 1000000
		self.thisptr = new MBHBCARMATask(p, q, numThreads, numBurn)

	def __dealloc__(self):
		del self.thisptr

	def reset_MBHBCARMATask(self, p, q, numBurn = None):
		if numBurn == None:
			numBurn = 1000000
		self.thisptr.reset_MBHBCARMATask(p, q, numBurn)

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

	def set_Epoch(self, epochIn, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.set_Epoch(epochIn, threadNum)

	def get_Epoch(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Epoch(threadNum)

	def get_Period(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Period(threadNum)

	def get_A1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_A1(threadNum)

	def get_A2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_A2(threadNum)

	def get_M1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_M1(threadNum)

	def get_M2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_M2(threadNum)

	def get_M12(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_M12(threadNum)

	def get_M2OverM1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_M2OverM1(threadNum)

	def get_RPeribothron1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RPeribothron1(threadNum)

	def get_RPeribothron2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RPeribothron2(threadNum)

	def get_RApobothron1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RApobothron1(threadNum)

	def get_RApobothron2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RApobothron2(threadNum)

	def get_RPeribothronTot(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RPeribothronTot(threadNum)

	def get_RApobothronTot(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RApobothronTot(threadNum)

	def get_RS1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RS1(threadNum)

	def get_RS2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RS2(threadNum)

	def get_Eccentricity(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Eccentricity(threadNum)

	def get_Omega1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Omega1(threadNum)

	def get_Omega2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Omega2(threadNum)

	def get_Inclination(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Inclination(threadNum)

	def get_Tau(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Tau(threadNum)

	def get_MeanAnamoly(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_MeanAnomoly(threadNum)

	def get_EccentricAnamoly(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_EccentricAnomoly(threadNum)

	def get_TrueAnamoly(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_TrueAnomoly(threadNum)

	def get_R1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_R1(threadNum)

	def get_R2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_R2(threadNum)

	def get_Theta1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Theta1(threadNum)

	def get_Theta2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Theta2(threadNum)

	def get_Beta1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Beta1(threadNum)

	def get_Beta2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_Beta2(threadNum)

	def get_RadialBeta1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RadialBeta1(threadNum)

	def get_RadialBeta2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_RadialBeta2(threadNum)

	def get_DopplerFactor1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_DopplerFactor1(threadNum)

	def get_DopplerFactor2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_DopplerFactor2(threadNum)

	def get_BeamingFactor1(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_BeamingFactor1(threadNum)

	def get_BeamingFactor2(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_BeamingFactor2(threadNum)

	def get_aH(self, sigmaStars, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_aH(sigmaStars, threadNum)

	def get_aGW(self, sigmaStars, rhoStars, H, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_aGW(sigmaStars, rhoStars, H, threadNum)

	def get_durationInHardState(self, sigmaStars, rhoStars, H, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_durationInHardState(sigmaStars, rhoStars, H, threadNum)

	def get_ejectedMass(self, sigmaStars, rhoStars, H, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_ejectedMass(sigmaStars, rhoStars, H, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_BeamedLC(self, numCadences, tolIR, startT, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcX not None, np.ndarray[double, ndim=1, mode='c'] lcP not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_BeamedLC(numCadences, tolIR, startT, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &lcX[0], &lcP[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_IntrinsicLC(self, numCadences, tolIR, startT, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcX not None, np.ndarray[double, ndim=1, mode='c'] lcP not None, burnSeed, distSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(numCadences, tolIR, startT, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &lcX[0], &lcP[0], burnSeed, distSeed, threadNum)

	'''
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def extend_IntrinsicLC(self, numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] lcX not None, np.ndarray[double, ndim=1, mode='c'] lcP not None, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.extend_IntrinsicLC(numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &lcX[0], &lcP[0], distSeed, threadNum)
	'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_meanFlux(self, fracIntrinsicVar, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.get_meanFlux(fracIntrinsicVar, threadNum)

	'''
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_ObservedLC(self, numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, burnSeed, distSeed, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_ObservedLC(numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], burnSeed, distSeed, noiseSeed, threadNum)
	'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def add_ObservationNoise(self, numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.add_ObservationNoise(numCadences, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)

	'''
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def extend_ObservationNoise(self, numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.extend_ObservationNoise(numCadences, cadenceNum, tolIR, fracIntrinsicVar, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)
	'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPrior(self, numCadences, meandt, tolIR, startT, maxSigma, minTimescale, maxTimescale, lowestFlux, highestflux, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, periodCenter, periodWidth, fluxCenter, fluxWidth, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPrior(numCadences, meandt, tolIR, startT, maxSigma, minTimescale, maxTimescale, lowestFlux, highestflux, &t[0], &x[0], &y[0], &yerr[0], &mask[0], periodCenter, periodWidth, fluxCenter, fluxWidth, threadNum)

	'''
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def update_LnPrior(self, numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.update_LnPrior(numCadences, cadenceNum, tolIR, maxSigma, minTimescale, maxTimescale, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)
	'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnLikelihood(self, numCadences, cadenceNum, tolIR, startT, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, periodCenter, periodWidth, fluxCenter, fluxWidth, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnLikelihood(numCadences, cadenceNum, tolIR, startT, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], periodCenter, periodWidth, fluxCenter, fluxWidth, threadNum)

	'''
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
	'''

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fit_CARMAModel(self, dt, numCadences, meandt, tolIR, maxSigma, minTimescale, maxTimescale, lowestFlux, highestFlux, startT, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPrior not None, np.ndarray[double, ndim=1, mode='c'] LnLikelihood not None, periodCenter, periodWidth, fluxCenter, fluxWidth):
		return self.thisptr.fit_MBHBCARMAModel(dt, numCadences, meandt, tolIR, maxSigma, minTimescale, maxTimescale, lowestFlux, highestFlux, startT, &t[0], &x[0], &y[0], &yerr[0], &mask[0], nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPrior[0], &LnLikelihood[0], periodCenter, periodWidth, fluxCenter, fluxWidth)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def smooth_RTS(self, numCadences, cadenceNum, tolIR, startT, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, np.ndarray[double, ndim=1, mode='c'] X not None, np.ndarray[double, ndim=1, mode='c'] P not None, np.ndarray[double, ndim=1, mode='c'] XSmooth not None, np.ndarray[double, ndim=1, mode='c'] PSmooth not None, np.ndarray[double, ndim=1, mode='c'] xSmooth not None, np.ndarray[double, ndim=1, mode='c'] xerrSmooth not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.smooth_RTS(numCadences, cadenceNum, tolIR, startT, &t[0], &x[0], &y[0], &yerr[0], &mask[0], &X[0], &P[0], &XSmooth[0], &PSmooth[0], &xSmooth[0], &xerrSmooth[0], threadNum)
