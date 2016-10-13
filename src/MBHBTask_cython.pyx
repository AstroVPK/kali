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

cdef extern from 'MBHBTask.hpp' namespace "kali":
	cdef cppclass MBHBTask:
		MBHBTask(int numThreads) except+
		int check_Theta(double *Theta, int threadNum);
		void get_Theta(double *Theta, int threadNum);
		int set_System(double *Theta, int threadNum);
		void get_setSystemsVec(int *setSystems);
		int print_System(int threadNum);

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

		int make_IntrinsicLC(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int add_ObservationNoise(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);
		double compute_LnPrior(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double compute_LnLikelihood(int numCadences, double dt, int cadenceNum, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int fit_MBHBModel(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);

cdef class MBHBTask_cython:
	cdef MBHBTask *thisptr

	def __cinit__(self, numThreads = None):
		if numThreads == None:
			numThreads = int(psutil.cpu_count(logical = True))
		self.thisptr = new MBHBTask(numThreads)

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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_setSystemsVec(self, np.ndarray[int, ndim=1, mode='c'] setSystems not None):
		self.thisptr.get_setSystemsVec(&setSystems[0])

	def print_System(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.print_System(threadNum)

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
	def make_IntrinsicLC(self, numCadences, dt, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(numCadences, dt, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def add_ObservationNoise(self, numCadences, dt, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.add_ObservationNoise(numCadences, dt, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnPrior(self, numCadences, dt, lowestFlux, highestFlux, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnPrior(numCadences, dt, lowestFlux, highestFlux, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def compute_LnLikelihood(self, numCadences, dt, cadenceNum, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.compute_LnLikelihood(numCadences, dt, cadenceNum, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def fit_MBHBModel(self, numCadences, dt, lowestFlux, highestFlux, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, np.ndarray[double, ndim=1, mode='c'] xStart not None, np.ndarray[double, ndim=1, mode='c'] Chain not None, np.ndarray[double, ndim=1, mode='c'] LnPosterior not None):
		return self.thisptr.fit_MBHBModel(numCadences, dt, lowestFlux, highestFlux, &t[0], &x[0], &y[0], &yerr[0], &mask[0], nwalkers, nsteps, maxEvals, xTol, mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, &xStart[0], &Chain[0], &LnPosterior[0])
