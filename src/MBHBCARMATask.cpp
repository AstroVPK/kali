#ifdef __INTEL_COMPILER
    #include <mathimf.h>
    #if defined __APPLE__ && defined __MACH__
        #include <malloc/malloc.h>
    #else
        #include <malloc.h>
    #endif
#else
    #include <math.h>
    #include <mm_malloc.h>
#endif
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include <stdio.h>
#include "MBHBCARMA.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "MBHBCARMATask.hpp"

//#define DEBUG_COMPUTELNLIKELIHOOD
#define DEBUG_FIT_MBHBCARMAMODEL

using namespace std;

int kali::MBHBCARMATask::r = 8;

kali::MBHBCARMATask::MBHBCARMATask(int pGiven, int qGiven, int numThreadsGiven, int numBurnGiven) {
	p = pGiven;
	q = qGiven;
	numThreads = numThreadsGiven;
	numBurn = numBurnGiven;
	Systems = new kali::MBHBCARMA[numThreads];
	setSystemsVec = static_cast<bool*>(_mm_malloc(numThreads*sizeof(double),64));
	ThetaVec = static_cast<double*>(_mm_malloc(numThreads*(kali::MBHBCARMATask::r + p + q + 1)*sizeof(double),64));
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		Systems[threadNum].allocMBHBCARMA(p,q);
		setSystemsVec[threadNum] = false;
		#pragma omp simd
		for (int i = 0; i < (kali::MBHBCARMATask::r + p + q + 1); ++i) {
			ThetaVec[i + threadNum*(kali::MBHBCARMATask::r + p + q + 1)] = 0.0;
			}
		}
	}

kali::MBHBCARMATask::~MBHBCARMATask() {
	if (ThetaVec) {
		_mm_free(ThetaVec);
		ThetaVec = nullptr;
		}
	if (setSystemsVec) {
		_mm_free(setSystemsVec);
		setSystemsVec = nullptr;
		}
	for (int tNum = 0; tNum < numThreads; ++tNum) {
		Systems[tNum].deallocMBHBCARMA();
		}
	delete[] Systems;
	}

int kali::MBHBCARMATask::reset_MBHBCARMATask(int pGiven, int qGiven, int numBurn) {
	int retVal = -1;
	p = pGiven;
	q = qGiven;
	numBurn = numBurn;
	if (ThetaVec) {
		_mm_free(ThetaVec);
		ThetaVec = nullptr;
		}
	ThetaVec = static_cast<double*>(_mm_malloc(numThreads*(kali::MBHBCARMATask::r + p + q + 1)*sizeof(double),64));
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		Systems[threadNum].deallocMBHBCARMA();
		Systems[threadNum].allocMBHBCARMA(p,q);
		setSystemsVec[threadNum] = false;
		#pragma omp simd
		for (int i = 0; i < (kali::MBHBCARMATask::r + p + q + 1); ++i) {
			ThetaVec[i + threadNum*(kali::MBHBCARMATask::r + p + q  +1)] = 0.0;
			}
		}
	retVal = 0;
	return retVal;
	}

int kali::MBHBCARMATask::get_numBurn() {return numBurn;}
void kali::MBHBCARMATask::set_numBurn(int numBurn) {numBurn = numBurn;}

int kali::MBHBCARMATask::check_Theta(double *Theta, int threadNum) {
	return Systems[threadNum].checkMBHBCARMAParams(Theta);
	}

double kali::MBHBCARMATask::get_dt(int threadNum) {return Systems[threadNum].get_dt();}

void kali::MBHBCARMATask::get_Theta(double *Theta, int threadNum) {
	for (int i = 0; i < (kali::MBHBCARMATask::r + p + q + 1); ++i) {
		Theta[i] = ThetaVec[i + threadNum*(kali::MBHBCARMATask::r + p + q + 1)];
		}
	}

int kali::MBHBCARMATask::set_System(double dt, double *Theta, int threadNum) {
	bool alreadySet = true;
	int retVal = -1;
	if (setSystemsVec[threadNum] == true) {
		if (Systems[threadNum].get_dt() != dt) {
			alreadySet = false;
			}
		for (int i = 0; i < (kali::MBHBCARMATask::r + p + q + 1); ++i) {
			if (ThetaVec[i + threadNum*(kali::MBHBCARMATask::r + p + q + 1)] != Theta[i]) {
				alreadySet = false;
				}
			}
		} else {
		alreadySet = false;
		}
	if (alreadySet == false) {
		int goodYN = Systems[threadNum].checkMBHBCARMAParams(Theta);
		if (goodYN == 1) {
			for (int i = 0; i < (kali::MBHBCARMATask::r + p + q + 1); ++i) {
				ThetaVec[i + threadNum*(kali::MBHBCARMATask::r + p + q + 1)] = Theta[i];
				}
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setMBHBCARMA(Theta);
			Systems[threadNum].solveMBHBCARMA();
			Systems[threadNum].resetState();
			retVal = 0;
			setSystemsVec[threadNum] = true;
			}
		} else {
		retVal = 0;
		}
	return retVal;
	}

int kali::MBHBCARMATask::reset_System(int threadNum) {
	int retVal = -1;
	Systems[threadNum].resetState();
	retVal = 0;
	return retVal;
	}

void kali::MBHBCARMATask::get_setSystemsVec(int *setSystems) {
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
		}
	}

int kali::MBHBCARMATask::print_System(int threadNum) {
	int retVal = 0;
    printf("a1: %+e (pc)\n", Systems[threadNum].getA1());
	printf("a2: %+e (pc)\n", Systems[threadNum].getA2());
	printf("Period: %+e (day) ==  %+e (year)\n", Systems[threadNum].getPeriod(), Systems[threadNum].getPeriod()/kali::Year);
	printf("m1: %+e (10^6 Solar Mass)\n", Systems[threadNum].getM1());
	printf("m2: %+e (10^6 Solar Mass)\n", Systems[threadNum].getM2());
	printf("Total Mass: %+e (10^6 Solar Mass)\n", Systems[threadNum].getM12());
	printf("Mass Ratio: %+e\n", Systems[threadNum].getM2OverM1());
	printf("rS1: %+e (pc)\n", Systems[threadNum].getRS1());
	printf("rS2: %+e (pc)\n", Systems[threadNum].getRS2());
	printf("Eccentricity: %+e\n", Systems[threadNum].getEccentricity());
	printf("rPeribothron: %+e (pc)\n", Systems[threadNum].getRPeribothronTot());
	printf("rApobothron: %+e (pc)\n", Systems[threadNum].getRApobothronTot());
	printf("Argument of periapsis (mass 1): %+e (degree)\n", Systems[threadNum].getOmega1());
	printf("Inclination: %+e (degree)\n", Systems[threadNum].getInclination());
	printf("Time of Periastron: %+e (day)\n", Systems[threadNum].getTau());
	//printf("Total Flux: %+e\n", Systems[threadNum].getTotalFlux());
	printf("dt: %+8.7e\n", Systems[threadNum].get_dt());
	printf("A\n");
	Systems[threadNum].printA();
	printf("\n");
	printf("w\n");
	Systems[threadNum].printw();
	printf("\n");
	printf("vr\n");
	Systems[threadNum].printvr();
	printf("\n");
	printf("B\n");
	Systems[threadNum].printB();
	printf("\n");
	printf("C\n");
	Systems[threadNum].printC();
	printf("\n");
	printf("expw\n");
	Systems[threadNum].printexpw();
	printf("\n");
	printf("F\n");
	Systems[threadNum].printF();
	printf("\n");
	printf("Q\n");
	Systems[threadNum].printQ();
	printf("\n");
	printf("T\n");
	Systems[threadNum].printT();
	printf("\n");
	printf("Sigma\n");
	Systems[threadNum].printSigma();
	printf("\n");
	printf("X\n");
	Systems[threadNum].printX();
	printf("\n");
	printf("P\n");
	Systems[threadNum].printP();
	printf("\n");
	return retVal;
	}


int kali::MBHBCARMATask::get_A(complex<double> *A, int threadNum) {
	int retVal = 0;
	const complex<double> *ptrToA = Systems[threadNum].getA();
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		for (int colCtr = 0; colCtr < p; ++colCtr) {
			A[rowCtr + p*colCtr] = ptrToA[rowCtr + p*colCtr];
			}
		}
	return retVal;
	}

int kali::MBHBCARMATask::get_B(complex<double> *B, int threadNum) {
	int retVal = 0;
	const complex<double> *ptrToB = Systems[threadNum].getB();
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		B[rowCtr] = ptrToB[rowCtr];
		}
	return retVal;
	}

int kali::MBHBCARMATask::get_Sigma(double *Sigma, int threadNum) {
	int retVal = 0;
	const double *ptrToSigma = Systems[threadNum].getSigma();
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		for (int colCtr = 0; colCtr < p; ++colCtr) {
			Sigma[rowCtr + p*colCtr] = ptrToSigma[rowCtr + p*colCtr];
			}
		}
	return retVal;
	}

int kali::MBHBCARMATask::get_X(double *newX, int threadNum) {
	int retVal = 0;
	Systems[threadNum].getX(newX);
	return retVal;
	}

int kali::MBHBCARMATask::set_X(double *newX, int threadNum) {
	int retVal = 0;
	Systems[threadNum].setX(newX);
	return retVal;
	}

int kali::MBHBCARMATask::get_P(double *newP, int threadNum) {
	int retVal = 0;
	Systems[threadNum].getP(newP);
	return retVal;
	}

int kali::MBHBCARMATask::set_P(double *newP, int threadNum) {
	int retVal = 0;
	Systems[threadNum].setP(newP);
	return retVal;
	}

int kali::MBHBCARMATask::make_IntrinsicLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int burnSeed, unsigned int distSeed, int threadNum) {
	int retVal = 0;
	Systems[threadNum].resetState();
	double old_dt = Systems[threadNum].get_dt();
	double* burnRand = static_cast<double*>(_mm_malloc(numBurn*p*sizeof(double),64));
	for (int i = 0; i < numBurn*p; i++) {
		burnRand[i] = 0.0;
		}
	Systems[threadNum].burnSystem(numBurn, burnSeed, burnRand);
	_mm_free(burnRand);
	double* distRand = static_cast<double*>(_mm_malloc(numCadences*p*sizeof(double),64));
	for (int i = 0; i < numCadences*p; i++) {
		distRand[i] = 0.0;
		}
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lcX = lcX;
	Data.lcP = lcP;
    Data.fracIntrinsicVar = fracIntrinsicVar;
    Data.fracNoiseToSignal = fracNoiseToSignal;
	kali::LnLikeData *ptr2Data = &Data;
	Systems[threadNum].simulateSystem(ptr2Data, distSeed, distRand);
	_mm_free(distRand);
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	return retVal;
	}

/*int kali::MBHBCARMATask::extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum) {
	int retVal = 0;
	double old_dt = Systems[threadNum].get_dt();
	double* distRand = static_cast<double*>(_mm_malloc((numCadences - cadenceNum - 1)*p*sizeof(double),64));
	for (int i = 0; i < (numCadences - cadenceNum - 1)*p; i++) {
		distRand[i] = 0.0;
		}
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lcX = lcX;
	Data.lcP = lcP;
	kali::LnLikeData *ptr2Data = &Data;
	Systems[threadNum].setX(lcX);
	Systems[threadNum].setP(lcP);
	Systems[threadNum].extendSystem(ptr2Data, distSeed, distRand);
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	_mm_free(distRand);
	return retVal;
}*/

double kali::MBHBCARMATask::get_meanFlux(double fracIntrinsicVar, int threadNum) {
	double meanFlux = -1.0;
	kali::LnLikeData Data;
	Data.fracIntrinsicVar = fracIntrinsicVar;
	kali::LnLikeData *ptr2Data = &Data;
	meanFlux = Systems[threadNum].getMeanFlux(ptr2Data);
	return meanFlux;
	}

int kali::MBHBCARMATask::make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum) {
	int retVal = 0;
	double old_dt = Systems[threadNum].get_dt();
	double* burnRand = static_cast<double*>(_mm_malloc(numBurn*p*sizeof(double),64));
	for (int i = 0; i < numBurn*p; ++i) {
		burnRand[i] = 0.0;
		}
	Systems[threadNum].burnSystem(numBurn, burnSeed, burnRand);
	_mm_free(burnRand);
	double* distRand = static_cast<double*>(_mm_malloc(numCadences*p*sizeof(double),64));
	for (int i = 0; i < numCadences*p; ++i) {
		distRand[i] = 0.0;
		}
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.fracIntrinsicVar = fracIntrinsicVar;
	Data.fracNoiseToSignal = fracNoiseToSignal;
	kali::LnLikeData *ptr2Data = &Data;
	Systems[threadNum].simulateSystem(ptr2Data, distSeed, distRand);
	_mm_free(distRand);
	double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	for (int i = 0; i < numCadences; i++) {
		noiseRand[i] = 0.0;
		}
	Systems[threadNum].observeNoise(ptr2Data, noiseSeed, noiseRand);
	_mm_free(noiseRand);
	return retVal;
	}

int kali::MBHBCARMATask::add_ObservationNoise(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
	int retVal = 0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.fracIntrinsicVar = fracIntrinsicVar;
	Data.fracNoiseToSignal = fracNoiseToSignal;
	kali::LnLikeData *ptr2Data = &Data;
	double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	for (int i = 0; i < numCadences; i++) {
		noiseRand[i] = 0.0;
		}
	Systems[threadNum].observeNoise(ptr2Data, noiseSeed, noiseRand);
	_mm_free(noiseRand);
	return retVal;
	}

/*int kali::MBHBCARMATask::extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
	int retVal = 0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.fracIntrinsicVar = fracIntrinsicVar;
	Data.fracNoiseToSignal = fracNoiseToSignal;
	kali::LnLikeData *ptr2Data = &Data;
	double* noiseRand = static_cast<double*>(_mm_malloc((numCadences - cadenceNum - 1)*sizeof(double),64));
	for (int i = 0; i < (numCadences - cadenceNum -1); i++) {
		noiseRand[i] = 0.0;
		}
	Systems[threadNum].extendObserveNoise(ptr2Data, noiseSeed, noiseRand);
	_mm_free(noiseRand);
	return retVal;
} */

double kali::MBHBCARMATask::compute_LnPrior(int numCadences, double meandt, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth, int threadNum) {
	double LnPrior = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.tolIR = tolIR;
    Data.meandt = meandt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.maxSigma = maxSigma;
	Data.minTimescale = minTimescale;
	Data.maxTimescale = maxTimescale;
    Data.lowestFlux = lowestFlux;
    Data.highestFlux = highestFlux;
    Data.periodCenter = periodCenter;
    Data.periodWidth = periodWidth;
    Data.fluxCenter = fluxCenter;
    Data.fluxWidth = fluxWidth;
	kali::LnLikeData *ptr2Data = &Data;
	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
	return LnPrior;
	}

/* double kali::MBHBCARMATask::update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	double LnPrior = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.maxSigma = maxSigma;
	Data.minTimescale = minTimescale;
	Data.maxTimescale = maxTimescale;
	kali::LnLikeData *ptr2Data = &Data;
	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
	return LnPrior;
}*/

double kali::MBHBCARMATask::compute_LnLikelihood(int numCadences, int cadenceNum, double tolIR, double startT, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth, int threadNum) {
	double LnLikelihood = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
    Data.startT = startT;
	Data.lcX = lcX;
	Data.lcP = lcP;
    Data.periodCenter = periodCenter;
    Data.periodWidth = periodWidth;
    Data.fluxCenter = fluxCenter;
    Data.fluxWidth = fluxWidth;
	kali::LnLikeData *ptr2Data = &Data;
	double old_dt = Systems[threadNum].get_dt();
	Systems[threadNum].set_dt(t[1] - t[0]);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	cadenceNum = Data.cadenceNum;
	Systems[threadNum].set_dt(old_dt);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	return LnLikelihood;
	}

/*
double kali::MBHBCARMATask::update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
	double LnLikelihood = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.currentLnLikelihood = currentLnLikelihood;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lcX = lcX;
	Data.lcP = lcP;
	kali::LnLikeData *ptr2Data = &Data;
	double old_dt = Systems[threadNum].get_dt();
	Systems[threadNum].set_dt(t[cadenceNum + 1] - t[cadenceNum]);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	Systems[threadNum].setX(lcX);
	Systems[threadNum].setP(lcP);
	LnLikelihood = Systems[threadNum].updateLnLikelihood(ptr2Data);
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	cadenceNum = Data.cadenceNum;
	Systems[threadNum].set_dt(old_dt);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	return LnLikelihood;
	}

double kali::MBHBCARMATask::compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
	double LnPrior = 0.0, LnLikelihood = 0.0, LnPosterior = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.maxSigma = maxSigma;
	Data.minTimescale = minTimescale;
	Data.maxTimescale = maxTimescale;
	kali::LnLikeData *ptr2Data = &Data;
	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
	double old_dt = Systems[threadNum].get_dt();
	Systems[threadNum].set_dt(t[1] - t[0]);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
	LnPosterior = LnPrior + LnLikelihood;
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	cadenceNum = Data.cadenceNum;
	Systems[threadNum].set_dt(old_dt);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	return LnPosterior;
	}

double kali::MBHBCARMATask::update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
	double LnPrior = 0.0, LnLikelihood = 0.0, LnPosterior = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.currentLnLikelihood = currentLnLikelihood;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.maxSigma = maxSigma;
	Data.minTimescale = minTimescale;
	Data.maxTimescale = maxTimescale;
	kali::LnLikeData *ptr2Data = &Data;
	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
	double old_dt = Systems[threadNum].get_dt();
	Systems[threadNum].set_dt(t[cadenceNum + 1] - t[cadenceNum]);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	Systems[threadNum].setX(lcX);
	Systems[threadNum].setP(lcP);
	LnLikelihood = Systems[threadNum].updateLnLikelihood(ptr2Data);
	LnPosterior = LnPrior + LnLikelihood;
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	cadenceNum = Data.cadenceNum;
	Systems[threadNum].set_dt(old_dt);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	return LnPosterior;
	}


void kali::MBHBCARMATask::compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum) {
	Systems[threadNum].computeACVF(numLags, Lags, ACVF);
	}
*/

int kali::MBHBCARMATask::fit_MBHBCARMAModel(double dt, int numCadences, double meandt, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double lowestFlux, double highestFlux, double startT, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior, double periodCenter, double periodWidth, double fluxCenter, double fluxWidth) {
	omp_set_num_threads(numThreads);
	int ndims = kali::MBHBCARMATask::r + p + q + 1;
	int threadNum = omp_get_thread_num();
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
    Data.meandt = meandt;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
    Data.startT = startT;
	Data.maxSigma = maxSigma;
	Data.minTimescale = minTimescale;
	Data.maxTimescale = maxTimescale;
    Data.lowestFlux = lowestFlux;
    Data.highestFlux = highestFlux;
    Data.periodCenter = periodCenter;
    Data.periodWidth = periodWidth;
    Data.fluxCenter = fluxCenter;
    Data.fluxWidth = fluxWidth;
	/*
    #ifdef DEBUG_FIT_MBHBCARMAMODEL
		#pragma omp critical
		{
		printf("fit_MBHBCARMAModel - threadNum: %d; maxSigma: %e\n", threadNum, maxSigma);
		printf("fit_MBHBCARMAModel - threadNum: %d; minTimescale: %e\n", threadNum, minTimescale);
		printf("fit_MBHBCARMAModel - threadNum: %d; maxTimescale: %e\n", threadNum, maxTimescale);
		}
	#endif
    */
	kali::LnLikeData *ptr2Data = &Data;
	kali::LnLikeArgs Args;
	Args.numThreads = numThreads;
	Args.Data = ptr2Data;
	Args.Systems = nullptr;
	void* p2Args = nullptr;
	Args.Systems = Systems;
	p2Args = &Args;
	double LnLikeVal = 0.0;
	double *initPos = nullptr, *offsetArr = nullptr;
	vector<vector<double>> xVec (numThreads, vector<double>(ndims));
	initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
	int nthreads = numThreads;
	nlopt::opt *optArray[numThreads];
	for (int i = 0; i < numThreads; ++i) {
		//optArray[i] = new nlopt::opt(nlopt::LN_BOBYQA, ndims); // Fastest
		optArray[i] = new nlopt::opt(nlopt::LN_NELDERMEAD, ndims); // Slower
		//optArray[i] = new nlopt::opt(nlopt::LN_COBYLA, ndims); // Slowest
		optArray[i]->set_max_objective(kali::calcLnPosterior, p2Args);
		optArray[i]->set_maxeval(maxEvals);
		optArray[i]->set_xtol_rel(xTol);
		//optArray[i]->set_maxtime(60.0); // Timeout after 60 sec.
		}
	double *max_LnPosterior = static_cast<double*>(_mm_malloc(numThreads*sizeof(double),64));
	kali::MBHBCARMA *ptrToSystems = Systems;
	#pragma omp parallel for default(none) shared(dt, nwalkers, ndims, optArray, initPos, xStart, t, ptrToSystems, xVec, max_LnPosterior, p2Args)
	for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
		int threadNum = omp_get_thread_num();
		max_LnPosterior[threadNum] = 0.0;
		xVec[threadNum].clear();
		set_System(t[1] - t[0], &xStart[walkerNum*ndims], threadNum);
		for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
			xVec[threadNum].push_back(xStart[walkerNum*ndims + dimCtr]);
			}
		#ifdef DEBUG_FIT_MBHBCARMAMODEL
			#pragma omp critical
			{
			fflush(0);
			printf("pre-opt xVec[%d][%d]: ", walkerNum, threadNum);
			for (int dimNum = 0; dimNum < ndims - 1; ++dimNum) {
				printf("%e, ", xVec[threadNum][dimNum]);
				}
			printf("%e", xVec[threadNum][ndims - 1]);
			max_LnPosterior[threadNum] = kali::calcLnPosterior(&xStart[walkerNum*ndims], p2Args);
			printf("; init_LnPosterior: %17.16e\n", max_LnPosterior[threadNum]);
			fflush(0);
			max_LnPosterior[threadNum] = 0.0;
			}
		#endif
		nlopt::result yesno = optArray[threadNum]->optimize(xVec[threadNum], max_LnPosterior[threadNum]);
		#ifdef DEBUG_FIT_MBHBCARMAMODEL
			#pragma omp critical
			{
			fflush(0);
			printf("post-opt xVec[%d][%d]: ", walkerNum, threadNum);
			for (int dimNum = 0; dimNum < ndims - 1; ++dimNum) {
				printf("%e, ", xVec[threadNum][dimNum]);
				}
			printf("%e", xVec[threadNum][ndims  - 1]);
			printf("; max_LnPosterior: %17.16e\n", max_LnPosterior[threadNum]);
			fflush(0);
			}
		#endif
		for (int dimNum = 0; dimNum < ndims; ++dimNum) {
			initPos[walkerNum*ndims + dimNum] = xVec[threadNum][dimNum];
			}
		}
	for (int i = 0; i < numThreads; ++i) {
		delete optArray[i];
		}
	_mm_free(max_LnPosterior);
	kali::EnsembleSampler newEnsemble = kali::EnsembleSampler(ndims, nwalkers, nsteps, numThreads, mcmcA, kali::calcLnPosterior, p2Args, zSSeed, walkerSeed, moveSeed);
	newEnsemble.runMCMC(initPos);
	_mm_free(initPos);
	newEnsemble.getChain(Chain);
	newEnsemble.getLnLike(LnPosterior);
	return 0;
	}

/*
int kali::MBHBCARMATask::smooth_RTS(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, int threadNum) {
	int successYN = -1;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.cadenceNum = cadenceNum;
	Data.tolIR = tolIR;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lcX = lcX;
	Data.lcP = lcP;
	kali::LnLikeData *ptr2Data = &Data;
	#ifdef DEBUG_COMPUTELNLIKELIHOOD
		for (int i = 0; i < numCadences; ++i) {
			printf("y[%d]: %+8.7e\n", i, y[i]);
			}
	#endif
	double old_dt = Systems[threadNum].get_dt();
	Systems[threadNum].set_dt(t[1] - t[0]);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	successYN = Systems[threadNum].RTSSmoother(ptr2Data, XSmooth, PSmooth);
	Systems[threadNum].getX(lcX);
	Systems[threadNum].getP(lcP);
	cadenceNum = Data.cadenceNum;
	Systems[threadNum].set_dt(old_dt);
	Systems[threadNum].solveMBHBCARMA();
	Systems[threadNum].resetState();
	return successYN;
}*/
