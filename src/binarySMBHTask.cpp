#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include <stdio.h>
#include "binarySMBH.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "binarySMBHTask.hpp"

//#define DEBUG_COMPUTELNLIKELIHOOD
//#define DEBUG_FIT_BEAMEDMODEL
//#define DEBUG_SETSYSTEM

using namespace std;

int lenTheta = 9;

binarySMBHTask::binarySMBHTask(int numThreadsGiven) {
	numThreads = numThreadsGiven;
	Systems = new binarySMBH[numThreads];
	setSystemsVec = static_cast<bool*>(_mm_malloc(numThreads*sizeof(double),64));
	ThetaVec = static_cast<double*>(_mm_malloc(numThreads*lenTheta*sizeof(double),64)); // We fix alpha1 and alpha2
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		setSystemsVec[threadNum] = false;
		#pragma omp simd
		for (int i = 0; i < lenTheta; ++i) {
			ThetaVec[i + threadNum*lenTheta] = 0.0;
			}
		}
	}

binarySMBHTask::~binarySMBHTask() {
	if (ThetaVec) {
		_mm_free(ThetaVec);
		ThetaVec = nullptr;
		}
	if (setSystemsVec) {
		_mm_free(setSystemsVec);
		setSystemsVec = nullptr;
		}
	delete[] Systems;
	}

int binarySMBHTask::check_Theta(double *Theta, int threadNum) {
	return Systems[threadNum].checkBinarySMBHParams(Theta);
	}

void binarySMBHTask::get_Theta(double *Theta, int threadNum) {
	for (int i = 0; i < lenTheta; ++i) {
		Theta[i] = ThetaVec[i + threadNum*lenTheta];
		}
	}

int binarySMBHTask::set_System(double *Theta, int threadNum) {
	bool alreadySet = true;
	int retVal = -1;
	if (setSystemsVec[threadNum] == true) {
		for (int i = 0; i < lenTheta; ++i) {
			if (ThetaVec[i + threadNum*lenTheta] != Theta[i]) {
				alreadySet = false;
				}
			}
		} else {
		alreadySet = false;
		}
	#ifdef DEBUG_SETSYSTEM
		printf("alreadySet: %d\n", alreadySet);
	#endif
	if (alreadySet == false) {
		int goodYN = Systems[threadNum].checkBinarySMBHParams(Theta);
		if (goodYN == 1) {
			for (int i = 0; i < lenTheta; ++i) {
				ThetaVec[i + threadNum*lenTheta] = Theta[i];
				}
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].setBinarySMBH(Theta);
			retVal = 0;
			setSystemsVec[threadNum] = true;
			}
		} else {
		retVal = -1;
		}
	return retVal;
	}

int binarySMBHTask::reset_System(double timeGiven, int threadNum) {
	int retVal = -1;
	Systems[threadNum](timeGiven*Day);
	retVal = 0;
	return retVal;
	}

void binarySMBHTask::get_setSystemsVec(int *setSystems) {
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
		}
	}

void binarySMBHTask::print_System(int threadNum) {
	Systems[threadNum].print();
	}

double binarySMBHTask::get_Period(int threadNum) {
	return Systems[threadNum].getPeriod();
	}

int binarySMBHTask::make_IntrinsicLC(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	LnLikeData *ptr2Data = &Data;
	Systems[threadNum].simulateSystem(ptr2Data);
	return 0;
	}

int binarySMBHTask::add_ObservationNoise(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.fracNoiseToSignal = fracNoiseToSignal;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	LnLikeData *ptr2Data = &Data;
	double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	#pragma omp simd
	for (int i = 0; i < numCadences; i++) {
		noiseRand[i] = 0.0;
		}
	Systems[threadNum].observeNoise(ptr2Data, noiseSeed, noiseRand);
	_mm_free(noiseRand);
	return 0;
	}