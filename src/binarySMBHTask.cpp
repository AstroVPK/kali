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
#define DEBUG_FIT_BINARYSMBHMODEL
//#define DEBUG_SETSYSTEM

using namespace std;

int lenTheta = 8;

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

void binarySMBHTask::get_setSystemsVec(int *setSystems) {
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
		}
	}

void binarySMBHTask::print_System(int threadNum) {
	Systems[threadNum].print();
	}

void binarySMBHTask::set_Epoch(double epochIn, int threadNum) {
	Systems[threadNum].setEpoch(epochIn);
	}

double binarySMBHTask::get_Epoch(int threadNum) {
	return Systems[threadNum].getEpoch();
	}

double binarySMBHTask::get_Period(int threadNum) {
	return Systems[threadNum].getPeriod();
	}

double binarySMBHTask::get_A1(int threadNum) {
	return Systems[threadNum].getA1();
	}

double binarySMBHTask::get_A2(int threadNum) {
	return Systems[threadNum].getA2();
	}

double binarySMBHTask::get_M1(int threadNum) {
	return Systems[threadNum].getM1();
	}

double binarySMBHTask::get_M2(int threadNum) {
	return Systems[threadNum].getM2();
	}

double binarySMBHTask::get_RPeri1(int threadNum) {
	return Systems[threadNum].getRPeri1();
	}

double binarySMBHTask::get_RPeri2(int threadNum) {
	return Systems[threadNum].getRPeri2();
	}

double binarySMBHTask::get_RApo1(int threadNum) {
	return Systems[threadNum].getRApo1();
	}

double binarySMBHTask::get_RApo2(int threadNum) {
	return Systems[threadNum].getRApo2();
	}

double binarySMBHTask::get_RPeriTot(int threadNum) {
	return Systems[threadNum].getRPeriTot();
	}

double binarySMBHTask::get_RApoTot(int threadNum) {
	return Systems[threadNum].getRApoTot();
	}

double binarySMBHTask::get_RS1(int threadNum) {
	return Systems[threadNum].getRS1();
	}

double binarySMBHTask::get_RS2(int threadNum) {
	return Systems[threadNum].getRS2();
	}

double binarySMBHTask::get_Eccentricity(int threadNum) {
	return Systems[threadNum].getEccentricity();
	}

double binarySMBHTask::get_Omega1(int threadNum) {
	return Systems[threadNum].getOmega1();
	}

double binarySMBHTask::get_Omega2(int threadNum) {
	return Systems[threadNum].getOmega2();
	}

double binarySMBHTask::get_Inclination(int threadNum) {
	return Systems[threadNum].getInclination();
	}

double binarySMBHTask::get_Tau(int threadNum) {
	return Systems[threadNum].getTau();
	}

double binarySMBHTask::get_MeanAnomoly(int threadNum) {
	return Systems[threadNum].getMeanAnomoly();
	}

double binarySMBHTask::get_EccentricAnomoly(int threadNum) {
	return Systems[threadNum].getEccentricAnomoly();
	}

double binarySMBHTask::get_TrueAnomoly(int threadNum) {
	return Systems[threadNum].getTrueAnomoly();
	}

double binarySMBHTask::get_R1(int threadNum) {
	return Systems[threadNum].getR1();
	}

double binarySMBHTask::get_R2(int threadNum) {
	return Systems[threadNum].getR2();
	}

double binarySMBHTask::get_Theta1(int threadNum) {
	return Systems[threadNum].getTheta1();
	}

double binarySMBHTask::get_Theta2(int threadNum) {
	return Systems[threadNum].getTheta2();
	}

double binarySMBHTask::get_Beta1(int threadNum) {
	return Systems[threadNum].getBeta1();
	}

double binarySMBHTask::get_Beta2(int threadNum) {
	return Systems[threadNum].getBeta2();
	}

double binarySMBHTask::get_RadialBeta1(int threadNum) {
	return Systems[threadNum].getRadialBeta1();
	}

double binarySMBHTask::get_RadialBeta2(int threadNum) {
	return Systems[threadNum].getRadialBeta2();
	}

double binarySMBHTask::get_DopplerFactor1(int threadNum) {
	return Systems[threadNum].getDopplerFactor1();
	}

double binarySMBHTask::get_DopplerFactor2(int threadNum) {
	return Systems[threadNum].getDopplerFactor2();
	}

double binarySMBHTask::get_BeamingFactor1(int threadNum) {
	return Systems[threadNum].getBeamingFactor1();
	}

double binarySMBHTask::get_BeamingFactor2(int threadNum) {
	return Systems[threadNum].getBeamingFactor2();
	}

double binarySMBHTask::get_aH(double sigmaStars, int threadNum) {
	return Systems[threadNum].aH(sigmaStars);
	}

double binarySMBHTask::get_aGW(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].aGW(sigmaStars, rhoStars, H);
	}

double binarySMBHTask::get_durationInHardState(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].durationInHardState(sigmaStars, rhoStars, H);
	}

double binarySMBHTask::get_ejectedMass(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].ejectedMass(sigmaStars, rhoStars, H);
	}

int binarySMBHTask::make_IntrinsicLC(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	LnLikeData *ptr2Data = &Data;
	Systems[threadNum].simulateSystem(ptr2Data);
	return 0;
	}

int binarySMBHTask::add_ObservationNoise(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
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

double binarySMBHTask::compute_LnPrior(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	double LnPrior = 0.0;
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lowestFlux = lowestFlux;
	Data.highestFlux = highestFlux;
	LnLikeData *ptr2Data = &Data;

	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);

	return LnPrior;
	}

double binarySMBHTask::compute_LnLikelihood(int numCadences, double dt, int cadenceNum, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	double LnLikelihood = 0.0;
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.cadenceNum = cadenceNum;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	LnLikeData *ptr2Data = &Data;

	Systems[threadNum].setEpoch(t[0]);
	LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
	cadenceNum = Data.cadenceNum;

	return LnLikelihood;
	}

int binarySMBHTask::fit_BinarySMBHModel(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior) {
	#ifdef DEBUG_FIT_BINARYSMBHMODEL
		printf("numThreads: %d\n",numThreads);
	#endif
	omp_set_num_threads(numThreads);
	int ndims = lenTheta;
	int threadNum = omp_get_thread_num();
	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lowestFlux = lowestFlux;
	Data.highestFlux = highestFlux;
	LnLikeData *ptr2Data = &Data;
	LnLikeArgs Args;
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
		optArray[i]->set_max_objective(calcLnPosterior, p2Args);
		optArray[i]->set_maxeval(maxEvals);
		optArray[i]->set_xtol_rel(xTol);
		//optArray[i]->set_maxtime(60.0); // Timeout after 60 sec.
		}
	double *max_LnPosterior = static_cast<double*>(_mm_malloc(numThreads*sizeof(double),64));
	binarySMBH *ptrToSystems = Systems;
	#pragma omp parallel for default(none) shared(nwalkers, ndims, optArray, initPos, xStart, t, ptrToSystems, xVec, max_LnPosterior, p2Args)
	for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
		int threadNum = omp_get_thread_num();
		max_LnPosterior[threadNum] = 0.0;
		xVec[threadNum].clear();
		set_System(&xStart[walkerNum*ndims], threadNum);
		for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
			xVec[threadNum].push_back(xStart[walkerNum*ndims + dimCtr]);
			}
		#ifdef DEBUG_FIT_BINARYSMBHMODEL
			#pragma omp critical
			{
			fflush(0);
			printf("pre-opt xVec[%d][%d]: ", walkerNum, threadNum);
			for (int dimNum = 0; dimNum < ndims - 1; ++dimNum) {
				printf("%e, ", xVec[threadNum][dimNum]);
				}
			printf("%e", xVec[threadNum][ndims - 1]);
			max_LnPosterior[threadNum] = calcLnPosterior(&xStart[walkerNum*ndims], p2Args);
			printf("; init_LnPosterior: %17.16e\n", max_LnPosterior[threadNum]);
			fflush(0);
			max_LnPosterior[threadNum] = 0.0;
			}
		#endif
		nlopt::result yesno = optArray[threadNum]->optimize(xVec[threadNum], max_LnPosterior[threadNum]);
		#ifdef DEBUG_FIT_BINARYSMBHMODEL
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
	EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, numThreads, mcmcA, calcLnPosterior, p2Args, zSSeed, walkerSeed, moveSeed);
	newEnsemble.runMCMC(initPos);
	_mm_free(initPos);
	newEnsemble.getChain(Chain);
	newEnsemble.getLnLike(LnPosterior);
	return 0;
	}