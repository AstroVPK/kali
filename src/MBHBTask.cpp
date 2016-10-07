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
#include "MBHB.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "MBHBTask.hpp"

//#define DEBUG_COMPUTELNLIKELIHOOD
//#define DEBUG_FIT_MBHBMODEL
//#define DEBUG_SETSYSTEM

using namespace std;

int lenTheta = 8;

kali::MBHBTask::MBHBTask(int numThreadsGiven) {
	numThreads = numThreadsGiven;
	Systems = new kali::MBHB[numThreads];
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

kali::MBHBTask::~MBHBTask() {
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

int kali::MBHBTask::check_Theta(double *Theta, int threadNum) {
	return Systems[threadNum].checkMBHBParams(Theta);
	}

void kali::MBHBTask::get_Theta(double *Theta, int threadNum) {
	for (int i = 0; i < lenTheta; ++i) {
		Theta[i] = ThetaVec[i + threadNum*lenTheta];
		}
	}

int kali::MBHBTask::set_System(double *Theta, int threadNum) {
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
		int goodYN = Systems[threadNum].checkMBHBParams(Theta);
		if (goodYN == 1) {
			for (int i = 0; i < lenTheta; ++i) {
				ThetaVec[i + threadNum*lenTheta] = Theta[i];
				}
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].setMBHB(Theta);
			retVal = 0;
			setSystemsVec[threadNum] = true;
			}
		} else {
		retVal = -1;
		}
	return retVal;
	}

void kali::MBHBTask::get_setSystemsVec(int *setSystems) {
	for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
		setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
		}
	}

void kali::MBHBTask::print_System(int threadNum) {
	Systems[threadNum].print();
	}

void kali::MBHBTask::set_Epoch(double epochIn, int threadNum) {
	Systems[threadNum].setEpoch(epochIn);
	}

double kali::MBHBTask::get_Epoch(int threadNum) {
	return Systems[threadNum].getEpoch();
	}

double kali::MBHBTask::get_Period(int threadNum) {
	return Systems[threadNum].getPeriod();
	}

double kali::MBHBTask::get_A1(int threadNum) {
	return Systems[threadNum].getA1();
	}

double kali::MBHBTask::get_A2(int threadNum) {
	return Systems[threadNum].getA2();
	}

double kali::MBHBTask::get_M1(int threadNum) {
	return Systems[threadNum].getM1();
	}

double kali::MBHBTask::get_M2(int threadNum) {
	return Systems[threadNum].getM2();
	}

double kali::MBHBTask::get_M12(int threadNum) {
	return Systems[threadNum].getM12();
	}

double kali::MBHBTask::get_M2OverM1(int threadNum) {
	return Systems[threadNum].getM2OverM1();
	}

double kali::MBHBTask::get_RPeribothron1(int threadNum) {
	return Systems[threadNum].getRPeribothron1();
	}

double kali::MBHBTask::get_RPeribothron2(int threadNum) {
	return Systems[threadNum].getRPeribothron2();
	}

double kali::MBHBTask::get_RApobothron1(int threadNum) {
	return Systems[threadNum].getRApobothron1();
	}

double kali::MBHBTask::get_RApobothron2(int threadNum) {
	return Systems[threadNum].getRApobothron2();
	}

double kali::MBHBTask::get_RPeribothronTot(int threadNum) {
	return Systems[threadNum].getRPeribothronTot();
	}

double kali::MBHBTask::get_RApobothronTot(int threadNum) {
	return Systems[threadNum].getRApobothronTot();
	}

double kali::MBHBTask::get_RS1(int threadNum) {
	return Systems[threadNum].getRS1();
	}

double kali::MBHBTask::get_RS2(int threadNum) {
	return Systems[threadNum].getRS2();
	}

double kali::MBHBTask::get_Eccentricity(int threadNum) {
	return Systems[threadNum].getEccentricity();
	}

double kali::MBHBTask::get_Omega1(int threadNum) {
	return Systems[threadNum].getOmega1();
	}

double kali::MBHBTask::get_Omega2(int threadNum) {
	return Systems[threadNum].getOmega2();
	}

double kali::MBHBTask::get_Inclination(int threadNum) {
	return Systems[threadNum].getInclination();
	}

double kali::MBHBTask::get_Tau(int threadNum) {
	return Systems[threadNum].getTau();
	}

double kali::MBHBTask::get_MeanAnomoly(int threadNum) {
	return Systems[threadNum].getMeanAnomoly();
	}

double kali::MBHBTask::get_EccentricAnomoly(int threadNum) {
	return Systems[threadNum].getEccentricAnomoly();
	}

double kali::MBHBTask::get_TrueAnomoly(int threadNum) {
	return Systems[threadNum].getTrueAnomoly();
	}

double kali::MBHBTask::get_R1(int threadNum) {
	return Systems[threadNum].getR1();
	}

double kali::MBHBTask::get_R2(int threadNum) {
	return Systems[threadNum].getR2();
	}

double kali::MBHBTask::get_Theta1(int threadNum) {
	return Systems[threadNum].getTheta1();
	}

double kali::MBHBTask::get_Theta2(int threadNum) {
	return Systems[threadNum].getTheta2();
	}

double kali::MBHBTask::get_Beta1(int threadNum) {
	return Systems[threadNum].getBeta1();
	}

double kali::MBHBTask::get_Beta2(int threadNum) {
	return Systems[threadNum].getBeta2();
	}

double kali::MBHBTask::get_RadialBeta1(int threadNum) {
	return Systems[threadNum].getRadialBeta1();
	}

double kali::MBHBTask::get_RadialBeta2(int threadNum) {
	return Systems[threadNum].getRadialBeta2();
	}

double kali::MBHBTask::get_DopplerFactor1(int threadNum) {
	return Systems[threadNum].getDopplerFactor1();
	}

double kali::MBHBTask::get_DopplerFactor2(int threadNum) {
	return Systems[threadNum].getDopplerFactor2();
	}

double kali::MBHBTask::get_BeamingFactor1(int threadNum) {
	return Systems[threadNum].getBeamingFactor1();
	}

double kali::MBHBTask::get_BeamingFactor2(int threadNum) {
	return Systems[threadNum].getBeamingFactor2();
	}

double kali::MBHBTask::get_aH(double sigmaStars, int threadNum) {
	return Systems[threadNum].aH(sigmaStars);
	}

double kali::MBHBTask::get_aGW(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].aGW(sigmaStars, rhoStars, H);
	}

double kali::MBHBTask::get_durationInHardState(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].durationInHardState(sigmaStars, rhoStars, H);
	}

double kali::MBHBTask::get_ejectedMass(double sigmaStars, double rhoStars, double H, int threadNum) {
	return Systems[threadNum].ejectedMass(sigmaStars, rhoStars, H);
	}

int kali::MBHBTask::make_IntrinsicLC(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	kali::LnLikeData *ptr2Data = &Data;
	Systems[threadNum].simulateSystem(ptr2Data);
	return 0;
	}

int kali::MBHBTask::add_ObservationNoise(int numCadences, double dt, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.fracNoiseToSignal = fracNoiseToSignal;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	kali::LnLikeData *ptr2Data = &Data;
	double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	#pragma omp simd
	for (int i = 0; i < numCadences; i++) {
		noiseRand[i] = 0.0;
		}
	Systems[threadNum].observeNoise(ptr2Data, noiseSeed, noiseRand);
	_mm_free(noiseRand);
	return 0;
	}

double kali::MBHBTask::compute_LnPrior(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	double LnPrior = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lowestFlux = lowestFlux;
	Data.highestFlux = highestFlux;
	kali::LnLikeData *ptr2Data = &Data;

	LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);

	return LnPrior;
	}

double kali::MBHBTask::compute_LnLikelihood(int numCadences, double dt, int cadenceNum, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
	double LnLikelihood = 0.0;
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.cadenceNum = cadenceNum;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	kali::LnLikeData *ptr2Data = &Data;

	Systems[threadNum].setEpoch(t[0]);
	LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
	cadenceNum = Data.cadenceNum;

	return LnLikelihood;
	}

int kali::MBHBTask::fit_MBHBModel(int numCadences, double dt, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior) {
	#ifdef DEBUG_FIT_MBHBMODEL
		printf("numThreads: %d\n",numThreads);
	#endif
	omp_set_num_threads(numThreads);
	int ndims = lenTheta;
	int threadNum = omp_get_thread_num();
	kali::LnLikeData Data;
	Data.numCadences = numCadences;
	Data.dt = dt;
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
	Data.lowestFlux = lowestFlux;
	Data.highestFlux = highestFlux;
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
	kali::MBHB *ptrToSystems = Systems;
	#pragma omp parallel for default(none) shared(nwalkers, ndims, optArray, initPos, xStart, t, ptrToSystems, xVec, max_LnPosterior, p2Args)
	for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
		int threadNum = omp_get_thread_num();
		max_LnPosterior[threadNum] = 0.0;
		xVec[threadNum].clear();
		set_System(&xStart[walkerNum*ndims], threadNum);
		for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
			xVec[threadNum].push_back(xStart[walkerNum*ndims + dimCtr]);
			}
		#ifdef DEBUG_FIT_MBHBMODEL
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
		#ifdef DEBUG_FIT_MBHBMODEL
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
