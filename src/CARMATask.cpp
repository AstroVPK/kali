#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include <stdio.h>
#include "CARMA.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "CARMATask.hpp"

//#define DEBUG_COMPUTELNLIKELIHOOD
//#define DEBUG_FIT_CARMAMODEL

using namespace std;

	kali::CARMATask::CARMATask(int pGiven, int qGiven, int numThreadsGiven, int numBurnGiven) {
		p = pGiven;
		q = qGiven;
		numThreads = numThreadsGiven;
		numBurn = numBurnGiven;
		Systems = new kali::CARMA[numThreads];
		setSystemsVec = static_cast<bool*>(_mm_malloc(numThreads*sizeof(double),64));
		ThetaVec = static_cast<double*>(_mm_malloc(numThreads*(p + q + 1)*sizeof(double),64));
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			Systems[threadNum].allocCARMA(p,q);
			setSystemsVec[threadNum] = false;
			#pragma omp simd
			for (int i = 0; i < (p + q + 1); ++i) {
				ThetaVec[i + threadNum*(p + q  +1)] = 0.0;
				}
			}
		}

	kali::CARMATask::~CARMATask() {
		if (ThetaVec) {
			_mm_free(ThetaVec);
			ThetaVec = nullptr;
			}
		if (setSystemsVec) {
			_mm_free(setSystemsVec);
			setSystemsVec = nullptr;
			}
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].deallocCARMA();
			}
		delete[] Systems;
		}

	int kali::CARMATask::reset_CARMATask(int pGiven, int qGiven, int numBurn) {
		int retVal = -1;
		p = pGiven;
		q = qGiven;
		numBurn = numBurn;
		if (ThetaVec) {
			_mm_free(ThetaVec);
			ThetaVec = nullptr;
			}
		ThetaVec = static_cast<double*>(_mm_malloc(numThreads*(p + q + 1)*sizeof(double),64));
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			Systems[threadNum].deallocCARMA();
			Systems[threadNum].allocCARMA(p,q);
			setSystemsVec[threadNum] = false;
			#pragma omp simd
			for (int i = 0; i < (p + q + 1); ++i) {
				ThetaVec[i + threadNum*(p + q  +1)] = 0.0;
				}
			}
		retVal = 0;
		return retVal;
		}

	int kali::CARMATask::get_numBurn() {return numBurn;}
	void kali::CARMATask::set_numBurn(int numBurn) {numBurn = numBurn;}

	int kali::CARMATask::check_Theta(double *Theta, int threadNum) {
		return Systems[threadNum].checkCARMAParams(Theta);
		}

	double kali::CARMATask::get_dt(int threadNum) {return Systems[threadNum].get_dt();}

	void kali::CARMATask::get_Theta(double *Theta, int threadNum) {
		for (int i = 0; i < (p + q + 1); ++i) {
			Theta[i] = ThetaVec[i + threadNum*(p + q + 1)];
			}
		}

	int kali::CARMATask::set_System(double dt, double *Theta, int threadNum) {
		bool alreadySet = true;
		int retVal = -1;
		if (setSystemsVec[threadNum] == true) {
			if (Systems[threadNum].get_dt() != dt) {
				alreadySet = false;
				}
			for (int i = 0; i < (p + q + 1); ++i) {
				if (ThetaVec[i + threadNum*(p + q + 1)] != Theta[i]) {
					alreadySet = false;
					}
				}
			} else {
			alreadySet = false;
			}
		if (alreadySet == false) {
			int goodYN = Systems[threadNum].checkCARMAParams(Theta);
			if (goodYN == 1) {
				for (int i = 0; i < (p + q + 1); ++i) {
					ThetaVec[i + threadNum*(p + q + 1)] = Theta[i];
					}
				double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
				Systems[threadNum].set_dt(dt);
				Systems[threadNum].setCARMA(Theta);
				Systems[threadNum].solveCARMA();
				Systems[threadNum].resetState();
				retVal = 0;
				setSystemsVec[threadNum] = true;
				}
			} else {
			retVal = 0;
			}
		return retVal;
		}

	int kali::CARMATask::reset_System(int threadNum) {
		int retVal = -1;
		Systems[threadNum].resetState();
		retVal = 0;
		return retVal;
		}

	void kali::CARMATask::get_setSystemsVec(int *setSystems) {
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
			}
		}

	int kali::CARMATask::print_System(int threadNum) {
		int retVal = 0;
		printf("dt: %+8.7e\n",Systems[threadNum].get_dt());
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


	int kali::CARMATask::get_A(complex<double> *A, int threadNum) {
		int retVal = 0;
		const complex<double> *ptrToA = Systems[threadNum].getA();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			for (int colCtr = 0; colCtr < p; ++colCtr) {
				A[rowCtr + p*colCtr] = ptrToA[rowCtr + p*colCtr];
				}
			}
		return retVal;
		}

	int kali::CARMATask::get_B(complex<double> *B, int threadNum) {
		int retVal = 0;
		const complex<double> *ptrToB = Systems[threadNum].getB();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			B[rowCtr] = ptrToB[rowCtr];
			}
		return retVal;
		}

	int kali::CARMATask::get_Sigma(double *Sigma, int threadNum) {
		int retVal = 0;
		const double *ptrToSigma = Systems[threadNum].getSigma();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			for (int colCtr = 0; colCtr < p; ++colCtr) {
				Sigma[rowCtr + p*colCtr] = ptrToSigma[rowCtr + p*colCtr];
				}
			}
		return retVal;
		}

	int kali::CARMATask::get_X(double *newX, int threadNum) {
		int retVal = 0;
		Systems[threadNum].getX(newX);
		return retVal;
		}

	int kali::CARMATask::set_X(double *newX, int threadNum) {
		int retVal = 0;
		Systems[threadNum].setX(newX);
		return retVal;
		}

	int kali::CARMATask::get_P(double *newP, int threadNum) {
		int retVal = 0;
		Systems[threadNum].getP(newP);
		return retVal;
		}

	int kali::CARMATask::set_P(double *newP, int threadNum) {
		int retVal = 0;
		Systems[threadNum].setP(newP);
		return retVal;
		}

	int kali::CARMATask::make_IntrinsicLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int burnSeed, unsigned int distSeed, int threadNum) {
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
		kali::LnLikeData *ptr2Data = &Data;
		Systems[threadNum].simulateSystem(ptr2Data, distSeed, distRand);
		_mm_free(distRand);
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		return retVal;
		}

	int kali::CARMATask::extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum) {
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
		}

	double kali::CARMATask::get_meanFlux(double fracIntrinsicVar, int threadNum) {
		double meanFlux = -1.0;
		kali::LnLikeData Data;
		Data.fracIntrinsicVar = fracIntrinsicVar;
		kali::LnLikeData *ptr2Data = &Data;
		meanFlux = Systems[threadNum].getMeanFlux(ptr2Data);
		return meanFlux;
		}

	int kali::CARMATask::make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum) {
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

	int kali::CARMATask::add_ObservationNoise(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
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

	int kali::CARMATask::extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
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
		}

	double kali::CARMATask::compute_LnPrior(int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnPrior = 0.0;
		kali::LnLikeData Data;
		Data.numCadences = numCadences;
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
		}

	double kali::CARMATask::update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
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
		}

	double kali::CARMATask::compute_LnLikelihood(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
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
		Data.lcX = lcX;
		Data.lcP = lcP;
		kali::LnLikeData *ptr2Data = &Data;
		double old_dt = Systems[threadNum].get_dt();
		Systems[threadNum].set_dt(t[1] - t[0]);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		cadenceNum = Data.cadenceNum;
		Systems[threadNum].set_dt(old_dt);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		return LnLikelihood;
		}

	double kali::CARMATask::update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
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
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		Systems[threadNum].setX(lcX);
		Systems[threadNum].setP(lcP);
		LnLikelihood = Systems[threadNum].updateLnLikelihood(ptr2Data);
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		cadenceNum = Data.cadenceNum;
		Systems[threadNum].set_dt(old_dt);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		return LnLikelihood;
		}

	double kali::CARMATask::compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
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
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
		LnPosterior = LnPrior + LnLikelihood;
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		cadenceNum = Data.cadenceNum;
		Systems[threadNum].set_dt(old_dt);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		return LnPosterior;
		}

	double kali::CARMATask::update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum) {
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
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		Systems[threadNum].setX(lcX);
		Systems[threadNum].setP(lcP);
		LnLikelihood = Systems[threadNum].updateLnLikelihood(ptr2Data);
		LnPosterior = LnPrior + LnLikelihood;
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		cadenceNum = Data.cadenceNum;
		Systems[threadNum].set_dt(old_dt);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		return LnPosterior;
		}


	void kali::CARMATask::compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum) {
		Systems[threadNum].computeACVF(numLags, Lags, ACVF);
		}

	int kali::CARMATask::fit_CARMAModel(double dt, int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior) {
		omp_set_num_threads(numThreads);
		int ndims = p + q + 1;
		int threadNum = omp_get_thread_num();
		kali::LnLikeData Data;
		Data.numCadences = numCadences;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		Data.maxSigma = maxSigma;
		Data.minTimescale = minTimescale;
		Data.maxTimescale = maxTimescale;
		#ifdef DEBUG_FIT_CARMAMODEL
			#pragma omp critical
			{
			printf("fit_CARMAModel - threadNum: %d; maxSigma: %e\n", threadNum, maxSigma);
			printf("fit_CARMAModel - threadNum: %d; minTimescale: %e\n", threadNum, minTimescale);
			printf("fit_CARMAModel - threadNum: %d; maxTimescale: %e\n", threadNum, maxTimescale);
			}
		#endif
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
		kali::CARMA *ptrToSystems = Systems;
		#pragma omp parallel for default(none) shared(dt, nwalkers, ndims, optArray, initPos, xStart, t, ptrToSystems, xVec, max_LnPosterior, p2Args)
		for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
			int threadNum = omp_get_thread_num();
			max_LnPosterior[threadNum] = 0.0;
			xVec[threadNum].clear();
			set_System(t[1] - t[0], &xStart[walkerNum*ndims], threadNum);
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				xVec[threadNum].push_back(xStart[walkerNum*ndims + dimCtr]);
				}
			#ifdef DEBUG_FIT_CARMAMODEL
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
			#ifdef DEBUG_FIT_CARMAMODEL
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

	int kali::CARMATask::smooth_RTS(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, int threadNum) {
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
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		successYN = Systems[threadNum].RTSSmoother(ptr2Data, XSmooth, PSmooth);
		Systems[threadNum].getX(lcX);
		Systems[threadNum].getP(lcP);
		cadenceNum = Data.cadenceNum;
		Systems[threadNum].set_dt(old_dt);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		return successYN;
		}
