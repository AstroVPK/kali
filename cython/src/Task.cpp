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
#include "LC.hpp"
#include "Task.hpp"

//#define DEBUG_FIT_CARMAMODEL

using namespace std;

	Task::Task(int pGiven, int qGiven, int numThreadsGiven, int numBurnGiven) {
		p = pGiven;
		q = qGiven;
		numThreads = numThreadsGiven;
		numBurn = numBurnGiven;
		Systems = new CARMA[numThreads];
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

	Task::~Task() {
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
		delete[] setSystemsVec;
		}

	int Task::reset_Task(int pGiven, int qGiven, int numBurn) {
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

	int Task::get_numBurn() {return numBurn;}
	void Task::set_numBurn(int numBurn) {numBurn = numBurn;}

	int Task::check_Theta(double *Theta, int threadNum) {
		return Systems[threadNum].checkCARMAParams(Theta);
		}

	double Task::get_dt(int threadNum) {return Systems[threadNum].get_dt();}

	void Task::get_ThetaVec(double *Theta, int threadNum) {
		for (int i = 0; i < (p + q + 1); ++i) {
			Theta[i] = ThetaVec[i + threadNum*(p + q + 1)];
			}
		}

	int Task::set_System(double dt, double *Theta, int threadNum) {
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

	void Task::get_setSystemsVec(int *setSystems) {
		for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
			setSystems[threadNum] = static_cast<int>(setSystemsVec[threadNum]);
			}
		}

	int Task::print_System(int threadNum) {
		int retVal = 0;
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


	int Task::get_A(complex<double> *A, int threadNum) {
		int retVal = 0;
		const complex<double> *ptrToA = Systems[threadNum].getA();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			for (int colCtr = 0; colCtr < p; ++colCtr) {
				A[rowCtr + p*colCtr] = ptrToA[rowCtr + p*colCtr];
				}
			}
		return retVal;
		}

	int Task::get_B(complex<double> *B, int threadNum) {
		int retVal = 0;
		const complex<double> *ptrToB = Systems[threadNum].getB();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			B[rowCtr] = ptrToB[rowCtr];
			}
		return retVal;
		}

	int Task::get_Sigma(double *Sigma, int threadNum) {
		int retVal = 0;
		const double *ptrToSigma = Systems[threadNum].getSigma();
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			for (int colCtr = 0; colCtr < p; ++colCtr) {
				Sigma[rowCtr + p*colCtr] = ptrToSigma[rowCtr + p*colCtr];
				}
			}
		return retVal;
		}

	int Task::make_IntrinsicLC(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum) {
		int retVal = 0;
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
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		LnLikeData *ptr2Data = &Data;
		Systems[threadNum].observeSystem(ptr2Data, distSeed, distRand);
		_mm_free(distRand);
		Systems[threadNum].resetState();
		return retVal;
		}

	double Task::get_meanFlux(double fracIntrinsicVar, int threadNum) {
		double meanFlux = -1.0;
		LnLikeData Data;
		Data.fracIntrinsicVar = fracIntrinsicVar;
		LnLikeData *ptr2Data = &Data;
		meanFlux = Systems[threadNum].getMeanFlux(ptr2Data);
		return meanFlux;
		}

	int Task::make_ObservedLC(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum) {
		int retVal = 0;
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
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		Data.fracIntrinsicVar = fracIntrinsicVar;
		Data.fracNoiseToSignal = fracNoiseToSignal;
		LnLikeData *ptr2Data = &Data;
		Systems[threadNum].observeSystem(ptr2Data, distSeed, distRand);
		_mm_free(distRand);
		double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
		for (int i = 0; i < numCadences; i++) {
			noiseRand[i] = 0.0;
			}
		Systems[threadNum].addNoise(ptr2Data, noiseSeed, noiseRand);
		_mm_free(noiseRand);
		Systems[threadNum].resetState();
		return retVal;
		}

	int Task::add_ObservationNoise(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum) {
		int retVal = 0;
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		Data.fracIntrinsicVar = fracIntrinsicVar;
		Data.fracNoiseToSignal = fracNoiseToSignal;
		LnLikeData *ptr2Data = &Data;
		double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
		for (int i = 0; i < numCadences; i++) {
			noiseRand[i] = 0.0;
			}
		Systems[threadNum].addNoise(ptr2Data, noiseSeed, noiseRand);
		_mm_free(noiseRand);
		return retVal;
		}

	double Task::compute_LnPrior(int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnPrior = 0.0;
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		Data.maxSigma = maxSigma;
		Data.minTimescale = minTimescale;
		Data.maxTimescale = maxTimescale;
		LnLikeData *ptr2Data = &Data;
		LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
		Systems[threadNum].resetState();
		return LnPrior;
		}

	double Task::compute_LnLikelihood(int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnLikelihood = 0.0;
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		LnLikeData *ptr2Data = &Data;
		LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
		Systems[threadNum].resetState();
		return LnLikelihood;
		}

	double Task::compute_LnPosterior(int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnPrior = 0.0, LnLikelihood = 0.0, LnPosterior = 0.0;
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		Data.maxSigma = maxSigma;
		Data.minTimescale = minTimescale;
		Data.maxTimescale = maxTimescale;
		LnLikeData *ptr2Data = &Data;
		LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
		LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
		LnPosterior = LnPrior + LnLikelihood;
		Systems[threadNum].resetState();
		return LnPosterior;
		}

	int Task::fit_CARMAModel(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double scatterFactor, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior) {
		omp_set_num_threads(numThreads);
		int ndims = p + q + 1;
		int threadNum = omp_get_thread_num();
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.x = x;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		#ifdef DEBUG_FIT_CARMAMODEL
		#pragma omp critical
		{
		printf("fit_CARMAModel - threadNum: %d; maxSigma: %e\n", threadNum, maxSigma);
		printf("fit_CARMAModel - threadNum: %d; minTimescale: %e\n", threadNum, minTimescale);
		printf("fit_CARMAModel - threadNum: %d; maxTimescale: %e\n", threadNum, maxTimescale);
		}
		#endif
		Data.maxSigma = maxSigma;
		Data.minTimescale = minTimescale;
		Data.maxTimescale = maxTimescale;
		LnLikeData *ptr2Data = &Data;
		LnLikeArgs Args;
		Args.numThreads = numThreads;
		Args.Data = ptr2Data;
		Args.Systems = nullptr;
		void* p2Args = nullptr;
		Args.Systems = Systems;
		p2Args = &Args;
		double LnLikeVal = 0.0;
		double *initPos = nullptr, *offsetArr = nullptr, *deltaXTemp = nullptr;
		vector<vector<double>> xVec (numThreads, vector<double>(ndims));
		VSLStreamStatePtr *xStream = (VSLStreamStatePtr*)_mm_malloc(numThreads*sizeof(VSLStreamStatePtr),64);
		deltaXTemp = static_cast<double*>(_mm_malloc(ndims*numThreads*sizeof(double),64));
		initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
		int nthreads = numThreads;
		#pragma omp parallel for default(none) shared(nthreads, xStream, xSeed, nwalkers)
		for (int i = 0; i < numThreads; ++i) {
			vslNewStream(&xStream[i], VSL_BRNG_SFMT19937, xSeed);
			vslSkipAheadStream(xStream[i], i*(nwalkers/nthreads));
			}
		nlopt::opt *optArray[numThreads];
		for (int i = 0; i < numThreads; ++i) {
			optArray[i] = new nlopt::opt(nlopt::LN_NELDERMEAD, ndims);
			optArray[i]->set_max_objective(calcLnPosterior, p2Args);
			optArray[i]->set_maxeval(maxEvals);
			optArray[i]->set_xtol_rel(xTol);
			}
		double *max_LnPosterior = static_cast<double*>(_mm_malloc(numThreads*sizeof(double),64));
		CARMA *ptrToSystems = Systems;
		#pragma omp parallel for default(none) shared(dt, nwalkers, ndims, deltaXTemp, xStream, scatterFactor, optArray, initPos, xStart, ptrToSystems, xVec, max_LnPosterior)
		for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
			int threadNum = omp_get_thread_num();
			bool goodPoint = false;
			max_LnPosterior[threadNum] = 0.0;
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				deltaXTemp[threadNum*ndims + dimCtr] = 0.0;
				}
			do {
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream[threadNum], ndims, &deltaXTemp[threadNum*ndims], 0.0, scatterFactor);
				for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
					deltaXTemp[threadNum*ndims + dimCtr] += 1.0;
					deltaXTemp[threadNum*ndims + dimCtr] *= xStart[dimCtr];
					}
				if (set_System(dt, &deltaXTemp[threadNum*ndims], threadNum) == 0) {
					goodPoint = true;
					} else {
					goodPoint = false;
					}
				} while (goodPoint == false);
			xVec[threadNum].clear();
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				xVec[threadNum].push_back(deltaXTemp[threadNum*ndims + dimCtr]);
				}
			nlopt::result yesno = optArray[threadNum]->optimize(xVec[threadNum], max_LnPosterior[threadNum]);
			#ifdef DEBUG_FIT_CARMAMODEL
			#pragma omp critical
			{
			fflush(0);
			printf("xVec[%d][%d]: ", walkerNum, threadNum);
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
			vslDeleteStream(&xStream[i]);
			delete optArray[i];
			}
		_mm_free(xStream);
		_mm_free(deltaXTemp);
		_mm_free(max_LnPosterior);
		EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, numThreads, 2.0, calcLnPosterior, p2Args, zSSeed, walkerSeed, moveSeed);
		newEnsemble.runMCMC(initPos);
		_mm_free(initPos);
		newEnsemble.getChain(Chain);
		newEnsemble.getLnLike(LnPosterior);
		return 0;
		}