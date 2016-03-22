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

using namespace std;

	Task::Task(int pGiven, int qGiven, int numThreadsGiven, int numBurn) {
		p = pGiven;
		q = qGiven;
		numThreads = numThreadsGiven;
		numBurn = numBurn;
		Systems = new CARMA[numThreads];
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].allocCARMA(p,q);
			}
		}

	Task::~Task() {
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].deallocCARMA();
			}
		delete[] Systems;
		}

	int Task::getNumBurn() {return numBurn;}
	void Task::setNumBurn(int numBurn) {numBurn = numBurn;}

	int Task::checkParams(double *Theta, int threadNum) {
		return Systems[threadNum].checkCARMAParams(Theta);
		}

	void Task::setDT(double dt, int threadNum) {
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].set_dt(dt);
			}
		}

	int Task::printSystem(double dt, double *Theta, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);

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

			Systems[threadNum].solveCARMA();

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

			Systems[threadNum].resetState();

			printf("X\n");
			Systems[threadNum].printX();
			printf("\n");
			printf("P\n");
			Systems[threadNum].printP();
			printf("\n");
			} else {
			retVal = -1;
			}
		return retVal;
		}

	int Task::getSigma(double dt, double *Theta, double *Sigma, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			const double *ptrToSigma = Systems[threadNum].getSigma();
			for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
				for (int colCtr = 0; colCtr < p; ++colCtr) {
					Sigma[rowCtr + p*colCtr] = ptrToSigma[rowCtr + p*colCtr];
					}
				}
			} else {
			retVal = -1;
			}
		return retVal;
		}

	int Task::makeIntrinsicLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

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
			Data.y = x;
			Data.mask = mask;
			LnLikeData *ptr2Data = &Data;
			Systems[threadNum].observeSystem(ptr2Data, distSeed, distRand);
			_mm_free(distRand);
			} else {
			retVal = -1;
			}
		return retVal;
		}

	double Task::getMeanFlux(double dt, double *Theta, double fracIntrinsicVar, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta);
		double meanFlux = -1.0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			LnLikeData Data;
			Data.fracIntrinsicVar = fracIntrinsicVar;
			LnLikeData *ptr2Data = &Data;
			meanFlux = Systems[threadNum].getMeanFlux(ptr2Data);
			}
		return meanFlux;
		}

	int Task::makeObservedLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

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
			Data.y = y;
			Data.yerr = yerr;
			Data.mask = mask;
			Data.fracIntrinsicVar = fracIntrinsicVar;
			Data.fracSignalToNoise = fracSignalToNoise;
			LnLikeData *ptr2Data = &Data;
			Systems[threadNum].observeSystem(ptr2Data, distSeed, distRand);
			_mm_free(distRand);
	
			double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
			for (int i = 0; i < numCadences; i++) {
				noiseRand[i] = 0.0;
				}
			Systems[threadNum].addNoise(ptr2Data, noiseSeed, noiseRand);
			_mm_free(noiseRand);
			} else {
			retVal = -1;
			}
		return retVal;
		}

	double Task::computeLnPrior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnPrior = 0.0;
		int goodYN = Systems[threadNum].checkCARMAParams(Theta);
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			LnLikeData Data;
			Data.numCadences = numCadences;
			Data.IR = IR;
			Data.tolIR = tolIR;
			Data.t = t;
			Data.y = y;
			Data.yerr = yerr;
			Data.mask = mask;
			Data.maxSigma = maxSigma;
			Data.minTimescale = minTimescale;
			Data.maxTimescale = maxTimescale;
			LnLikeData *ptr2Data = &Data;
			LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);
			}
		return LnPrior;
		}

	double Task::computeLnLikelihood(double dt, double *Theta, int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnLikelihood = 0.0;
		int goodYN = Systems[threadNum].checkCARMAParams(Theta);
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			LnLikeData Data;
			Data.numCadences = numCadences;
			Data.IR = IR;
			Data.tolIR = tolIR;
			Data.t = t;
			Data.y = y;
			Data.yerr = yerr;
			Data.mask = mask;
			LnLikeData *ptr2Data = &Data;
			LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
			}
		return LnLikelihood;
		}

	double Task::computeLnPosterior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum) {
		double LnPosterior = 0.0;
		int goodYN = Systems[threadNum].checkCARMAParams(Theta);
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			LnLikeData Data;
			Data.numCadences = numCadences;
			Data.IR = IR;
			Data.tolIR = tolIR;
			Data.t = t;
			Data.y = y;
			Data.yerr = yerr;
			Data.mask = mask;
			Data.maxSigma = maxSigma;
			Data.minTimescale = minTimescale;
			Data.maxTimescale = maxTimescale;
			LnLikeData *ptr2Data = &Data;
			LnPosterior = Systems[threadNum].computeLnLikelihood(ptr2Data) + Systems[threadNum].computeLnPrior(ptr2Data);
			}
		return LnPosterior;
		}

	int Task::fitCARMA(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double scatterFactor, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior) {
		omp_set_num_threads(numThreads);
		int ndims = p + q + 1;
		int threadNum = omp_get_thread_num();
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
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
		double *max_LnLike = static_cast<double*>(_mm_malloc(numThreads*sizeof(double),64));
		CARMA *ptrToSystems = Systems;
		#pragma omp parallel for default(none) shared(nwalkers, ndims, deltaXTemp, xStream, scatterFactor, optArray, initPos, xStart, ptrToSystems, xVec, max_LnLike)
		for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
			int threadNum = omp_get_thread_num();
			bool goodPoint = false;
			max_LnLike[threadNum] = 0.0;
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				deltaXTemp[threadNum*ndims + dimCtr] = 0.0;
				}
			do {
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream[threadNum], ndims, &deltaXTemp[threadNum*ndims], 0.0, scatterFactor);
				for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
					deltaXTemp[threadNum*ndims + dimCtr] += 1.0;
					deltaXTemp[threadNum*ndims + dimCtr] *= xStart[dimCtr];
					}
				if (ptrToSystems[threadNum].checkCARMAParams(&deltaXTemp[threadNum*ndims]) == 1) {
					goodPoint = true;
					} else {
					goodPoint = false;
					}
				} while (goodPoint == false);
			xVec[threadNum].clear();
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				xVec[threadNum].push_back(deltaXTemp[threadNum*ndims + dimCtr]);
				}
			nlopt::result yesno = optArray[threadNum]->optimize(xVec[threadNum], max_LnLike[threadNum]);
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
		_mm_free(max_LnLike);
		EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, numThreads, 2.0, calcLnPosterior, p2Args, zSSeed, walkerSeed, moveSeed);
		newEnsemble.runMCMC(initPos);
		_mm_free(initPos);
		newEnsemble.getChain(Chain);
		newEnsemble.getLnLike(LnPosterior);
		return 0;
		}