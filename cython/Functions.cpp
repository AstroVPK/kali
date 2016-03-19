#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include "CARMA.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "rdrand.hpp"

//#define TIME_LNLIKE
//#define TIME_MCMC
//#define DEBUG_MAKEMOCKLC
//#define DEBUG_CFFI_MAKEMOCKLC
//#define DEBUG_COMPUTELNLIKE
//#define DEBUG_CFFI_COMPUTELNLIKE
#define DEBUG_FITCARMA 
//#define DEBUG_CFFI_FITCARMA

#if defined(DEBUG_MAKEMOCKLC) || defined(DEBUG_CFFI_MAKEMOCKLC) || defined(DEBUG_COMPUTELNLIKE) || defined(DEBUG_CFFI_COMPUTELNLIKE) || defined(DEBUG_FITCARMA) || defined(DEBUG_CFFI_FITCARMA)
#include <cstdio>
#endif

//#define DEBUG_MASK

using namespace std;

int getRandoms(int numRequested, unsigned int *Randoms) {
	int status = rdrand_get_n_32(numRequested, Randoms);
	return status;
	}

int testSystem(double dt, int p, int q, double *Theta) {
	int retVal = 1;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	SystemMaster.deallocCARMA();
	return goodYN;
	}

int printSystem(double dt, int p, int q, double *Theta) {
	int retVal = 1;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(dt);
		SystemMaster.setCARMA(Theta);

		printf("A\n");
		SystemMaster.printA();
		printf("\n");
		printf("w\n");
		SystemMaster.printw();
		printf("\n");
		printf("vr\n");
		SystemMaster.printvr();
		printf("\n");
		printf("B\n");
		SystemMaster.printB();
		printf("\n");
		printf("C\n");
		SystemMaster.printC();
		printf("\n");

		SystemMaster.solveCARMA();

		printf("expw\n");
		SystemMaster.printexpw();
		printf("\n");
		printf("F\n");
		SystemMaster.printF();
		printf("\n");
		printf("Q\n");
		SystemMaster.printQ();
		printf("\n");
		printf("T\n");
		SystemMaster.printT();
		printf("\n");
		printf("Sigma\n");
		SystemMaster.printSigma();
		printf("\n");

		SystemMaster.resetState();

		printf("X\n");
		SystemMaster.printX();
		printf("\n");
		printf("P\n");
		SystemMaster.printP();
		printf("\n");

		} else {
		retVal = 0;
		}
	SystemMaster.deallocCARMA();
	return retVal;
	}

int makeIntrinsicLC(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *x) {
	int retVal = 1;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(dt);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();
		double* burnRand = static_cast<double*>(_mm_malloc(numBurn*p*sizeof(double),64));
		for (int i = 0; i < numBurn*p; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
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
		SystemMaster.observeSystem(ptr2Data, distSeed, distRand);
		_mm_free(distRand);
		retVal = 0;
		} else {
		retVal = 1;
		}
	SystemMaster.deallocCARMA();
	return retVal;
	}

double getMeanFlux(int p, int q, double *Theta, double fracIntrinsicVar) {
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	double meanFlux = -1.0;
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(1.0);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();
		LnLikeData Data;
		Data.fracIntrinsicVar = fracIntrinsicVar;
		LnLikeData *ptr2Data = &Data;
		meanFlux = SystemMaster.getMeanFlux(ptr2Data);
		}
	SystemMaster.deallocCARMA();
	return meanFlux;
	}


int makeObservedLC(double dt, int p, int q, double *Theta, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {
	int retVal = 1;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(dt);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();
		double* burnRand = static_cast<double*>(_mm_malloc(numBurn*p*sizeof(double),64));
		for (int i = 0; i < numBurn*p; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
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
		SystemMaster.observeSystem(ptr2Data, distSeed, distRand);
		_mm_free(distRand);

		double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
		for (int i = 0; i < numCadences; i++) {
			noiseRand[i] = 0.0;
			}
		SystemMaster.addNoise(ptr2Data, noiseSeed, noiseRand);
		_mm_free(noiseRand);

		retVal = 0;
		} else {
		retVal = 1;
		}
	SystemMaster.deallocCARMA();
	return retVal;
	}

double computeLnLikelihood(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {
	double LnLikelihood = 0.0;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(dt);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = tolIR;
		Data.t = t;
		Data.y = y;
		Data.yerr = yerr;
		Data.mask = mask;
		LnLikeData *ptr2Data = &Data;
		LnLikelihood = SystemMaster.computeLnLikelihood(ptr2Data);
		SystemMaster.deallocCARMA();
		}
	return LnLikelihood;
	}

double computeLnPosterior(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale) {
	double LnPosterior = 0.0;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(p, q);
	int goodYN = SystemMaster.checkCARMAParams(Theta);
	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
		SystemMaster.set_dt(dt);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();
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
		LnPosterior = SystemMaster.computeLnLikelihood(ptr2Data) + SystemMaster.computeLnPrior(ptr2Data);
		SystemMaster.deallocCARMA();
		}
	return LnPosterior;
	}

int fitCARMA(double dt, int p, int q, bool IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnLike) {
	omp_set_num_threads(nthreads);
	int ndims = p + q + 1;
	int threadNum = omp_get_thread_num();

	//printf("minTimescale: %+4.3e\n",minTimescale);
	//printf("maxTimescale: %+4.3e\n",maxTimescale);

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
	Args.numThreads = nthreads;
	Args.Data = ptr2Data;
	Args.Systems = nullptr;
	void* p2Args = nullptr;
	CARMA Systems[nthreads];
	for (int tNum = 0; tNum < nthreads; ++tNum) {
		Systems[tNum].allocCARMA(p,q);
		Systems[tNum].set_dt(dt);
		}
	Args.Systems = Systems;
	p2Args = &Args;
	double LnLikeVal = 0.0;
	double *initPos = nullptr, *offsetArr = nullptr, *deltaXTemp = nullptr;
	vector<vector<double>> x (nthreads, vector<double>(ndims));
	VSLStreamStatePtr *xStream = (VSLStreamStatePtr*)_mm_malloc(nthreads*sizeof(VSLStreamStatePtr),64);
	deltaXTemp = static_cast<double*>(_mm_malloc(ndims*nthreads*sizeof(double),64));
	initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
	#pragma omp parallel for default(none) shared(nthreads, xStream, xSeed, nwalkers)
	for (int i = 0; i < nthreads; ++i) {
		vslNewStream(&xStream[i], VSL_BRNG_SFMT19937, xSeed);
		vslSkipAheadStream(xStream[i], i*(nwalkers/nthreads));
		}
	nlopt::opt *optArray[nthreads];
	for (int i = 0; i < nthreads; ++i) {
		optArray[i] = new nlopt::opt(nlopt::LN_NELDERMEAD, ndims);
		optArray[i]->set_max_objective(calcLnPosterior, p2Args);
		optArray[i]->set_maxeval(maxEvals);
		optArray[i]->set_xtol_rel(xTol);
		}
	double *max_LnLike = static_cast<double*>(_mm_malloc(nthreads*sizeof(double),64));
	#pragma omp parallel for default(none) shared(nwalkers, ndims, deltaXTemp, xStream, scatterFactor, optArray, initPos, xStart, Systems, x, max_LnLike)
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
			if (Systems[threadNum].checkCARMAParams(&deltaXTemp[threadNum*ndims]) == 1) {
				goodPoint = true;
				} else {
				goodPoint = false;
				}
			} while (goodPoint == false);
		x[threadNum].clear();
		for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
			x[threadNum].push_back(deltaXTemp[threadNum*ndims + dimCtr]);
			}
		nlopt::result yesno = optArray[threadNum]->optimize(x[threadNum], max_LnLike[threadNum]);
		for (int dimNum = 0; dimNum < ndims; ++dimNum) {
			initPos[walkerNum*ndims + dimNum] = x[threadNum][dimNum];
			}
		}
	for (int i = 0; i < nthreads; ++i) {
		vslDeleteStream(&xStream[i]);
		delete optArray[i];
		}
	_mm_free(xStream);
	_mm_free(deltaXTemp);
	_mm_free(max_LnLike);
	EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnPosterior, p2Args, zSSeed, walkerSeed, moveSeed);
	newEnsemble.runMCMC(initPos);
	_mm_free(initPos);
	for (int tNum = 0; tNum < nthreads; ++tNum) {
		Systems[tNum].deallocCARMA();
		}
	newEnsemble.getChain(Chain);
	newEnsemble.getLnLike(LnLike);
	return 0;
	}

extern "C" {

	extern int _getRandoms(int numRequested, unsigned int *Randoms) {
		return getRandoms(numRequested, Randoms);
	}

	extern unsigned int* _malloc_uint(int length) {
		unsigned int *mem = static_cast<unsigned int*>(_mm_malloc(length*sizeof(unsigned int),64));
		return mem;
		}

	extern void _free_uint(unsigned int *mem) {
		if (mem) {
			_mm_free(mem);
			}
		mem = nullptr;
		}

	extern int* _malloc_int(int length) {
		int *mem = static_cast<int*>(_mm_malloc(length*sizeof(int),64));
		return mem;
		}

	extern void _free_int(int *mem) {
		if (mem) {
			_mm_free(mem);
			}
		mem = nullptr;
		}

	extern double* _malloc_double(int length) {
		double *mem = static_cast<double*>(_mm_malloc(length*sizeof(double),64));
		return mem;
		}

	extern void _free_double(double *mem) {
		if (mem) {
			_mm_free(mem);
			}
		mem = nullptr;
		}

	extern int _testSystem(double dt, int p, int q, double *Theta) {
		return testSystem(dt, p, q, Theta);
		}

	extern int _printSystem(double dt, int p, int q, double *Theta) {
		return printSystem(dt, p, q, Theta);
		}

	extern int _makeIntrinsicLC(double dt, int p, int q, double *Theta, int IR, double tolIR, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *x) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return makeIntrinsicLC(dt, p, q, Theta, boolIR, tolIR, numBurn, numCadences, startCadence, burnSeed, distSeed, cadence, mask, t, x);
		}

	extern double _getMeanFlux(int p, int q, double *Theta, double fracIntrinsicVar) {
		return getMeanFlux(p, q, Theta, fracIntrinsicVar);
		}

	extern int _makeObservedLC(double dt, int p, int q, double *Theta, int IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return makeObservedLC(dt, p, q, Theta, boolIR, tolIR, fracIntrinsicVar,fracSignalToNoise, numBurn, numCadences, startCadence, burnSeed, distSeed, noiseSeed, cadence, mask, t, y, yerr);
		}

	extern double _computeLnLikelihood(double dt, int p, int q, double *Theta, int IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return computeLnLikelihood(dt, p, q, Theta, boolIR, tolIR, numCadences, cadence, mask, t, y, yerr);
		}

	extern double _computeLnPosterior(double dt, int p, int q, double *Theta, int IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return computeLnPosterior(dt, p, q, Theta, boolIR, tolIR, numCadences, cadence, mask, t, y, yerr, maxSigma, minTimescale, maxTimescale);
		}

	extern int _fitCARMA(double dt, int p, int q, int IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnLike) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return fitCARMA(dt, p, q, IR, tolIR, scatterFactor, numCadences, cadence, mask, t, y, yerr, maxSigma, minTimescale, maxTimescale, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, xStart, Chain, LnLike);
		}
	}
