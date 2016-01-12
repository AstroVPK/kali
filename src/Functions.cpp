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
		for (int i = 0; i < numBurn; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
		_mm_free(burnRand);
		double* distRand = static_cast<double*>(_mm_malloc(numCadences*p*sizeof(double),64));
		for (int i = 0; i < numCadences; i++) {
			distRand[i] = 0.0;
			}
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = IR;
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
		for (int i = 0; i < numBurn; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
		_mm_free(burnRand);
		double* distRand = static_cast<double*>(_mm_malloc(numCadences*p*sizeof(double),64));
		for (int i = 0; i < numCadences; i++) {
			distRand[i] = 0.0;
			}
		LnLikeData Data;
		Data.numCadences = numCadences;
		Data.IR = IR;
		Data.tolIR = IR;
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

double computeLnlike(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {
	double LnLike = 0.0;
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
		LnLike = SystemMaster.computeLnLike(ptr2Data);
		SystemMaster.deallocCARMA();
		}
	return LnLike;
	}

int fitCARMA(double dt, int p, int q, bool IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike) {
	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();

	LnLikeData Data;
	Data.numCadences = numCadences;
	Data.IR = IR;
	Data.tolIR = IR;
	Data.t = t;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;
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
	double *initPos = nullptr, *offsetArr = nullptr, *xTemp = nullptr;
	vector<double> x;
	VSLStreamStatePtr xStream, initStream;
	int ndims = p + q + 1;
	xTemp = static_cast<double*>(_mm_malloc(ndims*sizeof(double),64));
	vslNewStream(&xStream, VSL_BRNG_SFMT19937, xSeed);
	bool goodPoint = false;
	do {
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream, ndims, xTemp, 0.0, 1.0e-1);
		if (Systems[threadNum].checkCARMAParams(xTemp) == 1) {
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(xTemp);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();
			LnLikeVal = Systems[threadNum].computeLnLike(ptr2Data);
			goodPoint = true;
			} else {
			LnLikeVal = -infiniteVal;
			goodPoint = false;
			}
		} while (goodPoint == false);
	vslDeleteStream(&xStream);
	x.clear();
	for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
		x.push_back(xTemp[dimCtr]);
		}
	_mm_free(xTemp);
	nlopt::opt opt(nlopt::LN_NELDERMEAD, ndims);
	opt.set_max_objective(calcLnLike, p2Args);
	opt.set_maxeval(maxEvals);
	opt.set_xtol_rel(xTol);
	double max_LnLike = 0.0;
	nlopt::result yesno = opt.optimize(x, max_LnLike);
	initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
	offsetArr = static_cast<double*>(_mm_malloc(nwalkers*sizeof(double),64));
	for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
		offsetArr[walkerNum] = 0.0;
		for (int dimNum = 0; dimNum < ndims; ++dimNum) {
			initPos[walkerNum*ndims + dimNum] = 0.0;
			}
		}
	vslNewStream(&initStream, VSL_BRNG_SFMT19937, initSeed);
	for (int dimNum = 0; dimNum < ndims; ++dimNum) {
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, initStream, nwalkers, offsetArr, 0.0, x[dimNum]*scatterFactor);
		for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
			initPos[walkerNum*ndims + dimNum] = x[dimNum] + offsetArr[walkerNum];
			}
		}
	vslDeleteStream(&initStream);
	_mm_free(offsetArr);
	EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, zSSeed, walkerSeed, moveSeed);
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

	extern int _makeObservedLC(double dt, int p, int q, double *Theta, int IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return makeObservedLC(dt, p, q, Theta, boolIR, tolIR, fracIntrinsicVar,fracSignalToNoise, numBurn, numCadences, startCadence, burnSeed, distSeed, noiseSeed, cadence, mask, t, y, yerr);
		}

	extern double _computeLnlike(double dt, int p, int q, double *Theta, int IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return computeLnlike(dt, p, q, Theta, boolIR, tolIR, numCadences, cadence, mask, t, y, yerr);
		}

	extern int _fitCARMA(double dt, int p, int q, int IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike) {
		bool boolIR;
		if (IR == 0) {
			boolIR = false;
			} else {
			boolIR = true;
			}
		return fitCARMA(dt, p, q, IR, tolIR, scatterFactor, numCadences, cadence, mask, t, y, yerr, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, initSeed, Chain, LnLike);
		}
	}
