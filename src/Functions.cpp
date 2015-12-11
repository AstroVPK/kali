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

//#define TIME_LNLIKE
//#define TIME_MCMC
//#define DEBUG_MAKEMOCKLC
//#define DEBUG_CFFI_MAKEMOCKLC
//#define DEBUG_COMPUTELNLIKE
//#define DEBUG_CFFI_COMPUTELNLIKE
//#define DEBUG_FITCARMA 
//#define DEBUG_CFFI_FITCARMA

#if defined(DEBUG_MAKEMOCKLC) || defined(DEBUG_CFFI_MAKEMOCKLC) || defined(DEBUG_COMPUTELNLIKE) || defined(DEBUG_CFFI_COMPUTELNLIKE) || defined(DEBUG_FITCARMA) || defined(DEBUG_CFFI_FITCARMA)
#include <cstdio>
#endif

//#define DEBUG_MASK

using namespace std;

double makeMockLC(double dt, int p, int q, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {

	double LnLike = 0.0;

	#ifdef DEBUG_MAKEMOCKLC
	printf("makeMockLC: Creating SystemMaster...\n");
	#endif

	CARMA SystemMaster = CARMA();

	#ifdef DEBUG_MAKEMOCKLC
	printf("makeMockLC: SystemMaster created. Allocating memory...\n");
	#endif

	SystemMaster.allocCARMA(p, q);

	#ifdef DEBUG_MAKEMOCKLC
	printf("makeMockLC: Memory allocated. Checking CARMA params...\n");
	#endif

	int goodYN = 1;
	goodYN = SystemMaster.checkCARMAParams(Theta);

	#ifdef DEBUG_MAKEMOCKLC
	printf("makeMockLC: Checked CARMA Params.\n");
	#endif

	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: CARMA params good. Setting dt and C-ARMA model...\n");
		#endif

		SystemMaster.set_t(dt);
		SystemMaster.setCARMA(Theta);

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: dt and C-ARMA model set. Solving C-ARMA model...\n");
		#endif

		SystemMaster.solveCARMA();

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: C-ARMA model solved. Computing initial state...\n");
		#endif

		SystemMaster.resetState();

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: Initial state computed. Burning system...\n");
		#endif

		double* burnRand = static_cast<double*>(_mm_malloc(numBurn*sizeof(double),64));
		for (int i = 0; i < numBurn; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
		_mm_free(burnRand);

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: System burnt. Observing LC...\n");
		#endif

		double* distRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
		double* noiseRand = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

		for (int i = 0; i < numCadences; i++) {
			distRand[i] = 0.0;
			noiseRand[i] = 0.0;
			y[i] = 0.0;
			if (mask[i] == 1.0) {
				yerr[i] = noiseSigma;
				} else {
				yerr[i] = sqrtMaxDouble;
				} 
			}

		SystemMaster.observeSystem(numCadences, distSeed, noiseSeed, distRand, noiseRand, noiseSigma, y, mask);

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: LC Observed. Removing mean...\n");
		#endif

		_mm_free(distRand);
		_mm_free(noiseRand);

		double ySum = 0.0, yCounter = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			ySum += mask[i]*y[i];
			yCounter += mask[i];
			}
		double yMean = ySum/yCounter;
		for (int i = 0; i < numCadences; ++i) {
			y[i] -= mask[i]*yMean;
			}

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: Mean removed. Computing LnLike...\n");
		#endif

		LnLike = SystemMaster.computeLnLike(numCadences, y, yerr, mask);

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: Mean removed. De-allocating system...\n");
		#endif

		SystemMaster.deallocCARMA();

		} else {

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: CARMA params bad.\n");
		#endif

		LnLike = -infiniteVal;
		}
	SystemMaster.deallocCARMA();

	return LnLike;
	}

double computeLnLike(double dt, int p, int q, double *Theta, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {

	int LnLike;

	#ifdef DEBUG_MAKEMOCKLC
	printf("computeLnLike: Creating SystemMaster...\n");
	#endif

	CARMA SystemMaster = CARMA();

	#ifdef DEBUG_MAKEMOCKLC
	printf("computeLnLike: SystemMaster created. Allocating memory...\n");
	#endif

	SystemMaster.allocCARMA(p, q);

	#ifdef DEBUG_MAKEMOCKLC
	printf("computeLnLike: Memory allocated. Checking CARMA params...\n");
	#endif

	int goodYN = 1;
	goodYN = SystemMaster.checkCARMAParams(Theta);

	#ifdef DEBUG_MAKEMOCKLC
	printf("computeLnLike: Checked CARMA Params.\n");
	#endif

	if (goodYN == 1) {
		double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: CARMA params good. Setting dt and C-ARMA model...\n");
		#endif

		SystemMaster.set_t(dt);
		SystemMaster.setCARMA(Theta);

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: dt and C-ARMA model set. Solving C-ARMA model...\n");
		#endif

		SystemMaster.solveCARMA();

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: C-ARMA model solved. Computing initial state...\n");
		#endif

		SystemMaster.resetState();

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: Initial state computed. Computing LnLike...\n");
		#endif

		LnLike = SystemMaster.computeLnLike(numCadences, y, yerr, mask);

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: LnLike computed. Deallocating system...\n");
		#endif

		SystemMaster.deallocCARMA();

		} else {

		#ifdef DEBUG_MAKEMOCKLC
		printf("computeLnLike: CARMA params bad.\n");
		#endif

		LnLike = -infiniteVal;
		}

	SystemMaster.deallocCARMA();
	return LnLike;
	}

void fitCARMA(double dt, int p, int q, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike) {

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Creating Data...\n");
	#endif

	LnLikeData Data;
	Data.numPts = numCadences;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Data created. Creating Args...\n");
	#endif

	LnLikeArgs Args;
	Args.numThreads = nthreads;
	Args.Data = Data;
	Args.Systems = nullptr;

	void* p2Args = nullptr;

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Args created. Creating Systems...\n");
	#endif

	double *initPos = nullptr, *offsetArr = nullptr, *xTemp = nullptr;
	vector<double> x;
	VSLStreamStatePtr xStream, initStream;

	CARMA Systems[nthreads];

	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Systems created. Allocating memory...\n");
	#endif

	int ndims = p + q + 1;
	for (int tNum = 0; tNum < nthreads; tNum++) {
		Systems[tNum].allocCARMA(p,q);
		Systems[tNum].set_t(dt);
		}
	Args.Systems = Systems;

	p2Args = &Args;

	double LnLikeVal = 0.0;

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Memory allocated. Finding good initial value...\n");
	#endif

	xTemp = static_cast<double*>(_mm_malloc(ndims*sizeof(double),64));
	vslNewStream(&xStream, VSL_BRNG_SFMT19937, xSeed);
	bool goodPoint = false;
	do {
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream, ndims, xTemp, 0.0, 1e-1);
		if (Systems[threadNum].checkCARMAParams(xTemp) == 1) {
			Systems[threadNum].set_t(dt);
			Systems[threadNum].setCARMA(xTemp);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();
			LnLikeVal = Systems[threadNum].computeLnLike(numCadences, y, yerr, mask);
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

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Good initial value found. Finding minima...\n");
	#endif

	nlopt::opt opt(nlopt::LN_NELDERMEAD, ndims);
	opt.set_max_objective(calcLnLike, p2Args);
	opt.set_maxeval(maxEvals);
	opt.set_xtol_rel(xTol);
	double max_LnLike = 0.0;
	nlopt::result yesno = opt.optimize(x, max_LnLike);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Minima found. Initializing ensemble...\n");
	#endif

	EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, zSSeed, walkerSeed, moveSeed);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Ensemble initialized. Creating initial positions...\n");
	#endif

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
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, initStream, nwalkers, offsetArr, 0.0, x[dimNum]*1.0e-3);
		for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
			initPos[walkerNum*ndims + dimNum] = x[dimNum] + offsetArr[walkerNum];
			}
		}
	vslDeleteStream(&initStream);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Initial positions set. Running MCMC...\n");
	#endif

	newEnsemble.runMCMC(initPos);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: MCMC run complete. Extracting chain...\n");
	#endif

	newEnsemble.getChain(Chain);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: Chain extracted. Extarcting LnLike...\n");
	#endif

	newEnsemble.getLnLike(LnLike);

	#ifdef DEBUG_FITCARMA
	printf("fitCARMA: LnLike extracted. Freeing memory...\n");
	#endif

	for (int tNum = 0; tNum < nthreads; tNum++) {
		Systems[tNum].deallocCARMA();
		}
	_mm_free(initPos);
	_mm_free(offsetArr);
	}

extern "C" {

	extern double _makeMockLC(double dt, int p, int q, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("_makeMockLC: Calling makeMockLC...\n");
		#endif

		double LnLike = makeMockLC(dt, p, q, Theta, numBurn, numCadences, noiseSigma, startCadence, burnSeed, distSeed, noiseSeed, cadence, mask, t, y, yerr);

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("_makeMockLC: makeMockLC returned...\n");
		#endif

		return LnLike;
		}

	extern double _computeLnLike(double dt, int p, int q, double *Theta, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("_computeLnLike: Calling computeLnLike...\n");
		#endif

		double LnLike = computeLnLike(dt, p, q, Theta, numCadences, cadence, mask, t, y, yerr);

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("_computeLnLike: computeLnLike returned...\n");
		#endif

		return LnLike;
		}

	extern void _fitCARMA(double dt, int p, int q, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, unsigned int initSeed, double *Chain, double *LnLike) {

		#ifdef DEBUG_CFFI_FITCARMA
		printf("_fitCARMA: Calling fitCARMA...\n");
		#endif

		fitCARMA(dt, p, q, numCadences, cadence, mask, t, y, yerr, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, initSeed, Chain, LnLike);

		#ifdef DEBUG_CFFI_FITCARMA
		printf("_fitCARMA: fitCARMA returned...\n");
		#endif

		}
	}