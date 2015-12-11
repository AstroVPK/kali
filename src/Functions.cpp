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

#if defined DEBUG_MAKEMOCKLC || DEBUG_CFFI_MAKEMOCKLC || DEBUG_COMPUTELNLIKE || DEBUG_CFFI_COMPUTELNLIKE
#include <cstdio>
#endif

//#define DEBUG_MASK

using namespace std;

int makeMockLC(double dt, int p, int q, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {

	int retVal;

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
		printf("makeMockLC: Mean removed. De-allocating system...\n");
		#endif

		SystemMaster.deallocCARMA();

		retVal = 0;

		} else {

		#ifdef DEBUG_MAKEMOCKLC
		printf("makeMockLC: CARMA params bad.\n");
		#endif

		retVal = -1;
		}
	SystemMaster.deallocCARMA();
	return retVal;
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

extern "C" {

	extern int cffi_makeMockLC(double dt, int p, int q, double *Theta, int numBurn, int numCadences, double noiseSigma, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr) {

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("cffi_makeMockLC: Calling makeMockLC...\n");
		#endif

		int YesOrNo = makeMockLC(dt, p, q, Theta, numBurn, numCadences, noiseSigma, startCadence, burnSeed, distSeed, noiseSeed, cadence, mask, t, y, yerr);

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("cffi_makeMockLC: makeMockLC returned...\n");
		#endif

		return YesOrNo;
		}

	extern double cffi_computeLnLike(double dt, int p, int q, double *Theta, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr) {

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("cffi_computeLnLike: Calling computeLnLike...\n");
		#endif

		double LnLike = computeLnLike(dt, p, q, Theta, numCadences, cadence, mask, t, y, yerr);

		#ifdef DEBUG_CFFI_MAKEMOCKLC
		printf("cffi_computeLnLike: computeLnLike returned...\n");
		#endif

		return LnLike;
		}
	}