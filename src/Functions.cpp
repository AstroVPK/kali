#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <cstdio>
//#include <sys/sysinfo.h>
//#include <unistd.h>
//#include <cstdlib>
//#include <vector>
//#include <array>
//#include <tuple>
//#include <string>
//#include <cstring>
//#include <sstream>
//#include <iostream>
//#include <fstream>
#include <nlopt.hpp>
//#include <boost/system/error_code.hpp>
//#include <boost/system/system_error.hpp>
//#include <boost/system/linux_error.hpp>
//#include <boost/filesystem.hpp>
//#include <boost/io/detail/quoted_manip.hpp>
//#include "Acquire.hpp"
#include "CARMA.hpp"
//#include "Universe.hpp"
//#include "Kepler.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC
#define DEBUG_MAKEMOCKLC
#define DEBUG_CFFI_MAKEMOCKLC

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

		SystemMaster.set_t(dt);
		SystemMaster.setCARMA(Theta);
		SystemMaster.solveCARMA();
		SystemMaster.resetState();

		double* burnRand = static_cast<double*>(_mm_malloc(numBurn*sizeof(double),64));
		for (int i = 0; i < numBurn; i++) {
			burnRand[i] = 0.0;
			}
		SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
		_mm_free(burnRand);

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

		SystemMaster.deallocCARMA();

		retVal = 0;

		} else {
		retVal = -1;
		}
	SystemMaster.deallocCARMA();
	return retVal;
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

	}