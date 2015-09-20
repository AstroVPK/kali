#include <malloc.h>
#include <limits>
#include <mathimf.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <mkl.h>
#include <mkl_types.h>
#include "Correlation.hpp"
#include "DLAPACKE.hpp"
#include <stdio.h>

//#define DEBUG_PACF

using namespace std;

void ACVF(int numCadences, const double* const y, const double* const mask, double* acvf) {
	int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
	omp_set_num_threads(nthreads);
	/*! First remove the mean. */
	double sum = 0.0, numObs = 0.0;
	for (int cadCounter = 0; cadCounter < numCadences; ++cadCounter) {
		sum += mask[cadCounter]*y[cadCounter];
		numObs += mask[cadCounter];
		}
	double mean = sum/numObs;

	double* yScratch = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	#pragma omp parallel for simd default(none) shared(numCadences, yScratch, y, mean, mask)
	for (int cadCounter = 0; cadCounter < numCadences; ++cadCounter) {
		yScratch[cadCounter] = y[cadCounter] - mean*mask[cadCounter];
		}

	/*! Now, following "Spectrum Estimation with Missing Observations" by Richard H. Jones (Rcvd 1969, Rvsd 1971) we compute for each lag. */
	/*#pragma omp parallel for default(none) shared(numLags, acvf, numCadences, mask, yScratch, numObs)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acvf[lagCounter] = 0.0;
		double numPts = 0.0;
		for (int cadCounter = 0; cadCounter < numCadences - lagCounter; ++cadCounter) {
			acvf[lagCounter] += mask[cadCounter]*mask[cadCounter + lagCounter]*yScratch[cadCounter]*yScratch[cadCounter + lagCounter]; 
			numPts += mask[cadCounter]*mask[cadCounter + lagCounter];
			}
		acvf[lagCounter] /= numPts;
		}*/

	/*! Following "Modern Applied Statistics with S" by W.N. Venables & B.D. Ripley (4th Ed, 1992) we compute for each lag. */
	#pragma omp parallel for default(none) shared(acvf, numCadences, mask, yScratch, numObs)
	for (int lagCounter = 0; lagCounter < numCadences; ++lagCounter) {
		acvf[lagCounter] = 0.0;
		for (int cadCounter = 0; cadCounter < numCadences - lagCounter; ++cadCounter) {
			acvf[lagCounter] += mask[cadCounter]*mask[cadCounter + lagCounter]*yScratch[cadCounter]*yScratch[cadCounter + lagCounter]; 
			}
		acvf[lagCounter] /= numObs;
		}

	_mm_free(yScratch);
	}

void ACF(int numCadences, const double* const acvf, double* acf) {
	int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
	omp_set_num_threads(nthreads);
	#pragma omp parallel for simd default(none) shared(numCadences, acf, acvf)
	for (int lagCounter = 0; lagCounter < numCadences; ++lagCounter) {
		acf[lagCounter] = acvf[lagCounter]/acvf[0];
		}
	}

void PACF(int numCadences, int maxLag, const double* const acvf, double* pacf) {
	int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
	omp_set_num_threads(nthreads);

	pacf[0] = 1.0;

	double* R = static_cast<double*>(_mm_malloc(maxLag*sizeof(double),64));
	double* X = static_cast<double*>(_mm_malloc(maxLag*sizeof(double),64));
	double* Y = static_cast<double*>(_mm_malloc(maxLag*sizeof(double),64));

	#pragma omp parallel for simd default(none) shared(maxLag, acvf, R)
	for (int lagCounter = 0; lagCounter < maxLag; ++lagCounter) {
		R[lagCounter] = acvf[lagCounter]/acvf[0];

		#ifdef DEBUG_PACF
		printf("PACF - R[%d]: %f\n",lagCounter,R[lagCounter]);
		#endif

		}

	double inverseACVFZero = 1.0/acvf[0];

	#ifdef DEBUG_PACF
	printf("acvf[0]^-1: %f\n",inverseACVFZero);
	#endif

	pacf[1] = inverseACVFZero*R[1];

	#pragma omp parallel for default(none) shared(inverseACVFZero, numCadences, maxLag, R, X, Y, pacf)
	for (int lagNum = 2; lagNum < maxLag + 1; ++lagNum) {
		double detC = DLAPACKE_dszthp(lagNum-1, R, X, Y);

		#ifdef DEBUG_PACF
		printf("Computing inverse for lagNum = %d\n",lagNum);
		#endif

		pacf[lagNum] = 0.0;

		for (int colCounter = 0; colCounter < lagNum; ++colCounter) {

			#ifdef DEBUG_PACF
			for (int rowCounter = 0; rowCounter < lagNum; ++rowCounter) {
				printf("%f ",DLAPACKE_dszthi(rowCounter, colCounter, lagNum-1, X));
				}
			printf("\n");
			#endif

			pacf[lagNum] += DLAPACKE_dszthi(lagNum-1, colCounter, lagNum-1, X)*R[colCounter + 1];
			}
		pacf[lagNum] *= inverseACVFZero;
		}

	_mm_free(Y);
	_mm_free(X);
	_mm_free(R);
	}

void SF1(int numCadences, const double* const acvf, double* sf1) {
	int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
	omp_set_num_threads(nthreads);
	#pragma omp parallel for simd default(none) shared(numCadences, sf1, acvf)
	for (int lagCounter = 0; lagCounter < numCadences; ++lagCounter) {
		sf1[lagCounter] = 2.0*(acvf[0] - acvf[lagCounter]);
		}
	}