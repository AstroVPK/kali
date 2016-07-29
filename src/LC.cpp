#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include "LC.hpp"

//#define DEBUG_SF
//#define DEBUG_DACF


#if defined DEBUG_SF || defined DEBUG_DACF
	#include<cstdio>
#endif

using namespace std;

	LCData::LCData() {
		numCadences = 0;
		dt = 0.0;
		meandt = 0.0;
		mindt = 0.0;
		maxdt = 0.0;
		dtSmooth = 0.0;
		tolIR = 0.0;
		fracIntrinsicVar = 0.0;
		fracNoiseToSignal = 0.0;
		maxSigma = 0.0;
		minTimescale = 0.0;
		maxTimescale = 0.0;
		t = nullptr;
		x = nullptr;
		y = nullptr;
		yerr = nullptr;
		mask = nullptr;
		lcXSim = nullptr;
		lcPSim = nullptr;
		lcXComp = nullptr;
		lcPComp = nullptr;
	}

	int LCData::acvf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *acvfVals, double *acvfErrVals, int threadNum) {
		double meanVal = 0.0;
		double errInMean = 0.0, meanErr = 0.0, meanErrSq = 0.0;
		double count = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			meanVal += maskIn[i]*yIn[i];
			errInMean += maskIn[i]*pow(yerrIn[i], 2.0);
			count += maskIn[i];
			}
		if (count > 0.0) {
			meanVal /= count;
			}
		meanErr = sqrt(errInMean)/count;
		meanErrSq = pow(meanErr, 2.0);
		#pragma omp parallel for default(none) shared(numCadences, dt, count, tIn, yIn, yerrIn, maskIn, lagVals, acvfVals, acvfErrVals, meanVal, meanErr, meanErrSq)
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			lagVals[lagCad] = lagCad*dt;
			double fSq = 0.0, t1 = 0.0, t2 = 0.0, errSum = 0.0;
			for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
				acvfVals[lagCad] += maskIn[pointNum]*maskIn[pointNum + lagCad]*(yIn[pointNum] - meanVal)*(yIn[pointNum + lagCad] - meanVal);
				fSq = pow((yIn[pointNum + lagCad] - meanVal)*(yIn[pointNum] - meanVal), 2.0);
				t1 = (pow(yerrIn[pointNum + lagCad], 2.0) + meanErrSq)/pow(yIn[pointNum + lagCad] - meanVal, 2.0);
				t2 = (pow(yerrIn[pointNum], 2.0) + meanErrSq)/pow(yIn[pointNum] - meanVal, 2.0);
				errSum += maskIn[pointNum]*maskIn[pointNum + lagCad]*fSq*(t1 + t2);
				}
			if (count > 0.0) {
				acvfVals[lagCad] /= count;
				acvfErrVals[lagCad] = sqrt(errSum/count);
				}
			}
		return 0;
		}

	int LCData::acf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *acfVals, double *acfErrVals, int threadNum) {
		double meanVal = 0.0;
		double errInMean = 0.0, meanErr = 0.0, meanErrSq = 0.0;
		double count = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			meanVal += maskIn[i]*yIn[i];
			errInMean += maskIn[i]*pow(yerrIn[i], 2.0);
			count += maskIn[i];
			}
		if (count > 0.0) {
			meanVal /= count;
			}
		meanErr = sqrt(errInMean)/count;
		meanErrSq = pow(meanErr, 2.0);
		#pragma omp parallel for default(none) shared(numCadences, dt, count, tIn, yIn, yerrIn, maskIn, lagVals, acfVals, acfErrVals, meanVal, meanErr, meanErrSq)
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			lagVals[lagCad] = lagCad*dt;
			double fSq = 0.0, t1 = 0.0, t2 = 0.0, errSum = 0.0;
			for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
				acfVals[lagCad] += maskIn[pointNum]*maskIn[pointNum + lagCad]*(yIn[pointNum] - meanVal)*(yIn[pointNum + lagCad] - meanVal);
				fSq = pow((yIn[pointNum + lagCad] - meanVal)*(yIn[pointNum] - meanVal), 2.0);
				t1 = (pow(yerrIn[pointNum + lagCad], 2.0) + meanErrSq)/pow(yIn[pointNum + lagCad] - meanVal, 2.0);
				t2 = (pow(yerrIn[pointNum], 2.0) + meanErrSq)/pow(yIn[pointNum] - meanVal, 2.0);
				errSum += maskIn[pointNum]*maskIn[pointNum + lagCad]*fSq*(t1 + t2);
				}
			if (count > 0.0) {
				acfVals[lagCad] /= count;
				acfErrVals[lagCad] = sqrt(errSum/count);
				}
			}
		double acvfFirst = acfVals[0], constErr = pow(acfErrVals[0]/acfVals[0], 2.0);
		#pragma omp parallel for default(none) shared(numCadences, dt, tIn, yIn, yerrIn, maskIn, lagVals, acfVals, acfErrVals, acvfFirst, constErr)
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			double acfHolder = acfVals[lagCad]/acvfFirst;
			acfErrVals[lagCad] = (acfVals[lagCad]/acvfFirst)*sqrt(pow(acfErrVals[lagCad]/acfVals[lagCad], 2.0) + constErr);
			acfVals[lagCad] = acfHolder;
			}
		return 0;
		}

	int LCData::sf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *sfVals, double *sfErrVals, int threadNum) {
		#ifdef DEBUG_SF
			printf("numCadences: %d\n",numCadences);
			printf("dt: %f\n",dt);
		#endif
		#pragma omp parallel for default(none) shared(numCadences, dt, tIn, yIn, yerrIn, maskIn, lagVals, sfVals, sfErrVals)
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			#ifdef DEBUG_SF
				printf("lagCad: %d\n",lagCad);
				printf("numcadences - lagCad: %d\n",numCadences - lagCad);
			#endif
			lagVals[lagCad] = lagCad*dt;
			double valSum = 0.0, errSum = 0.0, count = 0.0;
			for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
				#ifdef DEBUG_SF
					if ((mask[pointNum] == 1.0) and (mask[pointNum + lagCad] == 1.0)) {
						printf("\n");
						printf("y[%d]: %f\n", pointNum + lagCad, yIn[pointNum + lagCad]);
						printf("y[%d]: %f\n", pointNum, yIn[pointNum]);
						printf("term: %f\n", maskIn[pointNum]*maskIn[pointNum + lagCad]*pow((yIn[pointNum + lagCad] - yIn[pointNum]), 2.0));
						}
				#endif
				valSum += maskIn[pointNum]*maskIn[pointNum + lagCad]*pow((yIn[pointNum + lagCad] - yIn[pointNum]), 2.0);
				errSum += 2.0*maskIn[pointNum]*maskIn[pointNum + lagCad]*pow((yIn[pointNum + lagCad] - yIn[pointNum]), 2.0)*(pow(yIn[pointNum + lagCad], 2.0) + pow(yIn[pointNum], 2.0));
				count += maskIn[pointNum]*maskIn[pointNum + lagCad];
				}
			if (count > 0.0) {
				#ifdef DEBUG_SF
					printf("lagVals: %f\n",lagVals[lagCad]);
					printf("count: %f\n",count);
					printf("valSum: %f\n",valSum);
					printf("errSum: %f\n",errSum);
				#endif
				sfVals[lagCad] = valSum/count;
				sfErrVals[lagCad] = sqrt(errSum)/count;
				} else {
				sfVals[lagCad] = 0.0;
				sfErrVals[lagCad] = 0.0;
				}
			}
		return 0;
		}

	int LCData::dacf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, int numBins, double *lagVals, double *dacfVals, double *dacfErrVals, int threadNum) {
		/*!
		DACF of Edelson Krolik 1988
		*/
		// Compute the mean
		double meanVal = 0.0, count = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			meanVal += maskIn[i]*yIn[i];
			count += maskIn[i];
			}
		if (count > 0.0) {
			meanVal /= count;
			}
		// Compute the variance
		double varVal = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			varVal += pow(maskIn[i]*(yIn[i] - meanVal), 2.0);
			}
		if (count > 0.0) {
			varVal /= count;
			}
		#ifdef DEBUG_DACF
			printf("meanVal: %e\n", meanVal);
			printf("varVal: %e\n", varVal);
		#endif
		// Allocate array to hold all dacf pairs
		double *tPair = static_cast<double*>(_mm_malloc((numCadences*(numCadences + 1)/2)*sizeof(double),64));
		double *yPair = static_cast<double*>(_mm_malloc((numCadences*(numCadences + 1)/2)*sizeof(double),64));
		double *maskPair = static_cast<double*>(_mm_malloc((numCadences*(numCadences + 1)/2)*sizeof(double),64));
		// Compute UDCF for all pairs
		#pragma omp parallel for default(none) shared(numCadences, tIn, yIn, yerrIn, maskIn, meanVal, varVal, tPair, yPair, maskPair)
		for (int i = 0; i < numCadences; ++i) {
			#ifdef DEBUG_DACF
				int threadNum = omp_get_thread_num();
			#endif
			#pragma omp simd
			for (int j = 0; j < numCadences - i; ++j) {
				tPair[j + i*(numCadences - i)] = tIn[i] - tIn[j];
				yPair[j + i*(numCadences - i)] = (maskIn[i]*maskIn[j]*(yIn[i] - meanVal)*(yIn[j] - meanVal))/sqrt((varVal - pow(yerr[i],2.0))*(varVal - pow(yerr[j],2.0)));
				maskPair[j + i*(numCadences - i)] = maskIn[i]*maskIn[j];
				}
			}
		//Now loop through all the dacf bins
		#pragma omp parallel for default(none) shared(numCadences, dt, numBins, tIn, tPair, yPair, maskPair, lagVals, dacfVals, dacfErrVals)
		for (int binCtr = 0; binCtr < numBins; ++binCtr) {
			double binStart = 0.0;
			if (binCtr != 0) {
				binStart = lagVals[binCtr] - (0.5*(lagVals[binCtr] - lagVals[binCtr-1]));
				}
			int minSpacing = static_cast<int>(binStart/dt);
			double binEnd = tIn[numCadences-1] - tIn[0];
			if (binCtr != (numBins - 1)) {
				binEnd = lagVals[binCtr] + (0.5*(lagVals[binCtr+1] - lagVals[binCtr]));
				}
			int maxSpacing = static_cast<int>(binEnd/dt);
			double numPairs = 0.0;
			#ifdef DEBUG_DACF
				printf("dacf - threadNum: %d; binStart: %e\n",threadNum, binStart);
				printf("dacf - threadNum: %d; lagVals[%d]: %e\n",threadNum, binCtr, lagVals[binCtr]);
				printf("dacf - threadNum: %d; binEnd: %e\n",threadNum, binEnd);
			#endif
			// Loop through the pair arrays and add their contents if the tPair indicates to do so
			for (int pairCtr = 0; pairCtr < (numCadences*(numCadences + 1)/2); ++pairCtr) {
				if ((tPair[pairCtr] >= binStart) and (tPair[pairCtr] < binEnd)) {
					dacfVals[binCtr] += maskPair[pairCtr]*yPair[pairCtr];
					numPairs += maskPair[pairCtr];
					}
				}
			if (numPairs > 0.0) {
				dacfVals[binCtr] /= numPairs;
				}
			// Now loop through again for the errors
			for (int pairCtr = 0; pairCtr < (numCadences*(numCadences + 1)/2); ++pairCtr) {
				if ((tPair[pairCtr] >= binStart) and (tPair[pairCtr] < binEnd)) {
					dacfErrVals[binCtr] += pow(dacfVals[binCtr] - yPair[pairCtr], 2.0);
					}
				}
			if (numPairs > 0.0) {
				dacfErrVals[binCtr] /= sqrt((numPairs - 1.0)*static_cast<double>((numCadences - 1)));
				}
			dacfErrVals[binCtr] = sqrt(dacfErrVals[binCtr]);
			}
		// Free the pair arrays
		if (tPair) {
			_mm_free(tPair);
			tPair = nullptr;
			}
		if (yPair) {
			_mm_free(yPair);
			yPair = nullptr;
			}
		if (maskPair) {
			_mm_free(maskPair);
			maskPair = nullptr;
			}
		return 0;
		}