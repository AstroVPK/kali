#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include "LC.hpp"

//#define DEBUG_SF

#ifdef DEBUG_SF
	#include<cstdio>
#endif

using namespace std;

	LCData::LCData() {
		numCadences = 0;
		dt = 0.0;
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
		double errInMean = 0.0, meanErr = 0.0, meanErrSq = 0.0, fSq = 0.0, t1 = 0.0, t2 = 0.0, errSum = 0.0;
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
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			lagVals[lagCad] = lagCad*dt;
			errSum = 0.0;
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

	int LCData::sf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *sfVals, double *sfErrVals, int threadNum) {
		double count = 0.0, valSum = 0.0, errSum = 0.0;
		#ifdef DEBUG_SF
			printf("numCadences: %d\n",numCadences);
			printf("dt: %f\n",dt);
		#endif
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			#ifdef DEBUG_SF
				printf("lagCad: %d\n",lagCad);
				printf("numcadences - lagCad: %d\n",numCadences - lagCad);
			#endif
			lagVals[lagCad] = lagCad*dt;
			valSum = 0.0;
			errSum = 0.0;
			count = 0.0;
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

	int LCData::dacvf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, int numBins, double *lagVals, double *acvfVals, double *acvfErrVals, int threadNum) {

		// Compute the mean
		double meanVal = 0.0, count = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			meanVal += maskIn[i]*yIn[i];
			count += maskIn[i];
			}
		if (count > 0.0) {
			meanVal /= count;
			}

		// For the 1st Bin
		double binStart = 0.0;
		int minSpacing = static_cast<int>(binStart/dt);
		double binEnd = lagVals[0] + (0.5*(lagVals[1] - lagVals[0]));
		int maxSpacing = static_cast<int>(binEnd/dt);
		int pairsCtr = 0;
		for (int ptCtr = 0; ptCtr < numCadences; ++ptCtr) {
			for (int lagCtr = ptCtr + minSpacing; lagCtr < max(ptCtr + maxSpacing, numCadences); ++lagCtr) {
				acvfVals[0] = acvfVals[0] + ((maskIn[ptCtr]*(yIn[ptCtr] - meanVal))*(maskIn[ptCtr+lagCtr]*(yIn[ptCtr+lagCtr] - meanVal)));
				pairsCtr = pairsCtr + (maskIn[ptCtr]*maskIn[ptCtr+lagCtr]);
				}
			}
		if (pairsCtr > 0) {
			acvfVals[0] = acvfVals[0]/pairsCtr;
			}

		// For all but the 1st and last bins
		for (int binCtr = 1; binctr < numBins - 1; ++binCtr) {
			double binStart = lagVals[binCtr] - (0.5*(lagVals[binCtr] - lagVals[binCtr-1]));
			int minSpacing = static_cast<int>(binStart/dt);
			double binEnd = lagVals[binCtr] + (0.5*(lagVals[binCtr+1] - lagVals[binCtr]));
			int maxSpacing = static_cast<int>(binEnd/dt);
			int pairsCtr = 0;
			for (int ptCtr = 0; ptCtr < numCadences; ++ptCtr) {
				for (int lagCtr = ptCtr + minSpacing; lagCtr < max(ptCtr + maxSpacing, numCadences); ++lagCtr) {
					acvfVals[binCtr] = acvfVals[binCtr] + ((maskIn[ptCtr]*(yIn[ptCtr] - meanVal))*(maskIn[ptCtr+lagCtr]*(yIn[ptCtr+lagCtr] - meanVal)));
					pairsCtr = pairsCtr + (maskIn[ptCtr]*maskIn[ptCtr+lagCtr]);
					}
				}
			if (pairsCtr > 0) {
				acvfVals[binCtr] = acvfVals[binCtr]/pairsCtr;
				}
			}

		// For the last bins
		double binStart = lagVals[binCtr] - (0.5*(lagVals[binCtr] - lagVals[binCtr-1]));
		int minSpacing = static_cast<int>(binStart/dt);
		double binEnd = tIn[numCadences-1] - tIn[0];
		int maxSpacing = static_cast<int>(binEnd/dt);
		int pairsCtr = 0;
		for (int ptCtr = 0; ptCtr < numCadences; ++ptCtr) {
			for (int lagCtr = ptCtr + minSpacing; lagCtr < max(ptCtr + maxSpacing, numCadences); ++lagCtr) {
				acvfVals[numBins-1] = acvfVals[numBins-1] + ((maskIn[ptCtr]*(yIn[ptCtr] - meanVal))*(maskIn[ptCtr+lagCtr]*(yIn[ptCtr+lagCtr] - meanVal)));
				pairsCtr = pairsCtr + (maskIn[ptCtr]*maskIn[ptCtr+lagCtr]);
				}
			}
		if (pairsCtr > 0) {
			acvfVals[numBins-1] = acvfVals[numBins-1]/pairsCtr;
			}

		}