#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include "LC.hpp"

using namespace std;

	LCData::LCData() {
		numCadences = 0;
		dt = 0.0;
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

	int LCData::acvf(double *lagVals, double *acvfVals, double *acvfErrVals, int threadNum) {
		double meanVal = 0.0;
		double errInMean = 0.0, meanErr = 0.0, meanErrSq = 0.0, fSq = 0.0, t1 = 0.0, t2 = 0.0, errSum = 0.0;
		double count = 0.0;
		for (int i = 0; i < numCadences; ++i) {
			meanVal += mask[i]*y[i];
			errInMean += mask[i]*pow(yerr[i], 2.0);
			count += mask[i];
			}
		meanVal /= count;
		meanErr = sqrt(errInMean)/count;
		meanErrSq = pow(meanErr, 2.0);
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			lagVals[lagCad] = lagCad*dt;
			errSum = 0.0;
			for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
				acvfVals[lagCad] += mask[pointNum]*mask[pointNum + lagCad]*(y[pointNum] - meanVal)*(y[pointNum + lagCad] - meanVal);
				fSq = pow((y[pointNum + lagCad] - meanVal), 2.0)*pow((y[pointNum] - meanVal), 2.0);
				t1 = (pow(yerr[pointNum + lagCad], 2.0) + meanErrSq)/pow(y[pointNum + lagCad] - meanVal, 2.0);
				t2 = (pow(yerr[pointNum], 2.0) + meanErrSq)/pow(y[pointNum] - meanVal, 2.0);
				errSum += mask[pointNum]*mask[pointNum + lagCad]*fSq*(t1 + t2);
				}
			acvfVals[lagCad] /= count;
			acvfErrVals[lagCad] = sqrt(errSum)/count;
			}
		return 0;
		}

	int LCData::sf(double *lagVals, double *sfVals, double *sfErrVals, int threadNum) {
		double count = 0.0, errSum = 0.0;
		for (int lagCad = 0; lagCad < numCadences; ++lagCad) {
			lagVals[lagCad] = lagCad*dt;
			errSum = 0.0;
			count = 0.0;
			for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
				sfVals[lagCad] += mask[pointNum]*mask[pointNum + lagCad]*pow((y[pointNum + lagCad] - y[pointNum]), 2.0);
				errSum += 2.0*mask[pointNum]*mask[pointNum + lagCad]*pow((y[pointNum + lagCad] - y[pointNum]), 2.0)*(pow(y[pointNum + lagCad], 2.0) + pow(y[pointNum], 2.0));
				count += mask[pointNum]*mask[pointNum + lagCad];
				}
			sfVals[lagCad] /= count;
			sfErrVals[lagCad] = sqrt(errSum)/count;
			}
		return 0;
		}