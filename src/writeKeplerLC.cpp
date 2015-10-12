#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <nlopt.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/system/linux_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/detail/quoted_manip.hpp>
#include "Acquire.hpp"
#include "CARMA.hpp"
#include "Universe.hpp"
#include "Kepler.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

//#define DEBUG_MASK

using namespace std;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: fitARMA" << endl;
	cout << "Purpose: Program to to read Kepler LCs into y.dat format" << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	double maxDouble = numeric_limits<double>::max();
	double sqrtMaxDouble = sqrt(maxDouble);

	string basePath;
	AcquireDirectory(cout,cin,"Path to working directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "Output directory: " << basePath << endl;

	string keplerPath, keplerID, keplerObjPath, keplerObjCalibratedFile, keplerEpochList;
	cout << "Fit N points (offset by R points from the first point) from a Kepler light curve to an ARMA model." << endl;

	keplerPath = getenv("KEPLER_PATH");
	cout << "KEPLER_PATH: " << keplerPath << endl;

	do {
		AcquireInput(cout,cin,"KeplerID: ","Invalid value.\n",keplerID);
		keplerObjPath = keplerPath+keplerID+"/";
		} while (!boost::filesystem::exists(path(keplerObjPath)));

	cout << "Path to Kepler obj: " << keplerObjPath << endl;

	do {
		AcquireInput(cout,cin,"EpochList file (kplr0xxxxxxxx-epochList.dat): ","Invalid value.\n",keplerEpochList);
		} while ((!boost::filesystem::exists(path(keplerEpochList))) and (!boost::filesystem::is_regular_file(path(keplerPath+keplerID+"/"+keplerEpochList))));

	int cbvORpdc = 2;
	do {
		AcquireInput(cout,cin,"PDCSAP_FLUX or CBVSAP_FLUX (0/1): ","Invalid value.\n",cbvORpdc);
		} while (abs(cbvORpdc) > 1);

	array<double,3> loc = {0.0, 0.0, 0.0};
	Equatorial keplerPos = Equatorial(loc);
	KeplerObj newguy(keplerID, keplerPath, keplerPos);
	bool forceCalibrate = false;
	int stitchMethod = -1;

	keplerObjCalibratedFile = keplerObjPath+keplerID+"-calibrated.dat";
	tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray;
	if ((!boost::filesystem::exists(path(keplerObjCalibratedFile))) or (!boost::filesystem::is_regular_file(path(keplerObjCalibratedFile)))) {
		cout << "Calibrated data file not found. Calibration must be performed!" << endl;
		do {
			cout << "Stitching method to use?" << endl;
			cout << "[0]: No stitching" << endl;
			cout << "[1]: Match endpoints across quarters" << endl;
			cout << "[2]: Match averaged points across quarters" << endl;
			AcquireInput(cout,cin,"stitchMethod: ","Invalid value!\n",stitchMethod);
			} while ((stitchMethod < 0) && (stitchMethod > 2));
		} else {
		AcquireInput(cout,cin,"Calibrated data file located. Recalibrate? (Re-calibration is required if this version of -calibrated.dat does not correspond to the .dat files desired): ","Invalid value.\n",forceCalibrate);
		if (forceCalibrate == true) {
			do {
				cout << "Stitching method to use?" << endl;
				cout << "[0]: No stitching" << endl;
				cout << "[1]: Match endpoints across quarters" << endl;
				cout << "[2]: Match averaged points across quarters" << endl;
				AcquireInput(cout,cin,"stitchMethod: ","Invalid value!\n",stitchMethod);
				} while ((stitchMethod < 0) && (stitchMethod > 2));
			}
		}
	dataArray = newguy.getData(keplerEpochList,cbvORpdc,forceCalibrate,stitchMethod);
	newguy.setProperties(dataArray);
	int numCadences = newguy.getNumCadences();
	double* keplerMask = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	newguy.setMask(dataArray,keplerMask);
	int firstCadence = newguy.getFirstCadence(), lastCadence = newguy.getLastCadence(), startCadence = 0, offSet = 0;

	int numObs = 0;
	cout << keplerID << " has " << numCadences << " cadences." << endl;
	do {
		AcquireInput(cout,cin,"Number of cadences to write? (Must be greater than 0): ","Invalid value.\n",numObs);
		} while (numObs <= 0); 

	do {
		cout << "The first cadence in the lightcurve of " << keplerID << " is " << firstCadence << endl;
		cout << "The last cadence in the lightcurve of " << keplerID << " is " << lastCadence << endl;
		cout << "To observe " << numObs << " cadences in the lightcurve, the start cadence must be no greater than " << lastCadence + 1 - numObs << endl;
		cout << "Pick a start cadence mask between " << firstCadence << " and " << lastCadence + 1 - numObs << endl;
		AcquireInput(cout,cin,"startCadence: ","Invalid value.\n",startCadence);
		} while ((startCadence < firstCadence) or (startCadence > (lastCadence + 1 - numObs)));

	offSet = startCadence-firstCadence;
	int* cadence = static_cast<int*>(_mm_malloc(numObs*sizeof(int),64));
	double* mask = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	for (int obsCounter = 0; obsCounter < numObs; ++obsCounter) {
		mask[obsCounter] = keplerMask[obsCounter+offSet];
		cadence[obsCounter] = get<0>(dataArray)[obsCounter+offSet][0];
		if (mask[obsCounter] == 1.0) {
			y[obsCounter] = get<1>(dataArray)[obsCounter+offSet][1];
			yerr[obsCounter] = get<1>(dataArray)[obsCounter+offSet][2];
			} else {
			y[obsCounter] = 0.0;
			yerr[obsCounter] = sqrtMaxDouble;
			}
		}
	_mm_free(keplerMask);

	cout << "Computing and removing average flux" << endl;
	double ySum = 0.0, yCounter = 0.0;
	for (int i = 0; i < numObs; ++i) {
		ySum += mask[i]*y[i];
		yCounter += mask[i];
		}
	double yMean = ySum/yCounter;
	for (int i = 0; i < numObs; ++i) {
		y[i] -= mask[i]*yMean;
		}
	cout << "Actual number of observations: " << yCounter << endl;
	cout << "Mean Flux: " << yMean << endl;

	cout << "Writing light curve..." << endl;
	string yPath = basePath+"y.dat";
	ofstream yFile;
	yFile.open(yPath);
	yFile.precision(16);
	yFile << noshowpos << fixed << "numCadences: " << numObs << endl;
	yFile << noshowpos << fixed << "numObservations: " << static_cast<int>(yCounter) << endl;
	yFile << noshowpos << scientific << "meanFlux: " << yMean << endl;
	for (int i = 0; i < numObs-1; i++) {
		yFile << noshowpos << scientific << cadence[i] << " " << mask[i] << " " << y[i] << " " << yerr[i] << endl;
		}
	yFile << noshowpos << scientific << cadence[numObs-1] << " " << mask[numObs-1] << " " << y[numObs-1] << " " << yerr[numObs-1];
	yFile.close();

	cout << endl;
	cout << "Kepler data written to " << basePath + "y.dat" << endl;
	cout << "Program exiting...Have a nice day!" << endl; 

	_mm_free(cadence);
	_mm_free(mask);
	_mm_free(y);
	_mm_free(yerr);
	}