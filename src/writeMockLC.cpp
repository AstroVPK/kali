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
#include "Kalman.hpp"
#include "Universe.hpp"
#include "Kepler.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

//#define DEBUG_MASK

using namespace std;
using namespace nlopt;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: testMethod" << endl;
	cout << "Purpose: Program to create and write a mock LC with a given set of ARMA parameters." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	string basePath;
	AcquireDirectory(cout,cin,"Path to output directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "Output directory: " << basePath << endl;

	string keplerPath, keplerID, keplerObjPath, keplerObjCalibratedFile;
	cout << "Use a given Kepler AGN to create the mask of missing values for the mock light curve" << endl;

	keplerPath = getenv("KEPLER_PATH");
	cout << "KEPLER_PATH: " << keplerPath << endl;

	do {
		AcquireInput(cout,cin,"KeplerID: ","Invalid value.\n",keplerID);
		keplerObjPath = keplerPath+keplerID+"/";
		} while (!boost::filesystem::exists(path(keplerObjPath)));

	cout << "Path to Kepler obj: " << keplerObjPath << endl;

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
		AcquireInput(cout,cin,"Calibrated data file located. Recalibrate? (Recommended that re-calibration not be performed!): ","Invalid value.\n",forceCalibrate);
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
	dataArray = newguy.getData(forceCalibrate,stitchMethod);
	newguy.setProperties(dataArray);
	int numCadences = newguy.getNumCadences();
	double* keplerMask = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	newguy.setMask(dataArray,keplerMask);

	int firstCadence = newguy.getFirstCadence(), lastCadence = newguy.getLastCadence(), startCadence = 0, offSet = 0;

	#ifdef DEBUG_MASK
	for (int i = 0; i < numObs; ++i) {
		printf("mask[%d]: %f\n",i+firstCadence,keplerMask[i]);
		}
	#endif

	/*int numHost = sysconf(_SC_NPROCESSORS_ONLN);
	cout << numHost << " hardware thread contexts detected." << endl;
	int nthreads = 0;
	AcquireInput(cout,cin,"Number of OmpenMP threads to use: ","Invalid value!\n",nthreads);
	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();*/

	cout << "Create a test light curve with known parameters - make an ARMA light curve with p AR and q MA co-efficients." << endl;
	int pMaster = 0, qMaster = 0;
	while (pMaster < 1) {
		AcquireInput(cout,cin,"Number of AR coefficients p: ","Invalid value.\n",pMaster);
	}
	if (pMaster > 1) {
		qMaster = pMaster;
		while ((qMaster >= pMaster) or (qMaster < 0)) {
			cout << "The number of MA coefficients (q) must be less than the number of AR coefficients (p) if the system is to \n correspond to a C-ARMA process." << endl;
			cout << "Please select q < " << pMaster << endl;
			AcquireInput(cout,cin,"Number of MA coefficients q: ","Invalid value.\n",qMaster);
			}
		} else {
		qMaster = 0;
		}
	cout << "Creating ARMA model with " << pMaster << " AR components and " << qMaster << " MA components." << endl;
	DLM SystemMaster = DLM();
	SystemMaster.allocDLM(pMaster, qMaster);
	cout << "Allocated " << SystemMaster.allocated << " bytes for the model!" << endl;

	double* ThetaMaster = static_cast<double*>(_mm_malloc((pMaster+qMaster+1)*sizeof(double),64));
	#pragma omp parallel for simd default(none) shared(pMaster,qMaster,ThetaMaster)
	for (int i = 0; i < pMaster+qMaster+1; i++) {
		ThetaMaster[i] = 0.0;
		}
	cout << "Set the values of the ARMA model parameters." << endl;

	int goodYN = 0;
	while (goodYN == 0) {
		while (ThetaMaster[0] <= 0.0) {
			cout << "Set the standard deviation of the disturbances (sigma_dist) such that sigma_dist > 0.0" << endl;
			AcquireInput(cout,cin,"Set the value of sigma_dist: ","Invalid value.\n",ThetaMaster[0]);
			}

		string inStr;
		for (int i = 1; i < 1+pMaster; i++) {
			cout << "Set the value of phi_" << i;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "Set the value of theta_" << i-pMaster;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		SystemMaster.setDLM(ThetaMaster);
		cout << endl;
		cout << "Checking to see if the system is stable, invertible, not-redundant, and reasonable..." << endl;
		goodYN = SystemMaster.checkARMAParams(ThetaMaster);

		double *RealAR, *ImagAR, *RealMA, *ImagMA;

		cout << "AR Polynomial Roots" << endl;
		cout.precision(4);
		if (pMaster == 1) {
			cout << noshowpos << fixed << "Root1: " << showpos << 1.0/ThetaMaster[1] << 0.0 << "i; Magnitude: " << abs(1.0/ThetaMaster[1]) << endl;
			} else {
			SystemMaster.getARRoots(RealAR, ImagAR);
			for (int pCtr = 0; pCtr < pMaster; ++pCtr) {
				cout << showpos << fixed << "Root" << pCtr << ": " << showpos << RealAR[pCtr] << ImagAR[pCtr] << "i; Magnitude: " << sqrt(pow(RealAR[pCtr],2.0) + pow(ImagAR[pCtr],2.0)) << endl;
				}
			}
		cout.precision(4);
		if (qMaster > 0) {
			cout << endl;
			cout << "MA Polynomial Roots" << endl;
			if (qMaster == 1) {
				cout << showpos << fixed << "Root1: " << showpos << -1.0/ThetaMaster[pMaster+1] << 0.0 << "i; Magnitude: " << abs(-1.0/ThetaMaster[pMaster+1]) << endl;
				} else {
				SystemMaster.getMARoots(RealMA, ImagMA);
				for (int qCtr = 0; qCtr < qMaster; ++qCtr) {
					cout << showpos << fixed << "Root" << qCtr << ": " << showpos << RealMA[qCtr] << ImagMA[qCtr] << "i; Magnitude: " << sqrt(pow(RealMA[qCtr],2.0) + pow(ImagMA[qCtr],2.0)) << endl;
					}
				}
			}

		cout << noshowpos << endl;

		cout << "ARMA model parameters are ";
		if (goodYN == 0) {
			cout << "bad!" << endl;
			cout << "Redo!" << endl;
			cout << endl;
			} else {
			cout << "good!" << endl;
			cout << endl;
			}
		}
	cout << "Model is set to use the following parameters..." << endl;
	cout << "sigma_dist: " << ThetaMaster[0] << endl;
	for (int i = 1; i < 1+pMaster; i++) {
		cout << "phi_" << i << ": " << ThetaMaster[i] << endl;
		}
	for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
		cout << "theta_" << i-pMaster << ": " << ThetaMaster[i] << endl;
		}
	cout << endl;

	bool setSeedsYN = 0;
	unsigned int burnSeed = 1311890535, distSeed = 2603023340, noiseSeed = 2410288857;
	AcquireInput(cout,cin,"Supply seeds for LC? 1/0: ","Invalid value.\n",setSeedsYN);
	if (setSeedsYN) {
		burnSeed = 0, distSeed = 0, noiseSeed = 0;
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Burn-in phase disturbance seed: ","Invalid value.\n",burnSeed);
			} while (burnSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Observation phase disturbance seed: ","Invalid value.\n",distSeed);
			} while (distSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Observation phase noise seed: ","Invalid value.\n",noiseSeed);
			} while (noiseSeed <= 0);
		}
	cout << endl;

	int numBurn = 0;
	AcquireInput(cout,cin,"Number of burn-in steps for model: ","Invalid value.\n",numBurn);
	double* burnRand = static_cast<double*>(_mm_malloc(numBurn*sizeof(double),64));
	for (int i = 0; i < numBurn; i++) {
		burnRand[i] = 0.0;
		}
	cout << "Burning..." << endl;
	SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
	_mm_free(burnRand);
	cout << "Burn phase complete!" << endl;
	cout << endl;

	int numObs = 0;
	do {
		AcquireInput(cout,cin,"Number of cadences (Must be greater than 0): ","Invalid value.\n",numObs);
		} while (numObs <= 0); 
	double noiseSigma = 0.0;
	do {
		cout << "Set the standard deviation of the noise (sigma_noise) such that sigma_noise > 0.0" << endl;
		AcquireInput(cout,cin,"Observation noise sigma_noise: ","Invalid value.\n",noiseSigma);
		} while (noiseSigma <= 0.0);

	do {
		cout << "The first cadence in the lightcurve of " << keplerID << " is " << firstCadence << endl;
		cout << "The last cadence in the lightcurve of " << keplerID << " is " << lastCadence << endl;
		cout << "To observe " << numObs << " cadences in the LC, the start cadence must be no greater than " << lastCadence-numObs << endl;
		cout << "Pick the starting cadence for the mask between " << firstCadence << " and " << lastCadence-numObs << endl;
		AcquireInput(cout,cin,"startCadence: ","Invalid value.\n",startCadence);
		} while ((startCadence < firstCadence) or (startCadence >= (lastCadence-numObs)));

	offSet = startCadence-firstCadence;

	int* cadence = static_cast<int*>(_mm_malloc(numObs*sizeof(int),64));
	double* mask = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));

	for (int obsCounter = 0; obsCounter < numObs; ++obsCounter) {
		cadence[obsCounter] = get<0>(dataArray)[obsCounter+offSet][0];
		mask[obsCounter] = keplerMask[obsCounter+offSet];
		}
	_mm_free(keplerMask);

	#ifdef DEBUG_MASK
	cout << "offSet: " << offSet << endl;
	for (int i = 0; i < numObs; ++i) {
		printf("mask[%d]: %f\n",i+startCadence,mask[i]);
		}
	#endif

	double* distRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* noiseRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double maxDouble = numeric_limits<double>::max();
	double sqrtMaxDouble = sqrt(maxDouble);
	for (int i = 0; i < numObs; i++) {
		distRand[i] = 0.0;
		noiseRand[i] = 0.0;
		y[i] = 0.0;
		if (mask[i] == 1.0) {
			yerr[i] = noiseSigma;
			} else {
			yerr[i] = sqrtMaxDouble;
			} 
		}
	cout << "Observing..." << endl;
	SystemMaster.observeSystem(numObs, distSeed, noiseSeed, distRand, noiseRand, noiseSigma, y, mask);

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
	cout << "Number of observations: " << yCounter << endl;
	cout << "Mean Flux: " << yMean << endl;

	cout << "Writing y" << endl;
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
	_mm_free(distRand);
	_mm_free(noiseRand);

	cout << "Computing LnLike..." << endl;
	SystemMaster.resetState();

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	double timeBegLnLike = 0.0;
	double timeEndLnLike = 0.0;
	double timeTotLnLike = 0.0;
	timeBegLnLike = dtime();
	#endif

	double LnLike = SystemMaster.computeLnLike(numObs, y, yerr, mask);

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	timeEndLnLike = dtime();
	timeTotLnLike = timeEndLnLike - timeBegLnLike;
	cout << "Time taken: " << timeTotLnLike << endl;
	#endif

	cout << "LnLike: " << LnLike << endl;
	cout << endl;
	SystemMaster.deallocDLM();

	cout << endl;
	cout << "Deleting ARMA model..." << endl;
	cout << "Program exiting...Have a nice day!" << endl; 

	_mm_free(y);
	_mm_free(yerr);
	_mm_free(ThetaMaster);
	}
