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
//using namespace boost::filesystem;
using namespace nlopt;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: endToEndTest" << endl;
	cout << "Purpose: Program to test C-ARMA modelling (end-to-end test)." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institutions: University of Pennsylvania (Department of Physics & Astronomy) & Princeton University (Department of Astrophysical Sciences)" << endl;
	cout << "Email: vishal.kasliwal@gmail.com" << endl;
	cout << endl;

	string basePath;
	AcquireDirectory(cout,cin,"Path to CARMA directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "CARMA directory: " << basePath << endl;

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

	cout << "Create a test light curve with known parameters - make an ARMA light curve with p C-AR and q C-MA co-efficients." << endl;
	int pMaster = 0, qMaster = 0;
	while (pMaster < 1) {
		AcquireInput(cout,cin,"Number of C-AR coefficients p: ","Invalid value.\n",pMaster);
	}
	if (pMaster > 1) {
		qMaster = pMaster;
		while ((qMaster >= pMaster) or (qMaster < 0)) {
			cout << "The number of C-MA coefficients (q) must be less than the number of C-AR coefficients (p) if the system is to \n correspond to a C-ARMA process." << endl;
			cout << "Please select q < " << pMaster << endl;
			AcquireInput(cout,cin,"Number of C-MA coefficients q: ","Invalid value.\n",qMaster);
			}
		} else {
		qMaster = 0;
		}
	cout << "Creating C-ARMA model with " << pMaster << " C-AR components and " << qMaster << " C-MA components." << endl;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(pMaster, qMaster);
	cout << "Allocated " << SystemMaster.get_allocated() << " bytes for the model!" << endl;

	double* ThetaMaster = static_cast<double*>(_mm_malloc((pMaster+qMaster+1)*sizeof(double),64));
	#pragma omp parallel for simd default(none) shared(pMaster,qMaster,ThetaMaster)
	for (int i = 0; i < pMaster+qMaster+1; i++) {
		ThetaMaster[i] = 0.0;
		}

	double t_incr = 0.0;
	cout << "Set the sampling interval t_incr such that t_incr > 0.0" << endl;
	AcquireInput(cout,cin,"Set the value of t_incr: ","Invalid value.\n",t_incr);
	SystemMaster.set_t(t_incr);

	cout << "Set the values of the C-ARMA model parameters." << endl;

	int goodYN = 0;
	while (goodYN == 0) {

		for (int i = 0; i < pMaster; i++) {
			cout << "Set the value of a_" << i+1;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		AcquireInput(cout,cin,"Set the value of b_0: ","Invalid value.\n",ThetaMaster[pMaster]);

		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "Set the value of b_" << i-pMaster;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		cout << endl;
		cout << "Checking to see if the system is stable, invertible, not-redundant, and reasonable..." << endl;
		goodYN = SystemMaster.checkCARMAParams(ThetaMaster);

		complex<double> *CARRoots, *CMARoots;

		cout << "C-AR Polynomial Roots" << endl;
		cout.precision(4);
		SystemMaster.getCARRoots(CARRoots);
		for (int pCtr = 0; pCtr < pMaster; ++pCtr) {
			cout << showpos << fixed << "Root" << pCtr << ": " << showpos << CARRoots[pCtr] << "; Magnitude: " << abs(CARRoots[pCtr]) << endl;
			}
		cout << endl;
		cout << "C-AR Polynomial Roots" << endl;
		cout.precision(4);
		SystemMaster.getCMARoots(CMARoots);
		for (int qCtr = 0; qCtr < qMaster; ++qCtr) {
			cout << showpos << fixed << "Root" << qCtr << ": " << showpos << CMARoots[qCtr] << "; Magnitude: " << abs(CMARoots[qCtr]) << endl;
			}

		cout << noshowpos << endl;

		cout << "C-ARMA model parameters are ";
		if (goodYN == 0) {
			cout << "bad!" << endl;
			cout << "Redo!" << endl;
			cout << endl;
			} else {
			cout << "good!" << endl;
			cout << endl;
			}
		}

	cout << "System is set to use the following parameters..." << endl;
	cout.precision(4);
	for (int i = 0; i < pMaster; i++) {
		cout << noshowpos << scientific << "a_" << i+1 << ": " << ThetaMaster[i] << endl;
		}
	cout << "b_0: " << ThetaMaster[pMaster] << endl;
	for (int i = 1 + pMaster; i < 1 + pMaster + qMaster; i++) {
		cout << noshowpos << scientific << "b_" << i - pMaster << ": " << ThetaMaster[i] << endl;
		}
	cout << endl;

	SystemMaster.setCARMA(ThetaMaster);
	SystemMaster.set_t(t_incr);
	SystemMaster.solveCARMA();
	SystemMaster.resetState();

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
	SystemMaster.deallocCARMA();

	cout << endl;

	/*################################################################################################################################################*/

	cout << "Starting MCMC Phase." << endl;
	cout << endl;

	int pMax = 0, qMax = 0;
	do {
		AcquireInput(cout,cin,"Maximum number of AR coefficients to test: ","Invalid value.\n",pMax);
		} while (pMax <= 0);
	qMax = pMax - 1;
	cout << endl;

	int nwalkers = 2*(pMax+qMax), nsteps = 0;
	do {
		AcquireInput(cout,cin,"Number of walkers to use: ","Invalid value.\n",nwalkers);
		} while (nwalkers < 2*(pMax+qMax));
	do {
		AcquireInput(cout,cin,"Number of steps to take: ","Invalid value.\n",nsteps);
		} while (nsteps <= 0);

	setSeedsYN = 0;
	unsigned int zSSeed = 2229588325, walkerSeed = 3767076656, moveSeed = 2867335446, xSeed = 1413995162,initSeed = 3684614774;
	AcquireInput(cout,cin,"Supply seeds for MCMC? 1/0: ","Invalid value.\n",setSeedsYN);
	if (setSeedsYN) {
		zSSeed = 0, walkerSeed = 0, moveSeed = 0, initSeed = 0;
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Zs Seed: ","Invalid value.\n",zSSeed);
			} while (zSSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Walker choice seed: ","Invalid value.\n",walkerSeed);
			} while (walkerSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Bernoulli move seed: ","Invalid value.\n",moveSeed);
			} while (moveSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Initital position seed for NLOpt: ","Invalid value.\n",xSeed);
			} while (xSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Initital positions seed: ","Invalid value.\n",initSeed);
			} while (initSeed <= 0);
		}
	cout << endl;

	int ndims = 1;

	LnLikeData Data;
	Data.numPts = numObs;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;

	//DLM* Systems = static_cast<DLM*>(_mm_malloc(nthreads*sizeof(DLM),64));

	LnLikeArgs Args;
	Args.numThreads = nthreads;
	Args.Data = Data;
	Args.Systems = nullptr;

	void* p2Args = nullptr;

	double* initPos = nullptr;
	double* xTemp = nullptr;
	vector<double> x;
	VSLStreamStatePtr xStream, initStream;
	string myPath;
	ostringstream convertP, convertQ;
	int maxEvals = 1000;
	double xTol = 0.005;
	CARMA Systems[nthreads];

	for (int p = pMax; p > 0; --p) {
		for (int q = p-1; q >= 0; --q) {

	//for (int p = pMax; p > 1; --p) {
		//for (int q = p-1; q > 0; --q) {

			cout << endl;

			cout << "Running MCMC for p = " << p << " and q = " << q << endl;
			threadNum = omp_get_thread_num();
			//printf("testMethod - threadNum: %d\n",threadNum);

			ndims = p+q+1;

			//DLM Systems[nthreads];
			for (int tNum = 0; tNum < nthreads; tNum++) {
				//printf("testMethod - threadNum: %d; Address of Systems[%d]: %p\n",threadNum,tNum,&Systems[tNum]);
				//Systems[tNum] = DLM();
				Systems[tNum].allocCARMA(p,q);
				cout << "Allocated " << Systems[tNum].allocated << " bytes for Systems[" << tNum << "]!" << endl;
				}

			Args.Systems = Systems;

			p2Args = &Args;

			cout << "Finding test minima using NLOpt" << endl;
			xTemp = static_cast<double*>(_mm_malloc(ndims*sizeof(double),64));
			vslNewStream(&xStream, VSL_BRNG_SFMT19937, xSeed);
			bool goodPoint = false;
			do {
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream, ndims, xTemp, 0.0, 1e-3);
				if (Systems[threadNum].checkARMAParams(xTemp) == 1) {
					Systems[threadNum].setCARMA(xTemp);
					Systems[threadNum].set_t(t_incr);
					Systems[threadNum].solveCARMA();
					Systems[threadNum].resetState();
					LnLike = Systems[threadNum].computeLnLike(numObs, y, yerr, mask);
					goodPoint = true;
					} else {
					LnLike = -HUGE_VAL;
					goodPoint = false;
					}
				} while (goodPoint == false);
			vslDeleteStream(&xStream);

			x.clear();
			for (int dimCtr = 0; dimCtr < ndims; ++dimCtr) {
				x.push_back(xTemp[dimCtr]);
				}
			_mm_free(xTemp);

			opt opt(nlopt::LN_NELDERMEAD, p+q+1);
			opt.set_max_objective(calcLnLike, p2Args);
			opt.set_maxeval(maxEvals);
			opt.set_xtol_rel(xTol);
			double max_LnLike = 0.0;
			result yesno = opt.optimize(x, max_LnLike);

			cout << "NLOpt minimization done!" << endl;
			cout << "Best ARMA parameter values: ";
			for (int i = 0; i < (p+q+1); ++i) {
				cout << x[i] << " ";
				}
			cout << endl;
			cout << "Best LnLike: " << max_LnLike << endl;

			EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, zSSeed, walkerSeed, moveSeed);

			initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
			vslNewStream(&initStream, VSL_BRNG_SFMT19937, initSeed);
			//vslSkipAheadStream(initStream, nwalkers*(pMax+qMax+1)*(p*qMax+q));
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, initStream, nwalkers*ndims, initPos, 0.0, 1e-6);
			for (int walkerNum = 0; walkerNum < nwalkers; ++walkerNum) {
				#pragma omp simd
				for (int dimNum = 0; dimNum < ndims; ++dimNum) {
					initPos[walkerNum*ndims + dimNum] += x[dimNum];
					}
				}
			vslDeleteStream(&initStream);

			cout << "Running MCMC..." << endl;

			#ifdef TIME_MCMC
			#pragma omp barrier
			double timeBegMCMC = 0.0;
			double timeEndMCMC = 0.0;
			double timeTotMCMC = 0.0;
			timeBegMCMC = dtime();
			#endif
		
			newEnsemble.runMCMC(initPos);
		
			#ifdef TIME_MCMC
			#pragma omp barrier
			timeEndMCMC = dtime();
			timeTotMCMC = timeEndMCMC - timeBegMCMC;
			cout << "Time taken: " << timeTotMCMC/(60.0) << " (min)"<< endl;
			#endif

			printf("MCMC done. Writing result to ");
			convertP << p;
			convertQ << q;
			myPath = basePath + "mcmcOut_" + convertP.str() + "_" + convertQ.str() + ".dat";
			cout << myPath << endl;
			newEnsemble.writeChain(myPath,1);
			convertP.str("");
			convertQ.str("");
			printf("Result written!\n");
			fflush(0);

			//cout << "Deallocating " << allocated << " bytes from Systems..." << endl;
			for (int tNum = 0; tNum < nthreads; tNum++) {
				//printf("testMethod - threadNum: %d; Address of Systems[%d]: %p\n",threadNum,tNum,&Systems[tNum]);
				Systems[tNum].deallocCARMA();
				}
			//allocated = 0;
			//Args.Systems = nullptr;
			//p2Args = nullptr;
			_mm_free(initPos);
			//cout << endl;
			}
		}

	cout << endl;
	cout << "Deleting Systems..." << endl;
	cout << "Program exiting...Have a nice day!" << endl; 

	//_mm_free(Systems);
	_mm_free(y);
	_mm_free(yerr);
	_mm_free(ThetaMaster);
	}
