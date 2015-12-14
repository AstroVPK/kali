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
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
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
using namespace nlopt;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: fitCARMA" << endl;
	cout << "Purpose: Program to fit C-ARMA model to light curves" << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institutions: University of Pennsylvania (Department of Physics & Astronomy) & Princeton University (Department of Astrophysical Sciences)" << endl;
	cout << "Email: vishal.kasliwal@gmail.com" << endl;
	cout << endl;

	double maxDouble = numeric_limits<double>::max();
	double sqrtMaxDouble = sqrt(maxDouble);

	string basePath;
	vector<string> word(2);

	AcquireDirectory(cout,cin,"Path to working directory: ","Invalid path!\n",basePath);
	basePath += "/";

	string line, yFilePath = basePath + "y.dat";
	cout << "Input LC: " << yFilePath << endl;
	ifstream yFile;
	yFile.open(yFilePath);

	getline(yFile,line);
	istringstream record1(line);
	for (int i = 0; i < 2; ++i) {
		record1 >> word[i];
		}
	int numCadences = stoi(word[1]);
	cout << "numCadences: " << numCadences << endl;

	getline(yFile,line);
	istringstream record2(line);
	for (int i = 0; i < 2; ++i) {
		record2 >> word[i];
		}
	int numObservations = stoi(word[1]);
	cout << "numObservations: " << numObservations << endl;

	getline(yFile,line);
	istringstream record3(line);
	for (int i = 0; i < 2; ++i) {
		record3 >> word[i];
		}
	double meanFlux = stod(word[1]);
	cout << "meanFlux: " << meanFlux << endl;

	double t_incr = 0.0;
	cout << "Set the sampling interval t_incr such that t_incr > 0.0" << endl;
	AcquireInput(cout,cin,"Set the value of t_incr: ","Invalid value.\n",t_incr);

	int *cadence = static_cast<int*>(_mm_malloc(numCadences*sizeof(double),64));
	double *mask = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double *t = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double *y = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double *yerr = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	vector<string> wordNew(4);
	string lineNew;
	istringstream recordNew;
	cout.precision(16);
	int i = 0;
	while (!yFile.eof()) {
		getline(yFile,lineNew); 
		istringstream record(lineNew);
		for (int j = 0; j < 4; ++j) {
			record >> wordNew[j];
			}
		cadence[i] = stoi(wordNew[0]);
		mask[i] = stod(wordNew[1]);
		t[i] = i*t_incr;
		y[i] = stod(wordNew[2]);
		yerr[i] = stod(wordNew[3]);
		i += 1;
		} 

	/*################################################################################################################################################*/

	cout << "Starting MCMC Phase." << endl;
	cout << endl;

	int numHost = sysconf(_SC_NPROCESSORS_ONLN);
	cout << numHost << " hardware thread contexts detected." << endl;
	int nthreads = 0;
	AcquireInput(cout,cin,"Number of OmpenMP threads to use: ","Invalid value!\n",nthreads);

	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();

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

	bool setSeedsYN = 0;
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
	Data.numPts = numCadences;
	Data.IR = IR;
	Data.t = t;
	Data.y = y;
	Data.yerr = yerr;
	Data.mask = mask;

	LnLikeArgs Args;
	Args.numThreads = nthreads;
	Args.Data = Data;
	Args.Systems = nullptr;

	void* p2Args = nullptr;

	double *initPos = nullptr, *offsetArr = nullptr, *xTemp = nullptr;
	vector<double> x;
	VSLStreamStatePtr xStream, initStream;
	string myPath;
	ostringstream convertP, convertQ;
	int maxEvals = 1000;
	double xTol = 0.005;
	CARMA Systems[nthreads];

	double LnLike = 0.0;

	for (int p = pMax; p > 0; --p) {
		for (int q = p-1; q >= 0; --q) {
			cout << endl;

			cout << "Running MCMC for p = " << p << " and q = " << q << endl;
			threadNum = omp_get_thread_num();

			ndims = p+q+1;

			for (int tNum = 0; tNum < nthreads; tNum++) {
				Systems[tNum].allocCARMA(p,q);
				Systems[tNum].set_t(t_incr);
				cout << "Allocated " << Systems[tNum].get_allocated() << " bytes for Systems[" << tNum << "]!" << endl;
				}

			Args.Systems = Systems;

			p2Args = &Args;

			cout << "Finding test minima using NLOpt" << endl;
			xTemp = static_cast<double*>(_mm_malloc(ndims*sizeof(double),64));
			vslNewStream(&xStream, VSL_BRNG_SFMT19937, xSeed);
			bool goodPoint = false;
			do {
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream, ndims, xTemp, 0.0, 1e-1);
				if (Systems[threadNum].checkCARMAParams(xTemp) == 1) {
					Systems[threadNum].setCARMA(xTemp);
					Systems[threadNum].set_t(t_incr);
					Systems[threadNum].solveCARMA();
					Systems[threadNum].resetState();
					LnLike = Systems[threadNum].computeLnLike(numCadences, y, yerr, mask);
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
			cout << "Best C-ARMA parameter values: ";
			cout.precision(4);
			for (int i = 0; i < (p+q+1); ++i) {
				cout << noshowpos << scientific << x[i] << " ";
				}
			cout << endl;
			cout << "Best LnLike: " << max_LnLike << endl;

			EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, zSSeed, walkerSeed, moveSeed);

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

			for (int tNum = 0; tNum < nthreads; tNum++) {
				Systems[tNum].deallocCARMA();
				}
			_mm_free(initPos);
			_mm_free(offsetArr);
			}
		}

	cout << endl;
	cout << "Deleting Systems..." << endl;
	cout << "Program exiting...Have a nice day!" << endl; 

	_mm_free(cadence);
	_mm_free(mask);
	_mm_free(y);
	_mm_free(yerr);
	}