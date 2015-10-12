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
#include <sstream>
#include <iostream>
#include <fstream>
#include <nlopt.hpp>
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
	cout << "Program: plotARMARegions" << endl;
	cout << "Purpose: Program to visualize the valid parts of the ARMA parameter space." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	string basePath;
	AcquireDirectory(cout,cin,"Full path to output directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "Output directory: " << basePath << endl;

	int numHost = sysconf(_SC_NPROCESSORS_ONLN);
	cout << numHost << " hardware thread contexts detected." << endl;
	int nthreads = 0;
	AcquireInput(cout,cin,"Number of OmpenMP threads to use: ","Invalid value!\n",nthreads);
	//omp_set_dynamic(0);
	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();

	cout << "Determine valid parameter values for C-ARMA models with p C-AR and q C-MA co-efficients." << endl;

	cout << "Starting MCMC Phase." << endl;
	cout << endl;

	int pMax = 0, qMax = 0;
	do {
		AcquireInput(cout,cin,"Maximum number of C-AR coefficients to test: ","Invalid value.\n",pMax);
		} while (pMax <= 0);
	qMax = pMax - 1;
	cout << endl;

	double t_incr = 0.0;
	cout << "Set the sampling interval t_incr such that t_incr > 0.0" << endl;
	AcquireInput(cout,cin,"Set the value of t_incr: ","Invalid value.\n",t_incr);

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
	Data.numPts = 0;
	Data.y = nullptr;
	Data.yerr = nullptr;
	Data.mask = nullptr;

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
	double xTol = 0.01, LnLike = -HUGE_VAL;
	CARMA Systems[nthreads];

	for (int p = pMax; p > 0; --p) {
		for (int q = p-1; q > -1; --q) {

			cout << endl;

			cout << "Running MCMC for p = " << p << " and q = " << q << endl;
			threadNum = omp_get_thread_num();

			ndims = p+q+1;

			for (int tNum = 0; tNum < nthreads; tNum++) {
				Systems[tNum].allocCARMA(p,q);
				cout << "Allocated " << Systems[tNum].get_allocated() << " bytes for Systems[" << tNum << "]!" << endl;
				}

			Args.Systems = Systems;

			p2Args = &Args;

			cout << "Finding test minima using NLOpt" << endl;
			xTemp = static_cast<double*>(_mm_malloc(ndims*sizeof(double),64));
			vslNewStream(&xStream, VSL_BRNG_SFMT19937, xSeed);
			bool goodPoint = false;
			do {
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, xStream, ndims, xTemp, 0.0, 1e-3);
				/*Systems[threadNum].setCARMA(xTemp);
				Systems[threadNum].set_t(t_incr);
				Systems[threadNum].solveCARMA();
				Systems[threadNum].resetState();*/
				if (Systems[threadNum].checkCARMAParams(xTemp) == 1) {
					LnLike = 0.0;
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

			EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcCARMALnLike, p2Args, zSSeed, walkerSeed, moveSeed);

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
	}
