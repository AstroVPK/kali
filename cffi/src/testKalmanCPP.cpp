#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <string>
#include <iostream>
#include <fstream>
#include "Kalman.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

using namespace std;

int main() {
	cout << "Creating DLM" << endl; 
	DLM System;
	//int p = 3;
	//int q = 2;
	int p = 2;
	int q = 0;
	MKL_INT64 allocated = allocateDLM(System, p, q);
	cout << "Allocated " << allocated << " bytes for System!" << endl; 
	double* Theta = static_cast<double*>(_mm_malloc((p+q+1)*sizeof(double),64));
	/*Theta[0] = 2.0;
	Theta[1] = 1.25;
	Theta[2] = -0.3;
	Theta[3] = -0.25;
	Theta[4] = 1.15;
	Theta[5] = 0.95;*/
	Theta[0] = 2.5;
	Theta[1] = 1.25;
	Theta[2] = -0.3;
	setDLM(System, Theta);
	resetState(System);

	cout << "Checking params!" << endl;
	int good = checkARMAParams(System, Theta);
	cout << "Params good(1)/bad(0): " << good << endl;
	cout << endl;

	int numBurn = 1000, numObs = 2500;
	unsigned int burnSeed = 1311890535, distSeed = 2603023340, noiseSeed = 2410288857;
	double noiseSigma = 0.5;
	double* burnRand = static_cast<double*>(_mm_malloc(numBurn*sizeof(double),64));
	for (int i = 0; i < numBurn; i++) {
		burnRand[i] = 0.0;
		}

	double* distRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* noiseRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	for (int i = 0; i < numObs; i++) {
		distRand[i] = 0.0;
		noiseRand[i] = 0.0;
		y[i] = 0.0;
		yerr[i] = noiseSigma;
		}

	cout << "Burning..." << endl;
	burnSystem(System, numBurn, burnSeed, burnRand);

	cout << "Observing..." << endl;
	observeSystem(System, numObs, distSeed, noiseSeed, distRand, noiseRand, noiseSigma, y);

	cout << "Writing y" << endl;
	string yPath = "/home/exarkun/Desktop/y.dat";
	ofstream yFile;
	yFile.open(yPath);
	yFile.precision(16);
	for (int i = 0; i < numObs-1; i++) {
		yFile << noshowpos << scientific << y[i] << " " << yerr[i] << endl;
		}
	yFile << noshowpos << scientific << y[numObs-1] << " " << yerr[numObs-1] << endl;
	yFile.close();

	cout << "Computing LnLike..." << endl;
	resetState(System);

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	double timeBegLnLike = 0.0;
	double timeEndLnLike = 0.0;
	double timeTotLnLike = 0.0;
	timeBegLnLike = dtime();
	#endif

	double LnLike = computeLnLike(System, numObs, y, yerr);

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	timeEndLnLike = dtime();
	timeTotLnLike = timeEndLnLike - timeBegLnLike;
	cout << "Time taken: " << timeTotLnLike << endl;
	#endif

	cout << "LnLike: " << LnLike << endl;

	cout << "Deallocating " << allocated << " bytes from System..." << endl;  
	deallocateDLM(System);
	allocated = 0;
	_mm_free(burnRand);
	_mm_free(distRand);
	_mm_free(noiseRand);
	cout << endl;

	cout << "Starting MCMC" << endl;
	int nwalkers = 250, nsteps = 500, ndims = p+q+1, nthreads = 2;
	omp_set_dynamic(0);
	omp_set_num_threads(nthreads);

	LnLikeData Data;
	Data.numPts = numObs;
	Data.y = y;
	Data.yerr = yerr;

	DLM* Systems = static_cast<DLM*>(_mm_malloc(nthreads*sizeof(DLM),64));
	for (int threadNum = 0; threadNum < nthreads; threadNum++) {
		allocated += allocateDLM(Systems[threadNum],p,q);
		printf("testKalmanCPP - threadNum: %d; Address of System: %p\n",threadNum,(void*)&Systems[threadNum]);
		}
	cout << "Allocated " << allocated << " bytes for Systems!" << endl;

	LnLikeArgs Args;
	Args.numThreads = nthreads;
	Args.Data = Data;
	Args.Systems = Systems;

	void* p2Args = &Args;

	EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, 2229588325, 3767076656, 2867335446);
	double* initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
	VSLStreamStatePtr initStream;
	vslNewStream(&initStream, VSL_BRNG_SFMT19937, 3684614774);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, initStream, nwalkers*ndims, initPos, 0.0, 1e-3);

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

	cout << "MCMC done. Writing result..." << endl;
	string myPath = "/home/exarkun/Desktop/mcmcOut.dat";
	newEnsemble.writeChain(myPath,1);

	cout << "Deallocating " << allocated << " bytes from Systems..." << endl;
	for (int threadNum = 0; threadNum < nthreads; threadNum++) {
		deallocateDLM(Systems[threadNum]);
		}
	allocated = 0;
	cout << "Deleting Systems..." << endl;
	_mm_free(Systems);
	_mm_free(initPos);
	vslDeleteStream(&initStream);
	_mm_free(Theta);
	_mm_free(y);
	_mm_free(yerr);
	}