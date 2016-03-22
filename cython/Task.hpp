#ifndef TASK_HPP
#define TASK_HPP

#include <complex>
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>

#include "CARMA.hpp"
#include "LC.hpp"
#include "Constants.hpp"

using namespace std;

	class Task {
	private:
		int p;
		int q;
		int numThreads;
		int numBurn;
		CARMA *Systems;
	public:
		Task() = delete;
		Task(int p, int q, int numThreads, int numBurn);
		~Task();
		int getNumBurn();
		void setNumBurn(int numBurn);
		int checkParams(double *Theta, int threadNum);
		void setDT(double dt, int threadNum);
		int printSystem(double dt, double *Theta, int threadNum);
		int getSigma(double dt, double *Theta, double *Sigma, int threadNum);
		int makeIntrinsicLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum);
		double getMeanFlux(double dt, double *Theta, double fracIntrinsicVar, int threadNum);
		int  makeObservedLC(double dt, double *Theta, int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum);
		double computeLnPrior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double computeLnLikelihood(double dt, double *Theta, int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double computeLnPosterior(double dt, double *Theta, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int fitCARMA(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double scatterFactor, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);
		};

#endif