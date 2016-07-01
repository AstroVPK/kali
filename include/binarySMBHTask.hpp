#ifndef BINARYSMBHTASK_HPP
#define BINARYSMBHTASK_HPP

#include <complex>
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>

#include "binarySMBH.hpp"
#include "Constants.hpp"

using namespace std;

	class binarySMBHTask {
	private:
		int numThreads;
		binarySMBH *Systems;
		bool *setSystemsVec;
		double *ThetaVec;
	public:
		binarySMBHTask() = delete;
		binarySMBHTask(int numThreadsGiven);
		~binarySMBHTask();
		int check_Theta(double *Theta, int threadNum);
		void get_Theta(double *Theta, int threadNum);
		int set_System(double *Theta, int threadNum);
		int reset_System(double timeGiven, int threadNum);
		void get_setSystemsVec(int *setSystems);
		void print_System(int threadNum);
		double get_Period(int threadNum);

		int make_IntrinsicLC(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		//int extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum);
		//double get_meanFlux(double fracIntrinsicVar, int threadNum);
		//int  make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum);
		int add_ObservationNoise(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);
		//int extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);

		double compute_LnPrior(int numCadences, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		//double update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);

		double compute_LnLikelihood(int numCadences, int cadenceNum, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		//double update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);

		//double compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);
		//double update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);

		//void compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum);

		int fit_BinarySMBHModel(int numCadences, double lowestFlux, double highestFlux, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);

		//int smooth_Interpolate(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, int threadNum);
		};

#endif