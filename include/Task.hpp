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
		bool *setSystemsVec;
		double *ThetaVec;
	public:
		Task() = delete;
		Task(int pGiven, int qGiven, int numThreadsGiven, int numBurnGiven);
		~Task();
		int reset_Task(int pGiven, int qGiven, int numBurn);
		int get_numBurn();
		void set_numBurn(int numBurn);
		int check_Theta(double *Theta, int threadNum);
		double get_dt(int threadNum);
		void get_Theta(double *Theta, int threadNum);
		int set_System(double dt, double *Theta, int threadNum);
		int reset_System(int threadNum);
		void get_setSystemsVec(int *setSystems);
		int print_System(int threadNum);
		int get_A(complex<double> *A, int threadNum);
		int get_B(complex<double> *B, int threadNum);
		int get_Sigma(double *Sigma, int threadNum);
		int get_X(double *newX, int threadNum);
		int set_X(double *newX, int threadNum);
		int get_P(double *newP, int threadNum);
		int set_P(double *newP, int threadNum);

		int make_IntrinsicLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int burnSeed, unsigned int distSeed, int threadNum);
		int extend_IntrinsicLC(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, unsigned int distSeed, int threadNum);
		double get_meanFlux(double fracIntrinsicVar, int threadNum);
		int  make_ObservedLC(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum);
		int add_ObservationNoise(int numCadences, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);
		int extend_ObservationNoise(int numCadences, int cadenceNum, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);

		double compute_LnPrior(int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double update_LnPrior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);

		double compute_LnLikelihood(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);
		double update_LnLikelihood(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);

		double compute_LnPosterior(int numCadences, int cadenceNum, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);
		double update_LnPosterior(int numCadences, int cadenceNum, double currentLnLikelihood, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, int threadNum);

		void compute_ACVF(int numLags, double *Lags, double *ACVF, int threadNum);

		int fit_CARMAModel(double dt, int numCadences, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int nwalkers, int nsteps, int maxEvals, double xTol, double mcmcA, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);

		int smooth_RTS(int numCadences, int cadenceNum, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, double *lcX, double *lcP, double *XSmooth, double *PSmooth, int threadNum);
		};

#endif