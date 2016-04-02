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
		void get_ThetaVec(double *Theta, int threadNum);
		int set_System(double dt, double *Theta, int threadNum);
		void get_setSystemsVec(int *setSystems);
		//int print_System(double dt, double *Theta, int threadNum);
		int print_System(int threadNum);
		int get_A(complex<double> *A, int threadNum);
		int get_B(complex<double> *B, int threadNum);
		int get_Sigma(double *Sigma, int threadNum);
		int make_IntrinsicLC(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum);
		double get_meanFlux(double fracIntrinsicVar, int threadNum);
		int  make_ObservedLC(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int threadNum);
		int add_ObservationNoise(int numCadences, bool IR, double tolIR, double fracIntrinsicVar, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);
		double compute_LnPrior(int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double compute_LnLikelihood(int numCadences, bool IR, double tolIR, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		double compute_LnPosterior(int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int fit_CARMAModel(double dt, int numCadences, bool IR, double tolIR, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, double scatterFactor, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);
		};

#endif