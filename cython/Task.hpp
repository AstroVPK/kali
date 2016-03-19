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
		//~Task();
		void guard();
		int getNumBurn();
		void setNumBurn(int numBurn);
		int checkParams(double *Theta, int threadNum);
		void setDT(double dt, int threadNum);
		int printSystem(double dt, double *Theta, int threadNum);
		int makeIntrinsicLC(double *Theta, int numCadences, double dt, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, double maxSigma, double minTimescale, double maxTimescale, double *t, double *x, double *y, double *yerr, double *mask, unsigned int burnSeed, unsigned int distSeed, int threadNum);
		};

#endif