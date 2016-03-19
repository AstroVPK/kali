#ifndef TASK_HPP
#define TASK_HPP

#include <complex>
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>

#include "CARMA.hpp"
//#include "MCMC.hpp"
#include "Constants.hpp"
//#include "rdrand.hpp"

using namespace std;

	class LCData {
	public:
		int numCadences;
		bool IR;
		double tolIR;
		double dt;
		double fracIntrinsicVar;
		double fracSignalToNoise;
		double maxSigma;
		double minTimescale;
		double maxTimescale;
		double *t;
		double *x;
		double *y;
		double *yerr;
		double *mask;
		LCData() = default;
		~LCData();
	};

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
		int makeIntrinsicLC(double *Theta, LCData *ptrToWorkingLC, unsigned int burnSeed, unsigned int distSeed, int threadNum);
		};


#endif