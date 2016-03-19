#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include "CARMA.hpp"
//#include "MCMC.hpp"
#include "Constants.hpp"
//#include "rdrand.hpp"
#include "Task.hpp"

using namespace std;


	Task::Task(int p, int q, int numThreads, int numBurn) {
		p = p;
		q = q;
		numThreads = numThreads;
		numBurn = numBurn;
		Systems = new CARMA[numThreads];
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].allocCARMA(p,q);
			}
		}

	Task::~Task() {
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].deallocCARMA();
			delete[] Systems;
			}
		}

	int Task::getNumBurn() {return numBurn;}
	void Task::setNumBurn(int numBurn) {numBurn = numBurn;}

	int Task::checkParams(double *Theta, int threadNum) {
		return Systems[threadNum].checkCARMAParams(Theta);
		}

	void Task::setDT(double dt, int threadNum) {
		for (int tNum = 0; tNum < numThreads; ++tNum) {
			Systems[tNum].set_dt(dt);
			}
		}

	int Task::printSystem(double dt, double *Theta, int threadNum) {
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(dt);
			Systems[threadNum].setCARMA(Theta);

			printf("A\n");
			Systems[threadNum].printA();
			printf("\n");
			printf("w\n");
			Systems[threadNum].printw();
			printf("\n");
			printf("vr\n");
			Systems[threadNum].printvr();
			printf("\n");
			printf("B\n");
			Systems[threadNum].printB();
			printf("\n");
			printf("C\n");
			Systems[threadNum].printC();
			printf("\n");

			Systems[threadNum].solveCARMA();

			printf("expw\n");
			Systems[threadNum].printexpw();
			printf("\n");
			printf("F\n");
			Systems[threadNum].printF();
			printf("\n");
			printf("Q\n");
			Systems[threadNum].printQ();
			printf("\n");
			printf("T\n");
			Systems[threadNum].printT();
			printf("\n");
			printf("Sigma\n");
			Systems[threadNum].printSigma();
			printf("\n");

			Systems[threadNum].resetState();

			printf("X\n");
			Systems[threadNum].printX();
			printf("\n");
			printf("P\n");
			Systems[threadNum].printP();
			printf("\n");

			} else {
			retVal = -1;
			}
		return retVal;
		}

	int Task::makeIntrinsicLC(double *Theta, LCData *ptrToWorkingLC, unsigned int burnSeed, unsigned int distSeed, int threadNum) {
		LCData workingLC = *ptrToWorkingLC;
		int goodYN = Systems[threadNum].checkCARMAParams(Theta), retVal = 0;
		if (goodYN == 1) {
			double maxDouble = numeric_limits<double>::max(), sqrtMaxDouble = sqrt(maxDouble);
			Systems[threadNum].set_dt(workingLC.dt);
			Systems[threadNum].setCARMA(Theta);
			Systems[threadNum].solveCARMA();
			Systems[threadNum].resetState();

			double* burnRand = static_cast<double*>(_mm_malloc(numBurn*p*sizeof(double),64));
			for (int i = 0; i < numBurn*p; i++) {
				burnRand[i] = 0.0;
				}
			Systems[threadNum].burnSystem(numBurn, burnSeed, burnRand);
			_mm_free(burnRand);
			double* distRand = static_cast<double*>(_mm_malloc(workingLC.numCadences*p*sizeof(double),64));
			for (int i = 0; i < workingLC.numCadences*p; i++) {
				distRand[i] = 0.0;
				}
			LnLikeData Data;
			Data.numCadences = workingLC.numCadences;
			Data.IR = workingLC.IR;
			Data.tolIR = workingLC.tolIR;
			Data.t = workingLC.t;
			Data.y = workingLC.x;
			Data.mask = workingLC.mask;
			LnLikeData *ptr2Data = &Data;
			Systems[threadNum].observeSystem(ptr2Data, distSeed, distRand);
			_mm_free(distRand);
			} else {
			retVal = -1;
			}
		return retVal;
		}