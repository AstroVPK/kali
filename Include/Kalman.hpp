#ifndef KALMAN_HPP
#define KALMAN_HPP

#include <mkl.h>
#include <mkl_types.h>

using namespace std;

double calcARMALnLike(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcARMALnLike(double* walkerPos, void* vdPtr2LnLikeArgs);

double calcLnLike(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnLike(double* walkerPos, void* vdPtr2LnLikeArgs);

class DLM {
public:
	int allocated;
	int isStable;
	int isInvertible;
	int isNotRedundant;
	int isReasonable;
	int p;
	int q;
	int m;
	int mSq;
	lapack_int* ilo;
	lapack_int* ihi;
	//double* ARz;
	//double* MAz;
	double* ARMatrix;
	double* MAMatrix;
	double* ARScale;
	double* MAScale;
	double* ARTau;
	double* MATau;
	double* ARwr;
	double* ARwi;
	double* MAwr;
	double* MAwi;
	double* Theta;
	double* A;
	double* Awr;
	double* Awi;
	double* Avr;
	double* Ascale;
	double* B;
	double* I;
	double* F;
	double* FKron;
	double* FKronAF;
	double* FKronR;
	double* FKronC;
	lapack_int* FKronPiv;
	double* D;
	double* Q;
	double* H;
	double* R;
	double* K;
	double* X;
	double* P;
	double* XMinus;
	double* PMinus;
	double* VScratch;
	double* MScratch;
	DLM();
	~DLM();
	void allocDLM(int numP, int numQ);
	void deallocDLM();
	void setDLM(double* Theta);
	void integrateSystem(double dt);
	void resetState(double InitUncertainty);
	void resetState();
	int checkARMAParams(double* Theta);
	void getARRoots(double*& RealAR, double*& ImagARW);
	void getMARoots(double*& RealMA, double*& ImagMA);
	void burnSystem(int numBurn, unsigned int burnSeed, double* burnRand);
	double observeSystem(double distRand, double noiseRand);
	double observeSystem(double distRand, double noiseRand, double mask);
	void observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y);
	void observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y, double* mask);
	double computeLnLike(int numPts, double* y, double* yerr);
	double computeLnLike(int numPts, double* y, double* yerr, double* mask);
	void getResiduals(int numPts, double* y, double* r);
	void fixedIntervalSmoother(int numPts, double* y, double* r, double* x);
	};

struct LnLikeData {
	int numPts;
	double* y;
	double* yerr;
	double* mask;
	};

struct LnLikeArgs {
	int numThreads;
	DLM* Systems;
	LnLikeData Data;
	}; 

void viewMatrix(int nRows, int nCols, double* mat);

double dtime();

void kron(int m, int n, double* A, int p, int q, double* B, double* C);

#endif
