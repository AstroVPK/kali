#ifndef CARMA_HPP
#define CARMA_HPP

#include <complex>
#include <vector>
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>

using namespace std;

double calcCARMALnPrior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcCARMALnPrior(double* walkerPos, void* vdPtr2LnLikeArgs);

double calcLnPosterior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnPosterior(double* walkerPos, void* vdPtr2LnLikeArgs);

void getSigma(int numP, int numQ, double *Theta, double *SigmaOut);

struct LnLikeData {
	int numCadences;
	int cadenceNum;
	double tolIR;
	double meanDT;
	double fracIntrinsicVar;
	double fracNoiseToSignal;
	double maxSigma;
	double minTimescale;
	double maxTimescale;
	double currentLnPrior;
	double currentLnLikelihood;
	double currentLnPosterior;
	double *t;
	double *x;
	double *y;
	double *yerr;
	double *mask;
	double *lcX;
	double *lcP;
	};

class CARMA {
private:
	int allocated;
	int isStable;
	int isInvertible;
	int isNotRedundant;
	int hasUniqueEigenValues;
	int hasPosSigma;
	int p;
	int q;
	int pSq;
	int qSq;
	double dt; // This is the last used step time to compute F, D and Q.
	// ilo, ihi and abnrm are arrays of size 1 so they can be re-used by everything. No need to make multiple copies for A, CAR and CMA
	lapack_int *ilo; // len 1
	lapack_int *ihi; // len 1
	double *abnrm; // len 1

	// Arrays used to compute expm(A dt)
	complex<double> *w; // len p
	complex<double> *expw; // len pSq
	complex<double> *CARMatrix; // len pSq
	complex<double> *CMAMatrix; // len qSq
	complex<double> *CARw; // len p
	complex<double> *CMAw; // len q
	double *scale; // len p
	complex<double> *vr; // len pSq
	complex<double> *vrInv; // len pSq
	double *rconde; // len p
	double *rcondv; // len p
	lapack_int *ipiv; // len p

	double *Theta; /*!< Theta contains p CAR parameters followed by q+1 CMA parameters, i.e. \f$\Theta = [a_{1}, a_{2}, ..., a_{p-1}, a_{p}, b_{0}, b_{1}, ..., b_{q-1}, b_{q}]\f$, where we follow the notation in Brockwell 2001, Handbook of Statistics, Vol 19.*/
	complex<double> *A;
	complex<double> *B;
	complex<double> *C;
	double *I;
	double *F;
	complex<double> *ACopy;
	complex<double> *AScratch;
	complex<double> *AScratch2;
	complex<double> *BScratch;
	double *Sigma;
	double *Q;
	double* T;
	double *H;
	double *R;
	double *K;
	double *X;
	double *P;
	double *XMinus;
	double *PMinus;
	double *VScratch;
	double *MScratch;
public:
	CARMA();
	~CARMA();
	int get_p();
	int get_q();
	double get_dt();
	void set_dt(double new_dt);
	int get_allocated();

	void printX();
	void getX(double *newX);
	void setX(double *newX);
	void printP();
	void getP(double *newP);
	void setP(double *newP);

	void printdt();
	void printA();
	const complex<double>* getA() const;
	void printvr();
	const complex<double>* getvr() const;
	void printvrInv();
	const complex<double>* getvrInv() const;
	void printw();
	const complex<double>* getw() const;
	void printexpw();
	const complex<double>* getexpw() const;
	void printB();
	const complex<double>* getB() const;
	void printC();
	const complex<double>* getC() const;
	void printF();
	const double* getF() const;
	void printSigma();
	const double* getSigma() const;
	void printQ();
	const double* getQ() const;
	void printT();
	const double* getT() const;

	void allocCARMA(int numP, int numQ);
	void deallocCARMA();
	int checkCARMAParams(double* ThetaIn); /*!< Function to check the validity of the CARMA parameters. Theta contains \f$p\f$ CAR parameters followed by \f$q+1\f$ CMA parameters, i.e. \f$\Theta = [a_{1}, a_{2}, ..., a_{p-1}, a_{p}, b_{0}, b_{1}, ..., b_{q-1}, b_{q}]\f$, where we follow the notation in Brockwell 2001, Handbook of Statistics, Vol 19.*/
	void setCARMA(double* ThetaIn); /*!< Function to set a CARMA object with the given CARMA parameters. Theta contains p CAR parameters followed by q+1 CMA parameters, i.e. \f$\Theta = [a_{1}, a_{2}, ..., a_{p-1}, a_{p}, b_{0}, b_{1}, ..., b_{q-1}, b_{q}]\f$, where we follow the notation in Brockwell 2001, Handbook of Statistics, Vol 19.*/
	void solveCARMA();
	void resetState(double InitUncertainty);
	void resetState();
	void getCARRoots(complex<double>*& CARoots);
	void getCMARoots(complex<double>*& CMARoots);
	double getIntrinsicVar();

	void burnSystem(int numBurn, unsigned int burnSeed, double* burnRand);
	void simulateSystem(LnLikeData *ptr2LnLikeData, unsigned int distSeed, double *distRand);
	void extendSystem(LnLikeData *ptr2Data, unsigned int distSeed, double *distRand);
	double getMeanFlux(LnLikeData *ptr2Data);
	void observeNoise(LnLikeData *ptr2LnLikeData, unsigned int noiseSeed, double* noiseRand);
	void extendObserveNoise(LnLikeData *ptr2Data, unsigned int noiseSeed, double* noiseRand);
	double computeLnLikelihood(LnLikeData *ptr2LnLikeData);
	double updateLnLikelihood(LnLikeData *ptr2LnLikeData);
	double computeLnPrior(LnLikeData *ptr2LnLikeData);
	void computeACVF(int numLags, double *Lags, double* ACVF);
	int RTSSmoother(LnLikeData *ptr2Data, double *XSmooth, double *PSmooth);
	};

struct LnLikeArgs {
	int numThreads;
	CARMA *Systems;
	LnLikeData *Data;
	};

void zeroMatrix(int nRows, int nCols, int* mat);

void zeroMatrix(int nRows, int nCols, double* mat);

void zeroMatrix(int nRows, int nCols, complex<double>* mat);

void viewMatrix(int nRows, int nCols, int* mat);

void viewMatrix(int nRows, int nCols, double* mat);

void viewMatrix(int nRows, int nCols, vector<double> mat);

void viewMatrix(int nRows, int nCols, complex<double>* mat);

double dtime();

void kron(int m, int n, double* A, int p, int q, double* B, double* C);

#endif
