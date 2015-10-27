#include <malloc.h>
#include <sys/time.h>
#include <limits>
#include <mathimf.h>
#include <omp.h>
#include <complex>
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "boost/numeric/odeint.hpp"
#include "Constants.hpp"
#include "CARMA.hpp"

//#define TIMEALL
//#define TIMEPER
//#define TIMEFINE
//#define DEBUG
//#define DEBUG_LNLIKE
//#define DEBUG_FUNC
//#define DEBUG_CHECKARMAPARAMS
//#define DEBUG_SETCARMA
//#define DEBUG_SOLVECARMA
//#define DEBUG_FUNCTOR
//#define DEBUG_SETCARMA_DEEP
//#define DEBUG_BURNSYSTEM
//#define WRITE_BURNSYSTEM
//#define DEBUG_OBSSYSTEM
//#define DEBUG_CTORCARMA
//#define DEBUG_DTORCARMA
//#define DEBUG_ALLOCATECARMA
//#define DEBUG_DEALLOCATECARMA
//#define DEBUG_DEALLOCATECARMA_DEEP
//#define DEBUG_RESETSTATE
//#define DEBUG_CALCLNLIKE

#ifdef WRITE
#include <fstream>
#endif

using namespace std;

double calcCARMALnLike(const vector<double> &x, vector<double>& grad, void* p2Args) {
	/*! Used for computing good regions */
	if (!grad.empty()) {
		#pragma omp simd
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	CARMA* Systems = Args.Systems;

	int numPts = Data.numPts;
	double LnLike = 0;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	#ifdef DEBUG_CALCLNLIKE2
	printf("calcLnLike - threadNum: %d; Location: ",threadNum);
	#endif

	if (Systems[threadNum].checkCARMAParams(const_cast<double*>(&x[0])) == 1) {
		LnLike = 0.0;
		} else {
		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_CALCLNLIKE2
	printf("LnLike: %f\n",LnLike);
	fflush(0);
	#endif

	return LnLike;

	}

double calcCARMALnLike(double* walkerPos, void* func_args) {
	/*! Used for computing good regions */

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	CARMA* Systems = Args.Systems;

	int numPts = Data.numPts;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	double LnLike = 0.0;

	if (Systems[threadNum].checkCARMAParams(walkerPos) == 1) {

		#ifdef DEBUG_FUNC2
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; Systems[threadNum].get_p() + Systems[threadNum].get_q() + 1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System good!\n",threadNum);
		#endif

		LnLike = 0.0;
		} else {

		#ifdef DEBUG_FUNC2
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].get_p() + Systems[threadNum].get_q() + 1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System bad!\n",threadNum);
		#endif

		LnLike = -HUGE_VAL;
		}
	return LnLike;

	}

double calcLnLike(const vector<double> &x, vector<double>& grad, void* p2Args) {
	if (!grad.empty()) {
		#pragma omp simd
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	CARMA* Systems = Args.Systems;

	int numPts = Data.numPts;
	double LnLike = 0;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	#ifdef DEBUG_CALCLNLIKE
	printf("calcLnLike - threadNum: %d",threadNum);
	#endif

	if (Systems[threadNum].checkCARMAParams(const_cast<double*>(&x[0])) == 1) {
		Systems[threadNum].setCARMA(const_cast<double*>(&x[0]));
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		LnLike = Systems[threadNum].computeLnLike(numPts, y, yerr, mask);
		} else {
		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_CALCLNLIKE
	#pragma omp critical
	{
	printf("calcLnLike - threadNum: %d; t: %f\n",threadNum,Systems[threadNum].get_t());
	printf("calcLnLike - threadNum: %d; F\n",threadNum);
	Systems[threadNum].printF();
	printf("\n");
	printf("calcLnLike - threadNum: %d; D\n",threadNum);
	Systems[threadNum].printD();
	printf("\n");
	printf("calcLnLike - threadNum: %d; Q\n",threadNum);
	Systems[threadNum].printQ();
	printf("\n");
	printf("calcLnLike - threadNum: %d; LnLike: %f\n",threadNum,LnLike);
	printf("\n");
	fflush(0);
	}
	#endif

	return LnLike;

	}

double calcLnLike(double* walkerPos, void* func_args) {

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	CARMA* Systems = Args.Systems;

	int numPts = Data.numPts;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	double LnLike = 0.0;

	if (Systems[threadNum].checkCARMAParams(walkerPos) == 1) {

		#ifdef DEBUG_FUNC
		#pragma omp critical
		{
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].get_p() + Systems[threadNum].get_q() + 1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System good!\n",threadNum);
		}
		#endif

		Systems[threadNum].setCARMA(walkerPos);
		Systems[threadNum].solveCARMA();
		Systems[threadNum].resetState();
		LnLike = Systems[threadNum].computeLnLike(numPts, y, yerr, mask);
		} else {

		#ifdef DEBUG_FUNC
		#pragma omp critical
		{
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].get_p() + Systems[threadNum].get_q() + 1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System bad!\n",threadNum);
		}
		#endif

		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_FUNC
	#pragma omp critical
	{
	printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < Systems[threadNum].get_p() + Systems[threadNum].get_q() + 1; dimNum++) {
		printf("%f ",walkerPos[dimNum]);
		}
	printf("\n");
	printf("calcLnLike - threadNum: %d; t: %f\n",threadNum,Systems[threadNum].get_t());
	printf("calcLnLike - threadNum: %d; F\n",threadNum);
	Systems[threadNum].printF();
	printf("\n");
	printf("calcLnLike - threadNum: %d; D\n",threadNum);
	Systems[threadNum].printD();
	printf("\n");
	printf("calcLnLike - threadNum: %d; Q\n",threadNum);
	Systems[threadNum].printQ();
	printf("\n");
	printf("calcLnLike - threadNum: %d; LnLike: %f\n",threadNum,LnLike);
	printf("\n");
	fflush(0);
	}
	#endif

	return LnLike;
	}

void zeroMatrix(int nRows, int nCols, int* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		#pragma omp simd
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0;
			}
		}
	}

void zeroMatrix(int nRows, int nCols, double* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		#pragma omp simd
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0.0;
			}
		}
	}

void zeroMatrix(int nRows, int nCols, complex<double>* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		#pragma omp simd
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0.0;
			}
		}
	}

void viewMatrix(int nRows, int nCols, int* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%+d ",mat[j*nCols + i]);
			}
		printf("\n");
		}
	}

void viewMatrix(int nRows, int nCols, double* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%+E ",mat[j*nCols + i]);
			}
		printf("\n");
		}
	}

void viewMatrix(int nRows, int nCols, complex<double>* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%+E%+Ei ",mat[j*nCols + i].real(),mat[j*nCols + i].imag());
			}
		printf("\n");
		}
	}

double dtime() {
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,(struct timezone*)0);
	tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
	return( tseconds );
	}

void kron(int m, int n, double* A, int p, int q, double* B, double* C) {
	int alpha = 0;
	int beta = 0;
	int mp = m*p;
	int nq = n*q;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < p; k++) {
				#pragma omp simd
				for (int l = 0; l < q; l++) {
					alpha = p*i + k;
					beta = q*j + l;
					C[alpha + beta*nq] = A[i + j*n]*B[k + l*q];
					}
				}
			}
		}
	}

void expm(double xi, double* out) {
	/*#pragma omp simd
	for (int i = 0; i < p; ++i) {
		expw[i + i*p] = exp(dt*w[i]);
		}

	complex<double> alpha = 1.0+0.0i, beta = 0.0+0.0i;
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, vr, p, expw, p, &beta, A, p);
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, A, p, vrInv, p, &beta, AScratch, p);

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			out[rowCtr + colCtr*p] = FScratch[rowCtr + colCtr*p];
			}
		}*/
	}

CARMA::CARMA() {
	/*! Object that holds data and methods for performing C-ARMA analysis. DLM objects hold pointers to blocks of data that are set as required based on the size of the C-ARMA model.*/
	#ifdef DEBUG_CTORDLM
	int threadNum = omp_get_thread_num();
	printf("DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	allocated = 0;
	isStable = 1;
	isInvertible = 1;
	isNotRedundant = 1;
	hasUniqueEigenValues = 1;
	hasPosSigma = 1;
	p = 0;
	q = 0;
	pSq = 0;
	t = 0.0;

	ilo = nullptr; // len 1
	ihi = nullptr; // len 1
	abnrm = nullptr; // len 1

	// Arrays used to compute expm(A dt)
	w = nullptr; // len p
	expw = nullptr; // len pSq
	CARw = nullptr; // len p
	CMAw = nullptr; //len p
	scale = nullptr;
	vr = nullptr;
	vrInv = nullptr;
	rconde = nullptr;
	rcondv = nullptr;
	ipiv = nullptr;

	Theta = nullptr;
	A = nullptr;
	AScratch = nullptr;
	B = nullptr;
	BScratch = nullptr;
	I = nullptr;
	F = nullptr;
	/*FKron = nullptr;
	FKron_af = nullptr;
	FKron_r = nullptr;
	FKron_c = nullptr;
	FKron_ipiv = nullptr;
	FKron_rcond = nullptr;
	FKron_rpvgrw = nullptr;
	FKron_berr = nullptr;
	FKron_err_bnds_norm = nullptr;
	FKron_err_bnds_comp = nullptr;*/
	Sigma = nullptr;
	D = nullptr;
	Q = nullptr;
	H = nullptr;
	R = nullptr;
	K = nullptr;
	X = nullptr;
	P = nullptr;
	XMinus = nullptr;
	PMinus = nullptr;
	VScratch = nullptr;
	MScratch = nullptr;

	#ifdef DEBUG_CTORDLM
	printf("DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	}

CARMA::~CARMA() {

	#ifdef DEBUG_DTORDLM
	int threadNum = omp_get_thread_num();
	printf("~DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	allocated = 0;
	isStable = 1;
	isInvertible = 1;
	isNotRedundant = 1;
	hasUniqueEigenValues = 1;
	hasPosSigma = 1;
	p = 0;
	q = 0;
	pSq = 0;
	t = 0.0;

	ilo = nullptr;
	ihi = nullptr;
	abnrm = nullptr;
	w = nullptr;
	expw = nullptr;
	CARw = nullptr;
	CMAw = nullptr;
	scale = nullptr;
	vr = nullptr;
	vrInv = nullptr;
	rconde = nullptr;
	rcondv = nullptr;
	ipiv = nullptr;

	Theta = nullptr;
	A = nullptr;
	ACopy = nullptr;
	AScratch = nullptr;
	B = nullptr;
	BScratch = nullptr;
	I = nullptr;
	F = nullptr;
	/*FKron = nullptr;
	FKron_af = nullptr;
	FKron_r = nullptr;
	FKron_c = nullptr;
	FKron_ipiv = nullptr;
	FKron_rcond = nullptr;
	FKron_rpvgrw = nullptr;
	FKron_berr = nullptr;
	FKron_err_bnds_norm = nullptr;
	FKron_err_bnds_comp = nullptr;*/
	D = nullptr;
	Sigma = nullptr;
	Q = nullptr;
	H = nullptr;
	R = nullptr;
	K = nullptr;
	X = nullptr;
	P = nullptr;
	XMinus = nullptr;
	PMinus = nullptr;
	VScratch = nullptr;
	MScratch = nullptr;

	#ifdef DEBUG_DTORDLM
	printf("~DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	}

void CARMA::allocCARMA(int numP, int numQ) {

	#ifdef DEBUG_ALLOCATEDLM
	int threadNum = omp_get_thread_num();
	printf("allocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	if ((numQ >= numP) or (numQ < 0)) {
		printf("FATAL LOGIC ERROR: numP MUST BE > numQ >= 0!\n");
		exit(1);
		}
	p = numP;
	q = numQ;
	allocated = 0;
	pSq = p*p;
	qSq = q*q;

	ilo = static_cast<lapack_int*>(_mm_malloc(1*sizeof(lapack_int),64));
	ihi = static_cast<lapack_int*>(_mm_malloc(1*sizeof(lapack_int),64));
	allocated += 2*sizeof(lapack_int);

	ilo[0] = 0;
	ihi[0] = 0;

	abnrm = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	allocated += sizeof(double);

	abnrm[0] = 0.0;

	ipiv = static_cast<lapack_int*>(_mm_malloc(p*sizeof(lapack_int),64));
	allocated += p*sizeof(lapack_int);

	scale = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	rconde = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	rcondv = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	allocated += 3*p*sizeof(double);

	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		ipiv[rowCtr] = 0;
		scale[rowCtr] = 0.0;
		rconde[rowCtr] = 0.0;
		rcondv[rowCtr] = 0.0;
		}

	w = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	CARw = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	B = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	BScratch = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	allocated += 4*p*sizeof(complex<double>);

	if (q > 0) {
		CMAw = static_cast<complex<double>*>(_mm_malloc(q*sizeof(complex<double>),64));
		CMAMatrix = static_cast<complex<double>*>(_mm_malloc(qSq*sizeof(complex<double>),64));
		allocated += q*sizeof(complex<double>);
		allocated += qSq*sizeof(complex<double>);

		for (int colCtr = 0; colCtr < q; ++colCtr) {
			CMAw[colCtr] = 0.0+0.0i;
			#pragma omp simd
			for (int rowCtr = 0; rowCtr < q; ++rowCtr) {
				CMAMatrix[rowCtr + colCtr*q] = 0.0+0.0i;
				}
			}
		}

	CARMatrix = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	expw = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	vr = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	vrInv = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	A = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	ACopy = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	AScratch = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	allocated += 7*pSq*sizeof(double);

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		w[colCtr] = 0.0+0.0i;
		CARw[colCtr] = 0.0+0.0i;
		B[colCtr] = 0.0;
		BScratch[colCtr] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			CARMatrix[rowCtr + colCtr*p] = 0.0+0.0i;
			expw[rowCtr + colCtr*p] = 0.0+0.0i;
			vr[rowCtr + colCtr*p] = 0.0+0.0i;
			vrInv[rowCtr + colCtr*p] = 0.0+0.0i;
			A[rowCtr + colCtr*p] = 0.0+0.0i;
			ACopy[rowCtr + colCtr*p] = 0.0+0.0i;
			AScratch[rowCtr + colCtr*p] = 0.0+0.0i;
			}
		}

	/*FKron_rcond = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	FKron_rpvgrw = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	FKron_berr = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	FKron_err_bnds_norm = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	FKron_err_bnds_comp = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	allocated += 5*sizeof(double);

	FKron_r = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	FKron_c = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	allocated += pSq*sizeof(double);

	FKron_ipiv = static_cast<lapack_int*>(_mm_malloc(pSq*pSq*sizeof(lapack_int),64));
	allocated += pSq*pSq*sizeof(lapack_int);

	FKron = static_cast<double*>(_mm_malloc(pSq*pSq*sizeof(double),64));
	FKron_af = static_cast<double*>(_mm_malloc(pSq*pSq*sizeof(double),64));
	allocated += pSq*pSq*sizeof(double);

	FKron_rcond[0] = 0.0;
	FKron_rpvgrw[0] = 0.0;
	FKron_berr[0] = 0.0;
	FKron_err_bnds_norm[0] = 0.0;
	FKron_err_bnds_comp[0] = 0.0;

	for (int colCtr = 0; colCtr < pSq; ++colCtr) {
		FKron_r[colCtr] = 0.0;
		FKron_c[colCtr] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < pSq; ++rowCtr) {
			FKron_ipiv[rowCtr + colCtr*pSq] = 0;
			FKron[rowCtr + colCtr*pSq] = 0.0;
			FKron_af[rowCtr + colCtr*pSq] = 0.0;
			}
		}*/

	Theta = static_cast<double*>(_mm_malloc((p + q + 1)*sizeof(double),64));
	allocated += (p+q+1)*sizeof(double);

	for (int rowCtr = 0; rowCtr < p + q + 1; ++rowCtr) {
		Theta[rowCtr] = 0.0;
		}

	D = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	H = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	K = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	X = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	XMinus = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	VScratch = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	allocated += 6*p*sizeof(double);

	I = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	F = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	Sigma = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	Q = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	P = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	PMinus = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	MScratch = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	allocated += 7*pSq*sizeof(double);

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		D[colCtr] = 0.0;
		H[colCtr] = 0.0;
		K[colCtr] = 0.0;
		X[colCtr] = 0.0;
		XMinus[colCtr] = 0.0;
		VScratch[colCtr] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			I[rowCtr + colCtr*p] = 0.0;
			F[rowCtr + colCtr*p] = 0.0;
			Sigma[rowCtr + colCtr*p] = 0.0;
			Q[rowCtr + colCtr*p] = 0.0;
			P[rowCtr + colCtr*p] = 0.0;
			PMinus[rowCtr + colCtr*p] = 0.0;
			MScratch[rowCtr + colCtr*p] = 0.0;
			}
		}

	R = static_cast<double*>(_mm_malloc(sizeof(double),64));
	allocated += sizeof(double);

	R[0] = 0.0;

	#pragma omp simd
	for (int i = 1; i < p; ++i) {
		A[i*p + (i - 1)] = 1.0;
		I[(i - 1)*p + (i - 1)] = 1.0;
		}
	I[(p - 1)*p + (p - 1)] = 1.0;

	#ifdef DEBUG_ALLOCATEDLM
	printf("allocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	}

void CARMA::deallocCARMA() {

	#ifdef DEBUG_DEALLOCATEDLM
	int threadNum = omp_get_thread_num();
	printf("deallocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	if (ilo) {
		_mm_free(ilo);
		ilo = nullptr;
		}

	if (ihi) {
		_mm_free(ihi);
		ihi = nullptr;
		}

	if (abnrm) {
		_mm_free(abnrm);
		abnrm = nullptr;
		}

	if (w) {
		_mm_free(w);
		w = nullptr;
	}

	if (expw) {
		_mm_free(expw);
		expw = nullptr;
		}

	if (CARMatrix) {
		_mm_free(CARMatrix);
		CARMatrix = nullptr;
		}

	if (CMAMatrix) {
		_mm_free(CMAMatrix);
		CMAMatrix = nullptr;
		}

	if (CARw) {
		_mm_free(CARw);
		CARw = nullptr;
		}

	if (CMAw) {
		_mm_free(CMAw);
		CMAw = nullptr;
		}

	if (scale) {
		_mm_free(scale);
		scale = nullptr;
		}

	if (vr) {
		_mm_free(vr);
		vr = nullptr;
		}

	if (vrInv) {
		_mm_free(vrInv);
		vrInv = nullptr;
		}

	if (rconde) {
		_mm_free(rconde);
		rconde = nullptr;
		}

	if (rcondv) {
		_mm_free(rcondv);
		rcondv = nullptr;
		}

	if (ipiv) {
		_mm_free(ipiv);
		ipiv = nullptr;
		}

	if (A) {
		_mm_free(A);
		A = nullptr;
		}

	if (ACopy) {
		_mm_free(ACopy);
		ACopy = nullptr;
		}

	if (AScratch) {
		_mm_free(AScratch);
		AScratch = nullptr;
		}

	if (B) {
		_mm_free(B);
		B = nullptr;
		}

	if (BScratch) {
		_mm_free(BScratch);
		BScratch = nullptr;
		}

	if (I) {
		_mm_free(I);
		I = nullptr;
		}

	if (F) {
		_mm_free(F);
		F = nullptr;
		}

/*	if (FKron) {
		_mm_free(FKron);
		FKron = nullptr;
		}

	if (FKron_af) {
		_mm_free(FKron_af);
		FKron_af = nullptr;
		}

	if (FKron_r) {
		_mm_free(FKron_r);
		FKron_r = nullptr;
		}

	if (FKron_c) {
		_mm_free(FKron_c);
		FKron_c = nullptr;
		}

	if (FKron_ipiv) {
		_mm_free(FKron_ipiv);
		FKron_ipiv = nullptr;
		}

	if (FKron_rcond) {
		_mm_free(FKron_rcond);
		FKron_rcond = nullptr;
		}

	if (FKron_rpvgrw) {
		_mm_free(FKron_rpvgrw);
		FKron_rpvgrw = nullptr;
		}

	if (FKron_berr) {
		_mm_free(FKron_berr);
		FKron_berr = nullptr;
		}

	if (FKron_err_bnds_norm) {
		_mm_free(FKron_err_bnds_norm);
		FKron_err_bnds_norm = nullptr;
		}

	if (FKron_err_bnds_comp) {
		_mm_free(FKron_err_bnds_comp);
		FKron_err_bnds_comp = nullptr;
		}*/

	if (Theta) {
		_mm_free(Theta);
		Theta = nullptr;
		}

	if (Sigma) {
		_mm_free(Sigma);
		Sigma = nullptr;
		}

	if (D) {
		_mm_free(D);
		D = nullptr;
		}

	if (Q) {
		_mm_free(Q);
		Q = nullptr;
		}

	if (H) {
		_mm_free(H);
		H = nullptr;
		}

	if (R) {
		_mm_free(R);
		R = nullptr;
		}

	if (K) {
		_mm_free(K);
		K = nullptr;
		}

	if (X) {
		_mm_free(X);
		X = nullptr;
		}

	if (P) {
		_mm_free(P);
		P = nullptr;
		}

	if (XMinus) {
		_mm_free(XMinus);
		XMinus = nullptr;
		}

	if (PMinus) {
		_mm_free(PMinus);
		PMinus = nullptr;
		}

	if (VScratch) {
		_mm_free(VScratch);
		VScratch = nullptr;
		}

	if (MScratch) {
		_mm_free(MScratch);
		MScratch = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM
	printf("deallocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif
	}

int CARMA::get_p() {
	return p;
	}

int CARMA::get_q() {
	return q;
	}

double CARMA::get_t() {
	return t;
	}

void CARMA::set_t(double t_incr) {
	t = t_incr;
	}

int CARMA::get_allocated() {
	return allocated;
	}

void CARMA::getCARRoots(complex<double>*& CARRoots) {
	CARRoots = CARw;
	}

void CARMA::getCMARoots(complex<double>*& CMARoots) {
	CMARoots = CMAw;
	}

void CARMA::printX() {
	viewMatrix(p,1,X);
	}

const double* CARMA::getX() const {
	return X;
	}

void CARMA::printP() {
	viewMatrix(p,p,P);
	}

const double* CARMA::getP() const {
	return P;
	}

void CARMA::printA() {
	viewMatrix(p,p,A);
	}

const complex<double>* CARMA::getA() const {
	return A;
	}

void CARMA::printB() {
	viewMatrix(p,1,B);
	}

const complex<double>* CARMA::getB() const {
	return B;
	}

void CARMA::printF() {
	viewMatrix(p,p,F);
	}

const double* CARMA::getF() const {
	return F;
	}

void CARMA::printD() {
	viewMatrix(p,1,D);
	}

const double* CARMA::getD() const {
	return D;
	}

void CARMA::printQ() {
	viewMatrix(p,p,Q);
	}

const double* CARMA::getQ() const {
	return Q;
	}

void CARMA::printSigma() {
	viewMatrix(p,p,Sigma);
	}

const double* CARMA::getSigma() const {
	return Sigma;
	}

/*!
 * Checks the validity of the supplied C-ARMA parameters.
 * @param[in]  Theta  \f$\Theta\f$ contains \f$p\f$ CAR parameters followed by \f$q+1\f$ CMA parameters, i.e. \f$\Theta = [a_{1}, a_{2}, ..., a_{p-1}, a_{p}, b_{0}, b_{1}, ..., b_{q-1}, b_{q}]\f$, where we follow the notation in Brockwell 2001, Handbook of Statistics, Vol 19
 */
int CARMA::checkCARMAParams(double *ThetaIn /**< [in]  */) {
	/*!< \brief Function to check the validity of the CARMA parameters.


	*/

	#ifdef DEBUG_CHECKARMAPARAMS
	int threadNum = omp_get_thread_num();
	printf("checkCARMAParams - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	isStable = 1;
	isInvertible = 1;
	isNotRedundant = 1;
	hasUniqueEigenValues = 1;
	hasPosSigma = 1;

	if (ThetaIn[p] <= 0.0) {
		hasPosSigma = 0;
		}

	for (int rowCtr = 0; rowCtr < p; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < p; colCtr++) {
			CARMatrix[rowCtr + p*colCtr] = 0.0 + 0.0i; // Reset matrix.
			}
		}

	CARMatrix[p*(p-1)] = -1.0*ThetaIn[p-1] + 0.0i; // The first row has no 1s so we just set the rightmost entry equal to -alpha_p
	#pragma omp simd
	for (int rowCtr = 1; rowCtr < p; rowCtr++) {
		CARMatrix[rowCtr+(p-1)*p] = -1.0*ThetaIn[p - 1 - rowCtr] + 0.0i; // Rightmost column of CARMatrix equals -alpha_k where 1 < k < p.
		CARMatrix[rowCtr+(rowCtr-1)*p] = 1.0; // CARMatrix has Identity matrix in bottom left.
		}
	ilo[0] = 0;
	ihi[0] = 0;
	abnrm[0] = 0.0;
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		CARw[rowCtr] = 0.0 + 0.0i;
		scale[rowCtr] = 0.0;
		rconde[rowCtr] = 0.0;
		rcondv[rowCtr] = 0.0;
		}
	#ifdef DEBUG_CHECKARMAPARAMS
	printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("checkCARMAParams - threadNum: %d; CARMatrix\n",threadNum);
	viewMatrix(p,p,CARMatrix);
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	lapack_int YesNo;
	//YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'N', 'N', p, CARMatrix, p, CARw, vrInv, p, vr, p, ilo, ihi, scale, abnrm, rconde, rcondv); // NOT WORKING!!!
	YesNo = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', p, CARMatrix, p, CARw, vrInv, p, vr, p);
	//YesNo = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', p, CARMatrix, p, CARw, vrInv, p, vr, p);

	for (int i = 0; i < p; i++) {

		#ifdef DEBUG_CHECKARMAPARAMS
		printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
			printf("%f ",Theta[dimNum]);
			}
		printf("\n");
		printf("checkCARMAParams - threadNum: %d; Root: %+f%+fi; Len: %f\n",threadNum, CARw[i].real(), CARw[i].imag(), abs(CARw[i]));
		#endif

		if (CARw[i].real() >= 0.0) {

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkCARMAParams - threadNum: %d; badRoot!!!: %+f%+fi; Len: %f\n",threadNum,CARw[i].real(), CARw[i].imag(),abs(CARw[i]));
			#endif

			isStable = 0;
			}

		for (int j = i+1; j < p; j++) { // Only need to check e-values against each other once.
			if (CARw[i] == CARw[j]) {
				hasUniqueEigenValues = 0;
				}
			}
		}

	if (q > 0) {
		for (int rowCtr = 0; rowCtr < q; ++rowCtr) {
			#pragma omp simd
			for (int colCtr = 0; colCtr < q; colCtr++) {
				CMAMatrix[rowCtr + q*colCtr] = 0.0; // Initialize matrix.
				}
			}
		CMAMatrix[(q-1)*q] = -1.0*ThetaIn[p]/ThetaIn[p + q]; // MAMatrix has -beta_q/-beta_0 at top right!
		#pragma omp simd
		for (int rowCtr = 1; rowCtr < q; ++rowCtr) {
			CMAMatrix[rowCtr + (q - 1)*q] = -1.0*ThetaIn[p + rowCtr]/ThetaIn[p + q]; // Rightmost column of MAMatrix has -MA coeffs.
			CMAMatrix[rowCtr + (rowCtr - 1)*q] = 1.0; // MAMatrix has Identity matrix in bottom left.
			}
		ilo[0] = 0;
		ihi[0] = 0;
		abnrm[0] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < q; ++rowCtr) {
			CMAw[rowCtr] = 0.0+0.0i;
			}
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			scale[rowCtr] = 0.0;
			rconde[rowCtr] = 0.0;
			rcondv[rowCtr] = 0.0;
			}
		#ifdef DEBUG_CHECKARMAPARAMS
		printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
			printf("%f ",Theta[dimNum]);
			}
		printf("\n");
		printf("checkCARMAParams - threadNum: %d; CMAMatrix\n",threadNum);
		viewMatrix(q,q,CMAMatrix);
		#endif

		//YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'V', 'N', q, CMAMatrix, q, CMAw, nullptr, 1, vr, q, ilo, ihi, scale, abnrm, rconde, rcondv); // NOT WORKING!!!!
		YesNo = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', q, CMAMatrix, q, CMAw, vrInv, q, vr, q);
		//YesNo = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', q, CMAMatrix, q, CMAw, vrInv, p, vr, p);

		for (int i = 0; i < q; i++) {

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkCARMAParams - threadNum: %d; Root: %+f%+fi; Len: %f\n",threadNum, CMAw[i].real(), CMAw[i].imag(), abs(CMAw[i]));
			#endif

			if (CMAw[i].real() > 0.0) {

				#ifdef DEBUG_CHECKARMAPARAMS
				printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
				for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
					printf("%f ",Theta[dimNum]);
					}
				printf("\n");
				printf("checkCARMAParams - threadNum: %d; badRoot!!!: %+f%+fi; Len: %f\n",threadNum,CMAw[i].real(), CMAw[i].imag(),abs(CMAw[i]));
				#endif

				isInvertible = 0;
				}
			}

		for (int i = 1; i < p; i++) {
			for (int j = 1; j < q; j++) {
				if (CARw[i] == CMAw[j]) {
					isNotRedundant = 0;
					}
				}
			}
		} else if (q == 0) {
		if (Theta[p] < 0) {
			isInvertible = 0;
			}
		} else {
		printf("FATAL LOGIC ERROR: numP MUST BE > numQ >= 0!\n");
		exit(1);
		}

	#ifdef DEBUG_CHECKARMAPARAMS
	printf("checkCARMAParams - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("checkCARMAParams - threadNum: %d; isStable: %d\n",threadNum,isStable);
	printf("checkCARMAParams - threadNum: %d; isInvertible: %d\n",threadNum,isInvertible);
	printf("checkCARMAParams - threadNum: %d; isNotRedundant: %d\n",threadNum,isNotRedundant);
	printf("checkCARMAParams - threadNum: %d; hasUniqueEigenValues: %d\n",threadNum,hasUniqueEigenValues);
	printf("checkCARMAParams - threadNum: %d; hasPosSigma: %d\n",threadNum,hasPosSigma);
	printf("\n");
	printf("\n");
	#endif

	return isStable*isInvertible*isNotRedundant*hasUniqueEigenValues*hasPosSigma;
	}

void CARMA::setCARMA(double *ThetaIn) {

	#ifdef DEBUG_SETCARMA
	int threadNum = omp_get_thread_num();
	printf("setCARMA - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p + q + 1; ++rowCtr) {
		Theta[rowCtr] = ThetaIn[rowCtr];
		}

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			A[rowCtr + colCtr*p] = 0.0;
			}
		}

	A[0] = -1.0*Theta[0];
	#pragma omp simd
	for (int i = 1; i < p; ++i) {
		A[i] = -1.0*Theta[i];
		A[i*p + (i - 1)] = 1.0;
		}

	cblas_zcopy(pSq, A, 1, ACopy, 1); // Copy A into ACopy so that we can keep a clean working version of it.

	#ifdef DEBUG_SETCARMA
	printf("setCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setCARMA - threadNum: %d; A\n",threadNum);
	viewMatrix(p,p,A);
	printf("\n");
	printf("setCARMA - threadNum: %d; ACopy\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	ilo[0] = 0;
	ihi[0] = 0;
	abnrm[0] = 0.0;
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		w[rowCtr] = 0.0+0.0i;
		scale[rowCtr] = 0.0;
		rconde[rowCtr] = 0.0;
		rcondv[rowCtr] = 0.0;
		#pragma omp simd
		for (int colCtr = 0; colCtr < p; ++colCtr) {
			vr[rowCtr + colCtr*p] = 0.0+0.0i;
			vrInv[rowCtr + colCtr*p] = 0.0+0.0i;
			}
		}

	lapack_int YesNo;
	//YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'V', 'N', p, A, p, w, nullptr, 1, vr, p, ilo, ihi, scale, abnrm, rconde, rcondv); // NOT WORKING!!!!
	YesNo = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', p, ACopy, p, w, vrInv, p, vr, p);

	YesNo = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'B', p, p, vr, p, vrInv, p);

	YesNo = LAPACKE_zgetrf(LAPACK_COL_MAJOR, p, p, vrInv, p, ipiv);

	YesNo = LAPACKE_zgetri(LAPACK_COL_MAJOR, p, vrInv, p, ipiv);

	#ifdef DEBUG_SETCARMA
	printf("setCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setCARMA - threadNum: %d; w\n",threadNum);
	viewMatrix(p,1,w);
	printf("\n");
	printf("setCARMA - threadNum: %d; vr\n",threadNum);
	viewMatrix(p,p,vr);
	printf("\n");
	#endif

	//#pragma omp simd
	for (int rowCtr = 0; rowCtr < q + 1; rowCtr++) {
		B[p - 1 - rowCtr] = Theta[p + rowCtr];
		}

	#ifdef DEBUG_SETCARMA
	printf("setCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setCARMA - threadNum: %d; B\n",threadNum);
	viewMatrix(p,1,B);
	printf("\n");
	#endif

	H[0] = 1.0;

	/*FKron_rcond[0] = 0.0;
	FKron_rpvgrw[0] = 0.0;
	FKron_berr[0] = 0.0;
	FKron_err_bnds_norm[0] = 0.0;
	FKron_err_bnds_comp[0] = 0.0;
	for (int colCtr = 0; colCtr < pSq; ++colCtr) {
		FKron_r[colCtr] = 0.0;
		FKron_c[colCtr] = 0.0;
		//#pragma omp simd
		for (int rowCtr = 0; rowCtr < pSq; ++rowCtr) {
			FKron_ipiv[rowCtr + colCtr*pSq] = 0;
			FKron[rowCtr + colCtr*pSq] = 0.0;
			FKron_af[rowCtr + colCtr*pSq] = 0.0;
			}
		}*/

	}

void CARMA::operator()(const vector<double> &x, vector<double> &dxdt, const double xi) {
	/*! \brief Compute and return the first column of expm(A*dt)*B*trans(B)*expm(trans(A)*dt)

	At every step, it is necessary to compute the conditional covariance matrix of the state given by \f$\textbf{\textsf{Q}} = \int_{t_{0}}^{t} \mathrm{e}^{\textbf{\textsf{A}}\chi} \mathbfit{B} \mathbfit{B}^{\top} \mathrm{e}^{\\textbf{\textsf{A}}^{\top}\chi} \mathrm{d}\chi\f$. Notice that the matrix \f$\textbf{\textsf{Q}}\f$ is symmetric positive definate and only the first column needfs to be computed.
	*/

	#ifdef DEBUG_FUNCTOR
	int threadNum = omp_get_thread_num();
	printf("() - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; xi: %f\n",threadNum,xi);
	printf("() - threadNum: %d; w (Before)\n",threadNum);
	viewMatrix(p,1,w);
	printf("\n");
	printf("() - threadNum: %d; expw (Before)\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	#endif

	// Start by computing expw = exp(w*t) where w is an e-value of A i.e. the doagonal of expw consists of the exponents of the e-values of A times t
	//#pragma omp simd
	for (int i = 0; i < p; ++i) {
		expw[i + i*p] = exp(xi*w[i]);
		}

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; w\n",threadNum);
	viewMatrix(p,1,w);
	printf("\n");
	printf("() - threadNum: %d; expw\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	#endif

	complex<double> alpha = 1.0+0.0i, beta = 0.0+0.0i;

	/*for (int colCtr = 0; colCtr < p; ++colCtr) {
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			A[rowCtr + colCtr*p] = 0.0+0.0i;
			}
		}*/

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; expw (Before)\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	printf("() - threadNum: %d; vr (Before)\n",threadNum);
	viewMatrix(p,p,vr);
	printf("\n");
	printf("() - threadNum: %d; vr*expw (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	// Begin by computing vr*expw. This is a pXp matrix. Store it in A.
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, vr, p, expw, p, &beta, ACopy, p);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; expw\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	printf("() - threadNum: %d; vr\n",threadNum);
	viewMatrix(p,p,vr);
	printf("\n");
	printf("() - threadNum: %d; vr*expw\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; vr*expw (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vrInv (Before)\n",threadNum);
	viewMatrix(p,p,vrInv);
	printf("\n");
	printf("() - threadNum: %d; (vr*expw)*vrInv (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	// Next compute expm(A*t) = (vr*expw)*vrInv. This is a pXp matrix. Store it in AScratch.
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, ACopy, p, vrInv, p, &beta, AScratch, p);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; vr*expw\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vrInv\n",threadNum);
	viewMatrix(p,p,vrInv);
	printf("\n");
	printf("() - threadNum: %d; (vr*expw)*vrInv\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; (vr*expw)*vrInv (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	printf("() - threadNum: %d; B (Before)\n",threadNum);
	viewMatrix(p,1,B);
	printf("\n");
	printf("() - threadNum: %d; ((vr*expw)*vrInv)*B (Before)\n",threadNum);
	viewMatrix(p,p,BScratch);
	printf("\n");
	#endif

	// Next compute expm(A*t)*B = ((vr*expw)*vrInv)*B. This is a pX1 vector. Store it in BScratch.
	cblas_zgemv(CblasColMajor, CblasNoTrans, p, p, &alpha, AScratch, p, B, 1, &beta, BScratch, 1);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; (vr*expw)*vrInv\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	printf("() - threadNum: %d; B\n",threadNum);
	viewMatrix(p,1,B);
	printf("\n");
	printf("() - threadNum: %d; ((vr*expw)*vrInv)*B\n",threadNum);
	viewMatrix(p,1,BScratch);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((vr*expw)*vrInv)*B (Before)\n",threadNum);
	viewMatrix(p,1,BScratch);
	printf("\n");
	printf("() - threadNum: %d; B (Before)\n",threadNum);
	viewMatrix(p,1,B);
	printf("\n");
	printf("() - threadNum: %d; (((vr*expw)*vrInv)*B)*trans(B) (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	// Next compute expm(A*t)*B*trans(B) = (((vr*expw)*vrInv)*B)*trans(B). This is a pXp matrix. Store it in A.
	//cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, 1, 1, &alpha, BScratch, p, B, 1, &beta, A, p);
	for (int colCtr = 0; colCtr < p; ++colCtr) {
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			ACopy[rowCtr + colCtr*p] = BScratch[rowCtr]*B[colCtr];
			}
		}

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((vr*expw)*vrInv)*B\n",threadNum);
	viewMatrix(p,1,BScratch);
	printf("\n");
	printf("() - threadNum: %d; B\n",threadNum);
	viewMatrix(p,1,B);
	printf("\n");
	printf("() - threadNum: %d; (((vr*expw)*vrInv)*B)*trans(B)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; (((vr*expw)*vrInv)*B)*trans(B) (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vrInv (Before)\n",threadNum);
	viewMatrix(p,p,vrInv);
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv) (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	// Next compute expm(A*t)*B*trans(B)*vrInv = ((((vr*expw)*vrInv)*B)*trans(B))*vrInv.  This is a pXp matrix. Store it in AScratch.
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, &alpha, ACopy, p, vrInv, p, &beta, AScratch, p);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; (((vr*expw)*vrInv)*B)*trans(B)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vrInv\n",threadNum);
	viewMatrix(p,p,vrInv);
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv) (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	printf("() - threadNum: %d; expw (Before)\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	printf("() - threadNum: %d; (((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	// Next compute  expm(A*t)*B*trans(B)*trans(vrInv)*expw = (((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw.  This is a pXp matrix. Store it in A.
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, AScratch, p, expw, p, &beta, ACopy, p);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	printf("() - threadNum: %d; expw\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	printf("() - threadNum: %d; (((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv) (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vr (Before)\n",threadNum);
	viewMatrix(p,p,vr);
	printf("\n");
	printf("() - threadNum: %d; ((((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw)*trans(vr) (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	// Finally compute  expm(A*t)*B*trans(B)*expm(trans(A)*t) = ((((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw)*trans(A).  This is a pXp matrix. Store it in AScratch.
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, &alpha, ACopy, p, vr, p, &beta, AScratch, p);

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; ((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	printf("() - threadNum: %d; vr\n",threadNum);
	viewMatrix(p,p,vr);
	printf("\n");
	printf("() - threadNum: %d; ((((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw)*trans(vr)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; dxdt (Before)\n",threadNum);
	viewMatrix(p,1,&dxdt[0]);
	printf("\n");
	#endif

	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		dxdt[rowCtr] = AScratch[rowCtr + rowCtr*p].real();
		}

	#ifdef DEBUG_FUNCTOR
	printf("() - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("() - threadNum: %d; dxdt (Before)\n",threadNum);
	viewMatrix(p,1,&dxdt[0]);
	printf("\n");
	#endif

	}

void CARMA::solveCARMA() {

	#ifdef DEBUG_SOLVECARMA
	int threadNum = omp_get_thread_num();
	printf("solveCARMA - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; w\n",threadNum);
	viewMatrix(p,1,w);
	printf("\n");
	printf("solveCARMA - threadNum: %d; expw\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	#endif

	// First compute expm(A*t)
	#pragma omp simd
	for (int i = 0; i < p; ++i) {
		expw[i + i*p] = exp(t*w[i]);
		}

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; expw\n",threadNum);
	viewMatrix(p,p,expw);
	printf("\n");
	#endif

	complex<double> alpha = 1.0+0.0i, beta = 0.0+0.0i;

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; vr*expw (Before)\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, vr, p, expw, p, &beta, ACopy, p);

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; vr*expw\n",threadNum);
	viewMatrix(p,p,ACopy);
	printf("\n");
	#endif

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; (vr*expw)*vrInv (Before)\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, ACopy, p, vrInv, p, &beta, AScratch, p);

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveCARMA - threadNum: %d; (vr*expw)*vrInv\n",threadNum);
	viewMatrix(p,p,AScratch);
	printf("\n");
	#endif

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveDLM - threadNum: %d; F (Before)\n",threadNum);
	viewMatrix(p,p,F);
	printf("\n");
	#endif

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			F[rowCtr + colCtr*p] = AScratch[rowCtr + colCtr*p].real();
			}
		}

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveDLM - threadNum: %d; F\n",threadNum);
	viewMatrix(p,p,F);
	printf("\n");
	#endif

	// Now compute Q by integrating expm(A*t)*B*trans(B)*expm(trans(A)*t) from 0 to t
	vector<double> initX(p); 
	size_t steps = boost::numeric::odeint::integrate(*this, initX, 0.0, t, 1.0e-6*t);
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		D[rowCtr] = sqrt(initX[rowCtr]);
		}

	// Finally compute Q
	//cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasTrans, p, 1, 1, &alpha, BScratch, p, BScratch, 1, &beta, Q, p);
	for (int colCtr = 0; colCtr < p; ++colCtr) {
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			Q[rowCtr + colCtr*p] = D[rowCtr]*D[colCtr];
			}
		}

	#ifdef DEBUG_SOLVECARMA
	printf("solveCARMA - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("solveDLM - threadNum: %d; Q\n",threadNum);
	viewMatrix(p,p,Q);
	printf("\n");
	#endif
	}

void CARMA::resetState(double InitUncertainty) {

	#ifdef DEBUG_RESETSTATE
	int threadNum = omp_get_thread_num();
	printf("resetState - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	for (int i = 0; i < p; i++) {
		X[i] = 0.0;
		XMinus[i] = 0.0;
		VScratch[i] = 0.0;
		#pragma omp simd
		for (int j = 0; j < p; j++) {
			P[i*p+j] = 0.0;
			PMinus[i*p+j] = 0.0;
			MScratch[i*p+j] = 0.0;
			}
		P[i*p+i] = InitUncertainty;
		}
	}

void CARMA::resetState() {

	#ifdef DEBUG_RESETSTATE
	int threadNum = omp_get_thread_num();
	printf("resetState - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	// Compute P by integrating expm(A*t)*B*trans(B)*expm(trans(A)*t) from 0 to infinity.
	vector<double> initX(p);

	size_t steps = boost::numeric::odeint::integrate(*this, initX, 0.0, 1.79769e+308, 1.0e-6);

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; steps: %lu\n",threadNum,steps);
	#endif

	// Finally compute P and in the process, reset the others.
	for (int colCtr = 0; colCtr < p; ++colCtr) {
		X[colCtr] = 0.0;
		XMinus[colCtr] = 0.0;
		VScratch[colCtr] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			PMinus[rowCtr + colCtr*p] = 0.0;
			MScratch[rowCtr + colCtr*p] = 0.0;
			P[rowCtr + colCtr*p] = sqrt(initX[rowCtr])*sqrt(initX[colCtr]);
			}
		}

	}

/*void CARMA::resetState_old() {

	#ifdef DEBUG_RESETSTATE
	int threadNum = omp_get_thread_num();
	printf("resetState - threadNum: %d; Address of System: %p\n",threadNum,this);
	printf("\n");
	#endif

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		X[colCtr] = 0.0;
		XMinus[colCtr] = 0.0;
		VScratch[colCtr] = 0.0;
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			P[rowCtr + colCtr*p] = 0.0;
			PMinus[rowCtr + colCtr*p] = 0.0;
			MScratch[rowCtr + colCtr*p] = 0.0;
			}
		}

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; F (Before)\n",threadNum);
	viewMatrix(p,p,F);
	printf("\n");
	printf("resetState - threadNum: %d; FKron (Before)\n",threadNum);
	viewMatrix(pSq,pSq,FKron);
	printf("\n");
	#endif

	kron(p,p,F,p,p,F,FKron);
	for (int i = 0; i < pSq; i++) {
		#pragma omp simd
		for (int j = 0; j < pSq; j++) {
			FKron[i*pSq + j] *= -1.0;
			}
		FKron[i*pSq + i] += 1.0;
		}

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; F\n",threadNum);
	viewMatrix(p,p,F);
	printf("\n");
	printf("resetState - threadNum: %d; FKron\n",threadNum);
	viewMatrix(pSq,pSq,FKron);
	printf("\n");
	#endif

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; Q (Before)\n",threadNum);
	viewMatrix(p,p,Q);
	printf("\n");
	printf("resetState - threadNum: %d; P (Before)\n",threadNum);
	viewMatrix(p,p,P);
	printf("\n");
	#endif

	lapack_int YesNo;
	char equed = 'N';
	cblas_dcopy(pSq, Q, 1, P, 1);

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; Q\n",threadNum);
	viewMatrix(p,p,Q);
	printf("\n");
	printf("resetState - threadNum: %d; P\n",threadNum);
	viewMatrix(p,p,P);
	printf("\n");
	#endif

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; FKron (Before)\n",threadNum);
	viewMatrix(pSq,pSq,FKron);
	printf("\n");
	printf("resetState - threadNum: %d; Q (Before)\n",threadNum);
	viewMatrix(p,p,Q);
	printf("\n");
	printf("resetState - threadNum: %d; P (Before)\n",threadNum);
	viewMatrix(p,p,P);
	printf("\n");
	#endif

	//YesNo = LAPACKE_dgesvxx(LAPACK_COL_MAJOR, 'E', 'N', pSq, 1, FKron, pSq, FKron_af, pSq, FKron_ipiv, &equed, FKron_r, FKron_c, Q, pSq, P, pSq, FKron_rcond, FKron_rpvgrw, FKron_berr, 1, FKron_err_bnds_norm, FKron_err_bnds_comp, 0, nullptr); // Not Working!!!!
	YesNo = LAPACKE_dgesv(LAPACK_COL_MAJOR, pSq, 1, FKron, pSq, FKron_ipiv , P, pSq);

	#ifdef DEBUG_RESETSTATE
	printf("resetState - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("resetState - threadNum: %d; FKron\n",threadNum);
	viewMatrix(pSq,pSq,FKron);
	printf("\n");
	printf("resetState - threadNum: %d; Q\n",threadNum);
	viewMatrix(p,p,Q);
	printf("\n");
	printf("resetState - threadNum: %d; P\n",threadNum);
	viewMatrix(p,p,P);
	printf("\n");
	#endif

	}*/

void CARMA::burnSystem(int numBurn, unsigned int burnSeed, double* burnRand) {

	#ifdef DEBUG_BURNSYSTEM
	int threadNum = omp_get_thread_num();
	printf("burnSystem - threadNum: %d; Starting...\n",threadNum);
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	VSLStreamStatePtr burnStream;
	vslNewStream(&burnStream, VSL_BRNG_SFMT19937, burnSeed);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, burnStream, numBurn, burnRand, 0.0, 1.0); // Check
	vslDeleteStream(&burnStream);

	#ifdef WRITE_BURNSYSTEM
	string burnPath = "/home/vish/Desktop/burn.dat";
	ofstream burnFile;
	burnFile.open(burnPath);
	burnFile.precision(16);
	for (int i = 0; i < numBurn-1; i++) {
		burnFile << noshowpos << scientific << burnRand[i] << endl;
		}
	burnFile << noshowpos << scientific << burnRand[numBurn-1];
	burnFile.close();
	#endif

	for (int i = 0; i < numBurn; ++i) {

		#ifdef DEBUG_BURNSYSTEM
		printf("\n");
		printf("burnSystem - threadNum: %d; i: %d\n",threadNum,i);
		printf("\n");
		printf("burnSystem - threadNum: %d; F (Before)\n",threadNum);
		viewMatrix(p, p, F);
		printf("\n");
		printf("burnSystem - threadNum: %d; X (Before)\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		printf("burnSystem - threadNum: %d; F*X (Before)\n",threadNum);
		viewMatrix(p, 1, VScratch);
		printf("\n");
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, VScratch, 1); // VScratch = F*X

		#ifdef DEBUG_BURNSYSTEM
		printf("burnSystem - threadNum: %d; F\n",threadNum);
		viewMatrix(p, p, F);
		printf("\n");
		printf("burnSystem - threadNum: %d; X\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		printf("burnSystem - threadNum: %d; F*X\n",threadNum);
		viewMatrix(p, 1, VScratch);
		printf("\n");
		#endif

		#ifdef DEBUG_BURNSYSTEM
		printf("burnSystem - threadNum: %d; F*X (Before)\n",threadNum);
		viewMatrix(p, 1, VScratch);
		printf("\n");
		printf("burnSystem - threadNum: %d; X (Before)\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		#endif

		cblas_dcopy(p, VScratch, 1, X, 1); // X = VScratch

		#ifdef DEBUG_BURNSYSTEM
		printf("burnSystem - threadNum: %d; F*X\n",threadNum);
		viewMatrix(p, 1, VScratch);
		printf("\n");
		printf("burnSystem - threadNum: %d; X\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		#endif

		#ifdef DEBUG_BURNSYSTEM
		printf("burnSystem - threadNum: %d; D (Before)\n",threadNum);
		viewMatrix(p,1,D);
		printf("\n");
		printf("burnSystem - threadNum: %d; w: %+f\n",threadNum,burnRand[i]);
		printf("\n");
		printf("burnSystem - threadNum: %d; X (Before)\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		#endif

		cblas_daxpy(p, burnRand[i], D, 1, X, 1); // X = w*D + X

		#ifdef DEBUG_BURNSYSTEM
		printf("burnSystem - threadNum: %d; D\n",threadNum);
		viewMatrix(p,1,D);
		printf("\n");
		printf("burnSystem - threadNum: %d; w: %+f\n",threadNum,burnRand[i]);
		printf("\n");
		printf("burnSystem - threadNum: %d; X\n",threadNum);
		viewMatrix(p, 1, X);
		printf("\n");
		#endif

		}
	}

double CARMA::observeSystem(double distRand, double noiseRand) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, VScratch, 1);
	cblas_dcopy(p, VScratch, 1, X, 1);
	cblas_daxpy(p, distRand, D, 1, X, 1);
	return cblas_ddot(p, H, 1, X, 1) + noiseRand;
	}

double CARMA::observeSystem(double distRand, double noiseRand, double mask) {

	double result;
	if (mask != 0.0) {
		result = observeSystem(distRand, noiseRand);
		} else {
		result = 0.0;
		}
	return result;
	}

void CARMA::observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	VSLStreamStatePtr distStream __attribute__((aligned(64)));
	VSLStreamStatePtr noiseStream __attribute__((aligned(64)));
	vslNewStream(&distStream, VSL_BRNG_SFMT19937, distSeed);
	vslNewStream(&noiseStream, VSL_BRNG_SFMT19937, noiseSeed);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, distStream, numObs, distRand, 0.0, 1.0); // Check Theta[p] == old distSigma
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, noiseStream, numObs, noiseRand, 0.0, noiseSigma);

	#ifdef WRITE
	string distPath = "/home/exarkun/Desktop/dist.dat";
	string noisePath = "/home/exarkun/Desktop/noise.dat";
	ofstream distFile, noiseFile;
	distFile.open(distPath);
	noiseFile.open(noisePath);
	distFile.precision(16);
	noiseFile.precision(16);
	for (int i = 0; i < numObs-1; i++) {
		distFile << noshowpos << scientific << distRand[i] << endl;
		noiseFile << noshowpos << scientific << noiseRand[i] << endl;
		}
	distFile << noshowpos << scientific << distRand[numObs-1];
	noiseFile << noshowpos << scientific << noiseRand[numObs-1];
	distFile.close();
	noiseFile.close();
	#endif

	for (int i = 0; i < numObs; i++) {

		#ifdef DEBUG_OBSSYSTEM
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << distRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(p, 1, X);
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, VScratch, 1);
		cblas_dcopy(p, VScratch, 1, X, 1);
		cblas_daxpy(p, distRand[i], D, 1, X, 1);

		#ifdef DEBUG_OBSSYSTEM
		cout << "X_+" << endl;
		viewMatrix(p, 1, X);
		cout << "Noise: " << noiseRand[i] << endl;
		#endif

		//y[i] = cblas_ddot(p, H, 1, X, 1) + noiseRand[i];
		y[i] = X[0] + noiseRand[i];

		#ifdef DEBUG_OBSSYSTEM
		cout << "y" << endl;
		cout << y[i] << endl;
		cout << endl;
		#endif

		}
	vslDeleteStream(&distStream);
	vslDeleteStream(&noiseStream);
	}

void CARMA::observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y, double* mask) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	VSLStreamStatePtr distStream __attribute__((aligned(64)));
	VSLStreamStatePtr noiseStream __attribute__((aligned(64)));
	vslNewStream(&distStream, VSL_BRNG_SFMT19937, distSeed);
	vslNewStream(&noiseStream, VSL_BRNG_SFMT19937, noiseSeed);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, distStream, numObs, distRand, 0.0, 1.0); // Check Theta[p] = distSigma
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, noiseStream, numObs, noiseRand, 0.0, noiseSigma);

	#ifdef WRITE
	string distPath = "/home/exarkun/Desktop/dist.dat";
	string noisePath = "/home/exarkun/Desktop/noise.dat";
	ofstream distFile, noiseFile;
	distFile.open(distPath);
	noiseFile.open(noisePath);
	distFile.precision(16);
	noiseFile.precision(16);
	for (int i = 0; i < numObs-1; i++) {
		distFile << noshowpos << scientific << distRand[i] << endl;
		noiseFile << noshowpos << scientific << noiseRand[i] << endl;
		}
	distFile << noshowpos << scientific << distRand[numObs-1];
	noiseFile << noshowpos << scientific << noiseRand[numObs-1];
	distFile.close();
	noiseFile.close();
	#endif

	for (int i = 0; i < numObs; i++) {

		#ifdef DEBUG_OBS
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << distRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(p, 1, X);
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, VScratch, 1);
		cblas_dcopy(p, VScratch, 1, X, 1);
		cblas_daxpy(p, distRand[i], D, 1, X, 1);

		#ifdef DEBUG_OBS
		cout << "X_+" << endl;
		viewMatrix(p, 1, X);
		cout << "Noise: " << noiseRand[i] << endl;
		#endif

		//y[i] = cblas_ddot(p, H, 1, X, 1) + noiseRand[i];
		y[i] = mask[i]*(X[0] + noiseRand[i]);

		#ifdef DEBUG_OBS
		cout << "y" << endl;
		cout << y[i] << endl;
		cout << endl;
		#endif

		}
	vslDeleteStream(&distStream);
	vslDeleteStream(&noiseStream);
	}

double CARMA::computeLnLike(int numPts, double* y, double* yerr) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	double LnLike = 0.0;
	double v = 0.0;
	double S = 0.0;
	double SInv = 0.0;

	#ifdef DEBUG_LNLIKE
	cout << endl;
	#endif

	for (int i = 0; i < numPts; i++) {
		R[0] = yerr[i]*yerr[i]; // Heteroskedastic errors

		#ifdef DEBUG_LNLIKE
		cout << endl;
		cout << "Pt:" << i << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, X);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, P);
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, XMinus, 1); // Compute XMinus = F*X

		#ifdef DEBUG_LNLIKE
		cout << "XMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, XMinus);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, F, p, P, p, 0.0, MScratch, p); // Compute MScratch = F*P

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, 1.0, MScratch, p, F, p, 0.0, PMinus, p); // Compute PMinus = MScratch*F_Transpose

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, I, p, Q, p, 1.0, PMinus, p); // Compute PMinus = I*Q + PMinus;

		#ifdef DEBUG_LNLIKE
		cout << "PMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, PMinus);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "y[" << i << "]: " << y[i] << endl;
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "XMinus[0]: " << XMinus[0] << endl;
		cout << endl;
		#endif

		v = y[i] - XMinus[0]; // Compute v = y - H*X

		#ifdef DEBUG_LNLIKE
		cout << "v[" << i << "]: " << v << endl;
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasTrans, p, p, 1.0, PMinus, p, H, 1, 0.0, K, 1); // Compute K = PMinus*H_Transpose

		S = cblas_ddot(p, K, 1, H, 1) + R[0]; // Compute S = H*K + R

		#ifdef DEBUG_LNLIKE
		cout << "S[" << i << "]: " << S << endl;
		cout << endl;
		#endif

		SInv = 1.0/S;

		#ifdef DEBUG_LNLIKE
		cout << "inverseS[" << i << "]: " << SInv << endl;
		cout << endl;
		#endif

		cblas_dscal(p, SInv, K, 1); // Compute K = SInv*K

		#ifdef DEBUG_LNLIKE
		cout << "K" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, K);
		cout << endl;
		#endif

		for (int colCounter = 0; colCounter < p; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < p; rowCounter++) {
				MScratch[rowCounter*p+colCounter] = I[colCounter*p+rowCounter] - K[colCounter]*H[rowCounter]; // Compute MScratch = I - K*H
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "IMinusKH" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, MScratch);
		cout << endl;
		#endif

		cblas_dcopy(p, K, 1, VScratch, 1); // Compute VScratch = K

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, MScratch, p, XMinus, 1, y[i], VScratch, 1); // Compute X = VScratch*y[i] + MScratch*XMinus

		#ifdef DEBUG_LNLIKE
		cout << "VScratch == X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, VScratch);
		cout << endl;
		#endif

		cblas_dcopy(p, VScratch, 1, X, 1); // Compute X = VScratch

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, X);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, MScratch, p, PMinus, p, 0.0, P, p); // Compute P = IMinusKH*PMinus

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, 1.0, P, p, MScratch, p, 0.0, PMinus, p); // Compute PMinus = P*IMinusKH_Transpose

		for (int colCounter = 0; colCounter < p; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < p; rowCounter++) {
				P[colCounter*p+rowCounter] = PMinus[colCounter*p+rowCounter] + R[0]*K[colCounter]*K[rowCounter]; // Compute P = PMinus + K*R*K_Transpose
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, P);
		cout << endl;
		#endif

		LnLike += -0.5*SInv*pow(v,2.0) -0.5*log2(S)/log2OfE; // LnLike += -0.5*v*v*SInv -0.5*log(det(S)) -0.5*log(2.0*pi)

		#ifdef DEBUG_LNLIKE
		cout << "Add LnLike: " << -0.5*SInv*pow(v,2.0) -0.5*log2(S)/log2OfE << endl;
		cout << endl;
		#endif

		}
	LnLike += -0.5*numPts*log2Pi;
	return LnLike;
	}

double CARMA::computeLnLike(int numPts, double* y, double* yerr, double* mask) {

	double maxDouble = numeric_limits<double>::max();

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	double LnLike = 0.0;
	double ptCounter = 0.0;
	double v = 0.0;
	double S = 0.0;
	double SInv = 0.0;

	#ifdef DEBUG_LNLIKE
	cout << endl;
	#endif

	for (int i = 0; i < numPts; i++) {
		R[0] = yerr[i]*yerr[i]; // Heteroskedastic errors

		#ifdef DEBUG_LNLIKE
		cout << endl;
		cout << "Pt:" << i << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, X);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, P);
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, F, p, X, 1, 0.0, XMinus, 1); // Compute XMinus = F*X

		#ifdef DEBUG_LNLIKE
		cout << "XMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, XMinus);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, F, p, P, p, 0.0, MScratch, p); // Compute MScratch = F*P

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, 1.0, MScratch, p, F, p, 0.0, PMinus, p); // Compute PMinus = MScratch*F_Transpose

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, I, p, Q, p, 1.0, PMinus, p); // Compute PMinus = I*Q + PMinus;

		#ifdef DEBUG_LNLIKE
		cout << "PMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, PMinus);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "y[" << i << "]: " << y[i] << endl;
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "XMinus[0]: " << XMinus[0] << endl;
		cout << endl;
		#endif

		v = y[i] - XMinus[0]; // Compute v = y - H*X

		#ifdef DEBUG_LNLIKE
		cout << "v[" << i << "]: " << v << endl;
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasTrans, p, p, 1.0, PMinus, p, H, 1, 0.0, K, 1); // Compute K = PMinus*H_Transpose

		S = cblas_ddot(p, K, 1, H, 1) + R[0]; // Compute S = H*K + R

		#ifdef DEBUG_LNLIKE
		cout << "S[" << i << "]: " << S << endl;
		cout << endl;
		#endif

		SInv = 1.0/S;

		#ifdef DEBUG_LNLIKE
		cout << "inverseS[" << i << "]: " << SInv << endl;
		cout << endl;
		#endif

		cblas_dscal(p, SInv, K, 1); // Compute K = SInv*K

		#ifdef DEBUG_LNLIKE
		cout << "K" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, K);
		cout << endl;
		#endif

		for (int colCounter = 0; colCounter < p; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < p; rowCounter++) {
				MScratch[rowCounter*p+colCounter] = I[colCounter*p+rowCounter] - K[colCounter]*H[rowCounter]; // Compute MScratch = I - K*H
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "IMinusKH" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, MScratch);
		cout << endl;
		#endif

		cblas_dcopy(p, K, 1, VScratch, 1); // Compute VScratch = K

		cblas_dgemv(CblasColMajor, CblasNoTrans, p, p, 1.0, MScratch, p, XMinus, 1, y[i], VScratch, 1); // Compute X = VScratch*y[i] + MScratch*XMinus

		#ifdef DEBUG_LNLIKE
		cout << "VScratch == X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, VScratch);
		cout << endl;
		#endif

		cblas_dcopy(p, VScratch, 1, X, 1); // Compute X = VScratch

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(p, 1, X);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, MScratch, p, PMinus, p, 0.0, P, p); // Compute P = IMinusKH*PMinus

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, 1.0, P, p, MScratch, p, 0.0, PMinus, p); // Compute PMinus = P*IMinusKH_Transpose

		for (int colCounter = 0; colCounter < p; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < p; rowCounter++) {
				P[colCounter*p+rowCounter] = PMinus[colCounter*p+rowCounter] + R[0]*K[colCounter]*K[rowCounter]; // Compute P = PMinus + K*R*K_Transpose
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(p, p, P);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "LnLike: " << LnLike << endl;
		cout << "Add LnLike: " << mask[i]*(-0.5*SInv*pow(v,2.0) -0.5*log2(S)/log2OfE) << endl;
		cout << endl;
		#endif

		//LnLike += -0.5*SInv*pow(v,2.0) -0.5*log2(S)/log2OfE; // LnLike += -0.5*v*v*SInv -0.5*log(det(S)) -0.5*log(2.0*pi)
		LnLike += mask[i]*(-0.5*SInv*pow(v,2.0) -0.5*log2(S)/log2OfE); // LnLike += -0.5*v*v*SInv -0.5*log(det(S)) -0.5*log(2.0*pi)
		ptCounter += mask[i];

		#ifdef DEBUG_LNLIKE
		cout << "LnLike: " << LnLike << endl;
		cout << endl;
		#endif

		}
	//LnLike += -0.5*numPts*log2Pi;
	#ifdef DEBUG_LNLIKE
	cout << "LnLike: " << LnLike << endl;
	cout << "Adding " << -0.5*ptCounter*log2Pi << " to LnLike" << endl;
	#endif

	LnLike += -0.5*ptCounter*log2Pi;

	#ifdef DEBUG_LNLIKE
	cout << "LnLike: " << LnLike << endl;
	#endif

	return LnLike;
	}

void getPSD(int numFreqs, double *freqVals, double *PSDVals) {
	}

void getACF(int numTimes, double *timeVals, double *ACFVals) {
	}
