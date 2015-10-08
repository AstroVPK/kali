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
#include "Constants.hpp"
#include "CARMA.hpp"
#include <stdio.h>
#include <stdlib.h>

//#define TIMEALL
//#define TIMEPER
//#define TIMEFINE
//#define DEBUG
//#define DEBUG_LNLIKE
//#define WRITE
//#define DEBUG_FUNC
//#define DEBUG_SETDLM
//#define DEBUG_SETDLM_DEEP
//#define DEBUG_CHECKARMAPARAMS
//#define DEBUG_BURNSYSTEM
//#define DEBUG_CTORDLM
//#define DEBUG_DTORDLM
//#define DEBUG_ALLOCATEDLM
//#define DEBUG_DEALLOCATEDLM
//#define DEBUG_DEALLOCATEDLM_DEEP
//#define DEBUG_RESETSTATE
//#define DEBUG_CALCLNLIKE

#ifdef WRITE
#include <fstream>
#endif

using namespace std;

double calcCARMALnLike(const vector<double> &x, vector<double>& grad, void* p2Args) {
	if (!grad.empty()) {
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	DLM* Systems = Args.Systems;

	int numPts = Data.numPts;
	double LnLike = 0;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	#ifdef DEBUG_CALCLNLIKE
	printf("calcLnLike - threadNum: %d; Location: ",threadNum);
	#endif

	for (int i = 0; i < (Systems[threadNum].p+Systems[threadNum].q+1); ++i) {
		Systems[threadNum].Theta[i] = x[i];

		#ifdef DEBUG_CALCLNLIKE
		printf("%f ",Systems[threadNum].Theta[i]);
		#endif

		}

	#ifdef DEBUG_CALCLNLIKE
	printf("\n");
	fflush(0);
	#endif

	Systems[threadNum].setDLM(Systems[threadNum].Theta);
	Systems[threadNum].resetState();
	if (Systems[threadNum].checkARMAParams(Systems[threadNum].Theta) == 1) {
		LnLike = 0.0;
		} else {
		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_CALCLNLIKE
	printf("LnLike: %f\n",LnLike);
	fflush(0);
	#endif

	return LnLike;

	}

double calcCARMALnLike(double* walkerPos, void* func_args) {

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	DLM* Systems = Args.Systems;

	int numPts = Data.numPts;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	double LnLike = 0.0;

	Systems[threadNum].setDLM(walkerPos);
	Systems[threadNum].resetState();
	//Systems[threadNum].resetState(1e7);

	if (Systems[threadNum].checkARMAParams(walkerPos) == 1) {

		#ifdef DEBUG_FUNC
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].p+Systems[threadNum].q+1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System good!\n",threadNum);
		#endif

		LnLike = 0.0;
		} else {

		#ifdef DEBUG_FUNC
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].p+Systems[threadNum].q+1; dimNum++) {
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
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	DLM* Systems = Args.Systems;

	int numPts = Data.numPts;
	double LnLike = 0;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	#ifdef DEBUG_CALCLNLIKE
	printf("calcLnLike - threadNum: %d; Location: ",threadNum);
	#endif

	for (int i = 0; i < (Systems[threadNum].p+Systems[threadNum].q+1); ++i) {
		Systems[threadNum].Theta[i] = x[i];

		#ifdef DEBUG_CALCLNLIKE
		printf("%f ",Systems[threadNum].Theta[i]);
		#endif

		}

	#ifdef DEBUG_CALCLNLIKE
	printf("\n");
	fflush(0);
	#endif

	Systems[threadNum].setDLM(Systems[threadNum].Theta);
	Systems[threadNum].resetState();
	if (Systems[threadNum].checkARMAParams(Systems[threadNum].Theta) == 1) {
		LnLike = Systems[threadNum].computeLnLike(numPts, y, yerr, mask);
		} else {
		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_CALCLNLIKE
	printf("LnLike: %f\n",LnLike);
	fflush(0);
	#endif

	return LnLike;

	}

double calcLnLike(double* walkerPos, void* func_args) {

	int threadNum = omp_get_thread_num();

	LnLikeArgs* ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;

	int numThreads = Args.numThreads;
	LnLikeData Data = Args.Data;
	DLM* Systems = Args.Systems;

	int numPts = Data.numPts;
	double* y = Data.y;
	double* yerr = Data.yerr;
	double* mask = Data.mask;

	double LnLike = 0.0;

	Systems[threadNum].setDLM(walkerPos);
	Systems[threadNum].resetState();
	//Systems[threadNum].resetState(1e7);

	if (Systems[threadNum].checkARMAParams(walkerPos) == 1) {

		#ifdef DEBUG_FUNC
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].p+Systems[threadNum].q+1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System good!\n",threadNum);
		#endif

		LnLike = Systems[threadNum].computeLnLike(numPts, y, yerr, mask);
		} else {

		#ifdef DEBUG_FUNC
		printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < Systems[threadNum].p+Systems[threadNum].q+1; dimNum++) {
			printf("%f ",walkerPos[dimNum]);
			}
		printf("\n");
		printf("calcLnLike - threadNum: %d; System bad!\n",threadNum);
		#endif

		LnLike = -HUGE_VAL;
		}

	#ifdef DEBUG_FUNC
	printf("calcLnLike = threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < Systems[threadNum].p+Systems[threadNum].q+1; dimNum++) {
		printf("%f ",walkerPos[dimNum]);
		}
	printf("\n");
	printf("calcLnLike - threadNum: %d; LnLike: %f\n",threadNum, LnLike);
	#endif

	return LnLike;
	}

void zeroMatrix(int nRows, int nCols, int* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0;
			}
		}
	}

void zeroMatrix(int nRows, int nCols, lapack_int* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0;
			}
		}
	}

void zeroMatrix(int nRows, int nCols, double* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0.0;
			}
		}
	}

void zeroMatrix(int nRows, int nCols, complex<double>* mat) {
	for (int colNum = 0; colNum < nCols; ++colNum) {
		for (int rowNum = 0; rowNum < nRows; ++rowNum) {
			mat[rowNum + nRows*colNum] = 0.0;
			}
		}
	}

void viewMatrix(int nRows, int nCols, double* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%+f ",mat[j*nCols + i]);
			}
		printf("\n");
		}
	}

void viewMatrix(int nRows, int nCols, complex<double>* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%+f%+fi ",mat[j*nCols + i].real(),mat[j*nCols + i].imag());
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
	#pragma omp simd
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
		}
	}

CARMA::CARMA() {
	/*! Object that holds data and methods for performing C-ARMA analysis. DLM objects hold pointers to blocks of data that are set as required based on the size of the C-ARMA model.*/
	#ifdef DEBUG_CTORDLM
	int threadNum = omp_get_thread_num();
	printf("DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	allocated = 0;
	isStable = 0;
	isInvertible = 0;
	isNotRedundant = 0;
	hasUniqueEigenValues = 0;
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
	FKron = nullptr;
	FKron_af = nullptr;
	FKron_r = nullptr;
	FKron_c = nullptr;
	FKron_ipiv = nullptr;
	FKron_rcond = nullptr;
	FKron_rpvgrw = nullptr;
	FKron_berr = nullptr;
	FKron_err_bnds_norm = nullptr;
	FKron_err_bnds_comp = nullptr;
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
	isStable = 0;
	isInvertible = 0;
	isNotRedundant = 0;
	hasUniqueEigenValues = 0;
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
	AScratch = nullptr;
	B = nullptr;
	BScratch = nullptr;
	I = nullptr;
	F = nullptr;
	FKron = nullptr;
	FKron_af = nullptr;
	FKron_r = nullptr;
	FKron_c = nullptr;
	FKron_ipiv = nullptr;
	FKron_rcond = nullptr;
	FKron_rpvgrw = nullptr;
	FKron_berr = nullptr;
	FKron_err_bnds_norm = nullptr;
	FKron_err_bnds_comp = nullptr;
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

	abnrm = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	allocated += sizeof(double);

	ilo[0] = 0;
	ihi[0] = 0;
	abnrm[0] = 0.0;

	ipiv = static_cast<lapack_int*>(_mm_malloc(p*sizeof(lapack_int),64));
	allocated += p*sizeof(lapack_int);

	scale = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	rconde = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	rcondv = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	allocated += 3*p*sizeof(double);

	w = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	CARw = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	CMAw = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	B = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	BScratch = static_cast<complex<double>*>(_mm_malloc(p*sizeof(complex<double>),64));
	allocated += 4*p*sizeof(complex<double>);
	allocated += q*sizeof(complex<double>);

	CARMatrix = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	CMAMatrix = static_cast<complex<double>*>(_mm_malloc(qSq*sizeof(complex<double>),64));
	expw = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	vr = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	vrInv = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	A = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	AScratch = static_cast<complex<double>*>(_mm_malloc(pSq*sizeof(complex<double>),64));
	allocated += 6*pSq*sizeof(double);
	allocated += qSq*sizeof(complex<double>);

	for (int colCtr = 0; colCtr < q; ++colCtr) {
		CMAw[colCtr] = 0.0+0.0i;
		#pramgma omp simd
		for (int rowCtr = 0; rowCtr < q; ++rowctr) {
			CMAMatrix[rowCtr + colCtr*q] = 0.0+0.0i;
			}
		}

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		ipiv[colCtr] = 0;
		scale[colCtr] = 0.0;
		rconde[colCtr] = 0.0;
		rcondv[colCtr] = 0.0;
		w[colCtr] = 0.0+0.0i;
		CARw[colCtr] = 0.0+0.0i;
		B[colCtr] = 0.0;
		BScratch[colCtr] = 0.0;
		#pramgma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowctr) {
			CARMatrix[rowCtr + colCtr*p] = 0.0+0.0i;
			expw[rowCtr + colCtr*p] = 0.0+0.0i;
			vr[rowCtr + colCtr*p] = 0.0+0.0i;
			vrInv[rowCtr + colCtr*p] = 0.0+0.0i;
			A[rowCtr + colCtr*p] = 0.0+0.0i;
			AScratch[rowCtr + colCtr*p] = 0.0+0.0i;
			}
		}

	FKron_rcond = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
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
		}

	Theta = static_cast<double*>(_mm_malloc((p+q+1)*sizeof(double),64));
	allocated += (p+q+1)*sizeof(double);

	for (int rowCtr = 0; rowCtr < (p+q+1); ++rowCtr) {
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
	Q = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	P = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	PMinus = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	MScratch = static_cast<double*>(_mm_malloc(pSq*sizeof(double),64));
	allocated += 6*pSq*sizeof(double);

	for (int i = 0; i < m; i++) {
		D[i] = 0.0;
		H[i] = 0.0;
		K[i] = 0.0;
		X[i] = 0.0;
		XMinus[i] = 0.0;
		VScratch[i] = 0.0;
		#pragma omp simd
		for (int j = 0; j < m; j++) {
			I[i*m+j] = 0.0;
			F[i*m+j] = 0.0;
			Q[i*m+j] = 0.0;
			P[i*m+j] = 0.0;
			PMinus[i*m+j] = 0.0;
			MScratch[i*m+j] = 0.0;
			}
		}

	R = static_cast<double*>(_mm_malloc(sizeof(double),64));
	allocated += sizeof(double);

	R[0] = 0.0;

	#pragma omp simd
	for (int i = 1; i < m; i++) {
		A[i*m+(i-1)] = 1.0;
		I[(i-1)*m+(i-1)] = 1.0;
		}
	I[(m-1)*m+(m-1)] = 1.0;

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

	if (ARTau) {
		_mm_free(ARTau);
		ARTau = nullptr;
	}

	if (MATau) {
		_mm_free(MATau);
		MATau = nullptr;
		}

	if (ARScale) {
		_mm_free(ARScale);
		ARScale = nullptr;
		}

	if (MAScale) {
		_mm_free(MAScale);
		MAScale = nullptr;
		}

	if (ARMatrix) {
		_mm_free(ARMatrix);
		ARMatrix = nullptr;
		}

	if (MAMatrix) {
		_mm_free(MAMatrix);
		MAMatrix = nullptr;
		}

	if (ARwr) {
		_mm_free(ARwr);
		ARwr = nullptr;
		}

	if (ARwi) {
		_mm_free(ARwi);
		ARwi = nullptr;
		}

	if (MAwr) {
		_mm_free(MAwr);
		MAwr = nullptr;
		}

	if (MAwi) {
		_mm_free(MAwi);
		MAwi = nullptr;
		}

	if (Theta) {
		_mm_free(Theta);
		Theta = nullptr;
		}

	if (A) {
		_mm_free(A);
		A = nullptr;
		}

	if (Avr) {
		_mm_free(Avr);
		Avr = nullptr;
		}

	if (Awr) {
		_mm_free(Awr);
		Awr = nullptr;
		}

	if (Awi) {
		_mm_free(Awi);
		Awi = nullptr;
		}

	if (Ascale) {
		_mm_free(Ascale);
		Ascale = nullptr;
		}

	if (B) {
		_mm_free(B);
		B = nullptr;
		}

	if (I) {
		_mm_free(I);
		I = nullptr;
		}

	if (F) {
		_mm_free(F);
		F = nullptr;
		}

	if (FKron) {
		_mm_free(FKron);
		FKron = nullptr;
		}

	if (FKronPiv) {
		_mm_free(FKronPiv);
		FKronPiv = nullptr;
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

int CARMA::checkCARMAParams(double* Theta) {

	isStable = 1;
	isInvertible = 1;
	isNotRedundant = 1;
	hasUniqueEigenValues = 1;

	for (int rowCtr = 0; rowCtr < p; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < p; colCtr++) {
			CARMatrix[rowCtr + p*colCtr] = 0.0; // Reset matrix.
			}
		}
	#pragma omp simd
	CARMatrix[p*(p-1)] = -1.0*Theta[p-1]; // The first row has no 1s so we just set the rightmost entry equal to -alpha_p
	for (int rowCtr = 1; rowCtr < p; rowCtr++) {
		CARMatrix[rowCtr+(p-1)*p] = -1.0*Theta[p-1-rowCtr]; // Rightmost column of ARMatrix equals -alpha_k where 1 < k < p.
		CARMatrix[rowCtr+(rowCtr-1)*p] = 1.0; // ARMatrix has Identity matrix in bottom left.
		}
	ilo[0] = 0;
	ihi[0] = 0;
	abnrm[0] = 0.0;
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		CARw[rowCtr] = 0.0;
		scale[rowCtr] = 0.0;
		rconde[rowCtr] = 0.0;
		rcondv[rowCtr] = 0.0;
		}
	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; ARMatrix\n",threadNum);
	viewMatrix(p,p,ARMatrix);
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	lapack_int YesNo;

	YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'N', 'N', p, CARMatrix, p, CARw, nullptr, 1, nullptr, 1, ilo, ihi, scale, abnrm, rconde, rcondv);

	for (int i = 0; i < p; i++) {

		#ifdef DEBUG_CHECKARMAPARAMS
		printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
		printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
			printf("%f ",Theta[dimNum]);
			}
		printf("\n");
		printf("checkARMAParams - threadNum: %d; Root: %f\n",threadNum,pow(ARwr[i], 2.0) + pow(ARwi[i],2.0));
		#endif

		if (CARw[i].real() >= 0.0) {

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; badRoot!!!: %f\n",threadNum,pow(ARwr[i], 2.0) + pow(ARwi[i],2.0));
			#endif

			isStable = 0;
			}

		for (int j = i+1; j < p; j++) { // Only need to check e-values against each other once.
			if (CARw[i] == CARw[j]) {
				hasUniqueEigenValues = 0;
				}
			}
			
		}

	for (int rowCtr = 0; rowCtr < q; ++rowCtr) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < q; colCtr++) {
			MAMatrix[rowCtr + q*colCtr] = 0.0; // Initialize matrix.
			}
		}
	MAMatrix[(q-1)*q] = -1.0*Theta[p+q-1]/Theta[p]; // MAMatrix has -beta_q/-beta_0 at top right!
	#pragma omp simd
	for (int rowCtr = 1; rowCtr < q; ++rowCtr) {
		MAMatrix[rowCtr+(q-1)*q] = -1.0*Theta[p+q-1-rowCtr]/Theta[p]; // Rightmost column of MAMatrix has -MA coeffs.
		MAMatrix[rowCtr+(rowCtr-1)*q] = 1.0; // MAMatrix has Identity matrix in bottom left.
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

	YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'N', 'N', q, CMAMatrix, q, CMAw, nullptr, 1, nullptr, 1, ilo, ihi, scale, abnrm, rconde, rcondv);

	for (int i = 0; i < q; i++) {

		#ifdef DEBUG_CHECKARMAPARAMS
		printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
		printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
		for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
			printf("%f ",Theta[dimNum]);
			}
		printf("\n");
		printf("checkARMAParams - threadNum: %d; Root: %f\n",threadNum,pow(MAwr[i], 2.0) + pow(MAwi[i],2.0));
		#endif

		if (CMAw[i].real() > 0.0) {

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; badRoot!!!: %f\n",threadNum,pow(MAwr[i], 2.0) + pow(MAwi[i],2.0));
			#endif

			isInvertible = 0;
			}
		}

	for (int i = 1; i < p; i++) {
		for (int j = 1; j < q; j++) {
			if (CARw[i] == MAwr[j]) {
				isNotRedundant = 0;
				}
			}
		}

	#ifdef DEBUG_CHECKARMAPARAMS
	printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("checkARMAParams - threadNum: %d; isStable: %d\n",threadNum,isStable);
	printf("checkARMAParams - threadNum: %d; isInvertible: %d\n",threadNum,isInvertible);
	printf("checkARMAParams - threadNum: %d; isNotRedundant: %d\n",threadNum,isNotRedundant);
	printf("checkARMAParams - threadNum: %d; isReasonable: %d\n",threadNum,isReasonable);
	#endif

	return isStable*isInvertible*isNotRedundant*hasUniqueEigenValues;
	}

void CARMA::setCARMA(double* Theta) {

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			A[rowCtr + colCtr*p] = 0.0;
			}
		}

	A[0] = -1.0*Theta[0]
	#pragma omp simd
	for (int i = 1; i < p; ++i) {
		A[i] = -1.0*Theta[i];
		A[i*p+(i-1)] = 1.0;
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; A\n",threadNum);
	viewMatrix(m,m,A);
	printf("\n");
	#endif

	ilo[0] = 0;
	ihi[0] = 0;
	abnrm[0] = 0.0;
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		w[rowCtr] = 0.0;
		scale[rowCtr] = 0.0;
		rconde[rowCtr] = 0.0;
		rcondv[rowCtr] = 0.0;
		}

	lapack_int YesNo;
	YesNo = LAPACKE_zgeevx(LAPACK_COL_MAJOR, 'B', 'N', 'V', 'N', p, A, p, w, nullptr, 1, vr, p, ilo, ihi, scale, abnrm, rconde, rcondv);

	YesNo = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'B', p, p, vr, p, vrInv, p);
	YesNo = LAPACKE_zgetrf(LAPACK_COL_MAJOR, p, p, vrInv, p, ipiv);
	YesNo = LAPACKE_zgetri(LAPACK_COL_MAJOR, p, vrInv, p, ipiv);

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	printf("setDLM - threadNum: %d; B\n",threadNum);
	viewMatrix(m,1,B);
	#endif

	#pragma omp simd
	for (int i = 0; i < q; i++) {
		B[p-q+i] = Theta[p+i];
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; B\n",threadNum);
	viewMatrix(m,1,B);
	printf("\n");
	#endif

	H[0] = 1.0;

	for (int colCtr = 0; colCtr < pSq; ++colCtr) {
		#pragma omp simd
		for (int rowctr = 0; rowCtr < pSq; ++rowCtr) {
			FKron[rowCtr + pSq*colCtr] = 0.0;
			FKronPiv[rowCtr + pSq*colCtr] = 0;
			}
		}

	}

void CARMA::operator()(const vector<double> &x, vector<double> &dxdt, const double t) {
	/*! \brief Compute and return the first column of expm(A*dt)*B*trans(B)*expm(trans(A)*dt)

	At every step, it is necessary to compute the conditional covariance matrix of the state given by \f$\textbf{\textsf{Q}} = \int_{t_{0}}^{t} \mathrm{e}^{\textbf{\textsf{A}}\chi} \mathbfit{B} \mathbfit{B}^{\top} \mathrm{e}^{\\textbf{\textsf{A}}^{\top}\chi} \mathrm{d}\chi\f$. Notice that the matrix \f$\textbf{\textsf{Q}}\f$ is symmetric positive definate and only the first column needfs to be computed.
	*/

	// Start by computing expw = exp(w*t) where w is an e-value of A i.e. the doagonal of expw consists of the exponents of the e-values of A times t
	#pragma omp simd
	for (int i = 0; i < p; ++i) {
		expw[i + i*p] = exp(t*w[i]);
		}
	complex<double> alpha = 1.0+0.0i, beta = 0.0+0.0i;

	// Begin by computing vr*expw. This is a pXp matrix. Store it in A.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, vr, p, expw, p, &beta, A, p);

	// Next compute expm(A*t) = (vr*expw)*vrInv. This is a pXp matrix. Store it in AScratch.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, A, p, vrInv, p, &beta, AScratch, p);

	// Next compute expm(A*t)*B = ((vr*expw)*vrInv)*B. This is a pX1 vector. Store it in BScratch.
	cblas_zgemv(CblasColMajor, CblasNoTrans, p, p, &alpha, FScratch, p, B, 1, &beta, BScratch, 1);

	// Next compute expm(A*t)*B*trans(B) = (((vr*expw)*vrInv)*B)*trans(B). This is a pXp matrix. Store it in A.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasTrans, p, 1, 1, &alpha, BScratch, p, B, 1, &beta, A, p);

	// Next compute expm(A*t)*B*trans(B)*vrInv = ((((vr*expw)*vrInv)*B)*trans(B))*vrInv.  This is a pXp matrix. Store it in AScratch.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, &alpha, A, p, vrInv, p, &beta, AScratch, p);

	// Next compute  expm(A*t)*B*trans(B)*trans(vrInv)*expw = (((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw.  This is a pXp matrix. Store it in A.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, AScratch, p, expw, p, &beta, A, p);

	// Finally compute  expm(A*t)*B*trans(B)*expm(trans(A)*t) = ((((((vr*expw)*vrInv)*B)*trans(B))*trans(vrInv))*expw)*trans(A).  This is a pXp matrix. Store it in AScratch.
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasTrans, p, p, p, &alpha, A, p, vr, p, &beta, AScratch, p);

	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		dxdt[rowCtr] = AScratch[rowCtr].real();
		}

	}

void CARMA::solveCARMA(double t) {

	// First compute expm(A*t)
	#pragma omp simd
	for (int i = 0; i < p; ++i) {
		expw[i + i*p] = exp(t*w[i]);
		}

	complex<double> alpha = 1.0+0.0i, beta = 0.0+0.0i;
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, vr, p, expw, p, &beta, A, p);
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, &alpha, A, p, vrInv, p, &beta, AScratch, p);

	for (int colCtr = 0; colCtr < p; ++colCtr) {
		#pragma omp simd
		for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
			F[rowCtr + colCtr*p] = AScratch[rowCtr + colCtr*p].real();
			}
		}

	// Now compute Q by integrating expm(A*t)*B*trans(B)*expm(trans(A)*t) from 0 to t
	vector<double> initX(p); 
	//size_t steps = integrate((*this)(), initX, 0.0, t, 1.0e-6*t); // JohnS suggests that I try "this" 
	size_t steps = integrate(this, initX, 0.0, t, 1.0e-6*t);
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p; ++rowCtr) {
		BScratch[rowCtr] = sqrt(initX[rowCtr]);
		}

	// Finally compute Q
	cblas_zgemm3m(CblasColMajor, CblasNoTrans, CblasTrans, p, 1, 1, &alpha, BScratch, p, BScratch, 1, &beta, Q, p);
	}

void CARMA::resetState(double InitUncertainty) {

	for (int i = 0; i < m; i++) {
		X[i] = 0.0;
		XMinus[i] = 0.0;
		VScratch[i] = 0.0;
		#pragma omp simd
		for (int j = 0; j < m; j++) {
			P[i*m+j] = 0.0;
			PMinus[i*m+j] = 0.0;
			MScratch[i*m+j] = 0.0;
			}
		P[i*m+i] = InitUncertainty;
		}
	}

void CARMA::resetState() {

	for (int i = 0; i < m; i++) {
		X[i] = 0.0;
		XMinus[i] = 0.0;
		VScratch[i] = 0.0;
		#pragma omp simd
		for (int j = 0; j < m; j++) {
			P[i*m+j] = 0.0;
			PMinus[i*m+j] = 0.0;
			MScratch[i*m+j] = 0.0;
			}
		}

	kron(m,m,F,m,m,F,FKron);
	for (int i = 0; i < mSq; i++) {
		#pragma omp simd
		for (int j = 0; j < mSq; j++) {
			FKron[i*mSq + j] *= -1.0;
			}
		FKron[i*mSq + i] += 1.0;
		}

	lapack_int YesNo;
	cblas_dcopy(mSq, Q, 1, P, 1);
	YesNo = LAPACKE_dgesvxx(LAPACK_COL_MAJOR, 'E', 'N', pSq, 1, FKron, pSq, FKron_af, pSq, FKron_ipiv, 'N', FKron_r, FKronC_c, pSq, P, pSq,FKron_rcond, FKron_rpvgrw, FKron_berr, 1, FKron_err_bnds_norm, FKron_err_bnds_comp, 0, nullptr);
	}

void CARMA::getCARRoots(double*& RealAR, double*& ImagAR) {
	RealAR = ARwr;
	ImagAR = ARwi;
	}

void CARMA::getMARoots(double*& RealMA, double*& ImagMA) {
	RealMA = MAwr;
	ImagMA = MAwi;
	}

void CARMA::burnSystem(int numBurn, unsigned int burnSeed, double* burnRand) {

	#ifdef DEBUG_BURNSYSTEM
	int threadNum = omp_get_thread_num();
	printf("burnSystem - threadNum: %d; Starting...\n",threadNum);
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	VSLStreamStatePtr burnStream;
	vslNewStream(&burnStream, VSL_BRNG_SFMT19937, burnSeed);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, burnStream, numBurn, burnRand, 0.0, distSigma);
	vslDeleteStream(&burnStream);

	#ifdef WRITE
	string burnPath = "/home/exarkun/Desktop/burn.dat";
	ofstream burnFile;
	burnFile.open(burnPath);
	burnFile.precision(16);
	for (int i = 0; i < numBurn-1; i++) {
		burnFile << noshowpos << scientific << burnRand[i] << endl;
		}
	burnFile << noshowpos << scientific << burnRand[numBurn-1];
	burnFile.close();
	#endif

	for (MKL_INT64 i = 0; i < numBurn; i++) {

		#ifdef DEBUG_BURNSYSTEM
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << burnRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(m, 1, X);
		cout << "F" << endl;
		viewMatrix(m, m, F);
		cout << "VScratch" << endl;
		viewMatrix(m, 1, VScratch);
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, VScratch, 1); // VScratch = F*X

		#ifdef DEBUG_BURNSYSTEM
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << burnRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(m, 1, X);
		cout << "F" << endl;
		viewMatrix(m, m, F);
		cout << "VScratch" << endl;
		viewMatrix(m, 1, VScratch);
		#endif

		cblas_dcopy(m, VScratch, 1, X, 1); // X = VScratch

		#ifdef DEBUG_BURNSYSTEM
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << burnRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(m, 1, X);
		cout << "F" << endl;
		viewMatrix(m, m, F);
		cout << "VScratch" << endl;
		viewMatrix(m, 1, VScratch);
		#endif

		cblas_daxpy(m, burnRand[i], D, 1, X, 1); // X = w*D + X

		#ifdef DEBUG_BURNSYSTEM
		cout << endl;
		cout << "i: " << i << endl;
		cout << "Disturbance: " << burnRand[i] << endl;
		cout << "X_-" << endl;
		viewMatrix(m, 1, X);
		cout << "F" << endl;
		viewMatrix(m, m, F);
		cout << "VScratch" << endl;
		viewMatrix(m, 1, VScratch);
		#endif

		}
	}

double CARMA::observeSystem(double distRand, double noiseRand) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, VScratch, 1);
	cblas_dcopy(m, VScratch, 1, X, 1);
	cblas_daxpy(m, distRand, D, 1, X, 1);
	return cblas_ddot(m, H, 1, X, 1) + noiseRand;
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
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, distStream, numObs, distRand, 0.0, distSigma);
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
		viewMatrix(m, 1, X);
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, VScratch, 1);
		cblas_dcopy(m, VScratch, 1, X, 1);
		cblas_daxpy(m, distRand[i], D, 1, X, 1);

		#ifdef DEBUG_OBS
		cout << "X_+" << endl;
		viewMatrix(m, 1, X);
		cout << "Noise: " << noiseRand[i] << endl;
		#endif

		//y[i] = cblas_ddot(m, H, 1, X, 1) + noiseRand[i];
		y[i] = X[0] + noiseRand[i];

		#ifdef DEBUG_OBS
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
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, distStream, numObs, distRand, 0.0, distSigma);
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
		viewMatrix(m, 1, X);
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, VScratch, 1);
		cblas_dcopy(m, VScratch, 1, X, 1);
		cblas_daxpy(m, distRand[i], D, 1, X, 1);

		#ifdef DEBUG_OBS
		cout << "X_+" << endl;
		viewMatrix(m, 1, X);
		cout << "Noise: " << noiseRand[i] << endl;
		#endif

		//y[i] = cblas_ddot(m, H, 1, X, 1) + noiseRand[i];
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
		viewMatrix(m, 1, X);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, P);
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, XMinus, 1); // Compute XMinus = F*X

		#ifdef DEBUG_LNLIKE
		cout << "XMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, XMinus);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, F, m, P, m, 0.0, MScratch, m); // Compute MScratch = F*P

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, MScratch, m, F, m, 0.0, PMinus, m); // Compute PMinus = MScratch*F_Transpose

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, I, m, Q, m, 1.0, PMinus, m); // Compute PMinus = I*Q + PMinus;

		#ifdef DEBUG_LNLIKE
		cout << "PMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, PMinus);
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

		cblas_dgemv(CblasColMajor, CblasTrans, m, m, 1.0, PMinus, m, H, 1, 0.0, K, 1); // Compute K = PMinus*H_Transpose

		S = cblas_ddot(m, K, 1, H, 1) + R[0]; // Compute S = H*K + R

		#ifdef DEBUG_LNLIKE
		cout << "S[" << i << "]: " << S << endl;
		cout << endl;
		#endif

		SInv = 1.0/S;

		#ifdef DEBUG_LNLIKE
		cout << "inverseS[" << i << "]: " << SInv << endl;
		cout << endl;
		#endif

		cblas_dscal(m, SInv, K, 1); // Compute K = SInv*K

		#ifdef DEBUG_LNLIKE
		cout << "K" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, K);
		cout << endl;
		#endif

		for (int colCounter = 0; colCounter < m; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < m; rowCounter++) {
				MScratch[rowCounter*m+colCounter] = I[colCounter*m+rowCounter] - K[colCounter]*H[rowCounter]; // Compute MScratch = I - K*H
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "IMinusKH" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, MScratch);
		cout << endl;
		#endif

		cblas_dcopy(m, K, 1, VScratch, 1); // Compute VScratch = K

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, MScratch, m, XMinus, 1, y[i], VScratch, 1); // Compute X = VScratch*y[i] + MScratch*XMinus

		#ifdef DEBUG_LNLIKE
		cout << "VScratch == X" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, VScratch);
		cout << endl;
		#endif

		cblas_dcopy(m, VScratch, 1, X, 1); // Compute X = VScratch

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, X);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, MScratch, m, PMinus, m, 0.0, P, m); // Compute P = IMinusKH*PMinus

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, P, m, MScratch, m, 0.0, PMinus, m); // Compute PMinus = P*IMinusKH_Transpose

		for (int colCounter = 0; colCounter < m; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < m; rowCounter++) {
				P[colCounter*m+rowCounter] = PMinus[colCounter*m+rowCounter] + R[0]*K[colCounter]*K[rowCounter]; // Compute P = PMinus + K*R*K_Transpose
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, P);
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
		viewMatrix(m, 1, X);
		cout << endl;
		#endif

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, P);
		cout << endl;
		#endif

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, XMinus, 1); // Compute XMinus = F*X

		#ifdef DEBUG_LNLIKE
		cout << "XMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, XMinus);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, F, m, P, m, 0.0, MScratch, m); // Compute MScratch = F*P

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, MScratch, m, F, m, 0.0, PMinus, m); // Compute PMinus = MScratch*F_Transpose

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, I, m, Q, m, 1.0, PMinus, m); // Compute PMinus = I*Q + PMinus;

		#ifdef DEBUG_LNLIKE
		cout << "PMinus" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, PMinus);
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

		cblas_dgemv(CblasColMajor, CblasTrans, m, m, 1.0, PMinus, m, H, 1, 0.0, K, 1); // Compute K = PMinus*H_Transpose

		S = cblas_ddot(m, K, 1, H, 1) + R[0]; // Compute S = H*K + R

		#ifdef DEBUG_LNLIKE
		cout << "S[" << i << "]: " << S << endl;
		cout << endl;
		#endif

		SInv = 1.0/S;

		#ifdef DEBUG_LNLIKE
		cout << "inverseS[" << i << "]: " << SInv << endl;
		cout << endl;
		#endif

		cblas_dscal(m, SInv, K, 1); // Compute K = SInv*K

		#ifdef DEBUG_LNLIKE
		cout << "K" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, K);
		cout << endl;
		#endif

		for (int colCounter = 0; colCounter < m; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < m; rowCounter++) {
				MScratch[rowCounter*m+colCounter] = I[colCounter*m+rowCounter] - K[colCounter]*H[rowCounter]; // Compute MScratch = I - K*H
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "IMinusKH" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, MScratch);
		cout << endl;
		#endif

		cblas_dcopy(m, K, 1, VScratch, 1); // Compute VScratch = K

		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, MScratch, m, XMinus, 1, y[i], VScratch, 1); // Compute X = VScratch*y[i] + MScratch*XMinus

		#ifdef DEBUG_LNLIKE
		cout << "VScratch == X" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, VScratch);
		cout << endl;
		#endif

		cblas_dcopy(m, VScratch, 1, X, 1); // Compute X = VScratch

		#ifdef DEBUG_LNLIKE
		cout << "X" << endl;
		cout << "--------" << endl;
		viewMatrix(m, 1, X);
		cout << endl;
		#endif

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, MScratch, m, PMinus, m, 0.0, P, m); // Compute P = IMinusKH*PMinus

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, P, m, MScratch, m, 0.0, PMinus, m); // Compute PMinus = P*IMinusKH_Transpose

		for (int colCounter = 0; colCounter < m; colCounter++) {
			#pragma omp simd
			for (int rowCounter = 0; rowCounter < m; rowCounter++) {
				P[colCounter*m+rowCounter] = PMinus[colCounter*m+rowCounter] + R[0]*K[colCounter]*K[rowCounter]; // Compute P = PMinus + K*R*K_Transpose
				}
			}

		#ifdef DEBUG_LNLIKE
		cout << "P" << endl;
		cout << "--------" << endl;
		viewMatrix(m, m, P);
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
