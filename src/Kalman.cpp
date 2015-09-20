#include <malloc.h>
#include <sys/time.h>
#include <limits>
#include <mathimf.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_types.h>
#include <iostream>
#include <vector>
#include "Constants.hpp"
#include "Kalman.hpp"
#include <stdio.h>

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

double calcARMALnLike(const vector<double> &x, vector<double>& grad, void* p2Args) {
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

double calcARMALnLike(double* walkerPos, void* func_args) {

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

void viewMatrix(int nRows, int nCols, double* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			cout << mat[j*nCols + i] << " ";
			}
		cout << endl;
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

DLM::DLM() {

	#ifdef DEBUG_CTORDLM
	int threadNum = omp_get_thread_num();
	printf("DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	allocated = 0;
	isStable = 0;
	isInvertible = 0;
	isNotRedundant = 0;
	isReasonable = 0;
	p = 0;
	q = 0;
	m = 0;
	mSq = 0;
	distSigma = 0.0;
	ilo = nullptr;
	ihi = nullptr;
	ARz = nullptr;
	MAz = nullptr;
	ARMatrix = nullptr;
	MAMatrix = nullptr;
	ARScale = nullptr;
	MAScale = nullptr;
	ARTau = nullptr;
	MATau = nullptr;
	ARwr = nullptr;
	ARwi = nullptr;
	MAwr = nullptr;
	MAwi = nullptr;
	A = nullptr;
	B = nullptr;
	I = nullptr;
	F = nullptr;
	FKron = nullptr;
	FKronPiv = nullptr;
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

DLM::~DLM() {

	#ifdef DEBUG_DTORDLM
	int threadNum = omp_get_thread_num();
	printf("~DLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	allocated = 0;
	isStable = 0;
	isInvertible = 0;
	isNotRedundant = 0;
	isReasonable = 0;
	p = 0;
	q = 0;
	m = 0;
	mSq = 0;
	distSigma = 0.0;
	ilo = nullptr;
	ihi = nullptr;
	ARz = nullptr;
	MAz = nullptr;
	ARMatrix = nullptr;
	MAMatrix = nullptr;
	ARScale = nullptr;
	MAScale = nullptr;
	ARTau = nullptr;
	MATau = nullptr;
	ARwr = nullptr;
	ARwi = nullptr;
	MAwr = nullptr;
	MAwi = nullptr;
	Theta = nullptr;
	A = nullptr;
	B = nullptr;
	I = nullptr;
	F = nullptr;
	FKron = nullptr;
	FKronPiv = nullptr;
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

void DLM::allocDLM(int numP, int numQ) {

	#ifdef DEBUG_ALLOCATEDLM
	int threadNum = omp_get_thread_num();
	printf("allocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	p = numP;
	q = numQ;
	allocated = 0;
	m = static_cast<int>(fmax(static_cast<double>(p),static_cast<double>(q+1)));
	mSq = m*m;

	ilo = static_cast<lapack_int*>(_mm_malloc(sizeof(lapack_int),64));
	ihi = static_cast<lapack_int*>(_mm_malloc(sizeof(lapack_int),64));
	allocated += 2*sizeof(lapack_int);

	ilo[0] = 0;
	ihi[0] = 0;

	ARz = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	MAz = static_cast<double*>(_mm_malloc(1*sizeof(double),64));
	allocated += 2*sizeof(double);

	ARz[0] = 0.0;
	MAz[0] = 0.0;

	ARScale = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	ARwr = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	ARwi = static_cast<double*>(_mm_malloc(p*sizeof(double),64));
	allocated += 3*p*sizeof(double);

	#pragma omp simd
	for (int i = 0; i < p; i++) {
		ARScale[i] = 0.0;
		ARwr[i] = 0.0;
		ARwi[i] = 0.0;
		}

	MAScale = static_cast<double*>(_mm_malloc(q*sizeof(double),64));
	MAwr = static_cast<double*>(_mm_malloc(q*sizeof(double),64));
	MAwi = static_cast<double*>(_mm_malloc(q*sizeof(double),64));
	allocated += 3*q*sizeof(double);

	#pragma omp simd
	for (int i = 0; i < q; i++) {
		MAScale[i] = 0.0;
		MAwr[i] = 0.0;
		MAwi[i] = 0.0;
		}

	ARTau = static_cast<double*>(_mm_malloc((p-1)*sizeof(double),64));
	allocated += (p-1)*sizeof(double);

	#pragma omp simd
	for (int i = 0; i < p-1; i++) {
		ARTau[i] = 0.0;
		}

	MATau = static_cast<double*>(_mm_malloc((q-1)*sizeof(double),64));
	allocated += (q-1)*sizeof(double);

	#pragma omp simd
	for (int i = 0; i < q-1; i++) {
		MATau[i] = 0.0;
		}

	ARMatrix = static_cast<double*>(_mm_malloc(p*p*sizeof(double),64));
	for (int rowCtr = 0; rowCtr < p; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < p; colCtr++) {
			ARMatrix[rowCtr + p*colCtr] = 0.0; // Initialize matrix.
			}
		}
	allocated += p*p*sizeof(double);

	MAMatrix = static_cast<double*>(_mm_malloc(q*q*sizeof(double),64));
	for (int rowCtr = 0; rowCtr < q; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < q; colCtr++) {
			MAMatrix[rowCtr + q*colCtr] = 0.0; // Initialize matrix.
			}
		}
	allocated += q*q*sizeof(double);

	Theta = static_cast<double*>(_mm_malloc((p+q+1)*sizeof(double),64));
	for (int i = 0; i < (p+q+1); ++i) {
		Theta[i] = 0.0;
		}
	allocated += (p+q+1)*sizeof(double);

	A = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	B = static_cast<double*>(_mm_malloc(m*sizeof(double),64));

	I = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	F = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));

/*
	printf("allocDLM - threadNum: %d, Address of System: %p; Value of F: %p\n",threadNum,this,F);
	fflush(0);
	for(int i=0; i < m*m; ++i)
	{
		printf("address of F[%d]: 0x%x\n", i, &(F[i]));
		fflush(0);
	}
	
	
	//printf("allocDLM - guessed size of F: 0x%X\n", *((size_t*)(((void*)F));
	//printf("allocDLM - guessed size of F: 0x%X\n", *((size_t*)(((void*)F) - sizeof(size_t))) );
	//printf("allocDLM - causing trouble 0x%x\n", *((int*)(*((size_t*)(((void*)F) - sizeof(size_t))))) );
	//fflush(0);

	printf("***************************************\n");
	printf("Looking at F BEFORE FKron\n");
	viewMatrix(m,m,F);
	printf("***************************************\n");
	
	printf("Exploring memory of F in 8 byte increments\n");
	double* temp = F;
	while( temp != &F[m*m-1]+1) 
	{
		printf("Looking at address %p, dumped value: %f\n", temp, *temp);
		fflush(0);
		temp++;
	}
	
	printf("Exploring extra memory after F to FKron in 8 byte increments\n");
	
	MKL_INT64* temp1 = (MKL_INT64*)(&F[m*m-1])+1;
	while( temp1 != ((MKL_INT64*)F)+16) 
	{
		char buf[8];
		//memcpy(buf, temp1, 8);
		printf("Looking at address %p, dumped value: 0x%x\n", temp1, *temp1);
		fflush(0);
		temp1++;
	}
	*/
	FKron = static_cast<double*>(_mm_malloc(mSq*mSq*sizeof(double),64));
	/*
	printf("THIS IS THE VALUE OF FKRON!!!!!! %p\n", FKron);
	
	printf("***************************************\n");
	printf("Looking at F AFTER FKron\n");
	viewMatrix(m,m,F);
	printf("***************************************\n");
	
	printf("Exploring memory of F in 8 byte increments\n");
	temp = F;
	while( temp != &F[m*m-1]+1) 
	{
		printf("Looking at address %p, dumped value: %f\n", temp, *temp);
		fflush(0);
		temp++;
	}
	
	printf("Exploring extra memory after F to FKron in 8 byte increments\n");
	
	temp1 = (MKL_INT64*)(&F[m*m-1])+1;
	while( temp1 != (MKL_INT64*)FKron) 
	{
		//memcpy(buf, temp1, 8);
		printf("Looking at address %p, dumped value: 0x%x\n", temp1, *temp1);
		fflush(0);
		temp1++;
	}
	* */
	//FKron = static_cast<double*>(malloc(mSq*mSq*sizeof(double)));
	

	/*printf("FKron is mSq*mSq*sizeof(double) = %d bytes\n",mSq*mSq*sizeof(double));
	printf("allocDLM - threadNum: %d, Address of System: %p; Value of Fkron: %p\n",threadNum,this,FKron);
	fflush(0);
	printf("allocDLM - guessed size of FKron: 0x%X\n", *((size_t*)(((void*)FKron) - sizeof(size_t))) );
	fflush(0);*/

	//FKronPiv = static_cast<lapack_int*>(_mm_malloc((mSq+1)*sizeof(lapack_int),64));
	FKronPiv = static_cast<lapack_int*>(_mm_malloc(mSq*mSq*sizeof(lapack_int),64));
	Q = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	P = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	PMinus = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	MScratch = static_cast<double*>(_mm_malloc(m*m*sizeof(double),64));
	allocated += 6*m*m*sizeof(double);
	allocated += mSq*mSq*sizeof(double);
	allocated += mSq*mSq*sizeof(lapack_int);

	D = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	H = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	K = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	X = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	XMinus = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	VScratch = static_cast<double*>(_mm_malloc(m*sizeof(double),64));
	allocated += 6*m*sizeof(double);

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

	for (int i = 0; i < mSq; i++) {
		//FKronPiv[i] = 0;
		#pragma omp simd
		for (int j = 0; j < mSq; j++) {
			FKron[i*mSq + j] = 0.0;
			FKronPiv[i*mSq + j] = 0;
			}
		}
	//FKronPiv[mSq] = 0;

	R = static_cast<double*>(_mm_malloc(sizeof(double),64));
	allocated += sizeof(double);

	D[0] = 1.0;
	R[0] = 0.0;

	#pragma omp simd
	for (int i = 1; i < m; i++) {
		F[i*m+(i-1)] = 1.0;
		I[(i-1)*m+(i-1)] = 1.0;
		}
	I[(m-1)*m+(m-1)] = 1.0;

	#ifdef DEBUG_ALLOCATEDLM
	printf("allocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	}

void DLM::deallocDLM() {

	#ifdef DEBUG_DEALLOCATEDLM
	int threadNum = omp_get_thread_num();
	printf("deallocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	int stepNum = 0;
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 0
	stepNum += 1;
	#endif

	if (ilo) {
		_mm_free(ilo);
		ilo = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 1
	stepNum += 1;
	#endif

	if (ihi) {
		_mm_free(ihi);
		ihi = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 2
	stepNum += 1;
	#endif

	if (ARz) {
		_mm_free(ARz);
		ARz = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 3
	stepNum += 1;
	#endif

	if (MAz) {
		_mm_free(MAz);
		MAz = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 4
	stepNum += 1;
	#endif

	if (ARTau) {
		_mm_free(ARTau);
		ARTau = nullptr;
	}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 5
	stepNum += 1;
	#endif

	if (MATau) {
		_mm_free(MATau);
		MATau = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 6
	stepNum += 1;
	#endif

	if (ARScale) {
		_mm_free(ARScale);
		ARScale = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 7
	stepNum += 1;
	#endif

	if (MAScale) {
		_mm_free(MAScale);
		MAScale = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 8
	stepNum += 1;
	#endif

	if (ARMatrix) {
		_mm_free(ARMatrix);
		ARMatrix = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 9
	stepNum += 1;
	#endif

	if (MAMatrix) {
		_mm_free(MAMatrix);
		MAMatrix = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 10
	stepNum += 1;
	#endif

	if (ARwr) {
		_mm_free(ARwr);
		ARwr = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 11
	stepNum += 1;
	#endif

	if (ARwi) {
		_mm_free(ARwi);
		ARwi = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 12
	stepNum += 1;
	#endif

	if (MAwr) {
		_mm_free(MAwr);
		MAwr = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 13
	stepNum += 1;
	#endif

	if (MAwi) {
		_mm_free(MAwi);
		MAwi = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 14
	stepNum += 1;
	#endif

	if (Theta) {
		_mm_free(Theta);
		Theta = nullptr;
		}

	if (A) {
		_mm_free(A);
		A = nullptr;
		}

	if (B) {
		_mm_free(B);
		B = nullptr;
		}

	if (I) {
		_mm_free(I);
		I = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 15
	stepNum += 1;
	#endif

	/*printf("***************************************\n");
	printf("Looking at F\n");
	viewMatrix(m,m,F);
	printf("***************************************\n");

	printf("Exploring memory of F in 8 byte increments\n");
	double* temp = F;
	while( temp != &F[m*m-1]+1) 
	{
		printf("Looking at address %p, dumped value: %f\n", temp, *temp);
		fflush(0);
		temp++;
	}
	
	printf("Exploring extra memory after F to FKron in 4 byte increments\n");
	
	MKL_INT64* temp1 = (MKL_INT64*)(&F[m*m-1])+1;
	while( temp1 != (MKL_INT64*)FKron) 
	{
		char buf[8];
		//memcpy(buf, temp1, 8);
		printf("Looking at address %p, dumped value: 0x%x\n", temp1, *temp1);
		fflush(0);
		temp1++;
	}
	
	printf("deallocDLM - threadNum: %d, Address of System: %p; Address of F: %p\n",threadNum,this,F);
	fflush(0);*/

	if (F) {
		_mm_free(F);
		F = nullptr;
		}

	/*printf("deallocDLM - threadNum: %d, Address of System: %p; Address of F: %p\n",threadNum,this,F);
	fflush(0);*/

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 16
	stepNum += 1;
	#endif

	/*printf("deallocDLM - threadNum: %d, Address of System: %p; Address of FKRON: %p\n",threadNum,this,FKron);
	fflush(0);

	printf("***************************************\n");
	printf("Looking at FKron\n");
	viewMatrix(mSq,mSq,FKron);
	printf("***************************************\n");*/

	if (FKron) {
		_mm_free(FKron);
		FKron = nullptr;
		}

	/*printf("deallocDLM - threadNum: %d, Address of System: %p; Address of FKRON: %p\n",threadNum,this,FKron);
	fflush(0);*/

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 17
	stepNum += 1;
	#endif

	if (FKronPiv) {
		_mm_free(FKronPiv);
		FKronPiv = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 18
	stepNum += 1;
	#endif

	if (D) {
		_mm_free(D);
		D = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 19
	stepNum += 1;
	#endif

	if (Q) {
		_mm_free(Q);
		Q = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 20
	stepNum += 1;
	#endif

	if (H) {
		_mm_free(H);
		H = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 21
	stepNum += 1;
	#endif

	if (R) {
		_mm_free(R);
		R = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 22
	stepNum += 1;
	#endif

	if (K) {
		_mm_free(K);
		K = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 23
	stepNum += 1;
	#endif

	if (X) {
		_mm_free(X);
		X = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 24
	stepNum += 1;
	#endif

	if (P) {
		_mm_free(P);
		P = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 25
	stepNum += 1;
	#endif

	if (XMinus) {
		_mm_free(XMinus);
		XMinus = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 26
	stepNum += 1;
	#endif

	if (PMinus) {
		_mm_free(PMinus);
		PMinus = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 27
	stepNum += 1;
	#endif

	if (VScratch) {
		_mm_free(VScratch);
		VScratch = nullptr;
		} 

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 28
	stepNum += 1;
	#endif

	if (MScratch) {
		_mm_free(MScratch);
		MScratch = nullptr;
		}

	#ifdef DEBUG_DEALLOCATEDLM_DEEP
	printf("deallocDLM - threadNum: %d; Address of System: %p; Step: %d\n",threadNum,this,stepNum); // 29
	stepNum += 1;
	#endif

	#ifdef DEBUG_DEALLOCATEDLM
	printf("deallocDLM - threadNum: %d; Address of System: %p\n",threadNum,this);
	#endif

	}

void DLM::setDLM(double* Theta) {

	#ifdef DEBUG_SETDLM
	int threadNum = omp_get_thread_num();
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	#endif

	isStable = 1;
	isInvertible = 1;
	isNotRedundant = 1;
	isReasonable = 0;

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; isStable: %d\n",threadNum,isStable);
	printf("setDLM - threadNum: %d; isInvertible: %d\n",threadNum,isInvertible);
	printf("setDLM - threadNum: %d; isNotRedundant: %d\n",threadNum,isNotRedundant);
	printf("setDLM - threadNum: %d; isReasonable: %d\n",threadNum,isReasonable);
	#endif

	distSigma = Theta[0];

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; distSigma: %f\n",threadNum,distSigma);
	#endif

	for (int rowCtr = 0; rowCtr < p; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < p; colCtr++) {
			ARMatrix[rowCtr + p*colCtr] = 0.0; // Reset matrix.
			}
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

	ARMatrix[(p-1)*p] = 1.0/Theta[p]; // ARMatrix has -1 at top right!
	#pragma omp simd
	for (int rowCtr = 1; rowCtr < p; rowCtr++) {
		ARMatrix[rowCtr+(p-1)*p] = -1.0*Theta[rowCtr]/Theta[p]; // Rightmost column of ARMatrix has AR coeffs.
		ARMatrix[rowCtr+(rowCtr-1)*p] = 1.0; // ARMatrix has Identity matrix in bottom left.
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

	#pragma omp simd
	for (int rowCtr = 0; rowCtr < p-1; rowCtr++) {
		ARScale[rowCtr] = 0.0;
		ARwr[rowCtr] = 0.0;
		ARwi[rowCtr] = 0.0;
		ARTau[rowCtr] = 0.0;
		}
	ARScale[p-1] = 0.0;
	ARwr[p-1] = 0.0;
	ARwi[p-1] = 0.0;
	ARz[0] = 0.0;

	for (int rowCtr = 0; rowCtr < q; rowCtr++) {
		#pragma omp simd
		for (int colCtr = 0; colCtr < q; colCtr++) {
			MAMatrix[rowCtr + q*colCtr] = 0.0; // Initialize matrix.
			}
		}

	MAMatrix[(q-1)*q] = -1.0/Theta[p+q]; // MAMatrix has -1 at top right!
	#pragma omp simd
	for (int rowCtr = 1; rowCtr < q; rowCtr++) {
		MAMatrix[rowCtr+(q-1)*q] = -1.0*Theta[p+rowCtr]/Theta[p+q]; // Rightmost column of MAMatrix has -MA coeffs.
		MAMatrix[rowCtr+(rowCtr-1)*q] = 1.0; // MAMatrix has Identity matrix in bottom left.
		}
	#pragma omp simd
	for (int rowCtr = 0; rowCtr < q-1; rowCtr++) {
		MAScale[rowCtr] = 0.0;
		MAwr[rowCtr] = 0.0;
		MAwi[rowCtr] = 0.0;
		MATau[rowCtr] = 0.0;
		}
	MAScale[q-1] = 0.0;
	MAwr[q-1] = 0.0;
	MAwi[q-1] = 0.0;
	MAz[0] = 0.0;

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; F\n",threadNum);
	viewMatrix(m,m,F);
	#endif

	#pragma omp simd
	for (int i = 0; i < p; i++) {
		F[i] = Theta[1+i];
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; F\n",threadNum);
	viewMatrix(m,m,F);
	printf("\n");
	#endif

	for (int i = 0; i < mSq; i++) {
		#pragma omp simd
		for (int j = 0; j < mSq; j++) {
			FKron[i + mSq*j] = 0.0;
			FKronPiv[i + mSq*j] = 0;
			}
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	printf("setDLM - threadNum: %d; D\n",threadNum);
	viewMatrix(m,1,D);
	#endif

	#pragma omp simd
	for (int i = 0; i < q; i++) {
		D[i+1] = Theta[1+p+i];
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; D\n",threadNum);
	viewMatrix(m,1,D);
	printf("\n");
	#endif

	#ifdef DEBUG_SETDLM
	int threadNum = omp_get_thread_num();
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; Q\n",threadNum);
	viewMatrix(m,m,Q);
	#endif

	for (int i = 0; i < m; i++) {
		#pragma omp simd
		for (int j = 0; j < m; j++) {
			Q[i*m+j] = pow(distSigma,2.0)*D[i]*D[j];
			}
		}

	#ifdef DEBUG_SETDLM
	printf("setDLM - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("setDLM - threadNum: %d; Q\n",threadNum);
	viewMatrix(m,m,Q);
	printf("\n");
	#endif

	H[0] = 1.0;
	}

void DLM::resetState(double InitUncertainty) {

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

void DLM::resetState() {

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

	#ifdef DEBUG_RESETSTATE
	printf("***************************************\n");
	printf("Looking at FKron in reset before LAPACK\n");
	viewMatrix(mSq,mSq,FKron);
	printf("***************************************\n");
	#endif

/*****************************************THIS IS WHERE THE BUG IS****************************************/


	lapack_int YesNo;
	YesNo = LAPACKE_dgetrf(LAPACK_COL_MAJOR, mSq, mSq, FKron, mSq, FKronPiv);


/*****************************************THIS IS WHERE THE BUG IS****************************************/

	#ifdef DEBUG_RESETSTATE
	printf("***************************************\n");
	printf("Looking at FKron in reset after LAPACK\n");
	viewMatrix(mSq,mSq,FKron);
	printf("***************************************\n");
	#endif

	cblas_dcopy(mSq, Q, 1, P, 1);

	#ifdef DEBUG_RESETSTATE
	printf("***************************************\n");
	printf("Looking at FKron in reset after LAPACK\n");
	viewMatrix(mSq,mSq,FKron);
	printf("***************************************\n");
	#endif

	YesNo = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', mSq, 1, FKron, mSq, FKronPiv, P, mSq);

	#ifdef DEBUG_RESETSTATE
	printf("***************************************\n");
	printf("Looking at FKron in reset after LAPACK\n");
	viewMatrix(mSq,mSq,FKron);
	printf("***************************************\n");
	#endif

	}

int DLM::checkARMAParams(double* Theta) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);

	#ifdef DEBUG_CHECKARMAPARAMS
	int threadNum = omp_get_thread_num();
	printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
	printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	printf("checkARMAParams - threadNum: %d; distSigma: %f\n",threadNum,distSigma);
	#endif

	if (distSigma > 0.0) {
		isReasonable = 1;
		if ((p == 1) and ((Theta[1] >= 1.0) or (Theta[1] <= -1.0))) { // Single AR component. Just check if -1 < phi_1 < 1.
			isStable = 0;
		} else if (p > 1) { // Only have to do this if AR Poly is atleast 2nd Order.

			lapack_int YesNo;

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Address of ARMatrix: %p\n",threadNum,&ARMatrix);
			printf("checkARMAParams - threadNum: %d; ARMatrix\n",threadNum);
			viewMatrix(p,p,ARMatrix);
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking AR Matrix...\n",threadNum);
			#endif

			YesNo = LAPACKE_dgebal(LAPACK_COL_MAJOR, 'B', p, ARMatrix, p, ilo, ihi, ARScale);

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking AR Matrix...\n",threadNum);
			#endif

			YesNo = LAPACKE_dgehrd(LAPACK_COL_MAJOR, p, *ilo, *ihi, ARMatrix, p, ARTau);


			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking AR Matrix...\n",threadNum);
			#endif

			YesNo = LAPACKE_dhseqr(LAPACK_COL_MAJOR,'E', 'N', p, *ilo, *ihi, ARMatrix, p, ARwr, ARwi, ARz, 1);

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Done checking AR Matrix!\n",threadNum);
			#endif

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

				if (pow(ARwr[i], 2.0) + pow(ARwi[i],2.0) <= 1.0) {

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
				}
		}


	#ifdef DEBUG_CHECKARMAPARAMS
	printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
	printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
	for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
		printf("%f ",Theta[dimNum]);
		}
	printf("\n");
	#endif

		if ((q == 1) and ((Theta[p+1] >= 1.0) or (Theta[p+1] <= -1.0))) { // Single MA component. Just check if < 1.
			isInvertible = 0;
		} else if (q > 1) { // Only have to do this if MA Poly is atleast 2nd Order.

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking MA Matrix...\n",threadNum);
			#endif

			lapack_int YesNo;

			YesNo = LAPACKE_dgebal(LAPACK_COL_MAJOR, 'B', q, MAMatrix, q, ilo, ihi, MAScale);

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking MA Matrix...\n",threadNum);
			#endif

			YesNo = LAPACKE_dgehrd(LAPACK_COL_MAJOR, q, *ilo, *ihi, MAMatrix, q, MATau);

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Checking MA Matrix...\n",threadNum);
			#endif

			YesNo = LAPACKE_dhseqr(LAPACK_COL_MAJOR,'E', 'N', q, *ilo, *ihi, MAMatrix, q, MAwr, MAwi, MAz, 1);

			#ifdef DEBUG_CHECKARMAPARAMS
			printf("checkARMAParams - threadNum: %d; Address of System: %p\n",threadNum,(void*)&System);
			printf("checkARMAParams - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < p+q+1; dimNum++) {
				printf("%f ",Theta[dimNum]);
				}
			printf("\n");
			printf("checkARMAParams - threadNum: %d; Done checking MA Matrix!\n",threadNum);
			#endif

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

				if (pow(MAwr[i], 2.0) + pow(MAwi[i],2.0) <= 1.0) {

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
		}
		for (int i = 1; i < p; i++) {
			for (int j = 1; j < q; j++) {
				if ((ARwr[i] == MAwr[j]) and (ARwi[i] == MAwi[j])) {
					isNotRedundant = 0;
					}
				}
			}
		} else {
		isReasonable = 0;
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

	return isStable*isInvertible*isNotRedundant*isReasonable;
	}

void DLM::getARRoots(double*& RealAR, double*& ImagAR) {
	RealAR = ARwr;
	ImagAR = ARwi;
	}

void DLM::getMARoots(double*& RealMA, double*& ImagMA) {
	RealMA = MAwr;
	ImagMA = MAwi;
	}

void DLM::burnSystem(int numBurn, unsigned int burnSeed, double* burnRand) {

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

double DLM::observeSystem(double distRand, double noiseRand) {

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, F, m, X, 1, 0.0, VScratch, 1);
	cblas_dcopy(m, VScratch, 1, X, 1);
	cblas_daxpy(m, distRand, D, 1, X, 1);
	return cblas_ddot(m, H, 1, X, 1) + noiseRand;
	}

double DLM::observeSystem(double distRand, double noiseRand, double mask) {

	double result;
	if (mask != 0.0) {
		result = observeSystem(distRand, noiseRand);
		} else {
		result = 0.0;
		}
	return result;
	}

void DLM::observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y) {

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

void DLM::observeSystem(int numObs, unsigned int distSeed, unsigned int noiseSeed, double* distRand, double* noiseRand, double noiseSigma, double* y, double* mask) {

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

double DLM::computeLnLike(int numPts, double* y, double* yerr) {

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

double DLM::computeLnLike(int numPts, double* y, double* yerr, double* mask) {

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
