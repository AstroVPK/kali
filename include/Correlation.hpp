#ifndef CORRELATION_HPP
#define CORRELATION_HPP

#include <mkl.h>
#include <mkl_types.h>

using namespace std;

void viewMatrix(int nRows, int nCols, double* mat);

double dtime();

void ACVF(int numCadences, const double* const y, const double* const mask, double* acvf);

void ACF(int numCadences, const double* const acvf, double* acf);

void PACF(int numCadences, int maxLag, const double* const acvf, double* pacf);

void SF1(int numCadences, const double* const acvf, double* sf1);

#endif
