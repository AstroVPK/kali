#ifndef GAUSSMV_HPP
#define GAUSSMV_HPP

#include <mkl.h>
#include <mkl_types.h>

using namespace std;

double calcLnLike(double* walkerPos, void* vdPtr2LnLikeArgs);

struct GaussMV {
	double *mu, *var;
	};

struct LnLikeArgs {
	int numPts;
	double *y;
	GaussMV probDist;
	};