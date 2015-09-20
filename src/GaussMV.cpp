#include <malloc.h>
#include <sys/time.h>
#include <mathimf.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_types.h>
#include <iostream>
#include "Constants.hpp"
#include "GaussMV.hpp"

using namespace std;

double calcLnLike(double* x, void* vdPtr2LnLikeArgs) {