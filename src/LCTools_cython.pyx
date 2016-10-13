# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool


cdef extern from 'LC.hpp' namespace "kali":
	int acvf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *acvfVals, double *acvfErrvals)
	int acf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *acvfVals, double *acvfErrvals)
	int sf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double*maskIn, double *lagVals, double *sfVals, double *sfErrVals)
	int dacf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, int numBins, double *lagVals, double *acvfVals, double *acvfErrVals)



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ACVF(numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] acvfVals not None, np.ndarray[double, ndim=1, mode='c'] acvfErrVals not None):
	return acvf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], &lagVals[0], &acvfVals[0], &acvfErrVals[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ACF(numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] acfVals not None, np.ndarray[double, ndim=1, mode='c'] acfErrVals not None):
	return acf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], &lagVals[0], &acfVals[0], &acfErrVals[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_SF(numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] sfVals not None, np.ndarray[double, ndim=1, mode='c'] sfErrVals not None):
	return sf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], &lagVals[0], &sfVals[0], &sfErrVals[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_DACF(numCadences, dt, np.ndarray[double, ndim=1, mode='c'] tIn not None, np.ndarray[double, ndim=1, mode='c'] xIn not None, np.ndarray[double, ndim=1, mode='c'] yIn not None, np.ndarray[double, ndim=1, mode='c'] yerrIn not None, np.ndarray[double, ndim=1, mode='c'] maskIn not None, numBins, np.ndarray[double, ndim=1, mode='c'] lagVals not None, np.ndarray[double, ndim=1, mode='c'] dacfVals not None, np.ndarray[double, ndim=1, mode='c'] dacfErrVals not None):
	return dacf(numCadences, dt, &tIn[0], &xIn[0], &yIn[0], &yerrIn[0], &maskIn[0], numBins, &lagVals[0], &dacfVals[0], &dacfErrVals[0])
