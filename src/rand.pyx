# distutils: language = c++

import numpy as np
cimport numpy as np
import warnings
import cython
DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

cdef extern from 'rdrand.hpp':
	cdef int rdrand_get_n_32(unsigned int n, unsigned int* x)

@cython.boundscheck(False)
@cython.wraparound(False)
def rdrand(np.ndarray[unsigned int, ndim=1, mode="c"] inputArr not None):
	cdef int numRand = inputArr.shape[0]
	success = rdrand_get_n_32(numRand, &inputArr[0])
	if success != 1:
		warnings.warn('Intel RDRAND failed with error code %d! Using numpy.random'%(success))
		inputArr = np.random.randint(0, 4294967295, numRand)
	return success