# distutils: language = c++
import math
import cython
import numpy as np
import psutil
cimport numpy as np
from libcpp cimport bool

cdef extern from 'binarySMBHTask.hpp':
	cdef cppclass binarySMBHTask:
		binarySMBHTask(int numThreads) except+
		int check_Theta(double *Theta, int threadNum);
		void get_Theta(double *Theta, int threadNum);
		int set_System(double *Theta, int threadNum);
		int reset_System(double timeGiven, int threadNum);
		void get_setSystemsVec(int *setSystems);
		int print_System(int threadNum);

		int make_IntrinsicLC(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, int threadNum);
		int add_ObservationNoise(int numCadences, double fracNoiseToSignal, double *t, double *x, double *y, double *yerr, double *mask, unsigned int noiseSeed, int threadNum);

cdef class bSMBHTask:
	cdef binarySMBHTask *thisptr

	def __cinit__(self, numThreads = None):
		if numThreads == None:
			numThreads = int(psutil.cpu_count(logical = False))
		self.thisptr = new binarySMBHTask(numThreads)

	def __dealloc__(self):
		del self.thisptr

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def check_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.check_Theta(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_Theta(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.get_Theta(&Theta[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_System(self, np.ndarray[double, ndim=1, mode='c'] Theta not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.set_System(&Theta[0], threadNum)

	def reset_System(self, epoch, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.reset_System(epoch, threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_setSystemsVec(self, np.ndarray[int, ndim=1, mode='c'] setSystems not None):
		self.thisptr.get_setSystemsVec(&setSystems[0])

	def print_System(self, threadNum = None):
		if threadNum == None:
			threadNum = 0
		self.thisptr.print_System(threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def make_IntrinsicLC(self, numCadences, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.make_IntrinsicLC(numCadences, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], threadNum)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def add_ObservationNoise(self, numCadences, fracNoiseToSignal, np.ndarray[double, ndim=1, mode='c'] t not None, np.ndarray[double, ndim=1, mode='c'] x not None, np.ndarray[double, ndim=1, mode='c'] y not None, np.ndarray[double, ndim=1, mode='c'] yerr not None, np.ndarray[double, ndim=1, mode='c'] mask not None, noiseSeed, threadNum = None):
		if threadNum == None:
			threadNum = 0
		return self.thisptr.add_ObservationNoise(numCadences, fracNoiseToSignal, &t[0], &x[0], &y[0], &yerr[0], &mask[0], noiseSeed, threadNum)