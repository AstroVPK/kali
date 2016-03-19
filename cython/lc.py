#!/usr/bin/env python
"""	Module to implement a class that holds light curves.

	For a demonstration of the module, please run the module as a command line program eg.
	bash-prompt$ python lc.py --help
	and
	bash-prompt$ python lc.py
"""

import numpy as np
import abc
import pdb

import CARMATask

class epoch(object):
	def __init__(self, t, x, y, yerr, mask):
		self.t = t
		self.x = x
		self.y = y
		self.yerr = yerr
		self.mask = mask

	def __repr__(self):
		return u"lc.epoch(%f, %f, %f, %f, %f)"%(self.t, self.x, self.y, self.yerr, self.mask)

	def __str__(self):
		if self.mask == 1.0:
			return r't = %f MJD; intrinsic flux = %+f; observed flux = %+f; observed flux error = %+f'%(self.t, self.x, self.y, self.yerr)
		else:
			return r't = %f MJD; no data!'%(self.t)

	def __eq__(self, other):
		if type(other) is type(self):
			return self.__dict__ == other.__dict__
		return False

	def __neq__(self, other):
		if self == other:
			return False
		else:
			return True

class lc(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracSignalToNoise = 0.001, maxSigma = 1.0e2, minTimescale = 5.0e-1, maxTimescale = 5.0, supplied = None):
		self._numCadences = numCadences
		self.t = np.zeros(self.numCadences)
		self.x = np.zeros(self.numCadences)
		self.y = np.zeros(self.numCadences)
		self.yerr = np.zeros(self.numCadences)
		self.mask = np.zeros(self.numCadences)
		self._dt = dt
		self._IR = IR
		self._tolIR = tolIR
		self._fracIntrinsicVar = fracIntrinsicVar
		self._fracSignalToNoise = fracSignalToNoise
		self._maxSigma = maxSigma
		self._minTimescale = minTimescale
		self._maxTimescale = maxTimescale
		self._lcCython = CARMATask.lc(self.t, self.x, self.y, self.yerr, self.mask, dt = self.dt, IR = self._IR, tolIR = self._tolIR, fracIntrinsicVar = self._fracIntrinsicVar, fracSignalToNoise = self._fracSignalToNoise, maxSigma = self._maxSigma, minTimescale = self._minTimescale, maxTimescale = self._maxTimescale)
		if not supplied:
			for i in xrange(self._numCadences):
				self.t[i] = i*self._dt
				self.mask[i] = 1.0
		else:
			self._readlc(supplied)

	@property
	def numCadences(self):
		return self._numCadences

	@property
	def dt(self):
		return self._dt

	@property
	def IR(self):
		return self._IR

	@IR.setter
	def IR(self, value):
		self._IR = value
		self._lcCython.IR = value

	@property
	def tolIR(self):
		return self._tolIR

	@tolIR.setter
	def tolIR(self, value):
		self._tolIR = value
		self._lcCython.tolIR = value

	@property
	def fracIntrinsicVar(self):
		return self._fracIntrinsicVar

	@fracIntrinsicVar.setter
	def fracIntrinsicVar(self, value):
		self._fracIntrinsicVar = value
		self._lcCython.fracIntrinsicVar = value

	@property
	def fracSignalToNoise(self):
		return self._fracSignalToNoise

	@fracSignalToNoise.setter
	def fracSignalToNoise(self, value):
		self._fracSignalToNoise = value
		self._lcCython.fracSignalToNoise = value

	@property
	def maxSigma(self):
		return self._maxSigma

	@maxSigma.setter
	def fracSignalToNoise(self, value):
		self._maxSigma = value
		self._lcCython.maxSigma = value

	@property
	def minTimescale(self):
		return self._minTimescale

	@minTimescale.setter
	def minTimescale(self, value):
		self._minTimescale = value
		self._lcCython.minTimescale = value

	@property
	def maxTimescale(self):
		return self._maxTimescale

	@maxTimescale.setter
	def maxTimescale(self, value):
		self._maxTimescale = value
		self._lcCython.maxTimescale = value

	def __len__(self):
		return self._numCadences

	def __repr__(self):
		return u"lc.lc(%f, %s, %f, %f, %f, %f, %f, %f, %f)"%(self._numCadences, self._IR, self._tolIR, self._t_incr, self._fracIntrinsicVar, self._fracSignalToNoise, self._maxSigma, self._minTimescale, self._maxTimescale)

	def __str__(self):
		line = 'numCadences: %d\n'%(self._numCadences)
		line += 'Irregularly sampled: %s\n'%(self._IR)
		line += 'Tolerance for irregularity: %f\n'%(self._tolIR)
		line += 'Intrinsic variability fraction: %f\n'%(self._fracIntrinsicVar)
		line += 'Noise to signal: %f\n'%(self._fracSignalToNoise)
		line += 'Maximum allowed sigma multiplier: %f\n'%(self._maxSigma)
		line += 'Minimum allowed timescale: %f\n'%(self._minTimescale)
		line += 'Maximum allowed timescale: %f\n'%(self._maxTimescale)
		for i in xrange(self._numCadences - 1):
			line = line + str(self[i]) + '\n'
		line = line + str(self[self._numCadences - 1])
		return line

	def __eq__(self, other):
		if type(other) is type(self):
			return self.__dict__ == other.__dict__
		return False

	def __neq__(self, other):
		if self == other:
			return False
		else:
			return True

	def __getitem__(self, key):
		return epoch(self.t[key], self.x[key], self.y[key], self.yerr[key], self.mask[key])

	def __setitem__(self, key, val):
		if isinstance(val, epoch):
			self.t[key] = val.t
			self.x[key] = val.x
			self.y[key] = val.y
			self.yerr[key] = val.yerr
			self.mask[key] = val.mask

	@abc.abstractmethod
	def _readlc(self, supplied):
		raise NotImplementedError('Override readlc ')

class basicLC(lc):
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracSignalToNoise = 0.001, maxSigma = 1.0e2, minTimescale = 5.0e-1, maxTimescale = 5.0, supplied = None):
		super(basicLC, self).__init__(numCadences, dt, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, maxSigma, minTimescale, maxTimescale, supplied)

	def _readlc(self, supplied):
		data = np.loadtxt(supplied)