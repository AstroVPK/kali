#!/usr/bin/env python
"""	Module to implement a class that holds light curves.

	For a demonstration of the module, please run the module as a command line program eg.
	bash-prompt$ python lc.py --help
	and
	bash-prompt$ python lc.py
"""

import numpy as np
import abc
import psutil
import types
import pdb

import lib.rand as rand
import lib.CARMATask as CARMATask

class epoch(object):
	def __init__(self, t, x, y, yerr, mask):
		self.t = t
		self.x = x
		self.y = y
		self.yerr = yerr
		self.mask = mask

	def __repr__(self):
		return u"libcarma.epoch(%f, %f, %f, %f, %f)"%(self.t, self.x, self.y, self.yerr, self.mask)

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
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 5.0e-1, maxTimescale = 5.0, supplied = None):
		self._numCadences = numCadences
		self.t = np.zeros(self.numCadences)
		self.x = np.zeros(self.numCadences)
		self.y = np.zeros(self.numCadences)
		self.yerr = np.zeros(self.numCadences)
		self.mask = np.zeros(self.numCadences)
		self._dt = dt
		self._T = self.t[-1] - self.t[0]
		self._IR = IR
		self._tolIR = tolIR
		self._fracIntrinsicVar = fracIntrinsicVar
		self._fracNoiseToSignal = fracNoiseToSignal
		self._maxSigma = maxSigma
		self._minTimescale = minTimescale
		self._maxTimescale = maxTimescale
		self._lcCython = CARMATask.lc(self.t, self.x, self.y, self.yerr, self.mask, dt = self.dt, IR = self._IR, tolIR = self._tolIR, fracIntrinsicVar = self._fracIntrinsicVar, fracNoiseToSignal = self._fracNoiseToSignal, maxSigma = self._maxSigma, minTimescale = self._minTimescale, maxTimescale = self._maxTimescale)
		self._mean = np.mean(self.y)
		self._std = np.std(self.y)
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
	def fracNoiseToSignal(self):
		return self._fracNoiseToSignal

	@fracNoiseToSignal.setter
	def fracNoiseToSignal(self, value):
		self._fracNoiseToSignal = value
		self._lcCython.fracNoiseToSignal = value

	@property
	def maxSigma(self):
		return self._maxSigma

	@maxSigma.setter
	def maxSigma(self, value):
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
		return u"libcarma.lc(%f, %s, %f, %f, %f, %f, %f, %f, %f)"%(self._numCadences, self._IR, self._tolIR, self.t_incr, self._fracIntrinsicVar, self._fracSignalToNoise, self._maxSigma, self._minTimescale, self._maxTimescale)

	def __str__(self):
		line = 'numCadences: %d\n'%(self._numCadences)
		line += 'IR (Irregularly sampled): %s\n'%(self._IR)
		line += 'tolIR (Tolerance for irregularity): %f\n'%(self._tolIR)
		line += 'fracIntrinsicVar (Intrinsic variability fraction): %f\n'%(self._fracIntrinsicVar)
		line += 'fracNoiseToSignal (Noise to signal fraction): %f\n'%(self._fracSignalToNoise)
		line += 'maxSigma (Maximum allowed sigma multiplier): %f\n'%(self._maxSigma)
		line += 'minTimescale (Minimum allowed timescale factor): %f\n'%(self._minTimescale)
		line += 'maxTimescale (Maximum allowed timescale factor): %f\n'%(self._maxTimescale)
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
		raise NotImplementedError(r'Override readlc!')

class basicLC(lc):
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 1.0e2, minTimescale = 5.0e-1, maxTimescale = 5.0, supplied = None):
		super(basicLC, self).__init__(numCadences, dt, IR, tolIR, fracIntrinsicVar, fracNoiseToSignal, maxSigma, minTimescale, maxTimescale, supplied)

	def _readlc(self, supplied):
		data = np.loadtxt(supplied)

class task(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, p, q, nthreads = psutil.cpu_count(logical = False), nburn = 1000000, nwalkers = 25*psutil.cpu_count(logical = False), nsteps = 250, scatterFactor = 1.0e-1, maxEvals = 1000, xTol = 0.005):
		try:
			assert p > q, r'p must be greater than q'
			assert p >= 1, r'p must be greater than or equal to 1'
			assert type(p) is types.IntType, r'p must be an integer'
			assert q >= 0, r'q must be greater than or equal to 0'
			assert type(q) is types.IntType, r'q must be an integer'
			assert nthreads > 0, r'nthreads must be greater than 0'
			assert type(nthreads) is types.IntType, r'nthreads must be an integer'
			assert nburn >= 0, r'nburn must be greater than or equal to 0'
			assert type(nburn) is types.IntType, r'nburn must be an integer'
			assert nwalkers > 0, r'nwalkers must be greater than 0'
			assert type(nwalkers) is types.IntType, r'nwalkers must be an integer'
			assert nsteps > 0, r'nsteps must be greater than 0'
			assert type(nsteps) is types.IntType, r'nsteps must be an integer'
			assert scatterFactor > 0.0, r'scatterFactor must be greater than 0'
			assert type(scatterFactor) is types.FloatType, r'scatterFactor must be a float'
			assert maxEvals > 0, r'maxEvals must be greater than 0'
			assert type(maxEvals) is types.IntType, r'maxEvals must be an integer'
			assert xTol > 0.0, r'xTol must be greater than 0'
			assert type(xTol) is types.FloatType, r'xTol must be a float'
			self._p = p
			self._q = q
			self._ndims = self._p + self._q + 1
			self._nthreads = nthreads
			self._nburn = nburn
			self._nwalkers = nwalkers
			self._nsteps = nsteps
			self._scatterFactor = scatterFactor
			self._maxEvals = maxEvals
			self._xTol = xTol
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
			self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
			self._taskCython = CARMATask.CARMATask(self._p, self._q, self._nthreads, self._nburn)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def p(self):
		return self._p

	@p.setter
	def p(self, value):
		try:
			assert value > self._q, r'p must be greater than q'
			assert value >= 1, r'p must be greater than or equal to 1'
			assert type(value) is types.IntType, r'p must be an integer'
			self._taskCython.reset_Task(value, self._q, self._nburn)
			self._p = value
			self._ndims = self._p + self._q + 1
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def q(self):
		return self._q

	@q.setter
	def q(self, value):
		try:
			assert value > self._q, r'p must be greater than q'
			assert value >= 0, r'q must be greater than or equal to 0'
			assert type(value) is types.IntType, r'q must be an integer'
			self._taskCython.reset_Task(self._p, value, self._nburn)
			self._q = value
			self._ndims = self._p + self._q + 1
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def nthreads(self):
		return self._nthreads

	@property
	def nburn(self):
		return self._nburn

	@nburn.setter
	def nburn(self, nburnVal):
		try:
			assert value >= 0, r'nburn must be greater than or equal to 0'
			assert type(value) is types.IntType, r'nburn must be an integer'
			self._taskCython.reset_Task(self._p, self._q, value)
			self._nburn = value
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def nwalkers(self):
		return self._nwalkers

	@nwalkers.setter
	def nwalkers(self, value):
		try:
			assert value >= 0, r'nwalkers must be greater than or equal to 0'
			assert type(value) is types.IntType, r'nwalkers must be an integer'
			self._nwalkers = value
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
			self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def nsteps(self):
		return self._nsteps

	@nsteps.setter
	def nsteps(self, value):
		try:
			assert value >= 0, r'nsteps must be greater than or equal to 0'
			assert type(value) is types.IntType, r'nsteps must be an integer'
			self._nsteps = value
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
			self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def scatterFactor(self):
		return self._scatterFactor

	@scatterFactor.setter
	def scatterFactor(self, value):
		try:
			assert value >= 0.0, r'scatterFactor must be greater than or equal to 0.0'
			assert type(value) is types.FloatType, r'scatterFactor must be a float'
			self._scatterFactor = value
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def maxEvals(self):
		return self._maxEvals

	@maxEvals.setter
	def maxEvals(self, value):
		try:
			assert value >= 0, r'maxEvals must be greater than or equal to 0'
			assert type(value) is types.IntType, r'maxEvals must be an integer'
			self._maxEvals = value
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def xTol(self):
		return self._xTol

	@xTol.setter
	def xTol(self, value):
		try:
			assert value >= 0, r'xTol must be greater than or equal to 0'
			assert type(value) is types.FloatType, r'xTol must be a float'
			self._xTol = value
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def Chain(self):
		return np.reshape(self._Chain, newshape = (self._ndims, self._nwalkers, self._nsteps), order = 'F')

	@property
	def LnPosterior(self):
		return np.reshape(self._LnPosterior, newshape = (self._nwalkers, self._nsteps), order = 'F')

	def __repr__(self):
		return "libcarma.task(%d, %d, %d, %d, %d, %d, %f, %d, %f)"%(self._p, self._q, self._nthreads, self._nburn, self._nwalkers, self._nsteps, self._scatterFactor, self._maxEvals, self._xTol)

	def __str__(self):
		line = 'p: %d; q: %d; ndims: %d\n'%(self._p, self._q, self._ndims)
		line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
		line += 'nburn (Number of light curve steps to burn): %d\n'%(self._nburn)
		line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
		line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
		line += 'scatterFactor (Standard Deviation of factor by which to perturb from initial guess): %f\n'%(self._scatterFactor)
		line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for MCMC): %d\n'%(self._maxEvals)
		line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)
		return line

	def __eq__(self, other):
		if type(other) == task:
			if (self._p == other.p) and (self._q == other.q) and (self._nthreads == other.nthreads) and (self._nburn == other.nburn) and (self._nwalkers == other.nwalkers) and (self._nsteps == other.nsteps) and (self._scatterFactor == other.scatterFactor) and (self._maxEvals == other.maxEvals) and (self.xTol == other.xTol):
				return True
			else:
				return False
		else:
			return False

	def __neq__(self, other):
		if self == other:
			return False
		else:
			return True

	def reset(self, p = None, q = None, nburn = None, nwalkers = None, nsteps = None):
		if p is None:
			p = self._p
		if q is None:
			q = self._q
		if nburn is None:
			nburn = self._nburn
		if nwalkers is None:
			nwalkers = self._nwalkers
		if nsteps is None:
			nsteps = self._nsteps
		try:
			assert p > q, r'p must be greater than q'
			assert p >= 1, r'p must be greater than or equal to 1'
			assert type(p) is types.IntType, r'p must be an integer'
			assert q >= 0, r'q must be greater than or equal to 0'
			assert type(q) is types.IntType, r'q must be an integer'
			assert nburn >= 0, r'nburn must be greater than or equal to 0'
			assert type(nburn) is types.IntType, r'nburn must be an integer'
			assert nwalkers > 0, r'nwalkers must be greater than 0'
			assert type(nwalkers) is types.IntType, r'nwalkers must be an integer'
			assert nsteps > 0, r'nsteps must be greater than 0'
			assert type(nsteps) is types.IntType, r'nsteps must be an integer'
			self._taskCython.reset_Task(p, q, nburn)
			self._p = p
			self._q = q
			self._ndims = self._p + self._q + 1
			self._nburn = nburn
			self._nwalkers = nwalkers
			self._nsteps = nsteps
			self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
			self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
		except AssertionError as err:
			raise AttributeError(str(err))

	def check(self, Theta, tnum = None):
		if tnum is None:
			tnum = 0
		assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
		return bool(self._taskCython.check_Theta(Theta, tnum))

	def set(self, dt, Theta, tnum = None):
		if tnum is None:
			tnum = 0
		assert dt > 0.0, r'dt must be greater than 0.0'
		assert type(dt) is types.FloatType, r'dt must be a float'
		assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
		self._taskCython.set_System(dt, Theta, tnum)

	def dt(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_dt(tnum)

	def Theta(self, tnum = None):
		if tnum is None:
			tnum = 0
		Theta = np.zeros(self._ndims)
		self._taskCython.get_Theta(Theta, tnum)
		return Theta

	def list(self):
		setSystems = np.zeros(self._nthreads, dtype = 'int32')
		self._taskCython.get_setSystemsVec(setSystems)
		return setSystems.astype(np.bool_)

	def show(self, tnum = None):
		if tnum is None:
			tnum = 0
		self._taskCython.print_System(tnum)

	def Sigma(self, tnum = None):
		if tnum is None:
			tnum = 0
		Sigma = np.zeros(self._p*self._p)
		self._taskCython.get_Sigma(Sigma, tnum)
		return np.reshape(Sigma, newshape = (self._p, self._p), order = 'F')

	def simulate(self, numCadences, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 5.0e-1, maxTimescale = 5.0, burnSeed = None, distSeed = None, noiseSeed = None, noisy = False, tnum = None):
		if tnum is None:
			tnum = 0
		if noisy == False:
			intrinsicLC = basicLC(numCadences, dt = self._taskCython.get_dt(threadNum = tnum), IR = False, tolIR = 1.0e-3, fracIntrinsicVar = fracIntrinsicVar, fracNoiseToSignal = fracNoiseToSignal, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale)
			randSeed = np.zeros(1, dtype = 'uint32')
			if burnSeed is None:
				rand.rdrand(randSeed)
				burnSeed = randSeed[0]
			if distSeed is None:
				rand.rdrand(randSeed)
				distSeed = randSeed[0]
			self._taskCython.make_IntrinsicLC(intrinsicLC.numCadences, intrinsicLC.IR, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, burnSeed, distSeed, threadNum = tnum)
			intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0] 
			return intrinsicLC
		else:
			observedLC = basicLC(numCadences, dt = self._taskCython.get_dt(threadNum = tnum), IR = False, tolIR = 1.0e-3, fracIntrinsicVar = fracIntrinsicVar, fracNoiseToSignal = fracNoiseToSignal, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale)
			randSeed = np.zeros(1, dtype = 'uint32')
			if burnSeed is None:
				rand.rdrand(randSeed)
				burnSeed = randSeed[0]
			if distSeed is None:
				rand.rdrand(randSeed)
				distSeed = randSeed[0]
			if noiseSeed is None:
				rand.rdrand(randSeed)
				noiseSeed = randSeed[0]
			self._taskCython.make_ObservedLC(observedLC.numCadences, observedLC.IR, observedLC.tolIR, observedLC.fracIntrinsicVar, observedLC.fracNoiseToSignal, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, burnSeed, distSeed, noiseSeed, threadNum = tnum)
			observedLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
			return observedLC

	def observe(self, intrinsicLC, noiseSeed = None, tnum = None):
		if tnum is None:
			tnum = 0
		randSeed = np.zeros(1, dtype = 'uint32')
		if noiseSeed is None:
			rand.rdrand(randSeed)
			noiseSeed = randSeed[0]
		self._taskCython.add_ObservationNoise(intrinsicLC.numCadences, intrinsicLC.IR, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum = tnum)
		intrinsicLC._mean = np.mean(intrinsicLC.y)
		intrinsicLC._std = np.std(intrinsicLC.y)

	def logPrior(self, observedLC, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.compute_LnPrior(observedLC.numCadences, observedLC.IR, observedLC.tolIR, observedLC.maxSigma*observedLC._std, observedLC.minTimescale*observedLC._dt, observedLC.maxTimescale*observedLC._T, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)

	def logLikelihood(self, observedLC, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.compute_LnLikelihood(observedLC.numCadences, observedLC.IR, observedLC.tolIR, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)

	def logPosterior(self, observedLC, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.compute_LnPosterior(observedLC.numCadences, observedLC.IR, observedLC.tolIR, observedLC.maxSigma*observedLC._std, observedLC.minTimescale*observedLC._dt, observedLC.maxTimescale*observedLC._T, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)

	def fit(self, observedLC, xStart, zSSeed = None, walkerSeed = None, moveSeed = None, xSeed = None):
		randSeed = np.zeros(1, dtype = 'uint32')
		if zSSeed is None:
			rand.rdrand(randSeed)
			zSSeed = randSeed[0]
		if walkerSeed is None:
			rand.rdrand(randSeed)
			walkerSeed = randSeed[0]
		if moveSeed is None:
			rand.rdrand(randSeed)
			moveSeed = randSeed[0]
		if xSeed is None:
			rand.rdrand(randSeed)
			xSeed = randSeed[0]
		return self._taskCython.fit_CARMAModel(observedLC.dt, observedLC.numCadences, observedLC.IR, observedLC.tolIR, observedLC.maxSigma*observedLC._std, observedLC.minTimescale*observedLC._dt, observedLC.maxTimescale*observedLC._T, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, self.scatterFactor, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPosterior)

class basicTask(task):
	def __init__(self, p, q, nthreads = psutil.cpu_count(logical = False), nburn = 1000000, nwalkers = 25*psutil.cpu_count(logical = False), nsteps = 250, scatterFactor = 1.0e-1, maxEvals = 1000, xTol = 0.005):
		super(basicTask, self).__init__(p, q, nthreads, nburn, nwalkers, nsteps, scatterFactor, maxEvals, xTol)
