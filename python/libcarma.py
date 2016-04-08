#!/usr/bin/env python
"""	Module to perform basic C-ARMA modelling.

	For a demonstration of the module, please run the module as a command line program eg.
	bash-prompt$ python libcarma.py --help
	and
	bash-prompt$ python libcarma.py
"""

import numpy as np
import abc
import psutil
import types
import os
import pdb

import rand as rand
import CARMATask as CARMATask

class epoch(object):
	"""!
	\anchor epoch_
	
	\brief Class to hold individual epochs of a light curve.
	
	We wish to hold individual epochs in a light curve in an organized manner. This class lets us examine individual epochs and check for equality with other epochs. Two epochs are equal iff they have the same tiemstamp. Later on, we will implement some sort of unit system for the quantities (i.e. is the tiumestamp in sec, min, day, MJD etc...?)
	"""
	def __init__(self, t, x, y, yerr, mask):
		"""!
		\brief Initialize the epoch.
		
		Non-keyword arguments
		\param[in] t:    Timestamp.
		\param[in] x:    Intrinsic Flux i.e. the theoretical value of the underlying flux in the absence of measurement error.
		\param[in] y:    Observed Flux i.e. the observed value of the flux given a noise-to-signal level.
		\param[in] yerr: Error in Observed Flux i.e. the measurement error in the flux.
		\param[in] mask: Mask value at this epoch. 0.0 means that this epoch has a missing observation. 1.0 means that the observation exists.
		"""
		self.t = t ## Timestamp
		self.x = x ## Intrinsic flux
		self.y = y ## Observed flux
		self.yerr = yerr ## Error in Observed Flux
		self.mask = mask ## Mask value at this epoch

	def __repr__(self):
		"""!
		\brief Return a representation of the epoch such that eval(repr(someEpoch)) == someEpoch is True.
		"""
		return u"libcarma.epoch(%f, %f, %f, %f, %f)"%(self.t, self.x, self.y, self.yerr, self.mask)

	def __str__(self):
		"""!
		\brief Return a human readable representation of the epoch.
		"""
		if self.mask == 1.0:
			return r't = %f MJD; intrinsic flux = %+f; observed flux = %+f; observed flux error = %+f'%(self.t, self.x, self.y, self.yerr)
		else:
			return r't = %f MJD; no data!'%(self.t)

	def __eq__(self, other):
		"""!
		\brief Check for equality.
		
		Check for equality. Two epochs are equal iff the timestamps are equal.
		
		Non-keyword arguments
		\param[in] other: Another epoch or subclass of epoch.
		"""
		if type(other) is type(self):
			return self.t == other.t
		return False

	def __neq__(self, other):
		"""!
		\brief Check for inequality.
		
		Check for inequality. Two epochs are not equal iff the timestamps are not equal.
		
		Non-keyword arguments
		\param[in] other: Another epoch or subclass of epoch.
		"""
		if self == other:
			return False
		else:
			return True

class lc(object):
	"""!
	\anchor lc_
	
	\brief Class to hold light curve.
	
	ABC to model a light curve. Light curve objects consist of a number of properties and numpy arrays to hold the list of t, x, y, yerr, and mask.
	"""
	__metaclass__ = abc.ABCMeta
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 5.0e-1, maxTimescale = 2.0, supplied = None):
		"""!
		\brief Initialize a new light curve
		
		The constructor assumes that the light curve to be constructed is regular. There is no option to construct irregular light curves. Irregular light can be obtained by reading in a supplied irregular light curve. The constructor takes an optional keyword argument (supplied = <light curve file>) that is read in using the read method. This supplied light curve can be irregular. Typically, the supplied light curve is irregular either because the user created it that way, or because the survey that produced it sampled the sky at irregular intervals.
		
		Non-keyword arguments
		\param[in] numCadences: The number of cadences in the light curve
		
		Keyword arguments
		\param[in] dt:                The spacing between cadences.
		\param[in] IR:                Is the light curve irregular?
		\param[in] tolIR:             The tolerance level at which a given step in the lightcurve should be considered irregular for the purpose of solving the C-ARMA model. The C-ARMA model needs to be re-solved if abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where t_incr is the new increment in time and dt is the previous increment in time. If IR  == False, this parameter is not used.
		\param[in] fracIntrinsicVar:  The fractional variability of the source i.e. fracIntrinsicVar = sqrt(Sigma[0,0])/mean_flux.
		\param[in] fracNoiseToSignal: The fractional noise level i.e. fracNoiseToSignal = sigma_noise/flux. We assume that this level is fixed. In the future, we may wish to make this value flux dependent to make the noise model more realistic.
		\param[in] maxSigma:          The maximum allowed value of sqrt(Sigma[0,0]) = maxSigma*stddev(y) when fitting a C-ARMA model. Note that if the observed light curve is shorter than the de-correlation timescale, stddev(y) may be much smaller than sqrt(Sigma[0,0]) and hence maxSigma should be made larger in such cases.
		\param[in] minTimescale:      The shortest allowed timescale = minTimescale*dt. Note that if the observed light curve is very sparsely sampled, dt may be much larger than the actaul minimum timescale present and hence minTimescale should be made smaller in such cases.
		\param[in] maxTimescale:      The longest allowed timescale = maxTimescale*T. Note that if the observed light curve is shorter than the longest timescale present, T may be much smaller than the longest timescale and hence maxTimescale should be made larger in such cases.
		\param[in] supplied:          Reference for supplied light curve. Since this class is an ABC, individual subclasses must implement a read method and the format expected for supplied (i.e. full path or name etc...) will be determined by the subclass.
		"""
		self._numCadences = numCadences ## The number of cadences in the light curve. This is not the same thing as the number of actual observations as we can have missing observations.
		self.t = np.zeros(self.numCadences) ## Numpy array of timestamps.
		self.x = np.zeros(self.numCadences) ## Numpy array of intrinsic fluxes.
		self.y = np.zeros(self.numCadences) ## Numpy array of observed fluxes.
		self.yerr = np.zeros(self.numCadences) ## Numpy array of observed flux errors.
		self.mask = np.zeros(self.numCadences) ## Numpy array of mask values.
		self._dt = dt ## Increment between epochs.
		self._T = self.t[-1] - self.t[0] ## Total duration of the light curve.
		self._IR = IR ## Is the light curve irregular?
		self._tolIR = tolIR ## Tolerance on the irregularity. If IR == False, this parameter is not used. Otherwise, a timestep is irregular iff abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where t_incr is the new increment in time and dt is the previous increment in time.
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
			self.read(supplied)

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
		"""!
		\brief Return a representation of the lc such that eval(repr(someLC)) == someLC is True.
		"""
		return u"libcarma.lc(%f, %s, %f, %f, %f, %f, %f, %f, %f)"%(self._numCadences, self._dt, self._IR, self._tolIR, self._fracIntrinsicVar, self._fracNoiseToSignal, self._maxSigma, self._minTimescale, self._maxTimescale)

	def __str__(self):
		"""!
		\brief Return a human readable representation of the light curve.
		"""
		line = 'numCadences: %d\n'%(self._numCadences)
		line += 'IR (Irregularly sampled): %s\n'%(self._IR)
		line += 'tolIR (Tolerance for irregularity): %f\n'%(self._tolIR)
		line += 'fracIntrinsicVar (Intrinsic variability fraction): %f\n'%(self._fracIntrinsicVar)
		line += 'fracNoiseToSignal (Noise to signal fraction): %f\n'%(self._fracNoiseToSignal)
		line += 'maxSigma (Maximum allowed sigma multiplier): %f\n'%(self._maxSigma)
		line += 'minTimescale (Minimum allowed timescale factor): %f\n'%(self._minTimescale)
		line += 'maxTimescale (Maximum allowed timescale factor): %f\n'%(self._maxTimescale)
		for i in xrange(self._numCadences - 1):
			line = line + str(self[i]) + '\n'
		line = line + str(self[self._numCadences - 1])
		return line

	def __eq__(self, other):
		"""!
		\brief Check for equality.
		
		Check for equality. Two light curves are equal only iff all thier attributes are the same.
		
		Non-keyword arguments
		\param[in] other: Another lc or subclass of lc.
		"""
		if type(other) is type(self):
			return self.__dict__ == other.__dict__
		return False

	def __neq__(self, other):
		"""!
		\brief Check for inequality.
		
		Check for inequality. Two light curves are in-equal only iff all thier attributes are not the same.
		
		Non-keyword arguments
		\param[in] other: Another lc or subclass of lc.
		"""
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
	def read(self, supplied):
		raise NotImplementedError(r'Override read by subclassing lc!')

	@abc.abstractmethod
	def write(self, path = os.environ['PWD']):
		raise NotImplementedError(r'Override write by subclassing lc!')

class basicLC(lc):
	def __init__(self, numCadences, dt = 1.0, IR = False, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 1.0e2, minTimescale = 5.0e-1, maxTimescale = 5.0, supplied = None):
		super(basicLC, self).__init__(numCadences, dt, IR, tolIR, fracIntrinsicVar, fracNoiseToSignal, maxSigma, minTimescale, maxTimescale, supplied)

	def read(self, supplied):
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

	def acvf(self, start = 0.0, stop = 100.0, num = 100, endpoint = True, base  = 10.0, spacing = 'linear'):
		if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
			lags = np.logspace(np.log10(start)/np.log10(base), np.log10(stop)/np.log10(base), num  = num, endpoint = endpoint, base = base)
		elif spacing.lower() in ['linear', 'lin']:
			lags = np.linspace(start, stop, num  = num, endpoint = endpoint)
		else:
			raise RuntimeError('Unable to parse spacing')
		acvfs = np.zeros(num)
		self._taskCython.compute_ACVF(num, lags, acvfs)
		return lags, acvfs

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
