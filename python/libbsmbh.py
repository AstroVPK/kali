#!/usr/bin/env python
"""	Module to perform basic C-ARMA modelling.

	For a demonstration of the module, please run the module as a command line program eg.
	bash-prompt$ python libcarma.py --help
	and
	bash-prompt$ python libcarma.py
"""

import numpy as np
import math as math
import scipy.stats as spstats
import cmath as cmath
import random
import sys as sys
import abc as abc
import psutil as psutil
import types as types
import os as os
import reprlib as reprlib
import copy as copy
import warnings as warnings
import matplotlib.pyplot as plt
import pdb as pdb

try:
	import rand as rand
	import libcarma
	import bSMBHTask as bSMBHTask
	from util.mpl_settings import set_plot_params
except ImportError:
	print 'libbsmbh is not setup. Setup libbsmbh by sourcing bin/setup.sh'
	sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

class binarySMBHTask(object):
	lenTheta = 9
	G = 6.67408e-11
	c = 299792458.0
	pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
	Parsec = 3.0857e16
	Day = 86164.090530833
	Year = 31557600.0
	SolarMass = 1.98855e30

	def __init__(self, nthreads = psutil.cpu_count(logical = True), nwalkers = 25*psutil.cpu_count(logical = True), nsteps = 250, maxEvals = 10000, xTol = 0.005, mcmcA = 2.0):
		try:
			assert nthreads > 0, r'nthreads must be greater than 0'
			assert type(nthreads) is types.IntType, r'nthreads must be an integer'
			assert nwalkers > 0, r'nwalkers must be greater than 0'
			assert type(nwalkers) is types.IntType, r'nwalkers must be an integer'
			assert nsteps > 0, r'nsteps must be greater than 0'
			assert type(nsteps) is types.IntType, r'nsteps must be an integer'
			assert maxEvals > 0, r'maxEvals must be greater than 0'
			assert type(maxEvals) is types.IntType, r'maxEvals must be an integer'
			assert xTol > 0.0, r'xTol must be greater than 0'
			assert type(xTol) is types.FloatType, r'xTol must be a float'
			self._ndims = self.lenTheta
			self._nthreads = nthreads
			self._nwalkers = nwalkers
			self._nsteps = nsteps
			self._maxEvals = maxEvals
			self._xTol = xTol
			self._mcmcA = mcmcA
			self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
			self._LnPosterior = np.require(np.zeros(self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
			self._taskCython = bSMBHTask.bSMBHTask(self._nthreads)
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def nthreads(self):
		return self._nthreads

	@property
	def ndims(self):
		return self._ndims

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
	def mcmcA(self):
		return self._mcmcA

	@mcmcA.setter
	def mcmcA(self, value):
		try:
			assert value >= 0, r'mcmcA must be greater than or equal to 0.0'
			assert type(value) is types.FloatType, r'mcmcA must be a float'
			self._mcmcA = value
		except AssertionError as err:
			raise AttributeError(str(err))

	@property
	def Chain(self):
		return np.reshape(self._Chain, newshape = (self._ndims, self._nwalkers, self._nsteps), order = 'F')

	@property
	def LnPosterior(self):
		return np.reshape(self._LnPosterior, newshape = (self._nwalkers, self._nsteps), order = 'F')

	def __repr__(self):
		return "libsmbh.task(%d, %d, %d, %d, %f)"%(self._nthreads, self._nwalkers, self._nsteps, self._maxEvals, self._xTol)

	def __str__(self):
		line = 'ndims: %d\n'%(self._ndims)
		line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
		line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
		line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
		line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for MCMC): %d\n'%(self._maxEvals)
		line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)

	def __eq__(self, other):
		if type(other) == task:
			if (self._nthreads == other.nthreads) and (self._nwalkers == other.nwalkers) and (self._nsteps == other.nsteps) and (self._maxEvals == other.maxEvals) and (self.xTol == other.xTol):
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

	def reset(self, nwalkers = None, nsteps = None):
		if nwalkers is None:
			nwalkers = self._nwalkers
		if nsteps is None:
			nsteps = self._nsteps
		try:
			assert nwalkers > 0, r'nwalkers must be greater than 0'
			assert type(nwalkers) is types.IntType, r'nwalkers must be an integer'
			assert nsteps > 0, r'nsteps must be greater than 0'
			assert type(nsteps) is types.IntType, r'nsteps must be an integer'
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

	def set(self, Theta, tnum = None):
		if tnum is None:
			tnum = 0
		assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
		return self._taskCython.set_System(Theta, tnum)

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

	def period(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Period(tnum)

	def simulate(self, duration, dt = 0.1, fracNoiseToSignal = 0.001, tnum = None):
		if tnum is None:
			tnum = 0
		numCadences = int(round(float(duration)/dt))
		intrinsicLC = libcarma.basicLC(numCadences, dt = dt, fracNoiseToSignal = fracNoiseToSignal)
		self._taskCython.make_IntrinsicLC(intrinsicLC.numCadences, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, threadNum = tnum)
		intrinsicLC._simulatedCadenceNum = numCadences - 1
		intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0] 
		return intrinsicLC

	def observe(self, intrinsicLC, noiseSeed = None, tnum = None):
		if tnum is None:
			tnum = 0
		randSeed = np.zeros(1, dtype = 'uint32')
		if noiseSeed is None:
			rand.rdrand(randSeed)
			noiseSeed = randSeed[0]
		self._taskCython.add_ObservationNoise(intrinsicLC.numCadences, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum = tnum)

		count = int(np.sum(intrinsicLC.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(intrinsicLC.numCadences):
			y_meanSum += intrinsicLC.mask[i]*intrinsicLC.y[i]
			yerr_meanSum += intrinsicLC.mask[i]*intrinsicLC.yerr[i]
		if count > 0.0:
			intrinsicLC._mean = y_meanSum/count
			intrinsicLC._meanerr = yerr_meanSum/count
		else:
			intrinsicLC._mean = 0.0
			intrinsicLC._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(intrinsicLC.numCadences):
			y_stdSum += math.pow(intrinsicLC.mask[i]*intrinsicLC.y[i] - intrinsicLC._mean, 2.0)
			yerr_stdSum += math.pow(intrinsicLC.mask[i]*intrinsicLC.yerr[i] - intrinsicLC._meanerr, 2.0)
		if count > 0.0:
			intrinsicLC._std = math.sqrt(y_stdSum/count)
			intrinsicLC._stderr = math.sqrt(yerr_stdSum/count)
		else:
			intrinsicLC._std = 0.0
			intrinsicLC._stderr = 0.0

	def logPrior(self, observedLC, forced = False, tnum = None):
		if tnum is None:
			tnum = 0
		lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)])
		highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)])
		observedLC._logPrior =  self._taskCython.compute_LnPrior(observedLC.numCadences, lowestFlux, highestFlux, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)
		return observedLC._logPrior

	def logLikelihood(self, observedLC, forced = False, tnum = None):
		if tnum is None:
			tnum = 0
		observedLC._logPrior = self.logPrior(observedLC, forced = forced, tnum = tnum)
		if forced == True:
			observedLC._computedCadenceNum = -1
		if observedLC._computedCadenceNum == -1:
			observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(observedLC.numCadences, observedLC._computedCadenceNum, observedLC.t, observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask, tnum)
			observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
			observedLC._computedCadenceNum = observedLC.numCadences - 1
		else:
			pass
		return observedLC._logLikelihood

	def logPosterior(self, observedLC, forced = True, tnum = None):
		lnLikelihood = self.logLikelihood(observedLC, forced = forced, tnum = tnum)
		return observedLC._logPosterior

	def fit(self, observedLC, zSSeed = None, walkerSeed = None, moveSeed = None, xSeed = None):
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
		xStart = np.require(np.zeros(self.ndims*self.nwalkers), requirements=['F', 'A', 'W', 'O', 'E'])
		lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)])
		highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)])

		for walkerNum in xrange(self.nwalkers):
			noSuccess = True
			while noSuccess:
				m1Guess = math.pow(10.0, random.uniform(-1.0, 3.0)) # m1 in 1e6*SolarMass
				m2Guess = math.pow(10.0, random.uniform(-1.0, math.log10(m1Guess))) # m1 in 1e6SolarMass
				rS1Guess = ((2.0*self.G*m1Guess*1.0e6*self.SolarMass)/math.pow(self.c, 2.0)/self.Parsec) # rS1 in Parsec
				rS2Guess = ((2.0*self.G*m2Guess*1.0e6*self.SolarMass)/math.pow(self.c, 2.0)/self.Parsec) # rS2 in Parsec
				rPerGuess = math.pow(10.0, random.uniform(math.log10(10.0*(rS1Guess + rS1Guess)), 1.0)) #rPer in Parsec
				eGuess = random.uniform(0.0, 1.0)
				omegaGuess = random.uniform(0.0, 2*math.pi)
				inclinationGuess = random.uniform(0.0, math.pi)
				periodGuess = (2.0*math.pi*math.sqrt((math.pow((self.Parsec*rPerGuess)/(1.0 + eGuess), 3.0))/(self.G*(m1Guess + m2Guess)*1.0e6*self.SolarMass)))/self.Day
				tauGuess = random.uniform(0.0, periodGuess)
				totalFluxGuess = random.uniform(lowestFlux, highestFlux)
				fracBeamedFluxGuess = random.uniform(0.0, 1.0)
				ThetaGuess = np.array([rPerGuess, m1Guess, m2Guess, eGuess, omegaGuess, inclinationGuess, tauGuess, totalFluxGuess, fracBeamedFluxGuess])
				res = self.set(ThetaGuess)
				lnPrior = self.logPrior(observedLC)
				if res == 0 and lnPrior == 0.0:
					noSuccess = False
			for dimNum in xrange(self.ndims):
				xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]

		res = self._taskCython.fit_BinarySMBHModel(observedLC.numCadences, lowestFlux, highestFlux, observedLC.t, observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, self.mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPosterior)
		return res