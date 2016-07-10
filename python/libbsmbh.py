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
from scipy.interpolate import UnivariateSpline
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

from gatspy.periodic import LombScargleFast

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

def d2r(degree):
	return degree*(math.pi/180.0)

def r2d(radian):
	return radian*(180.0/math.pi)

class binarySMBHTask(object):
	lenTheta = 8
	G = 6.67408e-11
	c = 299792458.0
	pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
	twoPi = 2.0*pi
	Parsec = 3.0857e16
	Day = 86164.090530833
	Year = 31557600.0
	SolarMass = 1.98855e30

	def __init__(self, nthreads = psutil.cpu_count(logical = True), nwalkers = 25*psutil.cpu_count(logical = True), nsteps = 250, maxEvals = 10000, xTol = 0.1, mcmcA = 2.0):
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

	def __call__(self, epochVal, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.set_Epoch(epochVal, tnum)

	def epoch(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Epoch(tnum)

	def period(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Period(tnum)

	def a1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_A1(tnum)

	def a2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_A2(tnum)

	def m1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_M1(tnum)

	def m2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_M2(tnum)

	def rPeri1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RPeri1(tnum)

	def rPeri2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RPeri2(tnum)

	def rApo1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RApo1(tnum)

	def rApo2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RApo2(tnum)

	def rPeri(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RPeriTot(tnum)

	def rApo(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RApoTot(tnum)

	def rS1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RS1(tnum)

	def rS2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RS2(tnum)

	def eccentricity(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Eccentricity(tnum)

	def omega1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_omega1(tnum)

	def omega2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_omega2(tnum)

	def inclination(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Inclination(tnum)

	def tau(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Tau(tnum)

	def M(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_MeanAnomoly(tnum)

	def E(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_EccentricAnomoly(tnum)

	def nu(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_TrueAnomoly(tnum)

	def r1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_R1(tnum)

	def r2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_R2(tnum)

	def theta1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Theta1(tnum)

	def theta2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Theta2(tnum)

	def Beta1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Beta1(tnum)

	def Beta2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_Beta2(tnum)

	def radialBeta1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RadialBeta1(tnum)

	def radialBeta2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_RadialBeta2(tnum)

	def dopplerFactor1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_DopplerFactor1(tnum)

	def dopplerFactor2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_DopplerFactor2(tnum)

	def beamingFactor1(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_BeamingFactor1(tnum)

	def beamingFactor2(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_BeamingFactor2(tnum)

	def aH(self, sigmaStars = 200.0, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_aH(sigmaStars, tnum)

	def aGW(self, sigmaStars = 200.0, rhoStars = 1000.0, H = 16, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_aGW(sigmaStars, rhoStars, H, tnum)

	def durationInHardState(self, sigmaStars = 200.0, rhoStars = 1000.0, H = 16, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_durationInHardState(sigmaStars, rhoStars, H, tnum)

	def ejectedMass(self, sigmaStars = 200.0, rhoStars = 1000.0, H = 16, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_ejectedMass(sigmaStars, rhoStars, H, tnum)

	def simulate(self, duration, dt = None, fracNoiseToSignal = 0.001, tnum = None):
		if tnum is None:
			tnum = 0
		if dt is None:
			dt = self.period()/100.0
		numCadences = int(round(float(duration)/dt))
		intrinsicLC = libcarma.basicLC(numCadences, dt = dt, fracNoiseToSignal = fracNoiseToSignal)
		self._taskCython.make_IntrinsicLC(intrinsicLC.numCadences, intrinsicLC.dt, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, threadNum = tnum)
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
		self._taskCython.add_ObservationNoise(intrinsicLC.numCadences, intrinsicLC.dt, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum = tnum)

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

	def logPrior(self, observedLC, forced = True, tnum = None):
		if tnum is None:
			tnum = 0
		lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)])
		highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)])
		observedLC._logPrior =  self._taskCython.compute_LnPrior(observedLC.numCadences, observedLC.dt, lowestFlux, highestFlux, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)
		return observedLC._logPrior

	def logLikelihood(self, observedLC, forced = True, tnum = None):
		if tnum is None:
			tnum = 0
		observedLC._logPrior = self.logPrior(observedLC, forced = forced, tnum = tnum)
		if forced == True:
			observedLC._computedCadenceNum = -1
		if observedLC._computedCadenceNum == -1:
			observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(observedLC.numCadences, observedLC.dt, observedLC._computedCadenceNum, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)
			observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
			observedLC._computedCadenceNum = observedLC.numCadences - 1
		else:
			pass
		return observedLC._logLikelihood

	def logPosterior(self, observedLC, forced = True, tnum = None):
		lnLikelihood = self.logLikelihood(observedLC, forced = forced, tnum = tnum)
		return observedLC._logPosterior

		try:
			omega1 = math.acos((1.0/eccentricity)*((maxBetaParallel + minBetaParallel)/(maxBetaParallel - minBetaParallel)))
		except ValueError:
			pdb.set_trace()
		return omega1

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
		meanFlux = np.mean(observedLC.y[np.where(observedLC.mask == 1.0)])
		numIntrinsicFlux = 10
		maxPeriodFactor = 10.0
		intrinsicFlux = np.linspace(lowestFlux, highestFlux, num = numIntrinsicFlux)
		for f in xrange(3,numIntrinsicFlux-3):
			beamedLC = observedLC.copy()
			beamedLC.x = np.require(np.zeros(beamedLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			for i in xrange(beamedLC.numCadences):
				beamedLC.y[i] = observedLC.y[i]/intrinsicFlux[f]
				beamedLC.yerr[i] = observedLC.yerr[i]/intrinsicFlux[f]

			dopplerLC = beamedLC.copy()
			dopplerLC.x = np.require(np.zeros(dopplerLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			for i in xrange(observedLC.numCadences):
				dopplerLC.y[i] = math.pow(beamedLC.y[i], 1.0/3.44)
				dopplerLC.yerr[i] = 3.44*math.fabs((dopplerLC.y[i]*(1.0/3.44)*beamedLC.yerr[i])/beamedLC.y[i]) # 3.44 is a magic number here!

			dzdtLC = dopplerLC.copy()
			dzdtLC.x = np.require(np.zeros(dopplerLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			for i in xrange(observedLC.numCadences):
				dzdtLC.y[i] = 1.0 - (1.0/dopplerLC.y[i])
				dzdtLC.yerr[i] = 1.428641*math.fabs(dzdtLC.y[i])*(dopplerLC.yerr[i]/dopplerLC.y[i]) # 1.428641 is a magic number here!

			model = LombScargleFast().fit(dzdtLC.t, dzdtLC.y, dzdtLC.yerr)
			periods, power = model.periodogram_auto(nyquist_factor=100)
			model.optimizer.period_range=(2.0*dzdtLC.dt, maxPeriodFactor*dzdtLC.T)
			periodEst = model.best_period
			foldedLC = dzdtLC.fold(periodEst)
			foldedLC.x = np.require(np.zeros(foldedLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			spl = UnivariateSpline(foldedLC.t[np.where(foldedLC.mask == 1.0)], foldedLC.y[np.where(foldedLC.mask == 1.0)], 1.0/(2.5e2*foldedLC.yerr[np.where(foldedLC.mask == 1.0)]), k = 3, s = None, check_finite = True)
			splinedLC = foldedLC.copy()
			for i in xrange(splinedLC.numCadences):
				splinedLC.x[i] = spl(splinedLC.t[i])
			#splinedLC.x = savgol_filter(foldedLC.y, 5, 2)
			tMin = splinedLC.t[np.where(np.min(splinedLC.x) == splinedLC.x)[0][0]]
			tMax = splinedLC.t[np.where(np.max(splinedLC.x) == splinedLC.x)[0][0]]
			tZeros = spl.roots()
			pdb.set_trace()
			if tZeros.shape[0] != 2:
				continue
			if tMin < tMax:
				tFirst = tMin
				tSecond = tMax
				tMinLesstMax = True
			else:
				tFirst = tMax
				tSecond = tMin
				tMinLesstMax = False
			if tFirst < tZeros[0]:
				orderedTList = [tFirst, tZeros[0], tSecond, tZeros[1]]
				tFirstLesstZeros0 = True
			else:
				orderedTList = [tZeros[0], tFirst, tZeros[1], tSecond]
				tFirstLesstZeros0 = False
			pdb.set_trace()

		''''ThetaGuess = np.array([0.001, 75.0, 10.0, 0.0, 0.0, 90.0, 0.0, 100.0, 0.5])
		for dimNum in xrange(self.ndims):
			xStart[dimNum] = ThetaGuess[dimNum]
		for walkerNum in xrange(1, self.nwalkers):'''
		for walkerNum in xrange(self.nwalkers):
			noSuccess = True
			while noSuccess:
				m1Guess = math.pow(10.0, random.uniform(-1.0, 3.0)) # m1 in 1e6*SolarMass
				m2Guess = math.pow(10.0, random.uniform(-1.0, math.log10(m1Guess))) # m1 in 1e6SolarMass
				rS1Guess = ((2.0*self.G*m1Guess*1.0e6*self.SolarMass)/math.pow(self.c, 2.0)/self.Parsec) # rS1 in Parsec
				rS2Guess = ((2.0*self.G*m2Guess*1.0e6*self.SolarMass)/math.pow(self.c, 2.0)/self.Parsec) # rS2 in Parsec
				eccentricityGuess = random.uniform(0.0, 1.0)
				rPeriEst = math.pow(((self.G*math.pow(periodEst*self.Day, 2.0)*((m1Guess + m2Guess)*1.0e6*self.SolarMass)*math.pow(1.0 + eccentricityGuess, 3.0))/math.pow(self.twoPi, 2.0)),1.0/3.0)/self.Parsec
				a1Est = (m2Guess*rPeriEst)/((m1Guess + m2Guess)*(1.0 - eccentricityGuess));
				inclinationGuess = random.uniform(0.0, 90.0)
				tauGuess = random.uniform(0.0, periodEst)
				totalFluxGuess = 0.0
				while ((totalFluxGuess < lowestFlux) or (totalFluxGuess > highestFlux)):
					totalFluxGuess = random.gauss(meanFlux, (highestFlux - lowestFlux)/6.0)

				foldedLC = observedLC.fold(periodEst)
				pdb.set_trace()
				maxD = math.pow(np.max(foldedLC.y[np.where(observedLC.mask == 1.0)])/totalFlux,1.0/3.44)
				minD = math.pow(np.min(foldedLC.y[np.where(observedLC.mask == 1.0)])/totalFlux,1.0/3.44)
				Alpha = 1.0 - (1.0/maxD)
				Beta = 1.0 - (1.0/minD)
				OneOverE = ((maxBetaParallel + minBetaParallel)/(maxBetaParallel + minBetaParallel))

				#fracBeamedFluxGuess = random.uniform(0.0, 1.0)
				#ThetaGuess = np.array([rPerGuess, m1Guess, m2Guess, eGuess, omegaGuess, inclinationGuess, tauGuess, totalFluxGuess, fracBeamedFluxGuess])
				ThetaGuess = np.array([rPeriEst, m1Guess, m2Guess, eccentricityGuess, omega1Est, inclinationGuess, tauGuess, totalFluxGuess])
				res = self.set(ThetaGuess)
				lnPrior = self.logPrior(observedLC)
				if res == 0 and lnPrior == 0.0:
					noSuccess = False
			for dimNum in xrange(self.ndims):
				xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]

		res = self._taskCython.fit_BinarySMBHModel(observedLC.numCadences, observedLC.dt, lowestFlux, highestFlux, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, self.mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPosterior)
		return res