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
import operator as operator
import sys as sys
import abc as abc
import psutil as psutil
import types as types
import os as os
import reprlib as reprlib
import copy as copy
from scipy.interpolate import UnivariateSpline
import warnings as warnings
import matplotlib.pyplot as plt
import pdb as pdb

try:
	import rand as rand
	import CARMATask as CARMATask
	from util.mpl_settings import set_plot_params
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

ln10 = math.log(10)

def pogsonFlux(mag, magErr):
	flux = 3631.0*math.pow(10.0, (-1.0*mag)/2.5)
	fluxErr = (ln10/2.5)*flux*magErr
	return flux, fluxErr

def _f7(seq):
	"""http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order"""
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def MAD(self, a):
	medianVal = np.median(a)
	b = np.copy(a)
	for i in range(a.shape[0]):
		b[i]=abs(b[i] - medianVal)
	return np.median(b)

def _old_roots(p, q, Theta):
	ARPoly = np.zeros(p + 1)
	ARPoly[0] = 1.0
	for i in xrange(p):
		ARPoly[i + 1] = Theta[i]
	ARRoots = np.array(np.roots(ARPoly))
	MAPoly = np.zeros(q + 1)
	for i in xrange(q + 1):
		MAPoly[i] = Theta[p + q - i]
	MARoots = np.array(np.roots(MAPoly))
	Rho = np.zeros(p + q + 1, dtype = 'complex128')
	for i in xrange(p):
		Rho[i] = ARRoots[i]
	for i in xrange(q):
		Rho[p + i] = MARoots[i]
	Rho[p + q] = MAPoly[0]
	return Rho

def roots(p, q, Theta):
	ARPoly = np.zeros(p + 1)
	ARPoly[0] = 1.0
	for i in xrange(p):
		ARPoly[i + 1] = Theta[i]
	ARRoots = np.array(np.roots(ARPoly))
	MAPoly = np.zeros(q + 1)
	for i in xrange(q + 1):
		MAPoly[i] = Theta[p + q - i]
	MARoots = np.array(np.roots(MAPoly))
	Rho = np.zeros(p + q + 1, dtype = 'complex128')
	for i in xrange(p):
		Rho[i] = ARRoots[i]
	for i in xrange(q):
		Rho[p + i] = MARoots[i]
	Sigma = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
	ThetaC = np.require(np.array(Theta), requirements=['F', 'A', 'W', 'O', 'E'])
	CARMATask.get_Sigma(p, q, ThetaC, Sigma)
	Rho[p + q] = math.sqrt(Sigma[0])
	return Rho

def _old_coeffs(p, q, Rho):
	ARRoots = np.zeros(p, dtype = 'complex128')
	for i in xrange(p):
		ARRoots[i] = Rho[i]
	ARPoly = np.array(np.poly(ARRoots))
	MARoots = np.zeros(q, dtype = 'complex128')
	for i in xrange(q):
		MARoots[i] = Rho[p + i]
	if q == 0:
		MAPoly = np.ones(1)
	else:
		MAPoly = np.array(np.poly(MARoots))
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		for i in xrange(q + 1):
			MAPoly[i] = Rho[-1]*MAPoly[i]
	Theta = np.zeros(p + q + 1, dtype = 'float64')
	for i in xrange(p):
		Theta[i] = ARPoly[i + 1].real
	for i in xrange(q + 1):
		Theta[p + i] = MAPoly[q - i].real
	return Theta

def coeffs(p, q, Rho):
	ARRoots = np.zeros(p, dtype = 'complex128')
	for i in xrange(p):
		ARRoots[i] = Rho[i]
	ARPoly = np.array(np.poly(ARRoots))
	MARoots = np.zeros(q, dtype = 'complex128')
	for i in xrange(q):
		MARoots[i] = Rho[p + i]
	if q == 0:
		MAPoly = np.ones(1)
	else:
		MAPoly = np.array(np.poly(MARoots))
	ThetaPrime = np.require(np.array(ARPoly[1:].tolist() + MAPoly.tolist()[::-1]), requirements=['F', 'A', 'W', 'O', 'E'])
	SigmaPrime = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
	CARMATask.get_Sigma(p, q, ThetaPrime, SigmaPrime)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		Sigma00 = math.pow(Rho[p + q], 2.0)
	try:
		bQ = math.sqrt(Sigma00/SigmaPrime[0])
	except ValueError:
		bQ = 1.0
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		for i in xrange(q + 1):
			MAPoly[i] = bQ*MAPoly[i]
	Theta = np.zeros(p + q + 1, dtype = 'float64')
	for i in xrange(p):
		Theta[i] = ARPoly[i + 1].real
	for i in xrange(q + 1):
		Theta[p + i] = MAPoly[q - i].real
	return Theta

def timescales(p, q, Rho):
	imagPairs = 0
	for i in xrange(p):
		if Rho[i].imag != 0.0:
			imagPairs += 1
	numImag = imagPairs/2
	numReal = numImag + (p - imagPairs)
	decayTimescales = np.zeros(numReal)
	oscTimescales = np.zeros(numImag)
	realRoots = set(Rho[0:p].real)
	imagRoots = set(abs(Rho[0:p].imag)).difference(set([0.0]))
	realAR = sorted([1.0/abs(x) for x in realRoots])
	imagAR = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
	imagPairs = 0
	for i in xrange(q):
		if Rho[i].imag != 0.0:
			imagPairs += 1
	numImag = imagPairs/2
	numReal = numImag + (q - imagPairs)
	decayTimescales = np.zeros(numReal)
	oscTimescales = np.zeros(numImag)
	realRoots = set(Rho[p:p + q].real)
	imagRoots = set(abs(Rho[p:p + q].imag)).difference(set([0.0]))
	realMA = sorted([1.0/abs(x) for x in realRoots])
	imagMA = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
	return np.array(realAR + imagAR + realMA + imagMA + [Rho[p + q]])

class epoch(object):
	"""!
	\anchor epoch_
	
	\brief Class to hold individual epochs of a light curve.
	
	We wish to hold individual epochs in a light curve in an organized manner. This class lets us examine individual epochs and check for equality with other epochs. Two epochs are equal iff they have the same timestamp. Later on, we will implement some sort of unit system for the quantities (i.e. is the tiumestamp in sec, min, day, MJD etc...?)
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

	def __init__(self, numCadences = None, dt = None, meandt = None, mindt = None, dtSmooth = None, name = None, band = None, xunit = None, yunit = None, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 2.0, maxTimescale = 0.5, pSim = 0, qSim = 0, pComp = 0, qComp = 0, sampler = None, path = None, **kwargs):
		"""!
		\brief Initialize a new light curve

		The constructor assumes that the light curve to be constructed is regular. There is no option to construct irregular light curves. Irregular light can be obtained by reading in a supplied irregular light curve. The constructor takes an optional keyword argument (supplied = <light curve file>) that is read in using the read method. This supplied light curve can be irregular. Typically, the supplied light curve is irregular either because the user created it that way, or because the survey that produced it sampled the sky at irregular intervals.

		Non-keyword arguments
		\param[in] numCadences: The number of cadences in the light curve.
		\param[in] p          : The order of the C-ARMA model used.
		\param[in] q          : The order of the C-ARMA model used.

		Keyword arguments
		\param[in] dt:                The spacing between cadences.
		\param[in] dt:                The spacing between cadences after smoothing.
		\param[in] name:              The name of the light curve (usually the object's name).
		\param[in] band:              The name of the photometric band (eg. HSC-I or SDSS-g etc..).
		\param[in] xunit              Unit in which time is measured (eg. s, sec, seconds etc...).
		\param[in] yunit              Unit in which the flux is measured (eg Wm^{-2} etc...).
		\param[in] tolIR:             The tolerance level at which a given step in the lightcurve should be considered irregular for the purpose of solving the C-ARMA model. The C-ARMA model needs to be re-solved if abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where t_incr is the new increment in time and dt is the previous increment in time. If IR  == False, this parameter is not used.
		\param[in] fracIntrinsicVar:  The fractional variability of the source i.e. fracIntrinsicVar = sqrt(Sigma[0,0])/mean_flux.
		\param[in] fracNoiseToSignal: The fractional noise level i.e. fracNoiseToSignal = sigma_noise/flux. We assume that this level is fixed. In the future, we may wish to make this value flux dependent to make the noise model more realistic.
		\param[in] maxSigma:          The maximum allowed value of sqrt(Sigma[0,0]) = maxSigma*stddev(y) when fitting a C-ARMA model. Note that if the observed light curve is shorter than the de-correlation timescale, stddev(y) may be much smaller than sqrt(Sigma[0,0]) and hence maxSigma should be made larger in such cases.
		\param[in] minTimescale:      The shortest allowed timescale = minTimescale*dt. Note that if the observed light curve is very sparsely sampled, dt may be much larger than the actaul minimum timescale present and hence minTimescale should be made smaller in such cases.
		\param[in] maxTimescale:      The longest allowed timescale = maxTimescale*T. Note that if the observed light curve is shorter than the longest timescale present, T may be much smaller than the longest timescale and hence maxTimescale should be made larger in such cases.
		\param[in] supplied:          Reference for supplied light curve. Since this class is an ABC, individual subclasses must implement a read method and the format expected for supplied (i.e. full path or name etc...) will be determined by the subclass.
		\param[in] path:              Reference for supplied light curve. Since this class is an ABC, individual subclasses must implement a read method and the format expected for supplied (i.e. full path or name etc...) will be determined by the subclass.
		"""
		if name is not None and band is not None:
			self.read(name = name, band = band, path = path, **kwargs)
		else:
			self._numCadences = numCadences ## The number of cadences in the light curve. This is not the same thing as the number of actual observations as we can have missing observations.
			self._simulatedCadenceNum = -1 ## How many cadences have already been simulated.
			self._observedCadenceNum = -1 ## How many cadences have already been observed.
			self._computedCadenceNum = -1 ## How many cadences have been LnLikelihood'd already.
			self._pSim = pSim ## C-ARMA model used to simulate the LC.
			self._qSim = qSim ## C-ARMA model used to simulate the LC.
			self._pComp = pComp ## C-ARMA model used to simulate the LC.
			self._qComp = qComp ## C-ARMA model used to simulate the LC.
			self._isSmoothed = False ## Has the LC been smoothed?
			self._dtSmooth = 0.0
			self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
			self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed fluxes.
			self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed flux errors.
			self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
			self.XSim = np.require(np.zeros(self.pSim), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
			self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
			self.XComp = np.require(np.zeros(self.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
			self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
			self._name = str(name) ## The name of the light curve (usually the object's name).
			self._band = str(band) ## The name of the photometric band (eg. HSC-I or SDSS-g etc..).
			if str(xunit)[0] != '$':
				self._xunit = r'$' + str(xunit) + '$' ## Unit in which time is measured (eg. s, sec, seconds etc...).
			else:
				self._xunit = str(xunit)
			if str(yunit)[0] != '$':
				self._yunit = r'$' + str(yunit) + '$' ## Unit in which the flux is measured (eg Wm^{-2} etc...).
			else:
				self._yunit = str(yunit)
			self._tolIR = tolIR ## Tolerance on the irregularity. If IR == False, this parameter is not used. Otherwise, a timestep is irregular iff abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where t_incr is the new increment in time and dt is the previous increment in time.
			self._fracIntrinsicVar = fracIntrinsicVar
			self._fracNoiseToSignal = fracNoiseToSignal
			self._maxSigma = maxSigma
			self._minTimescale = minTimescale
			self._maxTimescale = maxTimescale
			for i in xrange(self._numCadences):
				self.t[i] = i*dt
				self.mask[i] = 1.0
			self._isRegular = True
			self._dt = float(self.t[1] - self.t[0])
			self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
			self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
			self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
			self._T = float(self.t[-1] - self.t[0])
		self._lcCython = CARMATask.lc(self.t, self.x, self.y, self.yerr, self.mask, self.XSim, self.PSim, self.XComp, self.PComp, dt = self._dt, meandt = self._meandt, mindt = self._mindt, maxdt = self._maxdt, dtSmooth = self._dtSmooth, tolIR = self._tolIR, fracIntrinsicVar = self._fracIntrinsicVar, fracNoiseToSignal = self._fracNoiseToSignal, maxSigma = self._maxSigma, minTimescale = self._minTimescale, maxTimescale = self._maxTimescale)
		if sampler is not None:
			self._sampler = sampler(self)
		else:
			self._sampler = None

		count = int(np.sum(self.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(self.numCadences):
			y_meanSum += self.mask[i]*self.y[i]
			yerr_meanSum += self.mask[i]*self.yerr[i]
		if count > 0.0:
			self._mean = y_meanSum/count
			self._meanerr = yerr_meanSum/count
		else:
			self._mean = 0.0
			self._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(self.numCadences):
			y_stdSum += math.pow(self.mask[i]*self.y[i] - self._mean, 2.0)
			yerr_stdSum += math.pow(self.mask[i]*self.yerr[i] - self._meanerr, 2.0)
		if count > 0.0:
			self._std = math.sqrt(y_stdSum/count)
			self._stderr = math.sqrt(yerr_stdSum/count)
		else:
			self._std = 0.0
			self._stderr = 0.0

	@property
	def numCadences(self):
		return self._numCadences

	@numCadences.setter
	def numCadences(self, value):
		self._lcCython.numCadences = value
		self._numCadences = value

	@property
	def simulatedCadenceNum(self):
		return self._simulatedCadenceNum

	@property
	def observedCadenceNum(self):
		return self._observedCadenceNum

	@property
	def computedCadenceNum(self):
		return self._computedCadenceNum

	@property
	def isRegular(self):
		return self._isRegular

	@property
	def isSmoothed(self):
		return self._isSmoothed

	@property
	def pSim(self):
		return self._pSim

	@pSim.setter
	def pSim(self, value):
		if value != self.pSim:
			newXSim = np.require(np.zeros(value), requirements=['F', 'A', 'W', 'O', 'E'])
			newPSim = np.require(np.zeros(value**2), requirements=['F', 'A', 'W', 'O', 'E'])
			large_number = math.sqrt(sys.float_info[0])
			if self.pSim > 0:
				if value > self.pSim:
					iterMax = self.pSim
				elif value < self.pSim:
					iterMax = value
				for i in xrange(iterMax):
					newXSim[i] = self.XSim[i]
					for j in xrange(iterMax):
						newPSim[i + j*value] = self.PSim[i + j*self.pSim]
			elif self.pSim == 0:
				for i in xrange(value):
					newXSim[i] = 0.0
					for j in xrange(value):
						newPSim[i + j*value] = 0.0
			self._pSim = value
			self.XSim = newXSim
			self.PSim = newPSim
		else:
			pass

	@property
	def pComp(self):
		return self._pComp

	@pComp.setter
	def pComp(self, value):
		if value != self.pComp:
			newXComp = np.require(np.zeros(value), requirements=['F', 'A', 'W', 'O', 'E'])
			newPComp = np.require(np.zeros(value**2), requirements=['F', 'A', 'W', 'O', 'E'])
			large_number = math.sqrt(sys.float_info[0])
			if self.pComp > 0:
				if value > self.pComp:
					iterMax = self.pComp
				elif value < self.pComp:
					iterMax = value
				for i in xrange(iterMax):
					newXComp[i] = self.XComp[i]
					for j in xrange(iterMax):
						newPComp[i + j*value] = self.PComp[i + j*self.pComp]
			elif self.pComp == 0.0:
				for i in xrange(value):
					newXComp[i] = 0.0
					for j in xrange(value):
						newPComp[i + j*value] = 0.0
			self._pComp = value
			self.XComp = newXComp
			self.PComp = newPComp
		else:
			pass

	@property
	def qSim(self):
		return self._qSim

	@qSim.setter
	def qSim(self, value):
		self._qSim = value

	@property
	def qComp(self):
		return self._qComp

	@qComp.setter
	def qComp(self, value):
		self._qComp = value

	@property
	def dt(self):
		return self._dt

	@dt.setter
	def dt(self, value):
		self._lcCython.dt = value
		self._dt = value

	@property
	def meandt(self):
		return self._meandt

	@meandt.setter
	def meandt(self, value):
		self._lcCython.meandt = value
		self._meandt = value

	@property
	def mindt(self):
		return self._mindt

	@mindt.setter
	def mindt(self, value):
		self._lcCython.mindt = value
		self._mindt = value

	@property
	def maxdt(self):
		return self._maxdt

	@maxdt.setter
	def maxdt(self, value):
		self._lcCython.maxdt = value
		self._maxdt = value

	@property
	def dtSmooth(self):
		return self._dtSmooth

	@dtSmooth.setter
	def dtSmooth(self, value):
		self._lcCython.dtSmooth = value
		self._dtSmooth = value

	@property
	def T(self):
		return self._T

	@T.setter
	def T(self, value):
		self._T = value

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, value):
		self._name = str(value)

	@property
	def band(self):
		return self._band

	@band.setter
	def band(self, value):
		self._band = str(value)

	@property
	def xunit(self):
		return self._xunit

	@xunit.setter
	def xunit(self, value):
		if str(value)[0] != r'$':
			self._xunit = r'$' + str(value) + r'$' 
		else:
			self._xunit = str(value)

	@property
	def yunit(self):
		return self._yunit

	@yunit.setter
	def yunit(self, value):
		if str(value)[0] != r'$':
			self._yunit = r'$' + str(value) + r'$' 
		else:
			self._yunit = str(value)

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

	@property
	def sampler(self):
		return str(self._sampler)

	@sampler.setter
	def sampler(self, value):
		self._sampler = eval(str(value).split('.')[-1])(self)

	@property
	def mean(self):
		"""!
		\brief Mean of the observations y.
		"""
		return self._mean

	@property
	def meanerr(self):
		"""!
		\brief Mean of the observation errors yerr.
		"""
		return self._meanerr

	@property
	def std(self):
		"""!
		\brief Standard deviation of the observations y.
		"""
		return self._std

	@property
	def stderr(self):
		"""!
		\brief Standard deviation of the observation errors yerr.
		"""
		return self._stderr

	def __len__(self):
		return self._numCadences

	def __repr__(self):
		"""!
		\brief Return a representation of the lc such that eval(repr(someLC)) == someLC is True.
		"""
		return u"libcarma.lc(%d, %f, %s, %s, %s, %s, %f, %f, %f, %f, %f, %f)"%(self._numCadences, self._dt, self._name, self._band, self._xunit, self._yunit, self._tolIR, self._fracIntrinsicVar, self._fracNoiseToSignal, self._maxSigma, self._minTimescale, self._maxTimescale)

	def __str__(self):
		"""!
		\brief Return a human readable representation of the light curve.
		"""
		line = ''
		line += '                                             Name: %s\n'%(self._name)
		line += '                                             Band: %s\n'%(self._band)
		line += '                                        Time Unit: %s\n'%(self._xunit)
		line += '                                        Flux Unit: %s\n'%(self._yunit)
		line += '                                      numCadences: %d\n'%(self._numCadences)
		line += '                                               dt: %e\n'%(self._dt)
		line += '                                                T: %e\n'%(self._T)
		line += '                                     mean of flux: %e\n'%(self._mean)
		line += '                           std. deviation of flux: %e\n'%(self._std)
		line += '                               mean of flux error: %e\n'%(self._meanerr)
		line += '                     std. deviation of flux error: %e\n'%(self._stderr)
		line += '               tolIR (Tolerance for irregularity): %e\n'%(self._tolIR)
		line += 'fracIntrinsicVar (Intrinsic variability fraction): %e\n'%(self._fracIntrinsicVar)
		line += '     fracNoiseToSignal (Noise to signal fraction): %e\n'%(self._fracNoiseToSignal)
		line += '      maxSigma (Maximum allowed sigma multiplier): %e\n'%(self._maxSigma)
		line += '  minTimescale (Minimum allowed timescale factor): %e\n'%(self._minTimescale)
		line += '  maxTimescale (Maximum allowed timescale factor): %e\n'%(self._maxTimescale)
		line += '\n'
		epochline = ''
		for i in xrange(self._numCadences - 1):
			epochline += str(self[i])
			epochline += '\n'
		epochline += str(self[self._numCadences - 1])
		line += reprlib.repr(epochline)
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
		"""!
		\brief Return an epoch.
		
		Return an epoch corresponding to the index.
		"""
		return epoch(self.t[key], self.x[key], self.y[key], self.yerr[key], self.mask[key])

	def __setitem__(self, key, val):
		"""!
		\brief Set an epoch.
		
		Set the epoch corresponding to the provided index using the values from the epoch val.
		"""
		if isinstance(val, epoch):
			self.t[key] = val.t
			self.x[key] = val.x
			self.y[key] = val.y
			self.yerr[key] = val.yerr
			self.mask[key] = val.mask
		self._mean = np.mean(self.y)
		self._std = np.std(self.y)
		self._meanerr = np.mean(self.yerr)
		self._stderr = np.std(self.yerr)

	def __iter__(self):
		"""!
		\brief Return a light curve iterator.
		
		Return a light curve iterator object making light curves iterable.
		"""
		return lcIterator(self.t, self.x, self.y, self.yerr, self.mask)

	@abc.abstractmethod
	def copy(self):
		"""!
		\brief Return a copy
		
		Return a (deep) copy of the object.
		"""
		raise NotImplementedError(r'Override copy by subclassing lc!')

	def __invert__(self):
		"""!
		\brief Return the lc without the mean.
		
		Return a new lc with the mean removed.
		"""
		lccopy = self.copy()
		lccopy.x -= lccopy._mean
		lccopy.y -= lccopy._mean

	def __pos__(self):
		"""!
		\brief Return + light curve.
		
		Return + light curve i.e. do nothing but just return a deepcopy of the object.
		"""
		return self.copy()

	def __neg__(self):
		"""!
		\brief Invert the light curve.
		
		Return a light curve with the delta fluxes flipped in sign.
		"""
		lccopy = self.copy()
		lccopy.x = -1.0*(self.x - np.mean(self.x)) + np.mean(self.x)
		lccopy.y = -1.0*(self.y - self._mean) + self._mean
		lccopy._mean = np.mean(lccopy.y)
		lccopy._std = np.std(lccopy.y)
		return lccopy

	def __abs__(self):
		"""!
		\brief Abs the light curve.
		
		Return a light curve with the abs of the delta fluxes.
		"""
		lccopy = self.copy()
		lccopy.x = np.abs(self.x - np.mean(self.x)) + np.mean(self.x)
		lccopy.y = np.abs(self.y - self._mean) + self._mean
		lccopy._mean = np.mean(lccopy.y)
		lccopy._std = np.std(lccopy.y)
		return lccopy

	def __add__(self, other):
		"""!
		\brief Add.
		
		Add another light curve or scalar to the light curve.
		"""
		lccopy = self.copy()
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			lccopy.x += other
			lccopy.y += other
			lccopy._mean = np.mean(lccopy.y)
			lccopy._std = np.std(lccopy.y)
		elif isinstance(other, lc):
			if other.numCadences == self.numCadences:
				lccopy.x += other.x
				lccopy.y += other.y
				lccopy.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
				lccopy._mean = np.mean(lccopy.y)
				lccopy._std = np.std(lccopy.y)
				lccopy._mean = np.mean(lccopy.yerr)
				lccopy._stderr = np.std(lccopy.yerr)
			else:
				raise ValueError('Light curves have un-equal length')
		else:
			raise NotImplemented
		return lccopy

	def __radd__(self, other):
		"""!
		\brief Add.
		
		Add a light curve to a scalar.
		"""
		return self + other

	def __sub__(self, other):
		"""!
		\brief Subtract.
		
		Subtract another light curve or scalar from the light curve.
		"""
		return self + (- other)

	def __rsub__(self, other):
		"""!
		\brief Subtract.
		
		Subtract a light curve from a scalar .
		"""
		return self + (- other)

	def __iadd__(self, other):
		"""!
		\brief Inplace add.
		
		Inplace add another light curve or scalar to the light curve.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			self.x += other
			self.y += other
			self._mean += other
		elif isinstance(other, lc):
			if other.numCadences == self.numCadences:
				self.x += other.x
				self.y += other.y
				self.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
				self._mean = np.mean(self.y)
				self._std = np.std(self.y)
				self._mean = np.mean(self.yerr)
				self._stderr = np.std(self.yerr)
			else:
				raise ValueError('Light curves have un-equal length')
		return self

	def __isub__(self, other):
		"""!
		\brief Inplace subtract.
		
		Inplace subtract another light curve or scalar from the light curve.
		"""
		return self.iadd( - other)

	def __mul__(self, other):
		"""!
		\brief Multiply.
		
		Multiply the light curve by a scalar.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			if type(other) is types.ComplexType:
				other = complex(other)
			else:
				other = float(other)
			lccopy = self.copy()
			lccopy.x *= other
			lccopy.y *= other
			lccopy.yerr *= other
			lccopy._mean += other
			lccopy._std *= other
			lccopy._meanerr *= other
			lccopy._stderr *= other
			return lccopy
		else:
			raise NotImplemented

	def __rmul__(self, other):
		"""!
		\brief Multiply.
		
		Multiply a scalar by the light curve.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			if type(other) is types.ComplexType:
				other = complex(other)
			else:
				other = float(other)
			return self*other
		else:
			raise NotImplemented

	def __div__(self, other):
		"""!
		\brief Divide.
		
		Divide the light curve by a scalar.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			if type(other) is types.ComplexType:
				other = complex(other)
			else:
				other = float(other)
			return self*(1.0/other)
		else:
			raise NotImplemented

	def __rdiv__(self, other):
		"""!
		\brief Divide  - not defined & not implemented.
		
		Divide a scalar by the light curve - not defined & not implemented.
		"""
		raise NotImplemented

	def __imul__(self, other):
		"""!
		\brief Inplace multiply.
		
		Inplace multiply a light curve by a scalar.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			if type(other) is types.ComplexType:
				other = complex(other)
			else:
				other = float(other)
			self.x *= other
			self.y *= other
			self.yerr *= other
			self._mean += other
			self._std *= other
			self._meanerr *= other
			self._stderr *= other
			return self
		else:
			raise NotImplemented

	def __idiv__(self, other):
		"""!
		\brief Inplace divide.
		
		Inplace divide a light curve by a scalar.
		"""
		if type(other) is types.IntType or type(other) is types.LongType or type(other) is types.FloatType or type(other) is types.ComplexType:
			if type(other) is types.ComplexType:
				other = complex(other)
			else:
				other = float(other)
			self.x *= (1.0/other)
			self.y *= (1.0/other)
			self.yerr *= (1.0/other)
			self._mean += (1.0/other)
			self._std *= (1.0/other)
			self._meanerr *= (1.0/other)
			self._stderr *= (1.0/other)
			return self
		else:
			raise NotImplemented

	@abc.abstractmethod
	def read(self, name, band, path = os.environ['PWD'], **kwargs):
		"""!
		\brief Read the light curve from disk.
		
		Not implemented!
		"""
		raise NotImplementedError(r'Override read by subclassing lc!')

	@abc.abstractmethod
	def write(self, name, band, path = os.environ['PWD'], **kwargs):
		"""!
		\brief Write the light curve to disk.
		
		Not implemented
		"""
		raise NotImplementedError(r'Override write by subclassing lc!')

	def regularize(self, newdt = None):
		"""!
		\brief Re-sample the light curve on a grid of spacing newdt
		
		Creates a new LC on gridding newdt and copies in the required points.
		"""
		if not self.isRegular:
			if not newdt:
				newdt = self.mindt/10.0
			if newdt > self.mindt:
				raise ValueError('newdt cannot be greater than mindt')
			newLC = self.copy()
			newLC.dt = newdt
			newLC.meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
			newLC.mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
			newLC.maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
			newLC.meandt = float(self.t[-1] - self.t[0])
			newLC.numCadences = int(math.ceil(self.T/newLC.dt))
			del newLC.t
			del newLC.x
			del newLC.y
			del newLC.yerr
			del newLC.mask
			newLC.t = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
			newLC.x = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			newLC.y = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed fluxes.
			newLC.yerr = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed flux errors.
			newLC.mask = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
			for i in xrange(newLC.numCadences):
				newLC.t[i] = i*newLC.dt + self.t[0]
			for i in xrange(self.numCadences):
				tOff = (self.t[i] - self.t[0])
				index = int(math.floor(tOff/newLC.dt))
				newLC.x[index] = self.x[i]
				newLC.y[index] = self.y[i]
				newLC.yerr[index] = self.yerr[i]
				newLC.mask[index] = 1.0
			count = int(np.sum(newLC.mask[i]))
			y_meanSum = 0.0
			yerr_meanSum = 0.0
			for i in xrange(newLC.numCadences):
				y_meanSum += newLC.mask[i]*newLC.y[i]
				yerr_meanSum += newLC.mask[i]*newLC.yerr[i]
			if count > 0:
				newLC._mean = y_meanSum/count
				newLC._meanerr = yerr_meanSum/count
			y_stdSum = 0.0
			yerr_stdSum = 0.0
			for i in xrange(newLC.numCadences):
				y_stdSum += math.pow(newLC.mask[i]*newLC.y[i] - newLC._mean, 2.0)
				yerr_stdSum += math.pow(newLC.mask[i]*newLC.yerr[i] - newLC._meanerr, 2.0)
			if count > 0:
				newLC._std = math.sqrt(y_stdSum/count)
				newLC._stderr = math.sqrt(yerr_stdSum/count)

			return newLC
		else:
			return self

	def sample(self, **kwargs):
		return self._sampler.sample(**kwargs)

	def acvf(self, newdt = None):
		if hasattr(self, '_acvflags') and hasattr(self, '_acvf') and hasattr(self, '_acvferr'):
			return self._acvflags, self._acvf, self._acvferr
		else:
			if not self.isRegular:
				useLC = self.regularize(newdt)
			else:
				useLC = self
			self._acvflags = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
			self._acvf = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			self._acvferr = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			useLC._lcCython.compute_ACVF(useLC.numCadences, useLC.dt, useLC.t, useLC.x, useLC.y, useLC.yerr, useLC.mask, self._acvflags, self._acvf, self._acvferr)
			return self._acvflags, self._acvf, self._acvferr

	def acf(self, newdt = None):
		if hasattr(self, '_acflags') and hasattr(self, '_acf') and hasattr(self, '_acferr'):
			return self._acflags, self._acf, self._acferr
		else:
			if not self.isRegular:
				useLC = self.regularize(newdt)
			else:
				useLC = self
			acvflags, acvf, acvferr = self.acvf(newdt)
			self._acflags = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			self._acf = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
			self._acferr = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			constErr = math.pow(acvferr[0]/acvf[0], 2.0)
			for i in xrange(useLC.numCadences):
				self._acflags[i] = acvflags[i]
				if acvf[i] != 0.0:
					self._acf[i] = acvf[i]/acvf[0]
					self._acferr[i] = (acvf[i]/acvf[0])*np.sqrt(np.power(acvferr[i]/acvf[i], 2.0) + constErr)
			return self._acflags, self._acf, self._acferr

	def dacf(self, newdt = None, nbins = None):
		if hasattr(self, '_dacflags') and hasattr(self, '_dacf') and hasattr(self, '_dacferr'):
			return self._dacflags, self._dacf, self._dacferr
		else:
			if not self.isRegular:
				useLC = self.regularize(newdt)
			else:
				useLC = self
			if nbins is None:
				nbins = int(useLC.numCadences/10)
			self._dacflags = np.require(np.linspace(start = 0, stop = useLC.T, num = nbins), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
			self._dacf = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			self._dacferr = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			useLC._lcCython.compute_DACF(useLC.numCadences, useLC.dt, useLC.t, useLC.x, useLC.y, useLC.yerr, useLC.mask, nbins, self._dacflags, self._dacf, self._dacferr)
			return self._dacflags, self.dacf, self.dacferr

	def sf(self, newdt = None):
		if hasattr(self, '_sflags') and hasattr(self, '_sf') and hasattr(self, '_sferr'):
			return self._sflags, self._sf, self._sferr
		else:
			if not self.isRegular:
				useLC = self.regularize(newdt)
			else:
				useLC = self
			self._sflags = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
			self._sf = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			self._sferr = np.require(np.zeros(useLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
			useLC._lcCython.compute_SF(useLC.numCadences, useLC.dt, useLC.t, useLC.x, useLC.y, useLC.yerr, useLC.mask, self._sflags, self._sf, self._sferr)
			return self._sflags, self._sf, self._sferr

	def plot(self, fig = -1, doShow = False, clearFig = True):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		if clearFig:
			plt.clf()
		if (np.sum(self.x) != 0.0) and (np.sum(self.y) == 0.0):
			plt.plot(self.t, self.x, color = '#984ea3', zorder = 0)
			plt.plot(self.t, self.x, color = '#984ea3', marker = 'o', markeredgecolor = 'none', zorder = 0)
		if (np.sum(self.x) == 0.0) and (np.sum(self.y) != 0.0):
			plt.errorbar(self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]], self.yerr[np.where(self.mask == 1.0)[0]], label = r'%s (%s-band)'%(self.name, self.band), fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
			plt.xlim(self.t[0], self.t[-1])
		if (np.sum(self.x) != 0.0) and (np.sum(self.y) != 0.0):
			plt.plot(self.t, self.x - np.mean(self.x) + np.mean(self.y[np.where(self.mask == 1.0)[0]]), color = '#984ea3', zorder = 0)
			plt.plot(self.t, self.x - np.mean(self.x) + np.mean(self.y[np.where(self.mask == 1.0)[0]]), color = '#984ea3', marker = 'o', markeredgecolor = 'none', zorder = 0)
			plt.errorbar(self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]], self.yerr[np.where(self.mask == 1.0)[0]], label = r'%s (%s-band)'%(self.name, self.band), fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
		plt.xlim(self.t[0], self.t[-1])
		if self.isSmoothed:
			plt.plot(self.tSmooth, self.xSmooth, color = '#4daf4a', marker = 'o', markeredgecolor = 'none', zorder = -5)
			plt.plot(self.tSmooth, self.xSmooth, color = '#4daf4a', zorder = -5)
			plt.fill_between(self.tSmooth, self.xSmooth - self.xerrSmooth, self.xSmooth + self.xerrSmooth, facecolor = '#ccebc5', alpha = 0.5, zorder = -5)
		plt.xlabel(self.xunit)
		plt.ylabel(self.yunit)
		plt.title(r'Light curve')
		plt.legend()
		if doShow:
			plt.show(False)
		return newFig

	def plotacvf(self, fig = -2, newdt = None, doShow = False):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		plt.plot(0.0, 0.0)
		if np.sum(self.y) != 0.0:
			lagsE, acvfE, acvferrE = self.acvf(newdt)
			if np.sum(acvfE) != 0.0:
				plt.errorbar(lagsE[0], acvfE[0], acvferrE[0], label = r'obs. Autocovariance Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				for i in xrange(1, lagsE.shape[0]):
					if acvfE[i] != 0.0:
						plt.errorbar(lagsE[i], acvfE[i], acvferrE[i], fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$ACVF$')
		plt.title(r'AutoCovariance Function')
		plt.legend(loc = 3)
		if doShow:
			plt.show(False)
		return newFig

	def plotacf(self, fig = -3, newdt = None, doShow = False):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		plt.plot(0.0, 0.0)
		if np.sum(self.y) != 0.0:
			lagsE, acfE, acferrE = self.acf(newdt)
			if np.sum(acfE) != 0.0:
				plt.errorbar(lagsE[0], acfE[0], acferrE[0], label = r'obs. Autocorrelation Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				for i in xrange(1, lagsE.shape[0]):
					if acfE[i] != 0.0:
						plt.errorbar(lagsE[i], acfE[i], acferrE[i], fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$ACF$')
		plt.title(r'AutoCorrelation Function')
		plt.legend(loc = 3)
		plt.ylim(-1.0, 1.0)
		if doShow:
			plt.show(False)
		return newFig

	def plotdacf(self, fig = -4, newdt = None, doShow = False):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		plt.plot(0.0, 0.0)
		if np.sum(self.y) != 0.0:
			lagsE, dacfE, dacferrE = self.dacf(newdt)
			if np.sum(dacfE) != 0.0:
				plt.errorbar(lagsE[0], dacfE[0], dacferrE[0], label = r'obs. Discrete Autocorrelation Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				for i in xrange(1, lagsE.shape[0]):
					if dacfE[i] != 0.0:
						plt.errorbar(lagsE[i], dacfE[i], dacferrE[i], fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$DACF$')
		plt.title(r'Discrete Autocorrelation Function')
		plt.legend(loc = 3)
		if doShow:
			plt.show(False)
		return newFig

	def plotsf(self, fig = -5, newdt = None, doShow = False):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		plt.loglog(1.0, 1.0)
		if np.sum(self.y) != 0.0:
			lagsE, sfE, sferrE = self.sf(newdt)
			if np.sum(sfE) != 0.0:
				plt.errorbar(lagsE[0], sfE[0], sferrE[0], label = r'obs. Structure Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				for i in xrange(1, lagsE.shape[0]):
					if dacfE[i] != 0.0:
						plt.errorbar(lagsE[i], sfE[i], sferrE[i], fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 10)
				plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$\log SF$')
		plt.title(r'Structure Function')
		plt.legend(loc = 2)
		if doShow:
			plt.show(False)
		return newFig

	def spline(self, ptFactor = 10, degree = 3):
		unObsUncertVal = math.sqrt(sys.float_info[0])
		newLC = self.copy()
		del newLC.t
		del newLC.x
		del newLC.y
		del newLC.yerr
		del newLC.mask
		newLC.numCadences = self.numCadences*ptFactor
		newLC.T = self.T
		newLC.dt = newLC.T/newLC.numCadences
		newLC.t = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.x = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.y = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.yerr = np.require(np.array(newLC.numCadences*[unObsUncertVal]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.mask = np.require(np.array(newLC.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
		spl = UnivariateSpline(self.t[np.where(self.mask == 1.0)], self.y[np.where(self.mask == 1.0)], 1.0/self.yerr[np.where(self.mask == 1.0)], k = degree, check_finite = True)
		for i in xrange(newLC.numCadences):
			newLC.t[i] = self.t[0] + i*newLC.dt
			newLC.x[i] = spl(newLC.t[i])
		return newLC

	def fold(self, foldPeriod, tStart = None):
		if tStart is None:
			tStart = self.t[0]
		numFolds = int(math.floor(self.T/foldPeriod))
		newLC = self.copy()
		del newLC.t
		del newLC.x
		del newLC.y
		del newLC.yerr
		del newLC.mask
		tList = list()
		xList = list()
		yList = list()
		yerrList = list()
		maskList = list()
		for i in xrange(self.numCadences):
			tList.append((self.t[i] - tStart)%foldPeriod)
			xList.append(self.x[i])
			yList.append(self.y[i])
			yerrList.append(self.yerr[i])
			maskList.append(self.mask[i])
		sortedLists = zip(*sorted(zip(tList,xList,yList,yerrList,maskList), key=operator.itemgetter(0)))
		newLC.t = np.require(np.array(sortedLists[0]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.x = np.require(np.array(sortedLists[1]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.y = np.require(np.array(sortedLists[2]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.yerr = np.require(np.array(sortedLists[3]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.mask = np.require(np.array(sortedLists[4]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.dt = float(newLC.t[1] - newLC.t[0])
		newLC.T = float(newLC.t[-1] - newLC.t[0])
		return newLC

	def bin(self, binRatio = 10):
		newLC = self.copy()
		newLC.numCadences = self.numCadences/binRatio
		del newLC.t
		del newLC.x
		del newLC.y
		del newLC.yerr
		del newLC.mask
		tList = list()
		xList = list()
		yList = list()
		yerrList = list()
		maskList = list()
		for i in xrange(newLC.numCadences):
			tList.append(np.mean(self.t[i*binRatio:(i + 1)*binRatio]))
			xList.append(np.mean(self.x[i*binRatio:(i + 1)*binRatio]))
			yList.append(np.mean(self.y[i*binRatio:(i + 1)*binRatio]))
			yerrList.append(math.sqrt(np.mean(np.power(self.yerr[i*binRatio:(i + 1)*binRatio], 2.0))))
			maskList.append(1.0)
		sortedLists = zip(*sorted(zip(tList,xList,yList,yerrList,maskList), key=operator.itemgetter(0)))
		newLC.t = np.require(np.array(sortedLists[0]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.x = np.require(np.array(sortedLists[1]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.y = np.require(np.array(sortedLists[2]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.yerr = np.require(np.array(sortedLists[3]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.mask = np.require(np.array(sortedLists[4]), requirements=['F', 'A', 'W', 'O', 'E'])
		newLC.dt = float(newLC.t[1] - newLC.t[0])
		newLC.T = float(newLC.t[-1] - newLC.t[0])
		return newLC

class lcIterator(object):
	def __init__(self, t, x, y, yerr, mask):
		self.t = t
		self.x = x
		self.y = y
		self.yerr = yerr
		self.mask = mask
		self.index = 0

	def next(self):
		"""!
		\brief Return the next epoch.
		
		To make light curves iterable, return the next epoch.
		"""
		try:
			nextEpoch = epoch(self.t[self.index], self.x[self.index], self.y[self.index], self.yerr[self.index], self.mask[self.index])
		except IndexError:
			raise StopIteration
		self.index += 1
		return nextEpoch

	def __iter__(self):
		return self

class basicLC(lc):

	def copy(self):
		lccopy = basicLC(self.numCadences, dt = self.dt, meandt = self.meandt, mindt = self.mindt, maxdt = self.maxdt, dtSmooth = self.dtSmooth, name = None, band = self.band, xunit = self.xunit, yunit = self.yunit, tolIR = self.tolIR, fracIntrinsicVar = self.fracIntrinsicVar, fracNoiseToSignal = self.fracNoiseToSignal, maxSigma = self.maxSigma, minTimescale = self.minTimescale, maxTimescale = self.maxTimescale)
		lccopy.t = np.copy(self.t)
		lccopy.x = np.copy(self.x)
		lccopy.y = np.copy(self.y)
		lccopy.yerr = np.copy(self.yerr)
		lccopy.mask = np.copy(self.mask)
		lccopy.pSim = np.copy(self.pSim)
		lccopy.qSim = np.copy(self.qSim)
		lccopy.pComp = np.copy(self.pComp)
		lccopy.qComp = np.copy(self.qComp)

		count = int(np.sum(lccopy.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(lccopy.numCadences):
			y_meanSum += lccopy.mask[i]*lccopy.y[i]
			yerr_meanSum += lccopy.mask[i]*lccopy.yerr[i]
		if count > 0.0:
			lccopy._mean = y_meanSum/count
			lccopy._meanerr = yerr_meanSum/count
		else:
			lccopy._mean = 0.0
			lccopy._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(lccopy.numCadences):
			y_stdSum += math.pow(lccopy.mask[i]*lccopy.y[i] - lccopy._mean, 2.0)
			yerr_stdSum += math.pow(lccopy.mask[i]*lccopy.yerr[i] - lccopy._meanerr, 2.0)
		if count > 0.0:
			lccopy._std = math.sqrt(y_stdSum/count)
			lccopy._stderr = math.sqrt(yerr_stdSum/count)
		else:
			lccopy._std = 0.0
			lccopy._stderr = 0.0

		return lccopy

	def read(self, name = None, band = None, pwd = None, **kwargs):
		pass

	def write(self, name = None, band = None, pwd = None, **kwargs):
		pass

class externalLC(basicLC):

	def _checkIsRegular(self):
		self._isRegular = True
		for i in xrange(1, self.numCadences):
			t_incr = self.t[i] - self.t[i-1]
			fracChange = abs((t_incr - self.dt)/((t_incr + self.dt)/2.0))
			if fracChange > self._tolIR:
				self._isRegular = False
				break

	def read(self, name, band, path = None, **kwargs):
		self._name = name
		self._band = band
		self._path = path
		t = kwargs.get('t')
		if t is not None:
			self.t = np.require(t, requirements=['F', 'A', 'W', 'O', 'E'])
		else:
			raise KeyError('Must supply key-word argument t!')
		self._numCadences = self.t.shape[0]
		self.x = np.require(kwargs.get('x', np.zeros(self.numCadences)), requirements=['F', 'A', 'W', 'O', 'E'])
		y = kwargs.get('y')
		if y is not None:
			self.y = np.require(y, requirements=['F', 'A', 'W', 'O', 'E'])
		else:
			raise KeyError('Must supply key-word argument y!')
		yerr = kwargs.get('yerr')
		if yerr is not None:
			self.yerr = np.require(yerr, requirements=['F', 'A', 'W', 'O', 'E'])
		else:
			raise Keyerror('Must supply key-word argument yerr!')
		mask = kwargs.get('mask')
		if mask is not None:
			self.mask = np.require(mask, requirements=['F', 'A', 'W', 'O', 'E'])
		else:
			raise Keyerror('Must supply key-word argument mask!')

		self._computedCadenceNum = -1
		self._tolIR = 1.0e-3
		self._fracIntrinsicVar = 0.0
		self._fracNoiseToSignal = 0.0
		self._maxSigma = 2.0
		self._minTimescale = 2.0
		self._maxTimescale = 0.5
		self._pSim = 0
		self._qSim = 0
		self._pComp = 0
		self._qComp = 0
		self._isSmoothed = False ## Has the LC been smoothed?
		self._dtSmooth = 0.0
		self.XSim = np.require(np.zeros(self.pSim), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self.XComp = np.require(np.zeros(self.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self._xunit = r'$t$ (d)' ## Unit in which time is measured (eg. s, sec, seconds etc...).
		self._yunit = r'who the f*** knows?' ## Unit in which the flux is measured (eg Wm^{-2} etc...).

		self._startT = float(self.t[0])
		self.t -= self._startT
		self._dt = float(self.t[1] - self.t[0])
		self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
		self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
		self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
		self._T = float(self.t[-1] - self.t[0])
		self._checkIsRegular()

		count = int(np.sum(self.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(self.numCadences):
			y_meanSum += self.mask[i]*self.y[i]
			yerr_meanSum += self.mask[i]*self.yerr[i]
		if count > 0.0:
			self._mean = y_meanSum/count
			self._meanerr = yerr_meanSum/count
		else:
			self._mean = 0.0
			self._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(self.numCadences):
			y_stdSum += math.pow(self.mask[i]*self.y[i] - self._mean, 2.0)
			yerr_stdSum += math.pow(self.mask[i]*self.yerr[i] - self._meanerr, 2.0)
		if count > 0.0:
			self._std = math.sqrt(y_stdSum/count)
			self._stderr = math.sqrt(yerr_stdSum/count)
		else:
			self._std = 0.0
			self._stderr = 0.0

class sampler(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, lcObj):
		"""!
		\brief Initialize the sampler.
		
		"""
		if isinstance(lcObj, lc):
			self.min_dt = np.min(lcObj.t[1:] - lcObj.t[:-1])
			self.max_T = lcObj.t[-1] - lcObj.t[0]
			self.lcObj = lcObj

	@abc.abstractmethod
	def sample(self, **kwargs):
		raise NotImplemented

class jumpSampler(sampler):

	def sample(self, **kwargs):
		returnLC = self.lcObj.copy()
		jumpVal = kwargs.get('jump', 1)
		newNumCadences = int(self.lcObj.numCadences/float(jumpVal))
		tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		for i in xrange(newNumCadences):
			tNew[i] = self.lcObj.t[jumpVal*i]
			xNew[i] = self.lcObj.x[jumpVal*i]
			yNew[i] = self.lcObj.y[jumpVal*i]
			yerrNew[i] = self.lcObj.yerr[jumpVal*i]
			maskNew[i] = self.lcObj.mask[jumpVal*i]
		returnLC.t = tNew
		returnLC.x = xNew
		returnLC.y = yNew
		returnLC.yerr = yerrNew
		returnLC.mask = maskNew
		returnLC._numCadences = newNumCadences
		returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
		returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
		returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
		returnLC._mindt = float(np.nanmin(returnLC.t))
		returnLC._maxdt = float(np.nanmax(returnLC.t))
		count = int(np.sum(returnLC.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_meanSum += returnLC.mask[i]*returnLC.y[i]
			yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
		if count > 0.0:
			returnLC._mean = y_meanSum/count
			returnLC._meanerr = yerr_meanSum/count
		else:
			returnLC._mean = 0.0
			returnLC._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
			yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
		if count > 0.0:
			returnLC._std = math.sqrt(y_stdSum/count)
			returnLC._stderr = math.sqrt(yerr_stdSum/count)
		else:
			returnLC._std = 0.0
			returnLC._stderr = 0.0
		return returnLC

class bernoulliSampler(sampler):

	def sample(self, **kwargs):
		returnLC = self.lcObj.copy()
		probVal = kwargs.get('probability', 1.0)
		keepArray = spstats.bernoulli.rvs(probVal, size = self.lcObj.numCadences)
		newNumCadences = np.sum(keepArray)
		tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		counter = 0
		for i in xrange(self.lcObj.numCadences):
			if keepArray[i] == 1:
				tNew[counter] = self.lcObj.t[i]
				xNew[counter] = self.lcObj.x[i]
				yNew[counter] = self.lcObj.y[i]
				yerrNew[counter] = self.lcObj.yerr[i]
				maskNew[counter] = self.lcObj.mask[i]
				counter += 1
		returnLC.t = tNew
		returnLC.x = xNew
		returnLC.y = yNew
		returnLC.yerr = yerrNew
		returnLC.mask = maskNew
		returnLC._numCadences = newNumCadences
		returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
		returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
		returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
		returnLC._mindt = float(np.nanmin(returnLC.t))
		returnLC._maxdt = float(np.nanmax(returnLC.t))
		count = int(np.sum(returnLC.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_meanSum += returnLC.mask[i]*returnLC.y[i]
			yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
		if count > 0.0:
			returnLC._mean = y_meanSum/count
			returnLC._meanerr = yerr_meanSum/count
		else:
			returnLC._mean = 0.0
			returnLC._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
			yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
		if count > 0.0:
			returnLC._std = math.sqrt(y_stdSum/count)
			returnLC._stderr = math.sqrt(yerr_stdSum/count)
		else:
			returnLC._std = 0.0
			returnLC._stderr = 0.0
		return returnLC

class matchSampler(sampler):

	def sample(self, **kwargs):
		returnLC = self.lcObj.copy()
		timeStamps = kwargs.get('timestamps', None)
		timeStampDeltas = timeStamps[1:] - timeStamps[:-1]
		SDSSLength = timeStamps[-1] - timeStamps[0]
		minDelta = np.min(timeStampDeltas)
		if minDelta < self.lcObj.dt:
			raise ValueError('Insufficiently dense sampling!')
		if SDSSLength > self.lcObj.T:
			raise ValueError('Insufficiently long lc!')
		newNumCadences = timeStamps.shape[0]
		tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		for i in xrange(newNumCadences):
			index = np.where(self.lcObj.t > timeStamps[i])[0][0]
			tNew[i] = self.lcObj.t[index]
			xNew[i] = self.lcObj.x[index]
			yNew[i] = self.lcObj.y[index]
			yerrNew[i] = self.lcObj.yerr[index]
			maskNew[i] = self.lcObj.mask[index]
		returnLC.t = tNew
		returnLC.x = xNew
		returnLC.y = yNew
		returnLC.yerr = yerrNew
		returnLC.mask = maskNew
		returnLC._numCadences = newNumCadences
		returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
		returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
		returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
		returnLC._mindt = float(np.nanmin(returnLC.t))
		returnLC._maxdt = float(np.nanmax(returnLC.t))
		count = int(np.sum(returnLC.mask))
		y_meanSum = 0.0
		yerr_meanSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_meanSum += returnLC.mask[i]*returnLC.y[i]
			yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
		if count > 0.0:
			returnLC._mean = y_meanSum/count
			returnLC._meanerr = yerr_meanSum/count
		else:
			returnLC._mean = 0.0
			returnLC._meanerr = 0.0
		y_stdSum = 0.0
		yerr_stdSum = 0.0
		for i in xrange(returnLC.numCadences):
			y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
			yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
		if count > 0.0:
			returnLC._std = math.sqrt(y_stdSum/count)
			returnLC._stderr = math.sqrt(yerr_stdSum/count)
		else:
			returnLC._std = 0.0
			returnLC._stderr = 0.0
		return returnLC

class task(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, p, q, nthreads = psutil.cpu_count(logical = True), nburn = 1000000, nwalkers = 25*psutil.cpu_count(logical = True), nsteps = 250, maxEvals = 10000, xTol = 0.001, mcmcA = 2.0):
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
			self._maxEvals = maxEvals
			self._xTol = xTol
			self._mcmcA = mcmcA
			self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
			self._LnPosterior = np.require(np.zeros(self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
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
	def rootChain(self):
		if hasattr(self, '_rootChain'):
			return self._rootChain
		else:
			Chain = self.Chain
			self._rootChain = np.require(np.zeros((self._ndims, self._nwalkers, self._nsteps), dtype = 'complex128'), requirements=['F', 'A', 'W', 'O', 'E'])
			for stepNum in xrange(self._nsteps):
				for walkerNum in xrange(self._nwalkers):
					self._rootChain[:, walkerNum, stepNum] = roots(self._p, self._q, Chain[:, walkerNum, stepNum])
		return self._rootChain

	@property
	def timescaleChain(self):
		if hasattr(self, '_timescaleChain'):
			return self._timescaleChain
		else:
			rootChain = self.rootChain
			self._timescaleChain = np.require(np.zeros((self._ndims, self._nwalkers, self._nsteps), dtype = 'float64'), requirements=['F', 'A', 'W', 'O', 'E'])
			for stepNum in xrange(self._nsteps):
				for walkerNum in xrange(self._nwalkers):
					self._timescaleChain[:, walkerNum, stepNum] = timescales(self._p, self._q, rootChain[:, walkerNum, stepNum])
		return self._timescaleChain

	@property
	def LnPosterior(self):
		return np.reshape(self._LnPosterior, newshape = (self._nwalkers, self._nsteps), order = 'F')

	def __repr__(self):
		return "libcarma.task(%d, %d, %d, %d, %d, %d, %d, %f)"%(self._p, self._q, self._nthreads, self._nburn, self._nwalkers, self._nsteps, self._maxEvals, self._xTol)

	def __str__(self):
		line = 'p: %d; q: %d; ndims: %d\n'%(self._p, self._q, self._ndims)
		line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
		line += 'nburn (Number of light curve steps to burn): %d\n'%(self._nburn)
		line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
		line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
		line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for MCMC): %d\n'%(self._maxEvals)
		line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)
		return line

	def __eq__(self, other):
		if type(other) == task:
			if (self._p == other.p) and (self._q == other.q) and (self._nthreads == other.nthreads) and (self._nburn == other.nburn) and (self._nwalkers == other.nwalkers) and (self._nsteps == other.nsteps) and (self._maxEvals == other.maxEvals) and (self.xTol == other.xTol):
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
		return self._taskCython.set_System(dt, Theta, tnum)

	def dt(self, tnum = None):
		if tnum is None:
			tnum = 0
		return self._taskCython.get_dt(tnum)

	def Theta(self, tnum = None):
		if tnum is None:
			tnum = 0
		Theta = np.require(np.zeros(self._ndims), requirements=['F', 'A', 'W', 'O', 'E'])
		self._taskCython.get_Theta(Theta, tnum)
		return Theta

	def list(self):
		setSystems = np.require(np.zeros(self._nthreads, dtype = 'int32'), requirements=['F', 'A', 'W', 'O', 'E'])
		self._taskCython.get_setSystemsVec(setSystems)
		return setSystems.astype(np.bool_)

	def show(self, tnum = None):
		if tnum is None:
			tnum = 0
		self._taskCython.print_System(tnum)

	def Sigma(self, tnum = None):
		if tnum is None:
			tnum = 0
		Sigma = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E'])
		self._taskCython.get_Sigma(Sigma, tnum)
		return np.reshape(Sigma, newshape = (self._p, self._p), order = 'F')

	def X(self, newX = None, tnum = None):
		if tnum is None:
			tnum = 0
		if not newX:
			X = np.zeros(self._p)
			self._taskCython.get_X(X, tnum)
			return np.reshape(X, newshape = (self._p), order = 'F')
		else:
			self._taskCython.set_X(np.reshape(X, newshape = (self._p), order = 'F'), tnum)
			return newX

	def P(self, newP = None, tnum = None):
		if tnum is None:
			tnum = 0
		if not newP:
			P = np.zeros(self._p*self._p)
			self._taskCython.get_P(P, tnum)
			return np.reshape(P, newshape = (self._p, self._p), order = 'F')
		else:
			self._taskCython.set_P(np.reshape(P, newshape = (self._p*self._p), order = 'F'), tnum)
			return newP

	def simulate(self, duration, tolIR = 1.0e-3, fracIntrinsicVar = 0.15, fracNoiseToSignal = 0.001, maxSigma = 2.0, minTimescale = 2.0, maxTimescale = 0.5, burnSeed = None, distSeed = None, noiseSeed = None, tnum = None):
		if tnum is None:
			tnum = 0
		numCadences = int(round(float(duration)/self._taskCython.get_dt(threadNum = tnum)))
		intrinsicLC = basicLC(numCadences, dt = self._taskCython.get_dt(threadNum = tnum), tolIR = tolIR, fracIntrinsicVar = fracIntrinsicVar, fracNoiseToSignal = fracNoiseToSignal, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale, pSim = self._p, qSim = self._q)
		randSeed = np.zeros(1, dtype = 'uint32')
		if burnSeed is None:
			rand.rdrand(randSeed)
			burnSeed = randSeed[0]
		if distSeed is None:
			rand.rdrand(randSeed)
			distSeed = randSeed[0]
		self._taskCython.make_IntrinsicLC(intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, intrinsicLC.XSim, intrinsicLC.PSim, burnSeed, distSeed, threadNum = tnum)
		intrinsicLC._simulatedCadenceNum = numCadences - 1
		intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
		return intrinsicLC

	def extend(self, intrinsicLC, duration, gap = None, distSeed = None, noiseSeed = None, tnum = None):
		if tnum is None:
			tnum = 0
		randSeed = np.zeros(1, dtype = 'uint32')
		if distSeed is None:
			rand.rdrand(randSeed)
			distSeed = randSeed[0]
		if noiseSeed is None:
			rand.rdrand(randSeed)
			noiseSeed = randSeed[0]
		if intrinsicLC.pSim != self.p:
			intrinsicLC.pSim = self.p
		if intrinsicLC.qSim != self.q:
			intrinsicLC.qSim = self.q
		if gap is None:
			gapSize = 0.0
		else:
			gapSize = gap
		oldNumCadences = intrinsicLC.numCadences
		gapNumCadences = int(round(float(gapSize)/self._taskCython.get_dt(threadNum = tnum)))
		extraNumCadences = int(round(float(duration + gapSize)/self._taskCython.get_dt(threadNum = tnum)))
		newNumCadences = intrinsicLC.numCadences + extraNumCadences
		newt = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		newx = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of intrinsic fluxes.
		newy = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed fluxes.
		newyerr = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of observed flux errors.
		newmask = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
		for i in xrange(intrinsicLC.numCadences):
			newt[i] = intrinsicLC.t[i]
			newx[i] = intrinsicLC.x[i]
			newy[i] = intrinsicLC.y[i]
			newyerr[i] = intrinsicLC.yerr[i]
			newmask[i] = intrinsicLC.mask[i]
		for i in xrange(intrinsicLC.numCadences, newNumCadences):
			newt[i] = newt[intrinsicLC.numCadences - 1] + gapSize + (i - intrinsicLC.numCadences + 1)*self._taskCython.get_dt(threadNum = tnum)
			newmask[i] = 1.0
		intrinsicLC._numCadences = newNumCadences
		self._taskCython.extend_IntrinsicLC(intrinsicLC.numCadences, intrinsicLC._simulatedCadenceNum, intrinsicLC._tolIR, intrinsicLC._fracIntrinsicVar, intrinsicLC._fracNoiseToSignal, newt, newx, newy, newyerr, newmask, intrinsicLC.XSim, intrinsicLC.PSim, distSeed, noiseSeed, threadNum = tnum)
		if gap is not None:
			old, gap, new = np.split(newt, [oldNumCadences, oldNumCadences + gapNumCadences])
			newt = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
			old, gap, new = np.split(newx, [oldNumCadences, oldNumCadences + gapNumCadences])
			newx = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
			old, gap, new = np.split(newy, [oldNumCadences, oldNumCadences + gapNumCadences])
			newy = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
			old, gap, new = np.split(newyerr, [oldNumCadences, oldNumCadences + gapNumCadences])
			newyerr = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
			old, gap, new = np.split(newmask, [oldNumCadences, oldNumCadences + gapNumCadences])
			newmask = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
		intrinsicLC._simulatedCadenceNum = newt.shape[0] - 1
		intrinsicLC._numCadences = newt.shape[0]
		intrinsicLC.t = newt
		intrinsicLC.x = newx
		intrinsicLC.y = newy
		intrinsicLC.yerr = newyerr
		intrinsicLC.mask = newmask

		count = int(np.sum(self.mask))
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

	def observe(self, intrinsicLC, noiseSeed = None, tnum = None):
		if tnum is None:
			tnum = 0
		randSeed = np.zeros(1, dtype = 'uint32')
		if noiseSeed is None:
			rand.rdrand(randSeed)
			noiseSeed = randSeed[0]
		if intrinsicLC._observedCadenceNum == -1:
			self._taskCython.add_ObservationNoise(intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum = tnum)
		else:
			self._taskCython.extend_ObservationNoise(intrinsicLC.numCadences, intrinsicLC.observedCadenceNum, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum = tnum)
			intrinsicLC._observedCadenceNum = intrinsicLC._numCadences - 1

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
		observedLC._logPrior =  self._taskCython.compute_LnPrior(observedLC.numCadences, observedLC.tolIR, observedLC.maxSigma*observedLC.std, observedLC.minTimescale*observedLC.mindt, observedLC.maxTimescale*observedLC.T, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)
		return observedLC._logPrior

	def logLikelihood(self, observedLC, forced = True, tnum = None):
		if tnum is None:
			tnum = 0
		observedLC.pComp = self.p
		observedLC.qComp = self.q
		observedLC._logPrior = self.logPrior(observedLC, forced = forced, tnum = tnum)
		if forced == True:
			observedLC._computedCadenceNum = -1
		if observedLC._computedCadenceNum == -1:
			for rowCtr in  xrange(observedLC.pComp):
				observedLC.XComp[rowCtr] = 0.0
				for colCtr in xrange(observedLC.pComp):
					observedLC.PComp[rowCtr + observedLC.pComp*colCtr] = 0.0
			observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(observedLC.numCadences, observedLC._computedCadenceNum, observedLC.tolIR, observedLC.t, observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask, observedLC.XComp, observedLC.PComp, tnum)
			observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
			observedLC._computedCadenceNum = observedLC.numCadences - 1
		elif observedLC._computedCadenceNum == observedLC.numCadences - 1:
			pass
		else:
			observedLC._logLikelihood = self._taskCython.update_LnLikelihood(observedLC.numCadences, observedLC._computedCadenceNum, observedLC._logLikelihood, observedLC.tolIR, observedLC.t, observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask, observedLC.XComp, observedLC.PComp, tnum)
			observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
			observedLC._computedCadenceNum = observedLC.numCadences - 1
		return observedLC._logLikelihood

	def logPosterior(self, observedLC, forced = True, tnum = None):
		lnLikelihood = self.logLikelihood(observedLC, forced = forced, tnum = tnum)
		return observedLC._logPosterior

	def acvf(self, start = 0.0, stop = 100.0, num = 100, endpoint = True, base  = 10.0, spacing = 'linear'):
		if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
			lags = np.logspace(np.log10(start)/np.log10(base), np.log10(stop)/np.log10(base), num  = num, endpoint = endpoint, base = base)
		elif spacing.lower() in ['linear', 'lin']:
			lags = np.linspace(start, stop, num  = num, endpoint = endpoint)
		else:
			raise RuntimeError('Unable to parse spacing')
		acvf = np.zeros(num)
		self._taskCython.compute_ACVF(num, lags, acvf)
		return lags, acvf

	def acf(self, start = 0.0, stop = 100.0, num = 100, endpoint = True, base  = 10.0, spacing = 'linear'):
		if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
			lags = np.logspace(np.log10(start)/np.log10(base), np.log10(stop)/np.log10(base), num  = num, endpoint = endpoint, base = base)
		elif spacing.lower() in ['linear', 'lin']:
			lags = np.linspace(start, stop, num  = num, endpoint = endpoint)
		else:
			raise RuntimeError('Unable to parse spacing')
		acvf = np.zeros(num)
		acf = np.zeros(num)
		self._taskCython.compute_ACVF(num, lags, acvf)
		acf = acvf/acvf[0]
		return lags, acf

	def sf(self, start = 0.0, stop = 100.0, num = 100, endpoint = True, base  = 10.0, spacing = 'linear'):
		if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
			lags = np.logspace(np.log10(start)/np.log10(base), np.log10(stop)/np.log10(base), num  = num, endpoint = endpoint, base = base)
		elif spacing.lower() in ['linear', 'lin']:
			lags = np.linspace(start, stop, num  = num, endpoint = endpoint)
		else:
			raise RuntimeError('Unable to parse spacing')
		acvf = np.zeros(num)
		sf = np.zeros(num)
		self._taskCython.compute_ACVF(num, lags, acvf)
		sf = 2.0*(acvf[0] - acvf)
		return lags, sf

	def plotacvf(self, fig = -2, LC = None, newdt = None, doShow = False, clearFig = True):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		if clearFig:
			plt.clf()
		if LC is not None:
			lagsM, acvfM = self.acvf(start = LC.dt, stop = LC.T, num = 1000, spacing = 'linear')
		else:
			lagsM, acvfM = self.acvf(start = 0.0, stop = 1000.0, num = 1000, spacing = 'linear')
		plt.plot(lagsM, acvfM, label = r'model Autocovariance Function', color = '#984ea3', zorder = 5)
		if LC is not None:
			if np.sum(LC.y) != 0.0:
				lagsE, acvfE, acvferrE = LC.acvf(newdt)
				if np.sum(acvfE) != 0.0:
					plt.errorbar(lagsE[1:], acvfE[1:], acvferrE[1:], label = r'obs. Autocovariance Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 0)
					plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$\log ACVF$')
		plt.title(r'Autocovariance Function')
		plt.legend(loc = 3)
		if doShow:
			plt.show(False)
		return newFig

	def plotacf(self, fig = -3, LC = None, newdt = None, doShow = False, clearFig = True):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		if clearFig:
			plt.clf()
		if LC is not None:
			lagsM, acfM = self.acf(start = LC.dt, stop = LC.T, num = 1000, spacing = 'linear')
		else:
			lagsM, acfM = self.acf(start = 0.0, stop = 1000.0, num = 1000, spacing = 'linear')
		plt.plot(lagsM, acfM, label = r'model Autocorrelation Function', color = '#984ea3', zorder = 5)
		if LC is not None:
			if np.sum(LC.y) != 0.0:
				lagsE, acfE, acferrE = LC.acf(newdt)
				if np.sum(acfE) != 0.0:
					plt.errorbar(lagsE[1:], acfE[1:], acferrE[1:], label = r'obs. Autocorrelation Function', fmt = 'o', capsize = 0, color = '#ff7f00', markeredgecolor = 'none', zorder = 0)
					plt.xlim(lagsE[1], lagsE[-1])
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$\log ACF$')
		plt.title(r'Autocorrelation Function')
		plt.legend(loc = 3)
		plt.ylim(-1.0, 1.0)
		if doShow:
			plt.show(False)
		return newFig

	def plotsf(self, fig = -4, LC = None, newdt = None, doShow = False, clearFig = True):
		newFig = plt.figure(fig, figsize = (fwid, fhgt))
		if clearFig:
			plt.clf()
		if LC is not None and np.sum(LC.y) != 0.0:
			lagsE, sfE, sferrE = LC.sf(newdt)
			lagsM, sfM = self.sf(start = lagsE[1], stop = lagsE[-1], num = 1000, spacing = 'log')
		else:
			lagsM, sfM = self.sf(start = 0.001, stop = 1000.0, num = 1000, spacing = 'log')
		plt.plot(np.log10(lagsM[1:]), np.log10(sfM[1:]), label = r'model Structure Function', color = '#984ea3', zorder = 5)
		if LC is not None:
			if np.sum(LC.y) != 0.0:
				if np.sum(sfE) != 0.0:
					plt.scatter(np.log10(lagsE[np.where(sfE != 0.0)[0]]), np.log10(sfE[np.where(sfE != 0.0)[0]]), marker = 'o', label = r'obs. Structure Function', color = '#ff7f00', edgecolors = 'none', zorder = 0)
		plt.xlim(math.log10(lagsM[1]), math.log10(lagsM[-1]))
		plt.xlabel(r'$\delta t$')
		plt.ylabel(r'$\log SF$')
		plt.title(r'Structure Function')
		plt.legend(loc = 2)
		if doShow:
			plt.show(False)
		return newFig

	def _psddenominator(self, freqs, order):
		nfreqs = freqs.shape[0]
		aList = self.Theta()[0:self.p].tolist()
		aList.insert(0, 1.0)
		psddenominator = np.zeros(nfreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*self.p)):
			aList.pop(0)
			return PSDVals
		else:
			for freq in xrange(nfreqs):
				val = 0.0
				for i in xrange(self.p + 1):
					j = 2*self.p - i - order
					if ((j >= 0) and (j < self.p + 1)):
						val += (aList[i]*aList[j]*((2.0*math.pi*1j*freqs[freq])**(2*self.p - (i + j)))*pow(-1.0, self.p - j)).real
					psddenominator[freq] = val
			aList.pop(0)
			return psddenominator

	def _psdnumerator(self, freqs, order):
		nfreqs = freqs.shape[0]
		bList = self.Theta()[self.p:self.p+self.q+1].tolist()
		psdnumerator = np.zeros(nfreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*self.q)):
			return psdnumerator
		else:
			for freq in xrange(nfreqs):
				val = 0.0
				for i in xrange(self.q + 1):
					j = 2*self.q - i - order
					if ((j >= 0) and (j < self.q + 1)):
						val += (bList[i]*bList[j]*((2.0*math.pi*1j*freqs[freq])**(2*self.q - (i + j)))*pow(-1.0, self.q - j)).real
					psdnumerator[freq] = val
			return psdnumerator

	def psd(self, start = 0.1, stop = 100.0, num = 100, endpoint = True, base  = 10.0, spacing = 'log'):
		if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
			freqs = np.logspace(np.log10(start)/np.log10(base), np.log10(stop)/np.log10(base), num  = num, endpoint = endpoint, base = base)
		elif spacing.lower() in ['linear', 'lin']:
			freqs = np.linspace(start, stop, num  = num, endpoint = endpoint)
		else:
			raise RuntimeError('Unable to parse spacing')
		maxDenomOrder = 2*self.p
		maxNumerOrder = 2*self.q

		psdnumeratorcomponent = np.zeros((num,(maxNumerOrder/2) + 1))
		psddenominatorcomponent = np.zeros((num,(maxDenomOrder/2) + 1))

		psdnumerator = np.zeros(num)
		psddenominator = np.zeros(num)
		psd = np.zeros(num)

		for orderVal in xrange(0, maxNumerOrder + 1, 2):
			psdnumeratorcomponent[:,orderVal/2] = self._psdnumerator(freqs, orderVal)

		for orderVal in xrange(0, maxDenomOrder + 1, 2):
			psddenominatorcomponent[:,orderVal/2] = self._psddenominator(freqs, orderVal)

		for freq in xrange(num):
			for orderVal in xrange(0, maxNumerOrder + 1, 2):
				psdnumerator[freq] += psdnumeratorcomponent[freq, orderVal/2]
			for orderVal in xrange(0, maxDenomOrder + 1, 2):
				psddenominator[freq] += psddenominatorcomponent[freq, orderVal/2]
			psd[freq] = psdnumerator[freq]/psddenominator[freq]
		return freqs, psd, psdnumerator, psddenominator, psdnumeratorcomponent, psddenominatorcomponent

	def fit(self, observedLC, zSSeed = None, walkerSeed = None, moveSeed = None, xSeed = None):
		observedLC.pComp = self.p
		observedLC.qComp = self.q
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
		minT = observedLC.mindt*observedLC.minTimescale
		maxT = observedLC.T*observedLC.maxTimescale
		minTLog10 = math.log10(minT)
		maxTLog10 = math.log10(maxT)

		for walkerNum in xrange(self.nwalkers):
			noSuccess = True
			sigmaFactor = 1.0e0
			RhoGuess = -1.0/np.power(10.0, ((maxTLog10 - minTLog10)*np.random.random(self.p + self.q + 1) + minTLog10))
			while noSuccess:
				RhoGuess[self.p + self.q] = sigmaFactor*observedLC.std
				ThetaGuess = coeffs(self.p, self.q, RhoGuess)
				res = self.set(observedLC.dt, ThetaGuess)
				lnPrior = self.logPrior(observedLC)
				if res == 0 and lnPrior == 0.0:
					noSuccess = False
				else:
					print 'SigmaTrial: %e'%(RhoGuess[self.p + self.q])
					sigmaFactor *= 0.31622776601 # sqrt(0.1)

			for dimNum in xrange(self.ndims):
				xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]
		res = self._taskCython.fit_CARMAModel(observedLC.dt, observedLC.numCadences, observedLC.tolIR, observedLC.maxSigma*observedLC.std, observedLC.minTimescale*observedLC.mindt, observedLC.maxTimescale*observedLC.T, observedLC.t, observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, self.mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPosterior)
		return res

	def smooth(self, observedLC, tnum = None):
		if tnum is None:
			tnum = 0
		if observedLC.dtSmooth is None or observedLC.dtSmooth == 0.0:
			observedLC.dtSmooth = observedLC.mindt/10.0

		observedLC.pComp = self.p
		observedLC.qComp = self.q

		t = observedLC.t.tolist() + np.linspace(start = observedLC.t[0], stop = observedLC.t[-1], num = int(math.ceil(observedLC.T/observedLC.dtSmooth)), endpoint = False).tolist()
		t.sort()
		t = _f7(t) # remove duplicates

		observedLC.numCadencesSmooth = len(t)

		observedLC.tSmooth = np.require(np.array(t), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.xSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.xerrSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.ySmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.yerrSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.maskSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.XSmooth = np.require(np.zeros(observedLC.numCadencesSmooth*observedLC.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.
		observedLC.PSmooth = np.require(np.zeros(observedLC.numCadencesSmooth*observedLC.pComp*observedLC.pComp), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of timestamps.

		unObsErr = math.sqrt(sys.float_info.max)

		obsCtr = 0
		for i in xrange(observedLC.numCadencesSmooth):
			if observedLC.tSmooth[i] == observedLC.t[obsCtr]:
				observedLC.xSmooth[i] = 0.0
				observedLC.xerrSmooth[i] = unObsErr
				observedLC.ySmooth[i] = observedLC.y[obsCtr]
				observedLC.yerrSmooth[i] = observedLC.yerr[obsCtr]
				observedLC.maskSmooth[i] = observedLC.mask[obsCtr]
				obsCtr += 1
			else:
				observedLC.xSmooth[i] = 0.0
				observedLC.xerrSmooth[i] = unObsErr
				observedLC.ySmooth[i] = 0.0
				observedLC.yerrSmooth[i] = unObsErr
				observedLC.maskSmooth[i] = 0.0

		preSmoothYMean = np.mean(observedLC.ySmooth[np.nonzero(observedLC.maskSmooth)])
		res = self._taskCython.smooth_RTS(observedLC.numCadencesSmooth, -1, observedLC.tolIR, observedLC.tSmooth, observedLC.xSmooth, observedLC.ySmooth - preSmoothYMean, observedLC.yerrSmooth, observedLC.maskSmooth, observedLC.XComp, observedLC.PComp, observedLC.XSmooth, observedLC.PSmooth, tnum)

		for i in xrange(observedLC.numCadencesSmooth):
			observedLC.xSmooth[i] = observedLC.XSmooth[i*observedLC.pComp] + preSmoothYMean
			try:
				observedLC.xerrSmooth[i] = math.sqrt(observedLC.PSmooth[i*observedLC.pComp*observedLC.pComp])
			except ValueError:
				observedLC.xerrSmooth[i] = 0.0
		observedLC._isSmoothed = True

		return res

class basicTask(task):
	def __init__(self, p, q, nthreads = psutil.cpu_count(logical = True), nburn = 1000000, nwalkers = 25*psutil.cpu_count(logical = True), nsteps = 250, maxEvals = 1000, xTol = 0.005):
		super(basicTask, self).__init__(p, q, nthreads, nburn, nwalkers, nsteps, maxEvals, xTol)
