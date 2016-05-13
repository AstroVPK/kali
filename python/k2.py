import math as math
import numpy as np
import astropy.io.fits as astfits
import os as os
import pdb

import libcarma as libcarma

# all classes should have inits for initializing 

class K2LC(libcarma.basicLC):

	def __init__(self):
		print("\n")
	def read(self, name, band = None, pwd = None, **kwargs):
	#open Van_k2 lc fits file 
		k2lc = astfits.open(pwd + name.split('.')[0] + '.fits')
		k2lc.info()
		table = k2lc[1].data
		#print table
		self.t = table[0:].field('T')
		rawF = table[0:].field('FRAW')
		self.x = np.zeros(len(rawF))
		self.y = table[0:].field('FCOR')
		arc = table[0:].field('ARCLENGTH')
		#self.yerr = 

	def write(self, name, pwd):
		print('read in Vanderberg lc')

class k2pdcsapLC(libcarma.basicLC):
	def read(self, name, band = None, path = None, **kwargs):
		self._tolIR = 1.0e-3
		self._fracIntrinsicVar = 0.0
		self._fracNoiseToSignal = 0.0
		self._maxSigma = 2.0
		self._minTimescale = 2.0
		self._maxTimescale = 0.5
		if path is None:
			path = os.environ['PWD']
		#open kepler k2 lc fits file
		k2lc = astfits.open(os.path.join(path, name))
		k2lc.info()
		table = k2lc[1].data
		cadenceIn = np.array(table[0:].field('CADENCENO'))
		tIn = np.array(table[0:].field('TIME'))
		yIn = np.array(table[0:].field('PDCSAP_FLUX'))
		yerrIn = np.array(table[0:].field('PDCSAP_FLUX_ERR'))
		maskIn = np.array(table[0:].field('SAP_QUALITY'))
		self._numCadences = tIn.shape[0]
		self._dt = np.nanmedian(tIn[1:] - tIn[:-1]) ## Increment between epochs.
		self._T = tIn[-1] - tIn[0] ## Total duration of the light curve.
		self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
		for i in xrange(self.numCadences):
			self.cadence[i] = cadenceIn[i]
			if maskIn[i] == 0:
				self.t[i] = tIn[i]
				self.y[i] = yIn[i]
				self.yerr[i] = yerrIn[i]
				self.mask[i] = 1.0
			else:
				if np.isnan(tIn[i]) == False:
					self.t[i] = tIn[i]
				else:
					self.t[i] = self.t[i - 1] + self._dt
				self.mask[i] = 0.0
		self._dt = np.nanmedian(self.t[1:] - self.t[:-1]) ## Increment between epochs.
		self._p = 0
		self._q = 0
		self.XSim = np.require(np.zeros(self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PSim = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self.XComp = np.require(np.zeros(self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PComp = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self._name = str(name.split('.')[0]) ## The name of the light curve (usually the object's name).
		self._band = str('Kep') ## The name of the photometric band (eg. HSC-I or SDSS-g etc..).
		self._xunit = r'$d$' ## Unit in which time is measured (eg. s, sec, seconds etc...).
		self._yunit = r'$who the f**** knows?$' ## Unit in which the flux is measured (eg Wm^{-2} etc...).

	def write(self, name, path = None, **kwrags):
		if path is None:
			path = os.envioron['PWD']
		print('read in Kepler Team lc')

class k2rawLC(libcarma.basicLC):
	def read(self, name, band = None, path = None, **kwargs):
		self._tolIR = 1.0e-3
		self._fracIntrinsicVar = 0.0
		self._fracNoiseToSignal = 0.0
		self._maxSigma = 2.0
		self._minTimescale = 2.0
		self._maxTimescale = 0.5
		if path is None:
			path = os.environ['PWD']
		#open kepler k2 lc fits file
		k2lc = astfits.open(os.path.join(path, name))
		k2lc.info()
		table = k2lc[1].data
		cadenceIn = np.array(table[0:].field('CADENCENO'))
		tIn = np.array(table[0:].field('TIME'))
		yIn = np.array(table[0:].field('SAP_FLUX'))
		yerrIn = np.array(table[0:].field('PDCSAP_FLUX_ERR'))
		maskIn = np.array(table[0:].field('SAP_QUALITY'))
		self._numCadences = tIn.shape[0]
		self._dt = np.nanmedian(tIn[1:] - tIn[:-1]) ## Increment between epochs.
		self._T = tIn[-1] - tIn[0] ## Total duration of the light curve.
		self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
		for i in xrange(self.numCadences):
			self.cadence[i] = cadenceIn[i]
			if maskIn[i] == 0:
				self.t[i] = tIn[i]
				self.y[i] = yIn[i]
				self.yerr[i] = yerrIn[i]
				self.mask[i] = 1.0
			else:
				if np.isnan(tIn[i]) is False:
					self.t[i] = tIn[i]
				else:
					self.t[i] = self.t[i - 1] + self._dt
				self.mask[i] = 0.0
		self._dt = np.nanmedian(self.t[1:] - self.t[:-1]) ## Increment between epochs.
		self._p = 0
		self._q = 0
		self.XSim = np.require(np.zeros(self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PSim = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self.XComp = np.require(np.zeros(self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
		self.PComp = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
		self._name = str(name.split('.')[0]) ## The name of the light curve (usually the object's name).
		self._band = str('Kep') ## The name of the photometric band (eg. HSC-I or SDSS-g etc..).
		self._xunit = r'$d$' ## Unit in which time is measured (eg. s, sec, seconds etc...).
		self._yunit = r'$who the f**** knows?$' ## Unit in which the flux is measured (eg Wm^{-2} etc...).

	def write(self, name, path = None, **kwargs):
		if path is None:
			path = os.envioron['PWD']
		print('read in Kepler Team lc')