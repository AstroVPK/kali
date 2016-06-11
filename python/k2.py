import math as math
import numpy as np
import astropy.io.fits as astfits
import os as os
import sys as sys
import pdb

try:
	import libcarma as libcarma
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

class k2LC(libcarma.basicLC):
	def read(self, name, band = None, path = None, **kwargs):
		sapORpdcsap = kwargs.get('lctype', 'sap').lower()
		self._computedCadenceNum = -1
		self._tolIR = 1.0e-3
		self._fracIntrinsicVar = 0.0
		self._fracNoiseToSignal = 0.0
		self._maxSigma = 2.0
		self._minTimescale = 2.0
		self._maxTimescale = 0.5
		if path is None:
			path = os.environ['PWD']
		with open(os.path.join(path, name),'r') as k2File:
			allLines = k2File.readlines()
		self._numCadences = len(allLines) - 1
		startT = -1.0
		lineNum = 1
		while startT == -1.0:
			words = allLines[lineNum].split()
			nextWords = allLines[lineNum + 1].split()
			if words[0] != '""' and nextWords[0] != '""':
				startT = float(words[0])
				dt = float(nextWords[0]) - float(words[0])
			else:
				lineNum += 1
		self.startT = startT
		self._dt = dt ## Increment between epochs.
		self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
		self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
		for i in xrange(self.numCadences):
			words = allLines[i +1].split()
			self.cadence[i] = int(words[2])
			if words[9] == '0':
				self.t[i] = float(words[0]) - self.startT
				if sapORpdcsap in ['sap', 'raw', 'uncal', 'un-cal', 'uncalibrated', 'un-calibrated']:
					try:
						self.y[i] = float(words[3])
						self.yerr[i] = float(words[4])
						self.mask[i] = 1.0
					except ValueError:
						self.y[i] = 0.0
						self.yerr[i] = math.sqrt(sys.float_info[0])
						self.mask[i] = 0.0
				elif sapORpdcsap  in ['pdcsap', 'mast', 'cal', 'calibrated']:
					try:
						self.y[i] = float(words[7])
						self.yerr[i] = float(words[8])
						self.mask[i] = 1.0
					except ValueError:
						self.y[i] = 0.0
						self.yerr[i] = math.sqrt(sys.float_info[0])
						self.mask[i] = 0.0
				else:
					raise ValueError('Unrecognized k2LC type')
			else:
				if words[0] != '""':
					self.t[i] = float(words[0]) - self.startT
				else:
					self.t[i] = self.t[i - 1] + self.dt
				self.yerr[i] = math.sqrt(sys.float_info[0])
				self.mask[i] = 0.0
		self._dt = float(np.nanmedian(self.t[1:] - self.t[:-1])) ## Increment between epochs.
		self._T = self.t[-1] - self.t[0] ## Total duration of the light curve.
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
		self._isRegular = True
		self._name = str(name.split('.')[0]) ## The name of the light curve (usually the object's name).
		self._band = str('Kep') ## The name of the photometric band (eg. HSC-I or SDSS-g etc..).
		self._xunit = r'$d$' ## Unit in which time is measured (eg. s, sec, seconds etc...).
		self._yunit = r'$who the f**** knows?$' ## Unit in which the flux is measured (eg Wm^{-2} etc...).
		count = int(np.sum(self.mask[i]))
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