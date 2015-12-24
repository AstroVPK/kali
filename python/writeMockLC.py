#!/usr/bin/env python
"""	Module that defines makeMockLCTask.

	For a demonstration of the module, please run the module as a command line program using 
	bash-prompt$ python makeMockLC.py --help
	and
	bash-prompt$ python makeMockLC.py $PWD/examples/taskTest taskTest01.ini
"""
import math as math
import cmath as cmath
import numpy as np
import copy as copy
import random as random
import ConfigParser as CP
import argparse as AP
import cffi as cffi
import os as os
import sys as sys
import pdb as pdb

from bin._libcarma import ffi
from python.task import Task

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class writeMockLCTask(Task):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""
	def __init__(self, WorkingDirectory = os.getcwd() + '/examples/', ConfigFile = 'Config.ini', DateTime = None):
		Task.__init__(self, WorkingDirectory = WorkingDirectory, ConfigFile = ConfigFile, DateTime = None)
		self.dt = None
		self.IR = False

	def _read_escChar(self):
		""" Attempts to set the escape charatcter to be used.
		"""
		try:
			self.escChar = self.parser.get('MISC', 'escChar')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.escChar = '#'

	def _setIR(self):
		t_incr = self.t[1:] - self.t[0:-1]
		for incr in t_incr:
			if abs((incr - self.dt)/((incr + self.dt)/2.0)) > self.tolIR:
				self.IR = True

	def _useTandMaskFile(self):
		"""	Attempts to find configuration parameter `tFile' in ConfigFile and reads in corresponding file of
			t values
		"""
		try:
			tFile = self.parser.get('LC', 'TandMaskFile')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			print str(Err)
			sys.exit(1)
		try:
			tFileStream = open(self.WorkingDirectory + tFile, 'r')
		except IOError as Err:
			print str(Err)
			sys.exit(1)
		counter = 0
		for line in tFileStream:
			if line[0] == self.escChar:
				continue
			self.cadence.append(counter)
			self.mask.append(float(line.rstrip("\n").split()[0]))
			self.t.append(float(line.rstrip("\n").split()[1]))
			self.y.append(0.0)
			self.yerr.append(0.0)
			counter += 1
		self.numCadences = len(self.t)
		self.T = self.t[-1] - self.t[0]
		self.cadence = np.array(self.cadence)
		self.mask = np.array(self.mask)
		self.t = np.array(self.t)
		self.y = np.array(self.y)
		self.yerr = np.array(self.yerr)
		t_incr = self.t[1:] - self.t[0:-1]
		self.dt = np.median(t_incr)
		self._setIR()
		return 0

	def _read_LCProps(self):
		"""	Attempts to read in the configuration parameters `dt', `T' or `numCadences', & `tStart'.
		"""
		try:
			self.tolIR = float(self.parser.get('LC', 'tolIR'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tolIR = 0.0
			print str(Err) + '. Using default tolIR = %+7.6e'%(self.tolIR)
		try:
			self.tStart = self.parser.get('LC', 'tStart')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tStart = 0.0
			print str(Err) + '. Using default tStart = %+7.6e (d)'%(self.tStart)

		DT = True
		try:
			self.dt = float(self.parser.get('LC', 'dt'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			DT = False
		T = True
		try:
			self.T = float(self.parser.get('LC', 'T'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			T = False
		NUMCADENCES = True
		try:
			self.numCadences = int(self.parser.get('LC', 'numCadences'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			NUMCADENCES = False

		if DT:
			if T or NUMCADENCES:
				if T:
					self.numCadences = int(self.T/self.dt)
				elif NUMCADENCES:
					self.T = float(self.numCadences)*self.dt
			elif T and NUMCADENCES:
				self.dt = self.T/float(self.numCadences)
			self.cadence = np.array(self.numCadences*[0])
			self.mask = np.array(self.numCadences*[1.0])
			self.t = np.array(self.numCadences*[0.0])
			self.y = np.array(self.numCadences*[0.0])
			self.yerr = np.array(self.numCadences*[0.0])
			for i in xrange(self.numCadences):
				self.cadence[i] = i
				self.mask[i] = 1.0
				self.t[i] = i*self.dt
		else:
			print 'Unable to determine light curve length. Attempting to read TandMask file'
			self.cadence = list()
			self.mask = list()
			self.t = list()
			self.y = list()
			self.yerr = list()
			self._useTandMaskFile()

		try:
			self.intrinsicVar = float(self.parser.get('LC', 'intrinsicVar'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.intrinsicVar = 0.1
			print str(Err) + '. Using default intrinsicVar = %+7.6e.'%(self.intrinsicVar)
		try:
			self.noiseLvl = float(self.parser.get('LC', 'noiseLvl'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.noiseLvl = 1.0e-18
			print str(Err) + '. Using default noiseLvl = %+7.6e.'%(self.noiseLvl)
		try:
			self.startCadence = self.parser.get('LC', 'startCadence')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.startCadence = 0
			print str(Err) + '. Using default startCadence = %d'%(self.startCadence)

	def _read_CARMAProps(self):
		"""	Attempts to parse AR roots and MA coefficients.
		"""
		ARRoots = list()
		ARPoly = list()
		MACoefs = list()
		self.ARCoefs = list()
		self.ARRoots = list()

		doneReadingARRoots = False
		pRoot = 0
		while not doneReadingARRoots:
			try:
				ARRoots.append(complex(self.parser.get('C-ARMA', 'r_%d'%(pRoot + 1))))
				pRoot += 1
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingARRoots = True

		doneReadingARPoly = False
		pPoly = 0
		while not doneReadingARPoly:
			try:
				ARPoly.append(float(self.parser.get('C-ARMA', 'a_%d'%(pRoot + 1))))
				pPoly += 1
			except ValueError as Err:
				print str(Err) + '. All AR polynomial coefficients must be real!'
				sys.exit(1)
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingARPoly = True

		if (pRoot == pPoly):
			aPoly = np.polynomial.polynomial.polyfromroots(ARRoots)
			aPoly = aPoly.tolist()
			aPoly.reverse()
			aPoly.pop(0)
			aPoly = [coeff.real for coeff in aPoly]
			for ARCoef, aPolyCoef in zip(ARPoly, aPoly):
				if abs((ARCoef - aPolyCoef)/((ARCoef + aPolyCoef)/2.0)) > 1.0e-6:
					print 'ARRoots and ARPolynomial supplied are not equivalent!'
					sys.exit(1)
			self.p = pRoot
			self.ARRoots = np.array(ARRoots)
			self.ARCoefs = np.array(ARPoly)
		elif (pRoot == 0) and (pPoly > 0):
			self.p = pPoly
			self.ARCoefs = aPoly
			ARPoly = copy.deepcopy(self.ARCoefs)
			self.ARCoefs = np.array(self.ARCoefs)
			ARPoly.insert(0,1.0)
			self.ARRoots = np.roots(ARPoly)
			self.ARRoots = np.array(self.ARRoots)
		elif (pRoot > 0) and (pPoly == 0):
			self.p = pRoot
			self.ARRoots = ARRoots
			aPoly = np.polynomial.polynomial.polyfromroots(ARRoots)
			self.ARRoots = np.array(self.ARRoots)
			aPoly = aPoly.tolist()
			aPoly.reverse()
			aPoly.pop(0)
			aPoly = [coeff.real for coeff in aPoly]
			self.ARCoefs = copy.deepcopy(aPoly)
			self.ARCoefs = np.array(self.ARCoefs)
		else:
			print 'ARRoots and ARPolynomial supplied are not equivalent!'
			sys.exit(1)

		doneReadingMACoefs = False
		self.q = -1
		while not doneReadingMACoefs:
			try:
				MACoefs.append(float(self.parser.get('C-ARMA', 'b_%d'%(self.q + 1))))
				self.q += 1
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingMACoefs = True
		self.MACoefs = np.array(MACoefs)

		if self.p < 1:
			print 'No C-AR roots supplied!'
			sys.exit(1)

		if self.q < 0:
			print 'No C-MA co-efficients supplied!'
			sys.exit(1)

		if self.p <= self.q:
			print 'Too many C-MA co-efficients! Exiting...'
			sys.exit(1)

		self.ndims = self.p + self.q + 1

		try:
			self.numBurn = int(self.parser.get('C-ARMA', 'numBurn'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numBurn = 1000000
			print str(Err) + '. Using default numBurn = %d'%(self.numBurn)

	def run(self):
		"""	Attempts to make the LC
		"""
		cadence_cffi = ffiObj.new("int[%d]"%(self.numCadences))
		mask_cffi = ffiObj.new("double[%d]"%(self.numCadences))
		t_cffi = ffiObj.new("double[%d]"%(self.numCadences))
		y_cffi = ffiObj.new("double[%d]"%(self.numCadences))
		yerr_cffi = ffiObj.new("double[%d]"%(self.numCadences))
		for i in xrange(self.numCadences):
			cadence_cffi[i] = self.cadence[i]
			mask_cffi[i] = self.mask[i]
			t_cffi[i] = self.t[i]
			y_cffi[i] = self.y[i]
			yerr_cffi[i] = self.yerr[i]
		Theta_cffi = ffiObj.new("double[%d]"%(self.p + self.q + 1))
		for i in xrange(self.p):
			Theta_cffi[i] = self.ARCoefs[i]
		for i in xrange(self.q + 1):
			Theta_cffi[self.p + i] = self.MACoefs[i]
		burnSeed = random.randint(0, 4294967295)
		distSeed = random.randint(0, 4294967295)
		noiseSeed = random.randint(0, 4294967295)
		#burnSeed = 1311890535
		#distSeed = 2603023340
		#noiseSeed = 2410288857
		if self.IR == True:
			IR = 1
		else:
			IR = 0
		yORn = C._makeIntrinsicLC(self.dt, self.p, self.q, Theta_cffi, IR, self.tolIR, self.numBurn, self.numCadences, self.startCadence, burnSeed, distSeed, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
		for i in xrange(self.numCadences):
			self.cadence[i] = cadence_cffi[i]
			self.mask[i] = mask_cffi[i]
			self.t[i] = t_cffi[i]
			self.y[i] = y_cffi[i]
			self.yerr[i] = yerr_cffi[i]

	'''self.t = None
	self.y = None
	self.yerr = None
	self.Mask = None
	self.Cadences = None
	if not self.DateTime:
		##sigmay = 1.2e-9
		SigmaY = self.MAPoly[0]
		##ma_coefs = np.array([1.0, 5.834])
		MACoefs = np.array([Poly/SigmaY for Poly in self.MAPoly])

		SigSqr = SigmaY**2/cmcmc.carma_variance(1.0, self.ARRoots, ma_coefs = MACoefs)

		self.t = np.arange(0.0, self.T, self.dt)
		self.y = cmcmc.carma_process(self.t, SigSqr, self.ARRoots, ma_coefs = MACoefs)
		self.y += np.array(self.t.shape[0]*[self.baseFlux])
		if not Cadences:
			self.Cadences = np.arange(self.t.shape[0])
		else:
			self.Cadences = Cadences
		noise = np.random.normal(loc = 0.0, scale = self.noiseLvl, size = self.Cadences.shape[0])
		self.y += noise
		self.yerr = np.array(self.Cadences.shape[0]*[self.noiseLvl])

		numMasked = 0
		if Mask:
			self.Mask = np.array(Mask)
			for i in xrange(self.Mask.shape[0]):
				if self.Mask[i] == 0.0:
					self.t[i] = 0.0
					self.y[i] = 0.0
					self.yerr[i] = 1.3407807929942596e+154
					numMasked += 1
		else:
			self.Mask = np.array(self.Cadences.shape[0]*[1.0])'''

	def write(self):
		self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
		outFile = open(self.LCFile, 'w')
		line = "ConfigFileHash: %s\n"%(self.ConfigFileHash)
		outFile.write(line)
		line = "numCadences: %d\n"%(self.t.shape[0])
		outFile.write(line)
		line = "numObservations: %d\n"%(self.t.shape[0] - numMasked)
		outFile.write(line)
		line = "meanFlux: %+17.16e\n"%(np.mean(self.y))
		outFile.write(line)
		line = "CadenceNum Mask t y yerr\n"
		outFile.write(line) 
		for i in xrange(self.Cadences.shape[0]-1):
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e\n"%(self.Cadences[i], self.Mask[i], self.t[i], self.y[i], self.yerr[i])
			outFile.write(line)
		line = "%d %1.0f %+17.16e %+17.16e %+17.16e"%(self.Cadences[self.Cadences.shape[0]-1], self.Mask[self.Cadences.shape[0]-1], self.t[self.Cadences.shape[0]-1], self.y[self.Cadences.shape[0]-1], self.yerr[self.Cadences.shape[0]-1])
		outFile.write(line)
		outFile.close()

	'''else:
		self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
		inFile = open(self.LCFile, 'rb')
		words = inFile.readline().rstrip('\n').split()
		LCHash = words[0]
		if (LCHash == self.ConfigFileHash):
			line = inFile.readline()
			line = line.rstrip('\n')
			words = line.split()
			numCadences = int(words[1])
			self.t = np.array(numCadences*[0.0])
			self.y = np.array(numCadences*[0.0])
			self.yerr = np.array(numCadences*[0.0])
			self.Cadences = np.array(numCadences*[0.0])
			self.Mask = np.array(numCadences*[0.0])
			line = inFile.readline()
			line = inFile.readline()
			for i in xrange(numCadences):
				words = inFile.readline().rstrip('\n').split()
				self.t[i] = i*self.dt
				self.Cadences[i] = int(words[0])
				self.Mask[i] = float(words[1])
				self.y[i] = float(words[2])
				self.yerr[i] = float(words[3])
		else:
			print "Hash mismatch! The ConfigFile %s in WorkingDirectory %s has changed and no longer matches that used to make the light curve. Exiting!"%(self.ConfigFile, self.WorkingDirectory)

	if self.plotLC == True:
		fig1 = plt.figure(1, figsize=(fwid, fhgt))
		ax1 = fig1.add_subplot(gs[:,:])
		ax1.ticklabel_format(useOffset = False)
		ax1.plot(self.t, self.y)
		ax1.set_xlabel(r'$t$ (d)')
		ax1.set_ylabel(r'Flux')
		if self.JPG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.jpg" , dpi = dpi)
		if self.PDF == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.pdf" , dpi = dpi)
		if self.EPS == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.eps" , dpi = dpi)
		if self.showFig == True:
			plt.show()
	return 0'''