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

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab

from bin._libcarma import ffi
from python.task import Task
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendMedium']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')
gs = gridspec.GridSpec(1000, 1000)

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class writeMockLCTask(Task):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""
	def __init__(self, WorkingDirectory, ConfigFile, DateTime = None):
		Task.__init__(self, WorkingDirectory = WorkingDirectory, ConfigFile = ConfigFile, DateTime = DateTime)
		self.LC.IR = None

	def _read_escChar(self):
		""" Attempts to set the escape charatcter to be used.
		"""
		try:
			self.escChar = self.parser.get('MISC', 'escChar')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.escChar = '#'

	def _read_plotOptions(self):
		"""	Attempts to read in the plot options to be used.
		"""
		try:
			self.makePlot = self.strToBool(self.plotParser.get('PLOT', 'makePlot'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.makePlot = False
		try:
			self.JPG = self.strToBool(self.plotParser.get('PLOT', 'JPG'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.JPG = False
		try:
			self.PDF = self.strToBool(self.plotParser.get('PLOT', 'PDF'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PDF = False
		try:
			self.EPS = self.strToBool(self.plotParser.get('PLOT', 'EPS'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EPS = False
		try:
			self.PNG = self.strToBool(self.plotParser.get('PLOT', 'PNG'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PNG = False
		try:
			self.showFig = self.strToBool(self.plotParser.get('PLOT', 'showFig'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showFig = False
		try:
			self.showDetail = self.strToBool(self.plotParser.get('PLOT', 'showDetail'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showDetail = True
		try:
			self.detailDuration = float(self.plotParser.get('PLOT', 'detailDuration'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.detailDuration = 1.0
		self.numPtsDetail = int(self.detailDuration/self.LC.dt)
		try:
			self.detailStart = self.plotParser.get('PLOT', 'detailStart')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.detailStart = 'random'
		if self.detailStart == 'random':
			self.detailStart = random.randint(0, self.LC.numCadences - self.numPtsDetail)
		else:
			self.detailStart = int(float(self.detailStart)/self.LC.dt)
			if self.detailStart > self.LC.numCadences - self.numPtsDetail:
				print "detailStart too large... Try reducing it."
				sys.exit(1)
		try:
			self.showEqn = self.strToBool(self.plotParser.get('PLOT', 'showEqn'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showEqn = True
		try:
			self.xLabel = self.plotParser.get('PLOT', 'xLabel')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xLabel = r'$t$~($d$)'
		try:
			self.yLabel = self.plotParser.get('PLOT', 'yLabel')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xLabel = r'$F$~($W m^{-2}$)'

	def _setIR(self):
		self.LC.IR = False
		for incr in self.LC.t_incr:
			if abs((incr - self.LC.dt)/((incr + self.LC.dt)/2.0)) > self.LC.tolIR:
				self.LC.IR = True

	def _usePatternFile(self, cadence, mask, t, x, y, yerr):
		"""	Attempts to find configuration parameter `tFile' in ConfigFile and reads in corresponding file of
			t values
		"""
		PatternFile = self.ConfigFile.split(".")[0] + '.pat'
		self.PatternFileHash = self.getHash(self.WorkingDirectory + self.PatternFile)
		try:
			PatternStream = open(self.WorkingDirectory + self.PatternFile, 'r')
		except IOError as Err:
			print str(Err)
			sys.exit(1)
		counter = 0
		numMasked = 0
		for line in PatternStream:
			if line[0] == self.escChar:
				continue
			cadence.append(counter)
			mask.append(float(line.rstrip("\n").split()[0]))
			if mask[-1] != 0.0:
				numMasked += 1
			t.append(float(line.rstrip("\n").split()[1]))
			x.append(0.0)
			y.append(0.0)
			yerr.append(0.0)
			counter += 1
		PatternStream.close()
		self.LC.numCadences = len(t)
		self.LC.T = t[-1] - t[0]
		self.LC.cadence = np.array(cadence)
		self.LC.mask = np.array(mask)
		self.LC.t = np.array(t)
		self.LC.x = np.array(x)
		self.LC.y = np.array(y)
		self.LC.yerr = np.array(yerr)
		self.LC.t_incr = self.LC.t[1:] - self.LC.t[0:-1]
		self.LC.dt = np.median(self.LC.t_incr)
		self.LC.numMasked = numMasked
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
			self.LC.tStart = 0.0
			print str(Err) + '. Using default tStart = %+7.6e (d)'%(self.LC.tStart)

		DT = True
		try:
			self.LC.dt = float(self.parser.get('LC', 'dt'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			DT = False
		T = True
		try:
			self.LC.T = float(self.parser.get('LC', 'T'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			T = False
		NUMCADENCES = True
		try:
			self.LC.numCadences = int(self.parser.get('LC', 'numCadences'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			NUMCADENCES = False

		if DT:
			if T or NUMCADENCES:
				if T:
					self.LC.numCadences = int(self.LC.T/self.LC.dt)
				elif NUMCADENCES:
					self.LC.T = float(self.LC.numCadences)*self.LC.dt
			elif T and NUMCADENCES:
				self.LC.dt = self.LC.T/float(self.LC.numCadences)
			self.LC.cadence = np.array(self.LC.numCadences*[0])
			self.LC.mask = np.array(self.LC.numCadences*[1.0])
			self.LC.t = np.array(self.LC.numCadences*[0.0])
			self.LC.x = np.array(self.LC.numCadences*[0.0])
			self.LC.y = np.array(self.LC.numCadences*[0.0])
			self.LC.yerr = np.array(self.LC.numCadences*[0.0])
			for i in xrange(self.LC.numCadences):
				self.LC.cadence[i] = i
				self.LC.mask[i] = 1.0
				self.LC.t[i] = i*self.LC.dt
			self.LC.numMasked = 0
			self.PatternFileHash = ''
		else:
			print 'Unable to determine light curve length. Attempting to read TandMask file'
			cadence = list()
			mask = list()
			t = list()
			x = list()
			y = list()
			yerr = list()
			self._usePatternFile(cadence, mask, t, x, y, yerr)

		try:
			self.LC.intrinsicVar = float(self.parser.get('LC', 'intrinsicVar'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.intrinsicVar = 0.1
			print str(Err) + '. Using default intrinsicVar = %+7.6e.'%(self.LC.intrinsicVar)
		try:
			self.LC.noiseLvl = float(self.parser.get('LC', 'noiseLvl'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.noiseLvl = 1.0e-18
			print str(Err) + '. Using default noiseLvl = %+7.6e.'%(self.LC.noiseLvl)
		try:
			self.LC.startCadence = self.parser.get('LC', 'startCadence')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.startCadence = 0
			print str(Err) + '. Using default startCadence = %d'%(self.LC.startCadence)

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

	def _read_TaskProps(self):
		try:
			self.doNoiseless = self.strToBool(self.parser.get('TASK', 'doNoiseless'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.doNoiseless = True
			print str(Err) + '. Using default doNoiseLess = %r'%(self.doNoiseless)
		if isinstance(self.doNoiseless, (int, float)):
			if self.doNoiseless == 0:
				self.doNoiseless = False
			else:
				self.doNoiseless = True
		elif isinstance(self.doNoiseless, str):
			if self.doNoiseless == 'No':
				self.doNoiseless = False
			else:
				self.doNoiseless = True
		elif isinstance(self.doNoiseless, bool):
			pass
		else:
			self.doNoiseless = True

	def _make_00_EqnString(self):
		"""	Attempts to construct a latex string consisting of the equation of the LC
		"""
		self.eqnStr = r'$'
		if self.p > 1:
			self.eqnStr += r'\mathrm{d}^{%d}F'%(self.p)
			for i in xrange(self.p - 2):
				self.eqnStr += r'%+4.3e\mathrm{d}^{%d}F'%(self.ARCoefs[i], self.p - 1 - i)
			self.eqnStr += r'%+4.3e\mathrm{d}F'%(self.ARCoefs[self.p - 2])
			self.eqnStr += r'%+4.3eF='%(self.ARCoefs[self.p - 1])
		elif self.p == 1:
			self.eqnStr += r'\mathrm{d}F'
			self.eqnStr += r'%+4.3eF='%(self.ARCoefs[0])
		if self.q >= 2:
			for i in xrange(self.q - 1):
				self.eqnStr += r'%4.3e\mathrm{d}^{%d}(\mathrm{d}W)'%(self.MACoefs[i], self.q - 1 - i)
			self.eqnStr += r'%+4.3e\mathrm{d}(\mathrm{d}W)'%(self.MACoefs[self.q - 2])
			self.eqnStr += r'%+4.3e(\mathrm{d}W)'%(self.MACoefs[self.q - 1])
		elif self.q == 1:
			self.eqnStr += r'%4.3e\mathrm{d}(\mathrm{d}W)'%(self.MACoefs[0])
			self.eqnStr += r'%+4.3e(\mathrm{d}W)'%(self.MACoefs[1])
		else:
			self.eqnStr += r'%4.3e(\mathrm{d}W)'%(self.MACoefs[0])
		self.eqnStr += r'$'

	def _make_01_LC(self):
		"""	Attempts to make the LC
		"""
		if self.DateTime == None:
			print 'Making LC'
			cadence_cffi = ffiObj.new('int[%d]'%(self.LC.numCadences))
			mask_cffi = ffiObj.new('double[%d]'%(self.LC.numCadences))
			t_cffi = ffiObj.new('double[%d]'%(self.LC.numCadences))
			x_cffi = ffiObj.new('double[%d]'%(self.LC.numCadences))
			y_cffi = ffiObj.new('double[%d]'%(self.LC.numCadences))
			yerr_cffi = ffiObj.new('double[%d]'%(self.LC.numCadences))
			for i in xrange(self.LC.numCadences):
				cadence_cffi[i] = self.LC.cadence[i]
				mask_cffi[i] = self.LC.mask[i]
				t_cffi[i] = self.LC.t[i]
				x_cffi[i] = self.LC.x[i]
				y_cffi[i] = self.LC.y[i]
				yerr_cffi[i] = self.LC.yerr[i]
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
			if self.LC.IR == True:
				IR = 1
			else:
				IR = 0
			if self.doNoiseless == True:
				yORn = C._makeIntrinsicLC(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.numBurn, self.LC.numCadences, self.LC.startCadence, burnSeed, distSeed, cadence_cffi, mask_cffi, t_cffi, x_cffi)
			yORn = C._makeObservedLC(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.LC.intrinsicVar, self.LC.noiseLvl, self.numBurn, self.LC.numCadences, self.LC.startCadence, burnSeed, distSeed, noiseSeed, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
			for i in xrange(self.LC.numCadences):
				self.LC.cadence[i] = cadence_cffi[i]
				self.LC.mask[i] = mask_cffi[i]
				self.LC.t[i] = t_cffi[i]
				self.LC.x[i] = x_cffi[i]
				self.LC.y[i] = y_cffi[i]
				self.LC.yerr[i] = yerr_cffi[i]
			self.LCFile = self.WorkingDirectory + self.prefix + '_LC.dat'
		else:
			print 'Reading in LC'
			self.LCFile = self.WorkingDirectory + self.prefix + '_LC.dat'
			inFile = open(self.LCFile, 'rb')
			words = inFile.readline().rstrip('\n').split()
			LCHash = words[1]
			if (LCHash == self.ConfigFileHash):
				inFile.readline()
				inFile.readline()
				words = inFile.readline().rstrip('\n').split()
				numCadences = int(words[1])
				self.LC.cadence = np.array(numCadences*[0.0])
				self.LC.mask = np.array(numCadences*[0.0])
				self.LC.t = np.array(numCadences*[0.0])
				self.LC.x = np.array(numCadences*[0.0])
				self.LC.y = np.array(numCadences*[0.0])
				self.LC.yerr = np.array(numCadences*[0.0])
				line = inFile.readline()
				line = inFile.readline()
				for i in xrange(numCadences):
					words = inFile.readline().rstrip('\n').split()
					self.LC.cadence[i] = int(words[0])
					self.LC.mask[i] = float(words[1])
					self.LC.t[i] = float(words[2])
					self.LC.x[i] = float(words[3])
					self.LC.y[i] = float(words[4])
					self.LC.yerr[i] = float(words[5])
			else:
				print "Hash mismatch! The ConfigFile %s in WorkingDirectory %s has changed and no longer matches that used to make the light curve. Exiting!"%(self.ConfigFile, self.WorkingDirectory)
				sys.exit(1)


	def _make_02_write(self):
		if self.DateTime == None:
			print 'Writing LC'
			self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
			outFile = open(self.LCFile, 'w')
			line = "ConfigFileHash: %s\n"%(self.ConfigFileHash)
			outFile.write(line)
			line = "PatternFileHash: %s\n"%(self.PatternFileHash)
			outFile.write(line)
			line = "numCadences: %d\n"%(self.LC.numCadences)
			outFile.write(line)
			line = "numObservations: %d\n"%(self.LC.numCadences - self.LC.numMasked)
			outFile.write(line)
			line = "meanFlux: %+17.16e\n"%(np.mean(self.LC.y))
			outFile.write(line)
			line = "cadence mask t x y yerr\n"
			outFile.write(line) 
			for i in xrange(self.LC.numCadences - 1):
				line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(self.LC.cadence[i], self.LC.mask[i], self.LC.t[i], self.LC.x[i], self.LC.y[i], self.LC.yerr[i])
				outFile.write(line)
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e"%(self.LC.cadence[self.LC.cadence.shape[0]-1], self.LC.mask[self.LC.cadence.shape[0]-1], self.LC.t[self.LC.cadence.shape[0]-1], self.LC.x[self.LC.cadence.shape[0]-1], self.LC.y[self.LC.cadence.shape[0]-1], self.LC.yerr[self.LC.cadence.shape[0]-1])
			outFile.write(line)
			outFile.close()
		else:
			pass

	def _make_03_plot(self):
		if self.makePlot == True:
			print 'Plotting LC'
			fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))

			ax1 = fig1.add_subplot(gs[:,:])
			ax1.ticklabel_format(useOffset = False)
			if self.doNoiseless == True:
				ax1.plot(self.LC.t, self.LC.x, color = '#7570b3', zorder = 5)
			ax1.errorbar(self.LC.t, self.LC.y, self.LC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
			yMax=np.max(self.LC.y[np.nonzero(self.LC.y[:])])
			yMin=np.min(self.LC.y[np.nonzero(self.LC.y[:])])
			ax1.set_xlabel(self.xLabel)
			ax1.set_ylabel(self.yLabel)
			ax1.set_xlim(self.LC.t[0],self.LC.t[-1])
			ax1.set_ylim(yMin,yMax)
			ax1.annotate(self.eqnStr, xy = (0.5, 0.1), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = 16, zorder = 100)

			if self.showDetail == True:
				ax2 = fig1.add_subplot(gs[50:299,700:949])
				ax2.ticklabel_format(useOffset = False)
				if self.doNoiseless == True:
					ax2.plot(self.LC.t[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.x[self.detailStart:self.detailStart+self.numPtsDetail], color = '#7570b3', zorder = 15)
				ax2.errorbar(self.LC.t[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.y[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.yerr[self.detailStart:self.detailStart+self.numPtsDetail], fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
				ax2.set_xlabel(self.xLabel)
				ax2.set_ylabel(self.yLabel)
				ax2.set_xlim(self.LC.t[self.detailStart],self.LC.t[self.detailStart + self.numPtsDetail])
				#yMax=np.max(self.LC.y[np.nonzero(self.LC.y[startLoc:startLoc + numPts])])
				#yMin=np.min(self.LC.y[np.nonzero(self.LC.y[startLoc:startLoc + numPts])])
				#ax2.set_ylim(yMin,yMax)

			if self.JPG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.jpg" , dpi = plot_params['dpi'])
			if self.PDF == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.pdf" , dpi = plot_params['dpi'])
			if self.EPS == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.eps" , dpi = plot_params['dpi'])
			if self.PNG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.png" , dpi = plot_params['dpi'])
			if self.showFig == True:
				plt.show()