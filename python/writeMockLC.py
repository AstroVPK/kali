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
import time as time
import pdb as pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab

from bin._libcarma import ffi
from python.task import SuppliedParametersTask
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXSmall']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')
gs = gridspec.GridSpec(1000, 1000)

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class writeMockLCTask(SuppliedParametersTask):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""
	def _read_plotOptions(self):
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
			self.LabelLCFontsize = self.strToBool(self.plotParser.get('PLOT', 'LabelLCFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LabelLCFontsize = 18
		try:
			self.DetailLabelLCFontsize = self.strToBool(self.plotParser.get('PLOT', 'DetailLabelLCFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.DetailLabelLCFontsize = 10
		try:
			self.showEqnLC = self.strToBool(self.plotParser.get('PLOT', 'showEqnLC'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showEqnLC = True
		try:
			self.showLnLike = self.strToBool(self.plotParser.get('PLOT', 'showLnLike'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showLnLike = True
		try:
			self.EqnLCLocY = float(self.plotParser.get('PLOT', 'EqnLCLocY'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EqnLCLocY = 0.1
		try:
			self.EqnLCFontsize = int(self.plotParser.get('PLOT', 'EqnLCFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EqnLCFontsize = 16
		try:
			self.showLegendLC = self.strToBool(self.plotParser.get('PLOT', 'showLegendLC'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showLegendLC = True
		try:
			self.LegendLCLoc = int(self.plotParser.get('PLOT', 'LegendLCLoc'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LegendLCLoc = 2
		try:
			self.LegendLCFontsize = int(self.plotParser.get('PLOT', 'LegendLCFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LegendLCFontsize = 12
		try:
			self.xLabelLC = self.plotParser.get('PLOT', 'xLabelLC')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xLabelLC = r'$t$~($d$)'
		try:
			self.yLabelLC = self.plotParser.get('PLOT', 'yLabelLC')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xLabelLC = r'$F$~($W m^{-2}$)'

	def _setIR(self):
		self.LC.IR = False
		for incr in self.LC.t_incr:
			if abs((incr - self.LC.dt)/((incr + self.LC.dt)/2.0)) > self.LC.tolIR:
				self.LC.IR = True

	'''def _usePatternFile(self, cadence, mask, t, x, y, yerr):
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
		return 0'''

	def _readLC(self, suppliedLC = None):
		logEntry = 'Reading in LC'
		self.echo(logEntry)
		self.log(logEntry)
		if suppliedLC == None:
			self.SuppliedLCFile = self.WorkingDirectory + self.prefix + '_LC.dat'
		else:
			self.SuppliedLCFile = self.WorkingDirectory + suppliedLC
			self.SuppliedLCHash = self.getHash(self.SuppliedLCFile)
		inFile = open(self.SuppliedLCFile, 'rb')
		words = inFile.readline().rstrip('\n').split()
		LCHash = words[1]
		if (LCHash == self.ConfigFileHash) or (suppliedLC != None):
			inFile.readline()
			self.LC.numCadences = int(inFile.readline().rstrip('\n').split()[1])
			self.LC.cadence = np.array(self.LC.numCadences*[0])
			self.LC.mask = np.array(self.LC.numCadences*[0.0])
			self.LC.t = np.array(self.LC.numCadences*[0.0])
			self.LC.x = np.array(self.LC.numCadences*[0.0])
			self.LC.y = np.array(self.LC.numCadences*[0.0])
			self.LC.yerr = np.array(self.LC.numCadences*[0.0])
			numObservations = int(inFile.readline().rstrip('\n').split()[1])
			self.LC.meanFlux = float(inFile.readline().rstrip('\n').split()[1])
			self.LC.LnLike = float(inFile.readline().rstrip('\n').split()[1])
			line = inFile.readline()
			for i in xrange(self.LC.numCadences):
				words = inFile.readline().rstrip('\n').split()
				self.LC.cadence[i] = int(words[0])
				self.LC.mask[i] = float(words[1])
				self.LC.t[i] = float(words[2])
				self.LC.x[i] = float(words[3])
				self.LC.y[i] = float(words[4])
				self.LC.yerr[i] = float(words[5])
			self.LC.T = self.LC.t[-1] - self.LC.t[0]
			self.LC.t_incr = self.LC.t[1:] - self.LC.t[0:-1]
			self.LC.dt = np.median(self.LC.t_incr)
			self.LC.numObservations = numObservations
			self._setIR()
			inFile.close()
		else:
			print "Hash mismatch! The ConfigFile %s in WorkingDirectory %s has changed and no longer matches that used to make the light curve. Exiting!"%(self.ConfigFile, self.WorkingDirectory)
			inFile.close()
			sys.exit(1)

	def _read_LCProps(self):
		"""	Attempts to read in the configuration parameters `dt', `T' or `numCadences', & `tStart'.
		"""
		try:
			self.LC.tolIR = float(self.parser.get('LC', 'tolIR'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.tolIR = 1.0e-3
			print str(Err) + '. Using default tolIR = %+7.6e'%(self.LC.tolIR)
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
			self.LC.numObservations = self.LC.numCadences
			self.PatternFileHash = ''
		else:
			print 'No dt supplied. Attempting to use suppliedLC'
			try:
				self.LC.SuppliedLC = self.parser.get('LC', 'suppliedLC')
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				self.LC.suppliedLC = None
				print str(Err)
				sys.exit(1)
			self._readLC(suppliedLC = self.LC.SuppliedLC)

			'''print 'Unable to determine light curve length. Attempting to read TandMask file'
			cadence = list()
			mask = list()
			t = list()
			x = list()
			y = list()
			yerr = list()
			self._usePatternFile(cadence, mask, t, x, y, yerr)'''

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
			self.LC.startCadence = int(self.parser.get('LC', 'startCadence'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.startCadence = 0
			print str(Err) + '. Using default startCadence = %d'%(self.LC.startCadence)

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

	def _make_01_LC(self):
		"""	Attempts to make the LC
		"""
		if self.DateTime == None:
			logEntry = 'Making LC'
			self.echo(logEntry)
			self.log(logEntry)
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
			self.LC.LnLike = C._computeLnlike(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.LC.numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
			for i in xrange(self.LC.numCadences):
				self.LC.cadence[i] = cadence_cffi[i]
				self.LC.mask[i] = mask_cffi[i]
				self.LC.t[i] = t_cffi[i]
				self.LC.x[i] = x_cffi[i]
				self.LC.y[i] = y_cffi[i]
				self.LC.yerr[i] = yerr_cffi[i]
			self.LCFile = self.WorkingDirectory + self.prefix + '_LC.dat'
		else:
			self._readLC()
			'''logEntry = 'Reading in LC'
			self.echo(logEntry)
			self.log(logEntry)
			self.LCFile = self.WorkingDirectory + self.prefix + '_LC.dat'
			inFile = open(self.LCFile, 'rb')
			words = inFile.readline().rstrip('\n').split()
			LCHash = words[1]
			if (LCHash == self.ConfigFileHash):
				inFile.readline()
				inFile.readline()
				numCadences = int(inFile.readline().rstrip('\n').split()[1])
				self.LC.cadence = np.array(numCadences*[0.0])
				self.LC.mask = np.array(numCadences*[0.0])
				self.LC.t = np.array(numCadences*[0.0])
				self.LC.x = np.array(numCadences*[0.0])
				self.LC.y = np.array(numCadences*[0.0])
				self.LC.yerr = np.array(numCadences*[0.0])
				line = inFile.readline()
				line = inFile.readline()
				self.LC.LnLike = float(inFile.readline().rstrip('\n').split()[1])
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
				sys.exit(1)'''

	def _make_02_write(self):
		if self.DateTime == None:
			logEntry = 'Writing LC'
			self.echo(logEntry)
			self.log(logEntry)
			self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
			outFile = open(self.LCFile, 'w')
			line = self.escChar + "ConfigFileHash: %s\n"%(self.ConfigFileHash)
			outFile.write(line)
			line = self.escChar + "SuppliedLCHash: %s\n"%(self.SuppliedLCHash)
			outFile.write(line)
			line = self.escChar + "numCadences: %d\n"%(self.LC.numCadences)
			outFile.write(line)
			line = self.escChar + "numObservations: %d\n"%(self.LC.numObservations)
			outFile.write(line)
			line = self.escChar + "meanFlux: %+17.16e\n"%(np.mean(self.LC.y))
			outFile.write(line)
			line = self.escChar + "LnLike: %+17.16e\n"%(self.LC.LnLike)
			outFile.write(line)
			line = self.escChar + "cadence mask t x y yerr\n"
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
			logEntry = 'Plotting LC'
			self.echo(logEntry)
			self.log(logEntry)
			fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
			ax1 = fig1.add_subplot(gs[:,:])
			ax1.ticklabel_format(useOffset = False)
			if self.doNoiseless == True:
				ax1.plot(self.LC.t, self.LC.x, color = '#7570b3', zorder = 5, label = r'Intrinsic LC')
			ax1.errorbar(self.LC.t, self.LC.y, self.LC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, Label = r'Observed LC')
			if self.showLegendLC == True:
				ax1.legend(loc = self.LegendLCLoc, ncol = 1, fancybox = True, fontsize = self.LegendLCFontsize)
			yMax=np.max(self.LC.y[np.nonzero(self.LC.y[:])])
			yMin=np.min(self.LC.y[np.nonzero(self.LC.y[:])])
			ax1.set_xlabel(self.xLabelLC)
			ax1.set_ylabel(self.yLabelLC)
			ax1.set_xlim(self.LC.t[0],self.LC.t[-1])
			ax1.set_ylim(yMin,yMax)
			if self.showEqnLC == True:
				ax1.annotate(self.eqnStr, xy = (0.5, 0.1), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)
			if self.showLnLike == True:
				ax1.annotate(r'$\ln \mathcal{L} = ' + self.formatFloat(self.LC.LnLike) + '$', xy = (0.5, 0.05), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)

			if self.showDetail == True:
				ax2 = fig1.add_subplot(gs[50:299,700:949])
				ax2.locator_params(nbins = 3)
				ax2.ticklabel_format(useOffset = False)
				if self.doNoiseless == True:
					ax2.plot(self.LC.t[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.x[self.detailStart:self.detailStart+self.numPtsDetail], color = '#7570b3', zorder = 15)
				ax2.errorbar(self.LC.t[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.y[self.detailStart:self.detailStart+self.numPtsDetail], self.LC.yerr[self.detailStart:self.detailStart+self.numPtsDetail], fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
				ax2.set_xlabel(self.xLabelLC)
				ax2.set_ylabel(self.yLabelLC)
				ax2.set_xlim(self.LC.t[self.detailStart],self.LC.t[self.detailStart + self.numPtsDetail])

			if self.JPG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.jpg" , dpi = self.dpi)
			if self.PDF == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.pdf" , dpi = self.dpi)
			if self.EPS == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.eps" , dpi = self.dpi)
			if self.PNG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.png" , dpi = self.dpi)
			if self.showFig == True:
				plt.show()
			fig1.clf()