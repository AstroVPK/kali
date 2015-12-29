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
from python.task import SuppliedParametersTask, SuppliedLCTask
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXSmall']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')
gs = gridspec.GridSpec(1000, 1000)

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_uint = ffiObj.new_allocator(alloc = C._malloc_uint, free = C._free_uint)
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class makeMockLCTask(SuppliedParametersTask, SuppliedLCTask):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""

	def _read_00_LCProps(self):
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

	def _read_01_TaskProps(self):
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
			Theta_cffi = ffiObj.new('double[%d]'%(self.p + self.q + 1))
			for i in xrange(self.p):
				Theta_cffi[i] = self.ARCoefs[i]
			for i in xrange(self.q + 1):
				Theta_cffi[self.p + i] = self.MACoefs[i]
			randomSeeds = ffiObj.new('unsigned int[3]')
			yORn = C._getRandoms(3, randomSeeds)
			if self.LC.IR == True:
				IR = 1
			else:
				IR = 0
			if self.doNoiseless == True:
				yORn = C._makeIntrinsicLC(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.numBurn, self.LC.numCadences, self.LC.startCadence, randomSeeds[0], randomSeeds[1], cadence_cffi, mask_cffi, t_cffi, x_cffi)
			yORn = C._makeObservedLC(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.LC.intrinsicVar, self.LC.noiseLvl, self.numBurn, self.LC.numCadences, self.LC.startCadence, randomSeeds[0], randomSeeds[1], randomSeeds[2], cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
			self.LC.LnLike = C._computeLnlike(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.LC.numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
			for i in xrange(self.LC.numCadences):
				self.LC.cadence[i] = cadence_cffi[i]
				self.LC.mask[i] = mask_cffi[i]
				self.LC.t[i] = t_cffi[i]
				self.LC.x[i] = x_cffi[i]
				self.LC.y[i] = y_cffi[i]
				self.LC.yerr[i] = yerr_cffi[i]
			self.LCFile = self.WorkingDirectory + self.prefix + '.lc'
		else:
			self._readLC()

	def _make_02_write(self):
		if self.DateTime == None:
			logEntry = 'Writing LC'
			self.echo(logEntry)
			self.log(logEntry)
			self.LCFile = self.WorkingDirectory + self.prefix + ".lc"
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
			for i in xrange(self.LC.numCadences):
				line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(self.LC.cadence[i], self.LC.mask[i], self.LC.t[i], self.LC.x[i], self.LC.y[i], self.LC.yerr[i])
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
			notMissing = np.where(self.LC.mask[:] == 1.0)[0]
			if self.doNoiseless == True:
				ax1.plot(self.LC.t[notMissing[:]], self.LC.x[notMissing[:]], color = '#7570b3', zorder = 5, label = r'Intrinsic LC')
			ax1.errorbar(self.LC.t[notMissing[:]], self.LC.y[notMissing[:]], self.LC.yerr[notMissing[:]], fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, Label = r'Observed LC')
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
				notMissingDetail = np.where(self.LC.mask[self.detailStart:self.detailStart+self.numPtsDetail] == 1.0)[0]
				if self.doNoiseless == True:
					ax2.plot(self.LC.t[notMissingDetail[:]], self.LC.x[notMissingDetail[:]], color = '#7570b3', zorder = 15)
				ax2.errorbar(self.LC.t[notMissingDetail[:]], self.LC.y[notMissingDetail[:]], self.LC.yerr[notMissingDetail[:]], fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
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