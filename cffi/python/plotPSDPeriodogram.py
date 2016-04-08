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
import socket
HOST = socket.gethostname()
print 'HOST: %s'%(str(HOST))
import os as os
try: 
	os.environ['DISPLAY']
except KeyError as Err:
	print "No display environment! Using matplotlib backend 'Agg'"
	import matplotlib
	matplotlib.use('Agg')
import sys as sys
import time as time
import gatspy.periodic as periodic
import scipy.signal as signal
import pdb as pdb

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab

from bin._libcarma import ffi
from python.task import SuppliedLCTask
from python.plotPSD import plotPSDTask
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXXXSmall']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')
gs = gridspec.GridSpec(1000, 1000)

ffiObj = cffi.FFI()
try:
	libcarmaPath = str(os.environ['LIBCARMA'])
except KeyError as Err:
	print str(Err) + '. Exiting....'
	sys.exit(1)
C = ffi.dlopen(libcarmaPath + '/bin/libcarma.so.1.0.0')
new_uint = ffiObj.new_allocator(alloc = C._malloc_uint, free = C._free_uint)
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class plotPSDPeriodogramTask(SuppliedLCTask,plotPSDTask):
	"""	Attempts to plot the PSD of the given C-ARMA model
	"""

	def _read_00_LnlikeProps(self):
		"""	Attempts to read in the configuration parameters `dt', `T' or `numCadences', & `tStart'.
		"""
		try:
			self.LC.tolIR = float(self.parser.get('LC', 'tolIR'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.tolIR = 1.0e-3
			print str(Err) + '. Using default tolIR = %+4.3e'%(self.LC.tolIR)
		try:
			self.LC.SuppliedLC = self.parser.get('LC', 'suppliedLC')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			print str(Err) + '. Exiting...'
			sys.exit(1)
		self._readLC(suppliedLC = self.LC.SuppliedLC)

	def _make_02_plotPSD(self):
		"""	Attempts to plot the PSD.
		"""
		fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))

		logEntry = 'Plotting periodogram of LC %s'%(self.SuppliedLCFile)
		self.echo(logEntry)
		self.log(logEntry)
		model = periodic.LombScargleFast().fit(self.LC.t, self.LC.y, self.LC.yerr)
		times, periodogram = model.periodogram_auto(nyquist_factor = 1)
		frequencies = 1.0/times
		for i in xrange(frequencies.shape[0]): 
			if frequencies[i] > self.freqs[0]: 
				start = i
				break

		periodogramScipy = signal.lombscargle(self.LC.t, self.LC.y, frequencies[start:])

		numerPSD = np.zeros((frequencies[start:].shape[0],(self.maxNumerOrder/2) + 2))
		denomPSD = np.zeros((frequencies[start:].shape[0],(self.maxDenomOrder/2) + 2))
		PSDNumerator = np.zeros((frequencies[start:].shape[0]))
		PSDDenominator = np.zeros((frequencies[start:].shape[0]))
		PSD = np.zeros((frequencies[start:].shape[0]))
		Ratio = np.zeros((frequencies[start:].shape[0]))
		RatioScipy = np.zeros((frequencies[start:].shape[0]))
		for orderVal in xrange(0, self.maxDenomOrder + 1, 2):
			denomPSD[:,orderVal/2] = self.getPSDDenominator(frequencies[start:], self.aList, orderVal)
		for orderVal in xrange(0, self.maxNumerOrder + 1, 2):
			numerPSD[:,orderVal/2] = self.getPSDNumerator(frequencies[start:], self.bList, orderVal)
		for freq in xrange(frequencies[start:].shape[0]):
			for orderVal in xrange(0, self.maxNumerOrder + 1, 2):
				PSDNumerator[freq] += numerPSD[freq, orderVal/2]
			for orderVal in xrange(0, self.maxDenomOrder + 1, 2):
				PSDDenominator[freq] += denomPSD[freq, orderVal/2]
			PSD[freq] = PSDNumerator[freq]/PSDDenominator[freq]
			Ratio[freq] = periodogram[start + freq]/PSD[freq]
			RatioScipy[freq] = periodogramScipy[freq]/PSD[freq]

		medRatio = np.median(Ratio)
		medRatioScipy = np.median(RatioScipy)
		plt.loglog(frequencies[start:], periodogram[start:]/Ratio[0], label = 'Gatspy periodogram')
		plt.loglog(frequencies[start:], periodogramScipy/medRatioScipy, label = 'Scipy periodogram')

		logEntry = 'Plotting PSD'
		self.echo(logEntry)
		self.log(logEntry)
		for orderVal in xrange(0, self.maxNumerOrder + 1, 2):
			plt.loglog(self.freqs, self.numerPSD[:,orderVal/2], linestyle = self.freqLine, color = self.color[orderVal/2], linewidth = self.freqLineWidth, zorder = 5)
			plt.annotate(r'$\nu^{%d}$'%(orderVal), xy = (self.freqs[int(self.fNum*self.fFracNumerX)], self.numerPSD[int(self.fNum*self.fFracNumerY),orderVal/2]), xycoords = 'data', xytext = (self.xTextPSDNumer*self.freqs[int(self.fNum*self.fFracNumerX)], self.yTextPSDNumer*self.numerPSD[int(self.fNum*self.fFracNumerY),orderVal/2]), textcoords = 'data', arrowprops = dict(arrowstyle = '->', connectionstyle = 'angle3, angleA = 0, angleB = 90'), ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnPSDFontsize, zorder = 100)
		for orderVal in xrange(0, self.maxDenomOrder + 1, 2):
			plt.loglog(self.freqs, 1.0/self.denomPSD[:,orderVal/2], linestyle = self.freqLine, color = self.color[orderVal/2], linewidth = self.freqLineWidth, zorder = 5)
			plt.annotate(r'$\nu^{-%d}$'%(orderVal), xy = (self.freqs[int(self.fNum*self.fFracDenomX)], 1.0/self.denomPSD[int(self.fNum*self.fFracDenomY),orderVal/2]), xycoords = 'data', xytext = (self.xTextPSDDenom*self.freqs[int(self.fNum*self.fFracDenomX)], self.yTextPSDDenom/self.denomPSD[int(self.fNum*self.fFracDenomY),orderVal/2]), textcoords = 'data', arrowprops = dict(arrowstyle = '->', connectionstyle = 'angle3, angleA = 0, angleB = 90'), ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnPSDFontsize, zorder = 100)
		plt.loglog(self.freqs, 1.0/self.PSDDenominator, linestyle = self.denominatorLine, color = self.denominatorColor, linewidth = self.denominatorLineWidth, zorder =10, label = r'$-\log_{10}D(\nu)$')
		plt.loglog(self.freqs, self.PSDNumerator, linestyle = self.numeratorLine, color = self.numeratorColor, linewidth = self.numeratorLineWidth, zorder =10, label = r'$log_{10}N(\nu)$')
		plt.loglog(self.freqs, self.PSD, linestyle = self.PSDLine, color = self.PSDColor, linewidth = self.PSDLineWidth, zorder = 15, label = r'$\log_{10}PSD(\nu) = \log_{10}N(\nu)-\log_{10}D(\nu)$')
		if self.showEqnPSD == True:
			plt.annotate(self.eqnStr, xy = (0.5, 0.9), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnPSDFontsize, zorder = 100)
		if self.showLegendPSD == True:
			plt.legend(loc = self.LegendPSDLoc, ncol = 1, fancybox = True, fontsize = self.LegendPSDFontsize)

		plt.xlabel(self.x1LabelPSD)
		plt.ylabel(self.yLabelPSD)

		if self.JPG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.jpg" , dpi = self.dpi)
		if self.PDF == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.pdf" , dpi = self.dpi)
		if self.EPS == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.eps" , dpi = self.dpi)
		if self.PNG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.png" , dpi = self.dpi)
		if self.showFig == True:
			plt.show()
		fig1.clf()