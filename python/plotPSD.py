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
import pdb as pdb

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

class plotPSDTask(SuppliedParametersTask):
	"""	Attempts to plot the PSD of the given C-ARMA model
	"""

	@staticmethod
	def getPSDDenominator(freqs, aList, order):
		pVal = len(aList)
		numFreqs = freqs.shape[0]
		aList.insert(0, 1.0)
		PSDVals = np.zeros(numFreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*pVal)):
			aList.pop(0)
			return PSDVals
		else:
			for freq in xrange(freqs.shape[0]):
				val = 0.0
				for i in xrange(pVal + 1):
					j = 2*pVal - i - order
					if ((j >= 0) and (j < pVal + 1)):
						val += (aList[i]*aList[j]*((2.0*math.pi*1j*freqs[freq])**(2*pVal - (i + j)))*pow(-1.0, pVal - j)).real
					PSDVals[freq] = val
			aList.pop(0)
			return PSDVals

	@staticmethod
	def getPSDNumerator(freqs, bList, order):
		qVal = len(bList) - 1
		numFreqs = freqs.shape[0]
		PSDVals = np.zeros(numFreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*qVal)):
			return PSDVals
		else:
			for freq in xrange(freqs.shape[0]):
				val = 0.0
				for i in xrange(qVal + 1):
					j = 2*qVal - i - order
					if ((j >= 0) and (j < qVal + 1)):
						val += (bList[i]*bList[j]*((2.0*math.pi*1j*freqs[freq])**(2*qVal - (i + j)))*pow(-1.0, qVal - j)).real
					PSDVals[freq] = val
			return PSDVals

	def _read_PSDPlotOptions(self):
		"""	Attempts to read PSD plot options
		"""
		try:
			self.fFracNumerX = float(self.plotParser.get('PLOT', 'fFracNumerX'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fFracNumerX = 0.1
			print str(Err) + '. Using default fFracNumerX = %d'%(self.fDivNumerX)
		try:
			self.fFracNumerY = float(self.plotParser.get('PLOT', 'fFracNumerY'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fFracNumerY = 0.1
			print str(Err) + '. Using default fFracNumerY = %d'%(self.fDivNumerY)
		try:
			self.fFracDenomX = float(self.plotParser.get('PLOT', 'fFracDenomX'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fFracDenomX = 0.9
			print str(Err) + '. Using default fFracDenomX = %d'%(self.fFracDenomX)
		try:
			self.fFracDenomY = float(self.plotParser.get('PLOT', 'fFracDenomY'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fFracDenomY = 0.9
			print str(Err) + '. Using default fFracDenomY = %d'%(self.fFracDenomY)
		try:
			self.xTextPSDNumer = float(self.plotParser.get('PLOT', 'xTextPSDNumer'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xTextPSDNumer = 2.5
			print str(Err) + '. Using default xTextPSDNumer = %3.2f'%(self.xTextPSDNumer)
		try:
			self.yTextPSDNumer = float(self.plotParser.get('PLOT', 'yTextPSDNumer'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.yTextPSDNumer = 1.0e-2
			print str(Err) + '. Using default yTextPSDNumer = %3.2f'%(self.yTextPSDNumer)
		try:
			self.xTextPSDDenom = float(self.plotParser.get('PLOT', 'xTextPSDDenom'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xTextPSDDenom = 2.5
			print str(Err) + '. Using default xTextPSDDenom = %3.2f'%(self.xTextPSDDenom)
		try:
			self.yTextPSDDenom = float(self.plotParser.get('PLOT', 'yTextPSDDenom'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.yTextPSDDenom = 1.0e+2
			print str(Err) + '. Using default yTextPSDDenom = %3.2f'%(self.yTextPSDDenom)
		try:
			self.showEqnPSD = self.strToBool(self.plotParser.get('PLOT', 'showEqnPSD'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showEqnPSD = True
		try:
			self.EqnPSDLocY = float(self.plotParser.get('PLOT', 'EqnPSDLocY'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EqnPSDLocY = 0.9
		try:
			self.EqnPSDFontsize = int(self.plotParser.get('PLOT', 'EqnPSDFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EqnPSDFontsize = 16
		try:
			self.showLegendPSD = self.strToBool(self.plotParser.get('PLOT', 'showLegendPSD'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showLegendPSD = True
		try:
			self.LegendPSDLoc = int(self.plotParser.get('PLOT', 'LegendPSDLoc'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LegendPSDLoc = 6
		try:
			self.LegendPSDFontsize = int(self.plotParser.get('PLOT', 'LegendPSDFontsize'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LegendPSDFontsize = 16
		try:
			self.freqLine = self.plotParser.get('PSDLINESTYLES', 'freqLine')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.freqLine = r'dotted'
			print str(Err) + '. Using default freqLine = %s'%(self.freqLine)
		try:
			self.freqLineWidth = int(self.plotParser.get('PSDLINESTYLES', 'freqLineWidth'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.freqLineWidth = 2
			print str(Err) + '. Using default freqLineWidth = %d'%(self.freqLineWidth)
		try:
			self.numeratorLine = self.plotParser.get('PSDLINESTYLES', 'numeratorLine')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numeratorLine = r'dashed'
			print str(Err) + '. Using default numeratorLine = %s'%(self.numeratorLine)
		try:
			self.numeratorLineWidth = int(self.plotParser.get('PSDLINESTYLES', 'numeratorLineWidth'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numeratorLineWidth = 4
			print str(Err) + '. Using default numeratorLineWidth = %d'%(self.numeratorLineWidth)
		try:
			self.denominatorLine = self.plotParser.get('PSDLINESTYLES', 'denominatorLine')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.denominatorLine = r'dashed'
			print str(Err) + '. Using default denominatorLine = %s'%(self.denominatorLine)
		try:
			self.denominatorLineWidth = int(self.plotParser.get('PSDLINESTYLES', 'denominatorLineWidth'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.denominatorLineWidth = 4
			print str(Err) + '. Using default denominatorLineWidth = %d'%(self.denominatorLineWidth)
		try:
			self.PSDLine = self.plotParser.get('PSDLINESTYLES', 'PSDLine')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PSDLine = r'solid'
			print str(Err) + '. Using default PSDLine = %s'%(self.PSDLine)
		try:
			self.PSDLineWidth = int(self.plotParser.get('PSDLINESTYLES', 'PSDLineWidth'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PSDLineWidth = 6
			print str(Err) + '. Using default PSDLineWidth = %s'%(self.PSDLineWidth)

		try:
			self.numColors = int(self.plotParser.get('PSDCOLORS', 'numColors'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numColors = 23
			print str(Err) + '. Using default numColors = %d'%(self.numColors)
		try:
			self.PSDColor = self.plotParser.get('PSDCOLORS', 'PSDColor')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PSDColor = r'#000000'
			print str(Err) + '. Using default PSDColor = %s'%(self.PSDColor)
		try:
			self.numeratorColor = self.plotParser.get('PSDCOLORS', 'numeratorColor')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numeratorColor = r'#0000ff'
			print str(Err) + '. Using default numeratorColor = %s'%(self.numeratorColor)
		try:
			self.denominatorColor = self.plotParser.get('PSDCOLORS', 'denominatorColor')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.denominatorColor = r'#ff0000'
			print str(Err) + '. Using default denominatorColor = %s'%(self.denominatorColor)
		try:
			self.x1LabelPSD = self.plotParser.get('PLOT', 'x1LabelPSD')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.x1LabelPSD = r'$\nu$~($d^{-1}$)'
			print str(Err) + '. Using default x1LabelPSD = %s'%(self.x1LabelPSD)
		try:
			self.x2LabelPSD = self.plotParser.get('PLOT', 'x2LabelPSD')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.x1LabelPSD = r'$t$~($d$)'
			print str(Err) + '. Using default x2LabelPSD = %s'%(self.x2LabelPSD)
		try:
			self.yLabelPSD = self.plotParser.get('PLOT', 'yLabelPSD')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.ylabelPSD = r'$P$~($W^{2} m^{-4} d$)'
			print str(Err) + '. Using default yLabelPSD = %s'%(self.yLabelPSD)
		self.color = self.numColors*['']
		for i in xrange(self.numColors):
			try:
				self.color[i] = self.plotParser.get('PSDCOLORS', '%d'%(2*i))
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				self.color[i] = None
				print str(Err) + '. No color supplied for freq^%d.'%(2*i)

	def _read_PSDOptions(self):
		"""	Attempts to read PSD options
		"""
		try:
			self.fMin = float(self.parser.get('PSD', 'fMin'))
			self.fMin = math.log10(self.fMin)
			self.tMax = -self.fMin
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			try:
				self.tMax = float(self.parser.get('PSD', 'tMax'))
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				print 'No fMin or tMax supplied!'
				sys.exit(1)
			self.tMax = math.log10(self.tMax)
			self.fMin = -self.tMax

		try:
			self.fMax = float(self.parser.get('PSD', 'fMax'))
			self.fMax = math.log10(self.fMax)
			self.tMin = -self.fMax
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			try:
				self.tMin = float(self.parser.get('PSD', 'tMin'))
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				print 'No fMax or tMin supplied!'
				sys.exit(1)
			self.tMin = math.log10(self.tMin)
			self.fMax = -self.tMin

		try:
			self.fNum = int(self.parser.get('PSD', 'fNum'))
			self.tNum = self.fNum
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			try:
				self.tNum = int(self.parser.get('PSD', 'tNum'))
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				print 'No fNum or tNum supplied!'
				sys.exit(1)
			self.fNum = self.tNum

	def _make_01_computePSD(self):
		"""	Attempts to compute the PSD.
		"""
		logEntry = 'Computing PSD'
		self.echo(logEntry)
		self.log(logEntry)
		self.freqs = np.logspace(self.fMin, self.fMax, self.fNum)
		self.aList = self.ARCoefs.tolist()
		self.bList = self.MACoefs.tolist()
		self.maxDenomOrder = 2*len(self.aList)
		self.maxNumerOrder = 2*(len(self.bList)-1)

		self.numerPSD = np.zeros((self.fNum,(self.maxNumerOrder/2) + 2))
		self.denomPSD = np.zeros((self.fNum,(self.maxDenomOrder/2) + 2))

		self.PSDNumerator = np.zeros((self.fNum))
		self.PSDDenominator = np.zeros((self.fNum))
		self.PSD = np.zeros((self.fNum))

		for orderVal in xrange(0, self.maxDenomOrder + 1, 2):
			self.denomPSD[:,orderVal/2] = self.getPSDDenominator(self.freqs, self.aList, orderVal)

		for orderVal in xrange(0, self.maxNumerOrder + 1, 2):
			self.numerPSD[:,orderVal/2] = self.getPSDNumerator(self.freqs, self.bList, orderVal)

		for freq in xrange(self.fNum):
			for orderVal in xrange(0, self.maxNumerOrder + 1, 2):
				self.PSDNumerator[freq] += self.numerPSD[freq, orderVal/2]
			for orderVal in xrange(0, self.maxDenomOrder + 1, 2):
				self.PSDDenominator[freq] += self.denomPSD[freq, orderVal/2]
			self.PSD[freq] = self.PSDNumerator[freq]/self.PSDDenominator[freq]

	def _make_02_plotPSD(self):
		"""	Attempts to plot the PSD.
		"""
		logEntry = 'Plotting PSD'
		self.echo(logEntry)
		self.log(logEntry)
		fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
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