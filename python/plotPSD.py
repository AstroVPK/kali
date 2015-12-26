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
from python.task import SuppliedParametersTask
from python.carma import *
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

	def _read_PSDOptions(self):
		"""	Attempts to read PSD options
		"""
		try:
			self.tMin = float(self.plotParser.get('PSD', 'tMin'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tMin = 0.01
		self.tMin = math.log10(self.tMin)
		try:
			self.tMax = float(self.plotParser.get('PSD', 'tMax'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tMax = 10000.0
		self.tMax = np.log10(self.tMax)
		try:
			self.tNum = float(self.plotParser.get('PSD', 'tNum'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tNum = 1000

		try:
			self.fMin = float(self.plotParser.get('PSD', 'fMin'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fMin = 1.0e-4
		self.fMin = math.log10(self.fMin)
		try:
			self.fMax = float(self.plotParser.get('PSD', 'fMax'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fMax = 100.0
		self.fMax = np.log10(self.fMax)
		try:
			self.fNum = float(self.plotParser.get('PSD', 'fNum'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.fNum = 1000

		self.freqs = np.logspace(self.fMin, self.fMax, self.fNum)

		maxDenomOrder = 2*len(self.ARCoefs)
		maxNumerOrder = 2*(len(self.MACoefs)-1)

		numerPSD = np.zeros((self.fNum,(maxNumerOrder/2) + 2))
		denomPSD = np.zeros((self.fNum,(maxDenomOrder/2) + 2))

		PSD = np.zeros((self.fNum))

		plt.figure(1)

		colormapping = {'0': '#a6cee3', '2': '#b2df8a', '4': '#fb9a99', '6': '#fdbf6f', '8': '#cab2d6', '10': '#ffff99'}

		bList = self.MACoefs.tolist()
		for orderVal in xrange(0, maxNumerOrder + 1, 2):
			numerPSD[:,orderVal/2] = self.getPSDNumerator(self.freqs, bList, orderVal)
			plt.loglog(self.freqs,numerPSD[:,orderVal/2],linestyle='--',color=colormapping[str(orderVal)],linewidth=2)

		aList = self.ARCoefs.tolist()
		for orderVal in xrange(0, maxDenomOrder + 1, 2):
			denomPSD[:,orderVal/2] = self.getPSDDenominator(self.freqs, aList, orderVal)
			plt.loglog(self.freqs,1.0/denomPSD[:,orderVal/2],linestyle='--',color=colormapping[str(orderVal)],linewidth=2)

		for freq in xrange(self.fNum):
			for orderVal in xrange(0, maxNumerOrder + 1, 2):
				numerPSD[freq, (maxNumerOrder/2) + 1] += numerPSD[freq, orderVal/2]
			for orderVal in xrange(0, maxDenomOrder + 1, 2):
				denomPSD[freq, (maxDenomOrder/2) + 1] += denomPSD[freq, orderVal/2]
			PSD[freq] = numerPSD[freq, (maxNumerOrder/2) + 1]/denomPSD[freq, (maxDenomOrder/2) + 1]

		plt.loglog(self.freqs, 1.0/denomPSD[:,(maxDenomOrder/2) + 1],linestyle=':',color='#e31a1c',linewidth=4)
		plt.loglog(self.freqs, numerPSD[:,(maxNumerOrder/2) + 1],linestyle=':',color='#1f78b4',linewidth=4)
		plt.loglog(self.freqs, PSD, linestyle='-',color='#000000',linewidth=4)

		'''if self.JPG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.jpg" , dpi = plot_params['dpi'])
		if self.PDF == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.pdf" , dpi = plot_params['dpi'])
		if self.EPS == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.eps" , dpi = plot_params['dpi'])
		if self.PNG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_PSD.png" , dpi = plot_params['dpi'])
		if self.showFig == True:'''
		plt.show()