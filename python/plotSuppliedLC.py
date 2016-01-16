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

class plotSuppliedLCTask(SuppliedParametersTask, SuppliedLCTask):
	"""	Read in and plot the supplied LC. 
	"""

	def _read_00_LCProps(self):
		"""	Attempts to read in the configuration parameters `dt', `T' or `numCadences', & `tStart'.
		"""
		try:
			self.LC.SuppliedLC = self.parser.get('LC', 'suppliedLC')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			print str(Err) + '. Exiting...'
			sys.exit(1)
		self._readLC(suppliedLC = self.LC.SuppliedLC)

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

	def _make_01_plot(self):
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
			ax1.errorbar(self.LC.t[notMissing[:]], self.LC.y[notMissing[:]], self.LC.yerr[notMissing[:]], fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC')
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
				notMissingDetail = np.where(self.LC.mask[self.detailStart:self.detailStart+self.numPtsDetail] == 1.0)[0] + self.detailStart
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