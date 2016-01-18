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
import operator as operator
import psutil
import multiprocessing
import pdb as pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab

from bin._libcarma import ffi
import python.util.triangle as triangle
from python.task import SuppliedLCTask
from python.task import SuppliedParametersTask
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXSmall']
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

class computeLnLikeTask(SuppliedLCTask,SuppliedParametersTask):

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

	def _make_01_computeLnlike(self):
		"""	Attempts to compute the log likelihood of the LC
		"""
		logEntry = 'Computing LnLike of LC %s'%(self.SuppliedLCFile)
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
		if not self.args:
			Theta_cffi = ffiObj.new('double[%d]'%(self.p + self.q + 1))
			for i in xrange(self.p):
				Theta_cffi[i] = self.ARCoefs[i]
			for i in xrange(self.q + 1):
				Theta_cffi[self.p + i] = self.MACoefs[i]
		else:
			self.p = int(self.kwargs['p'])
			self.q = int(self.kwargs['q'])
			Theta_cffi = ffiObj.new('double[%d]'%(self.p + self.q + 1))
			for i in xrange(self.p):
				Theta_cffi[i] = self.args[i]
			for i in xrange(self.q + 1):
				Theta_cffi[self.p + i] = self.args[self.p + i]
		randomSeeds = ffiObj.new('unsigned int[3]')
		yORn = self.rdrand(3, randomSeeds)
		if self.LC.IR == True:
			IR = 1
		else:
			IR = 0
		self.LC.LnLike = C._computeLnlike(self.LC.dt, self.p, self.q, Theta_cffi, IR, self.LC.tolIR, self.LC.numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)
		logEntry = 'For Theta = '
		if not self.args:
			for i in xrange(self.p):
				logEntry += '%+17.16e '%(float(self.ARCoefs[i]))
			for i in xrange(self.q + 1):
				logEntry += '%+17.16e '%(float(self.MACoefs[i]))
		else:
			for i in xrange(self.p):
				logEntry += '%+17.16e '%(float(self.args[i]))
			for i in xrange(self.q + 1):
				logEntry += '%+17.16e '%(float(self.args[self.p + i]))
		logEntry += 'LnLike = %+17.16e'%(float(self.LC.LnLike))
		self.echo(logEntry)
		self.log(logEntry)