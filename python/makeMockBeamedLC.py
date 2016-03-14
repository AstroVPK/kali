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
from python.binarySMBH import binarySMBH
from python.task import SuppliedParametersTask, SuppliedLCTask
from python.makeMockLC import makeMockLCTask
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

class makeMockBeamedLCTask(makeMockLCTask):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""
	def _read_01_BeamingPlotOptions(self):
		try:
			self.showBinaryDetail = self.strToBool(self.plotParser.get('PLOT', 'showBinaryDetail'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showBinaryDetail = True

	def _read_02_BeamingProps(self):
		"""	Attempts to read in the beaming configuration parameters.
		"""
		try:
			self.rPer = float(self.parser.get('BINARY', 'rPer'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.rPer = 1.0e-2
			print str(Err) + '. Using default rPer = %+3.2e parsec'%(self.rPer)
		try:
			self.m12 = float(self.parser.get('BINARY', 'm12'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.m12 = 1.0e7
			print str(Err) + '. Using default m12 = %+3.2e Solar Masses'%(self.m12)
		try:
			self.m2Om1 = float(self.parser.get('BINARY', 'm2/m1'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.m2Om1 = 1.0
			print str(Err) + '. Using default m2/m1 = %+3.2e'%(self.m2Om1)
		try:
			self.e = float(self.parser.get('BINARY', 'e'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.e = 0.0
			print str(Err) + '. Using default e = %+3.2e'%(self.e)
		try:
			self.omega = float(self.parser.get('BINARY', 'omega'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.omega = 0.0
			print str(Err) + '. Using default omega = %+3.2e'%(self.omega)
		try:
			self.i = float(self.parser.get('BINARY', 'i'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.i = 90.0
			print str(Err) + '. Using default i = %+3.2e'%(self.i)
		try:
			self.tau = float(self.parser.get('BINARY', 'tau'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.tau = 0.0
			print str(Err) + '. Using default tau = %+3.2e day'%(self.tau)
		try:
			self.alpha1 = float(self.parser.get('BINARY', 'alpha1'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.alpha1 = -0.44
			print str(Err) + '. Using default alpha1 = %+3.2e'%(self.alpha1)
		try:
			self.alpha2 = float(self.parser.get('BINARY', 'alpha2'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.alpha2 = -0.44
			print str(Err) + '. Using default alpha2 = %+3.2e'%(self.alpha2)

	def _make_01_beamedLC(self):
		"""	Attempts to make the LC
		"""
		logEntry = 'Beaming LC'
		self.echo(logEntry)
		self.log(logEntry)
		self.binary = binarySMBH(rPer = self.rPer, m12 = self.m12, q = self.m2Om1, e = self.e, omega = self.omega, i = self.i, tau = self.tau, alpha1 = self.alpha1, alpha2 = self.alpha2)
		self.LC.Period = self.binary.T
		Theta_cffi = ffiObj.new('double[%d]'%(self.p + self.q + 1))
		for i in xrange(self.p):
			Theta_cffi[i] = self.ARCoefs[i]
		for i in xrange(self.q + 1):
			Theta_cffi[self.p + i] = self.MACoefs[i]
		self.LC.meanFlux = C._getMeanFlux(self.p, self.q, Theta_cffi, self.LC.intrinsicVar)
		self.LC.bx = np.array(self.LC.numCadences*[0.0])
		startT = time.time()
		self.LC.beamFac1 = np.array(self.LC.numCadences*[0.0])
		self.LC.beamFac2 = np.array(self.LC.numCadences*[0.0])
		for i in xrange(self.LC.numCadences):
			self.LC.beamFac1[i], self.LC.beamFac2[i] = self.binary.beamingFactor(self.LC.t[i]*self.binary.Day)
			self.LC.bx[i] = self.LC.beamFac2[i]*(self.LC.x[i] + self.LC.meanFlux)
		stopT = time.time()
		diffT = stopT - startT
		logEntry = r'Beaming took %f sec = %f min = %f hrs'%(diffT, diffT/60.0, diffT/3600.0)
		self.echo(logEntry)
		self.log(logEntry)
		logEntry = 'Computing FFTs'
		self.echo(logEntry)
		self.log(logEntry)
		self.LC.fFFT = np.fft.fftfreq(self.LC.numCadences, self.LC.dt)
		self.LC.xFFT = np.fft.fft(self.LC.x + self.LC.meanFlux, norm = 'ortho')
		self.LC.xbFFT = np.fft.fft(self.LC.bx, norm = 'ortho')
		self.LC.xPSD = self.LC.xFFT*np.conj(self.LC.xFFT)
		self.LC.xbPSD = self.LC.xbFFT*np.conj(self.LC.xbFFT)

	def _make_01_plot(self):
		if self.makePlot == True:
			logEntry = 'Plotting beamed LC'
			self.echo(logEntry)
			self.log(logEntry)

			fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
			ax1 = fig1.add_subplot(gs[:,:])
			ax1.ticklabel_format(useOffset = False)
			notMissing = np.where(self.LC.mask[:] == 1.0)[0]
			ax1.plot(self.LC.t[notMissing[:]], self.LC.x[notMissing[:]] + self.LC.meanFlux, color = '#7570b3', zorder = 5, label = r'Intrinsic LC')
			ax1.plot(self.LC.t[notMissing[:]], self.LC.bx[notMissing[:]], color = '#d95f02', zorder = 10, label = r'Beamed Intrinsic LC')
			if self.showLegendLC == True:
				ax1.legend(loc = self.LegendLCLoc, ncol = 1, fancybox = True, fontsize = self.LegendLCFontsize)
			yMax=max(np.max(self.LC.bx[np.nonzero(self.LC.bx[:])]), np.max(self.LC.meanFlux + self.LC.x[np.nonzero(self.LC.x[:])]))
			yMin=min(np.min(self.LC.bx[np.nonzero(self.LC.bx[:])]), np.min(self.LC.meanFlux + self.LC.x[np.nonzero(self.LC.x[:])]))
			ax1.set_xlabel(self.xLabelLC)
			ax1.set_ylabel(self.yLabelLC)
			ax1.set_xlim(self.LC.t[0],self.LC.t[-1])
			ax1.set_ylim(yMin,yMax)
			if self.showEqnLC == True:
				ax1.annotate(self.eqnStr, xy = (0.5, 0.1), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)
			if self.showLnLike == True:
				ax1.annotate(r'$\ln \mathcal{L} = ' + self.formatFloat(self.LC.LnLike) + '$', xy = (0.5, 0.05), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)
			if self.showBinaryDetail == True:
				ax1.annotate(r'$\mathrm{M}_{12} = ' + self.formatFloat(self.m12) + '$; $r_{\mathrm{periapsis}} = ' + self.formatFloat(self.rPer) + '$; $T = ' + self.formatFloat(self.LC.Period/self.binary.Year) + '$', xy = (0.5, 0.20), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)
				ax1.annotate(r'$q = ' + self.formatFloat(self.m2Om1) + '$; $e = ' + self.formatFloat(self.e) + '$; $\omega = ' + self.formatFloat(self.omega) + '$; $i = ' + self.formatFloat(self.i) + '$', xy = (0.5, 0.15), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)

			if self.JPG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamedLC.jpg" , dpi = self.dpi)
			if self.PDF == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamedLC.pdf" , dpi = self.dpi)
			if self.EPS == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamedLC.eps" , dpi = self.dpi)
			if self.PNG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamedLC.png" , dpi = self.dpi)
			if self.showFig == True:
				plt.show()
			fig1.clf()

			fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
			ax1 = fig1.add_subplot(gs[:,:])
			ax1.ticklabel_format(useOffset = False)
			ax1.plot(self.LC.t, self.LC.beamFac1, color = '#7570b3', zorder = 5, label = r'$D^{3-\alpha}_{m_{1}}$')
			ax1.plot(self.LC.t, self.LC.beamFac2, color = '#d95f02', zorder = 10, label = r'$D^{3-\alpha}_{m_{2}}$')
			if self.showLegendLC == True:
				ax1.legend(loc = self.LegendLCLoc, ncol = 1, fancybox = True, fontsize = self.LegendLCFontsize)
			yMax=max(np.max(self.LC.beamFac1), np.max(self.LC.beamFac2))
			yMin=min(np.min(self.LC.beamFac1), np.min(self.LC.beamFac2))
			ax1.set_xlabel(self.xLabelLC)
			ax1.set_ylabel(r'$D^{3-\alpha}$')
			ax1.set_xlim(self.LC.t[0],self.LC.t[-1])
			ax1.set_ylim(yMin,yMax)
			if self.showBinaryDetail == True:
				ax1.annotate(r'$\mathrm{M}_{12} = ' + self.formatFloat(self.m12) + '$; $r_{\mathrm{periapsis}} = ' + self.formatFloat(self.rPer) + '$; $T = ' + self.formatFloat(self.LC.Period/self.binary.Year) + '$', xy = (0.5, 0.25), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)
				ax1.annotate(r'$q = ' + self.formatFloat(self.m2Om1) + '$; $e = ' + self.formatFloat(self.e) + '$; $\omega = ' + self.formatFloat(self.omega) + '$; $i = ' + self.formatFloat(self.i) + '$', xy = (0.5, 0.15), xycoords = 'axes fraction', textcoords = 'axes fraction', ha = 'center', va = 'center' ,multialignment = 'center', fontsize = self.EqnLCFontsize, zorder = 100)

			if self.JPG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingFactor.jpg" , dpi = self.dpi)
			if self.PDF == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingFactor.pdf" , dpi = self.dpi)
			if self.EPS == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingFactor.eps" , dpi = self.dpi)
			if self.PNG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingFactor.png" , dpi = self.dpi)
			if self.showFig == True:
				plt.show()
			fig1.clf()

			fig1 = plt.figure(10, figsize = (plot_params['fwid'], plot_params['fhgt']))
			ax1 = fig1.add_subplot(gs[:,0:399])
			ax1.ticklabel_format(useOffset = False)
			ax1.plot(np.log10(self.LC.fFFT[1:self.LC.fFFT.shape[0]/2]), np.log10(self.LC.xPSD[1:self.LC.fFFT.shape[0]/2].real), color = '#7570b3', zorder = 5, label = r'$PSD_{x}$')
			if self.showLegendLC == True:
				ax1.legend(loc = self.LegendLCLoc, ncol = 1, fancybox = True, fontsize = self.LegendLCFontsize)
			yMax = np.max(np.log10(self.LC.xPSD[1:self.LC.fFFT.shape[0]/2]))
			yMin = np.min(np.log10(self.LC.xPSD[1:self.LC.fFFT.shape[0]/2]))
			ax1.set_xlabel(r'$\log_{10} f$')
			ax1.set_ylabel(r'$\log_{10} PSD$')
			ax1.set_xlim(np.log10(self.LC.fFFT[1]), np.log10(self.LC.fFFT[(self.LC.fFFT.shape[0]/2) -1 ]))
			ax1.set_ylim(yMin,yMax)
			ax2 = fig1.add_subplot(gs[:,600:999])
			ax2.ticklabel_format(useOffset = False)
			ax2.plot(np.log10(self.LC.fFFT[1:self.LC.fFFT.shape[0]/2]), np.log10(self.LC.xbPSD[1:self.LC.fFFT.shape[0]/2].real), color = '#d95f02', zorder = 5, label = r'$PSD_{xb}$')
			if self.showLegendLC == True:
				ax2.legend(loc = self.LegendLCLoc, ncol = 1, fancybox = True, fontsize = self.LegendLCFontsize)
			yMax = np.max(np.log10(self.LC.xbPSD[1:self.LC.fFFT.shape[0]/2]))
			yMin = np.min(np.log10(self.LC.xbPSD[1:self.LC.fFFT.shape[0]/2]))
			ax2.set_xlabel(r'$\log_{10} f$')
			ax2.set_ylabel(r'$\log_{10} PSD$')
			ax2.set_xlim(np.log10(self.LC.fFFT[1]), np.log10(self.LC.fFFT[(self.LC.fFFT.shape[0]/2) -1 ]))
			ax2.set_ylim(yMin,yMax)

			if self.JPG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingPSD.jpg" , dpi = self.dpi)
			if self.PDF == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingPSD.pdf" , dpi = self.dpi)
			if self.EPS == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingPSD.eps" , dpi = self.dpi)
			if self.PNG == True:
				fig1.savefig(self.WorkingDirectory + self.prefix + "_beamingPSD.png" , dpi = self.dpi)
			if self.showFig == True:
				plt.show()
			fig1.clf()