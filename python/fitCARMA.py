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
import operator as operator
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

class fitCARMATask(SuppliedLCTask):
	"""	Fit a C-ARMA model to the supplied LC.
	"""

	def MAD(self, a):
		medianVal = np.median(a)
		b = np.copy(a)
		for i in range(a.shape[0]):
			b[i]=abs(b[i] - medianVal)
		return np.median(b)

	def _checkPQ(self):
		"""	Check to see if supplied pMax etc... are valid.
		"""
		if self.pMax < self.pMin:
			a = self.pMin
			self.pMin = self.pMax
			self.pMax = self.a
		try:
			qMaxInt = int(self.qMax)
			if qMaxInt < self.qMin:
				a = self.qMin
				self.qMin = self.qMax
				self.qMax = self.a
			if qMaxInt >= self.pMax:
				print 'Invalid pMax and qMax!'
				sys.exit(1)
		except ValueError as Err:
			pass

	def _writeMCMC(self):
		"""	Write the Chain and LnLike out
		"""
		chainFile = open(self.WorkingDirectory + self.prefix + '_%d'%(self.p) + '_%d'%(self.q) + '.mcmc', 'w')
		line = "nsteps: %d\n"%(self.nsteps)
		chainFile.write(line)
		line = "nwalkers: %d\n"%(self.nwalkers)
		chainFile.write(line)
		line = "ndim: %d; p: %d; q: %d\n"%(self.ndims, self.p, self.q)
		chainFile.write(line)

		for stepNum in xrange(self.nsteps):
			for walkerNum in xrange(self.nwalkers):
				line = "stepNum: %d; walkerNum: %d; "%(stepNum, walkerNum)
				for pNum in xrange(self.p):
					line += "%+17.16e "%(self.Chain[stepNum,walkerNum,pNum])
				for qNum in xrange(self.q + 1):
					line += "%+17.16e "%(self.Chain[stepNum,walkerNum,qNum])
				line += "%+17.16e\n"%(self.LnLike[stepNum,walkerNum])
			chainFile.write(line)
		chainFile.close()

	def _plotMCMC(self):
		"""	Plot the MCMC Chain.
		"""
		medianWalker = np.zeros((self.nsteps, self.ndims))
		medianDevWalker = np.zeros((self.nsteps, self.ndims))
		for i in range(self.nsteps):
			for k in range(self.ndims):
				medianWalker[i,k] = np.median(self.Chain[i,:,k])
				medianDevWalker[i,k] = self.MAD(self.Chain[i,:,k])
		stepArr = np.arange(self.nsteps)

		fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
		for k in range(self.ndims):
			plt.subplot(self.ndims, 1 , k + 1)
			for j in range(self.nwalkers):
				plt.plot(self.Chain[:,j,k], c = '#000000', alpha = 0.05, zorder = -5)
			plt.fill_between(stepArr[:], medianWalker[:,k] + medianDevWalker[:,k], medianWalker[:,k] - medianDevWalker[:,k], color = '#ff0000', edgecolor = '#ff0000', alpha = 0.5, zorder = 5)
			plt.plot(stepArr[:], medianWalker[:,k], c = '#dc143c', linewidth = 1, zorder = 10)
			plt.xlabel('stepNum')
			if (0 <= k < self.p):
				plt.ylabel("$a_{%d}$"%(k + 1))
			elif ((k >= self.p) and (k < self.ndims)):
				plt.ylabel("$b_{%d}$"%(k - self.p))
		if self.JPG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_Chain_%d_%d.jpg"%(self.p, self.q) , dpi = self.dpi)
		if self.PDF == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_Chain_%d_%d.pdf"%(self.p, self.q) , dpi = self.dpi)
		if self.EPS == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_Chain_%d_%d.eps"%(self.p, self.q) , dpi = self.dpi)
		if self.PNG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_Chain_%d_%d.png"%(self.p, self.q) , dpi = self.dpi)
		if self.showFig == True:
			plt.show()
		fig1.clf()

		samples = self.Chain[int(self.nsteps/2.0):,:,:].reshape((-1, self.ndims))
		sampleDeviances = self.Deviances[int(self.nsteps/2.0):,:].reshape((-1))
		DIC = 0.5*math.pow(np.std(sampleDeviances),2.0) + np.mean(sampleDeviances)
		self.dictDIC["%d %d"%(self.p, self.q)] = DIC
		lbls = list()
		for i in range(self.p):
			lbls.append("$a_{%d}$"%(i + 1))
		for i in range(self.q + 1):
			lbls.append("$b_{%d}$"%(i))
		fig2, quantiles, qvalues = triangle.corner(samples, labels = lbls, fig_title = "DIC: %f"%(self.dictDIC["%d %d"%(self.p, self.q)]), show_titles = True, title_args = {"fontsize": 12}, quantiles = [0.16, 0.5, 0.84], verbose = False, plot_contours = True, plot_datapoints = True, plot_contour_lines = False, pcolor_cmap = cm.gist_earth)
		if self.JPG == True:
			fig2.savefig(self.WorkingDirectory + self.prefix + "_MCMC_%d_%d.jpg"%(self.p, self.q) , dpi = self.dpi)
		if self.PDF == True:
			fig2.savefig(self.WorkingDirectory + self.prefix + "_MCMC_%d_%d.pdf"%(self.p, self.q) , dpi = self.dpi)
		if self.EPS == True:
			fig2.savefig(self.WorkingDirectory + self.prefix + "_MCMC_%d_%d.eps"%(self.p, self.q) , dpi = self.dpi)
		if self.PNG == True:
			fig2.savefig(self.WorkingDirectory + self.prefix + "_MCMC_%d_%d.png"%(self.p, self.q) , dpi = self.dpi)
		if self.showFig == True:
			plt.show()
		fig2.clf()

		try:
			Result = open(self.ResultFile,'a')
		except IOError as Err:
			print str(Err) + '. Exiting...'
		line1 = "p: %d; q: %d\n"%(self.p, self.q)
		Result.write(line1)
		line2 = "DIC: %e\n"%(self.dictDIC['%d %d'%(self.p, self.q)])
		Result.write(line2)
		for k in range(self.ndims):
			if (0 <= k < self.p):
				line3 = "a_%d\n"%(k)
			elif ((k >= self.p) and (k < self.ndims)):
				line3 = "b_%d\n"%(k - self.p)
			Result.write(line3)
			#self.fiftiethQ["%d %d %s"%(self.p, self.q, line.rstrip("\n"))] = float(qvalues[k][1])
			for i in range(len(quantiles)):
				line4 = "Quantile: %.2f; Value: %e\n"%(quantiles[i], qvalues[k][i])
				Result.write(line4)
		line5 = "\n"
		Result.write(line5)
		Result.close()

	def _read_00_LCProps(self):
		"""	Attempts to read in the configuration parameters `dt', `T' or `numCadences', & `tStart'.
		"""
		try:
			self.LC.tolIR = float(self.parser.get('LC', 'tolIR'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.LC.tolIR = 1.0e-3
			print str(Err) + '. Using default tolIR = %+4.3e'%(self.LC.tolIR)
		try:
			self.scatterFactor = float(self.parser.get('MCMC', 'scatterFactor'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.scatterFactor = 1.0e-6
			print str(Err) + '. Using default scatterFactor = %+4.3e'%(self.scatterFactor)
		try:
			self.pMax = int(self.parser.get('MCMC', 'pMax'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.pMax = 2
			print str(Err) + '. Using default pMax = %d'%(self.pMax)
		try:
			self.pMin = int(self.parser.get('MCMC', 'pMin'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.pMin = 1
			print str(Err) + '. Using default pMin = %d'%(self.pMin)
		try:
			self.qMax = self.parser.get('MCMC', 'qMax')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.qMax = 'pMax - 1'
			print str(Err) + '. Using default qMax = %d'%(self.qMax)
		try:
			self.qMin = int(self.parser.get('MCMC', 'qMin'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.qMin = 0
			print str(Err) + '. Using default qMin = %d'%(self.qMin)
		self._checkPQ()
		try:
			self.nthreads = int(self.parser.get('MCMC', 'nthreads'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.nthreads = (multiprocessing.cpu_count()/int(os.popen('lscpu').readlines()[5].rstrip('\n').split()[3]))
			print str(Err) + '. Using default nthreads = %d'%(self.nthreads)
		if self.nthreads > (multiprocessing.cpu_count()/int(os.popen('lscpu').readlines()[5].rstrip('\n').split()[3])):
			if self.nthreads > multiprocessing.cpu_count():
				print 'Using Intel Hyperthreading...'
			else:
				print 'More threads requested than available hardware threads...'
				print 'Caching may not be optimal.'
		elif self.nthreads < (multiprocessing.cpu_count()/int(os.popen('lscpu').readlines()[5].rstrip('\n').split()[3])):
			print '%d cores/%d cores will be used in computation.'%(self.nthreads,multiprocessing.cpu_count()/int(os.popen('lscpu').readlines()[5].rstrip('\n').split()[3]))
		try:
			self.nwalkers = int(self.parser.get('MCMC', 'nwalkers'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.nwalkers = 4
			print str(Err) + '. Using default nwalkers = %d'%(self.nwalkers)
		try:
			self.nsteps = int(self.parser.get('MCMC', 'nsteps'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.nsteps = 10
			print str(Err) + '. Using default nsteps = %d'%(self.nsteps)
		try:
			self.maxEvals = int(self.parser.get('MCMC', 'maxEvals'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.maxEvals = 1000
			print str(Err) + '. Using default maxEvals = %d'%(self.maxEvals)
		try:
			self.xTol = float(self.parser.get('MCMC', 'xTol'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.xTol = 160
			print str(Err) + '. Using default xTol = %+4.3e'%(self.xTol)
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

	def _doCARMAFit(self):
		logEntry = 'Computing C-ARMA fit for p = %d and q = %d'%(self.p, self.q)
		self.echo(logEntry)
		self.log(logEntry)
		randomSeeds = ffiObj.new('unsigned int[5]')
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
		self.ndims = self.p + self.q + 1
		yORn = self.rdrand(5, randomSeeds)
		Chain_cffi = ffiObj.new('double[%d]'%(self.nsteps*self.nwalkers*self.ndims))
		LnLike_cffi = ffiObj.new('double[%d]'%(self.nsteps*self.nwalkers))
		if self.LC.IR == True:
			IR = 1
		else:
			IR = 0
		C._fitCARMA(self.LC.dt, self.p, self.q, IR, self.LC.tolIR, self.scatterFactor, self.LC.numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi, self.nthreads, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, randomSeeds[0], randomSeeds[1], randomSeeds[2], randomSeeds[3], randomSeeds[4], Chain_cffi, LnLike_cffi)
		self.Chain = np.zeros((self.nsteps, self.nwalkers, self.ndims))
		self.LnLike = np.zeros((self.nsteps, self.nwalkers))
		self.Deviances = np.zeros((self.nsteps, self.nwalkers))
		for stepNum in xrange(self.nsteps):
			for walkerNum in xrange(self.nwalkers):
				self.LnLike[stepNum, walkerNum] = LnLike_cffi[walkerNum + stepNum*self.nwalkers]
				self.Deviances[stepNum,walkerNum] = -2.0*LnLike_cffi[walkerNum + stepNum*self.nwalkers]
				for dimNum in xrange(self.ndims):
					self.Chain[stepNum, walkerNum, dimNum] = Chain_cffi[dimNum + walkerNum*self.ndims + stepNum*self.ndims*self.nwalkers]
		self._writeMCMC()
		self._plotMCMC()

	def _make_00_runMCMC(self):
		self.dictDIC = dict()
		self.ResultFile = self.WorkingDirectory + self.prefix + '_CARMAResult.dat'
		try:
			Result = open(self.ResultFile,'w')
		except IOError as Err:
			print str(Err) + '. Exiting...'
		Result.close()
		if self.qMax.replace(' ','') == 'p-1':
			for p in xrange(self.pMin, self.pMax + 1):
				for q in xrange(self.qMin, p):
					self.p = p
					self.q = q
					self._doCARMAFit();
		else:
			self.qMax = int(self.qMax)
			for p in xrange(self.pMin, self.pMax + 1):
				for q in xrange(self.qMin, self.qMax + 1):
					if q >= p:
						break
					else:
						self.p = p
						self.q = q
						self._doCARMAFit()
		try:
			Result = open(self.ResultFile, 'a')
		except IOError as Err:
			print str(Err) + '. Exiting...'
		self.sortedDICVals = sorted(self.dictDIC.items(), key = operator.itemgetter(1))
		self.pBest = int(self.sortedDICVals[0][0].split()[0])
		self.qBest = int(self.sortedDICVals[0][0].split()[1])
		line = "Model Appropriateness (in descending order of appropriateness) & Relative Likelihood (i.e. Relative Likelihood of Minimal Infrmation Loss)\n"
		Result.write(line)
		line = "Model       DIC Value    Relative Likelihood\n"
		Result.write(line)
		line = "-----       ---------    -------------------\n"
		Result.write(line)
		for i in range(len(self.sortedDICVals)):
			RelProbOfMinInfLoss = 100.0*math.exp(0.5*(float(self.sortedDICVals[0][1]) - float(self.sortedDICVals[i][1])))
			line = '{:>4}   {:> 13.3f}    {:> 18.2f}%\n'.format(self.sortedDICVals[i][0], float(self.sortedDICVals[i][1]), RelProbOfMinInfLoss)
			Result.write(line);
		Result.close()