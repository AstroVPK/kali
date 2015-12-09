#!/usr/bin/env python
"""	Fucking awesome!!!
"""
import math as math
import cmath as cmath
import numpy as np
import socket
HOST = socket.gethostname()
if HOST == 'dirac.physics.drexel.edu':
	import matplotlib
	matplotlib.use('Agg')
import sys as sys
import time as time
import ConfigParser as CP
import argparse as AP
import hashlib as hashlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab
import cPickle as cPickle
import pdb

from mpl_settings import *
import carmcmc as cmcmc

goldenRatio=1.61803398875
fhgt=10.0
fwid=fhgt*goldenRatio

AnnotateXXLarge = 72
AnnotateXLarge = 48
AnnotateLarge = 32
AnnotateMedium = 28
AnnotateSmall = 24
AnnotateXSmall = 20
AnnotateXXSmall = 16

LegendLarge = 24
LegendMedium = 20
LegendSmall = 16

LabelXLarge = 32
LabelLarge = 28
LabelMedium = 24
LabelSmall = 20
LabelXSmall = 16

AxisXXLarge = 32
AxisXLarge = 28
AxisLarge = 24
AxisMedium = 20
AxisSmall = 16
AxisXSmall = 12
AxisXXSmall = 8

normalFontSize=32
smallFontSize=24
footnoteFontSize=20
scriptFontSize=16
tinyFontSize=12

LabelSize = LabelXLarge
AxisSize = AxisLarge
AnnotateSize = AnnotateXLarge
AnnotateSizeAlt = AnnotateMedium
AnnotateSizeAltAlt = AnnotateLarge
LegendSize = LegendMedium

gs = gridspec.GridSpec(1000, 1000) 

set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=AxisMedium,useTex='True')

class KellyCARMATask:
	def __init__(self, WorkingDirectory, ConfigFile, DateTime = None):
		self.RunTime = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
		self.WorkingDirectory = WorkingDirectory
		self.ConfigFile = ConfigFile
		self.preprefix = ConfigFile.split(".")[0]

		if DateTime:
			try:
				TestFile = open(WorkingDirectory + self.preprefix + '_' + DateTime + '_LC.dat', 'rb')
				self.DateTime = DateTime
				self.prefix = ConfigFile.split(".")[0] + "_" + self.DateTime
			except IOError:
				self.DateTime = None
				self.prefix = ConfigFile.split(".")[0] + "_" + self.RunTime
		else:
			self.DateTime = None
			self.prefix = ConfigFile.split(".")[0] + "_" + self.RunTime

		parser = CP.SafeConfigParser()
		parser.read(WorkingDirectory + ConfigFile)

		try:
			self.dt = float(parser.get('C-ARMA', 'dt'))
		except CP.NoOptionError as Err:
			self.dt = 1.0
			print Err + " Using default dt = %f (d)."%(self.dt)

		try:
			self.T = float(parser.get('C-ARMA', 'T'))
		except CP.NoOptionError as Err:
			self.T = 100.0
			print Err + " Using default T = %f (d)."%(self.T)

		try:
			self.baseFlux = float(parser.get('C-ARMA', 'baseFlux'))
		except CP.NoOptionError as Err:
			self.baseFlux = 0.0
			print str(Err) + " Using default baseFlux = %f (arb. units)."%(self.baseFlux)

		ARRoots = list()
		MAPoly = list()

		doneReadingARRoots = False
		self.p = 0
		while not doneReadingARRoots:
			try:
				ARRoots.append(complex(parser.get('C-ARMA', 'r_%d'%(self.p + 1))))
				self.p += 1
			except CP.NoOptionError as Err:
				doneReadingARRoots = True

		doneReadingMAPoly = False
		self.q = -1
		while not doneReadingMAPoly:
			try:
				MAPoly.append(float(parser.get('C-ARMA', 'b_%d'%(self.q + 1))))
				self.q += 1
			except CP.NoOptionError as Err:
				doneReadingMAPoly = True

		if self.p < 1:
			print "No C-AR roots supplied!"
			self.ARRoots = [10.0]
			self.p = 1
			print "Using ARRoots = " + str(self.ARRoots)

		if self.q < 0:
			print "No C-MA co-efficients supplied!"
			self.MAPoly = [1.0]
			self.q = 0
			print "Using MAPoly = " + str(self.MAPoly)

		if self.p <= self.q:
			print "Too many C-MA co-efficients!"
			sys.exit(1)

		self.ARRoots = np.array(ARRoots)
		self.MAPoly = np.array(MAPoly)

		try:
			self.noiseLvl = float(parser.get('C-ARMA', 'noiseLvl'))
		except CP.NoOptionError as Err:
			self.noiseLvl = self.MACoefs[0]/1.0e-9
			print Err + " Using noiseLvl = %4.3e corresponding to default S/N = 1e9."%(self.nLvl)

		try:
			self.nwalkers = int(parser.get('MCMC', 'nwalkers'))
		except CP.NoOptionError as Err:
			self.nwalkers = 100
			print Err + " Using default nwalkers = %d."%(nwalkers)

		try:
			self.nsteps = int(parser.get('MCMC', 'nsteps'))
		except CP.NoOptionError as Err:
			self.nsteps = 100
			print Err + " Using default nsteps = %d."%(nsteps)

	def writeLC(self, Mask = None, Cadences = None):
		"""	Create a C-ARMA light curve given a list of AR polynomial roots and a list of MA polynomial co-efficients. 
		"""
		self.t = None
		self.y = None
		self.yerr = None
		self.Mask = None
		self.Cadences = None
		if not self.DateTime:
			##sigmay = 1.2e-9
			SigmaY = self.MAPoly[0]
			##ma_coefs = np.array([1.0, 5.834])
			MACoefs = np.array([Poly/SigmaY for Poly in self.MAPoly])

			SigSqr = SigmaY**2/cmcmc.carma_variance(1.0, self.ARRoots, ma_coefs = MACoefs)

			self.t = np.arange(0.0, self.T, self.dt)
			self.y = cmcmc.carma_process(self.t, SigSqr, self.ARRoots, ma_coefs = MACoefs)
			self.y += np.array(self.t.shape[0]*[self.baseFlux])
			if not Cadences:
				self.Cadences = np.arange(self.t.shape[0])
			else:
				self.Cadences = Cadences
			noise = np.random.normal(loc = 0.0, scale = self.noiseLvl, size = self.Cadences.shape[0])
			self.y += noise
			self.yerr = np.array(self.Cadences.shape[0]*[self.noiseLvl])

			numMasked = 0
			if Mask:
				self.Mask = np.array(Mask)
				for i in xrange(self.Mask.shape[0]):
					if self.Mask[i] == 0.0:
						self.t[i] = 0.0
						self.y[i] = 0.0
						self.yerr[i] = 1.3407807929942596e+154
						numMasked += 1
			else:
				self.Mask = np.array(self.Cadences.shape[0]*[1.0])

			self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
			outFile = open(self.LCFile, 'w')
			line = "numCadences: %d\n"%(self.t.shape[0])
			outFile.write(line)
			line = "numObservations: %d\n"%(self.t.shape[0] - numMasked)
			outFile.write(line)
			line = "meanFlux: %+17.16e\n"%(np.mean(self.y))
			outFile.write(line)
			for i in xrange(self.Cadences.shape[0]-1):
				line = "%d %17.16e %17.16e %17.16e\n"%(self.Cadences[i], self.Mask[i], self.y[i], self.yerr[i])
				outFile.write(line)
			line = "%d %17.16e %17.16e %17.16e"%(self.Cadences[self.Cadences.shape[0]-1], self.Mask[self.Cadences.shape[0]-1], self.y[self.Cadences.shape[0]-1], self.yerr[self.Cadences.shape[0]-1])
			outFile.write(line)
			outFile.close()
		else:
			self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
			inFile = open(self.LCFile, 'rb')
			line = inFile.readline()
			line = line.rstrip('\n')
			words = line.split()
			numCadences = int(words[1])
			self.t = np.array(numCadences*[0.0])
			self.y = np.array(numCadences*[0.0])
			self.yerr = np.array(numCadences*[0.0])
			self.Cadences = np.array(numCadences*[0.0])
			self.Mask = np.array(numCadences*[0.0])
			line = inFile.readline()
			line = inFile.readline()
			for i in xrange(numCadences):
				words = inFile.readline().rstrip('\n').split()
				self.t[i] = i*self.dt
				self.Cadences[i] = int(words[0])
				self.Mask[i] = float(words[1])
				self.y[i] = float(words[2])
				self.yerr[i] = float(words[3])
		return 0

	def writeMCMCSamples(self, p, q):
		"""Read in a given pickled 'sample' file
	
		pickledSamplePath: Full path to pickled sample file output using Kelly's C-ARMA code.
	
		Assuming that using Kelly's C-ARMA code, we have executed
		>>> CARMA_Model = carmcmc.CarmaModel(t, y, yerr)
		>>> MLE, pqlist, AICc_list = CARMA_Model.choose_order(pmax, njobs=16)
		>>> sample = CARMA_Model.run_mcmc(nwalkers)
		where (t, y, yerr) is the lightcurve of the object that we are interested in. 'sample' is the MCMC chain 
		produced by Kelly's C-ARMA code. To save time and not re-run the Kelly C-ARMA code, we can save 'sample' 
		by pickling it using something like
		>>> Sample = open(pickledSamplePath, 'wb')
		>>> cPickle.dump(sample, Sample, -1)
		>>> Sample.close()
		This code requires the 'pickledSamplePath' that 'sample' is dumped to.
		"""
		SamplesFilePath = self.WorkingDirectory + self.prefix + "_MCMCSamples_%d_%d.pkl"%(p,q)
		try:
			SamplesFile = open(SamplesFilePath,'rb')
			self.MCMCSamples = cPickle.load(SamplesFile)
			SamplesFile.close()
		except IOError:
			CARMA_Model = cmcmc.CarmaModel(self.t, self.y, self.yerr, p = p, q = q)
			self.MCMCSamples = CARMA_Model.run_mcmc(self.nwalkers*self.nsteps)

			SamplesFile = open(SamplesFilePath, 'wb')
			cPickle.dump(self.MCMCSamples, SamplesFile, -1)
			SamplesFile.close()
		return 1

	def getCARMAParams(self):
		"""	Get the interesting parameters from a given MCMC Chains sample file.
		"""
		ARRoots = self.MCMCSamples.get_samples('ar_roots')
		ma_coefs = self.MCMCSamples.get_samples('ma_coefs')
		sigma = selfMCMCSamples.get_samples('sigma')
		LogLikelihood = -1.0*self.MCMCSamples.get_samples('loglik')
		LCMean = self.MCMCSamples.get_samples('mu')
		LCSigma = self.MCMCSamples.get_samples('var')
		MAPoly = ma_coefs*sigma
		return (LCMean, LCVar, ARPoly, MAPoly, LogLikelihood)

	def run(self):
		"""	
		"""
		self.writeLC()
		self.writeMCMCSamples(self.p, self.q)
		return 1

if __name__ == "__main__":
	parser = AP.ArgumentParser()
	parser.add_argument("pwd", help="Path to Working Directory")
	parser.add_argument("cf", help="Configuration File")
	parser.add_argument("--oldMCMC", help="DateTime of old LC and MCMC Chains to be used")
	args = parser.parse_args()
	if args.oldMCMC:
		newTask = KellyCARMATask(args.pwd, args.cf, args.oldMCMC)
	else:
		newTask = KellyCARMATask(args.pwd, args.cf)
	newTask.run()

	fig1 = plt.figure(1, figsize=(fwid, fhgt))
	ax1 = fig1.add_subplot(gs[:,:])
	ax1.ticklabel_format(useOffset = False)
	ax1.plot(newTask.t, newTask.y)
	ax1.set_xlabel(r'$t$ (d)')
	ax1.set_ylabel(r'Flux')

plt.show()
