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
import cffi as cffi
import os as os
import sys as sys
import pdb as pdb

from bin._libcarma import ffi
from python.task import Task

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class writeMockLCTask(Task):
	"""	Create a C-ARMA light curve with C-ARMA configuration supplied in the ConfigFile. 
	"""
	def __init__(self, WorkingDirectory = os.getcwd() + '/examples/', ConfigFile = 'Config.ini', DateTime = None):
		Task.__init__(self, WorkingDirectory = WorkingDirectory, ConfigFile = ConfigFile, DateTime = None)
		self.dt = None
		self.T = None
		self.numCadences = None
		self.t = None
		self.IR = None

	def _useTFile(self):
		try:
			tFile = self.Config["LC " + "tFile".lower()]
		except KeyError as Err:
			print str(Err)
			sys.exit(1)
		try:
			tFileStream = open(self.WorkingDirectory + tFile, 'r')
		except IOError as Err:
			print str(Err)
			sys.exit(1)
		for line in tFileStream:
			if line[0] == self.escChar:
				continue
			self.t.append(line.rstrip("\n").split()[0])
		self.numCadences = len(self.t)
		self.T = self.t[-1] - self.t[0]
		self.t = np.array(self.t)
		self.dt = np.median(self.t)
		return 0

	def parseConfig(self):
		try:
			self.escChar = self.Config["Misc " + "escChar".lower()]
		except KeyError as KeyErr:
			self.escChar = '#'
		try:
			self.tStart = self.Config["LC " + "tStart".lower()]
		except KeyError as KeyErr:
			self.tStart = 0.0
		try:
			self.dt = float(self.Config["LC dt"])
			self.IR = False
			try:
				self.T = float(self.Config["LC " + "T".lower()])
				self.numCadences = int(self.T/self.dt)
			except KeyError as Err:
				try:
					self.numCadences = int(self.Config["LC " + "T".lower()])
					self.T = float(self.numCadences)*self.dt
				except KeyError as KeyErr:
					self._useTFile()
		except KeyError as Err:
			self._useTFile()

		pdb.set_trace()

	'''self.t = None
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
		line = "ConfigFileHash: %s\n"%(self.ConfigFileHash)
		outFile.write(line)
		line = "numCadences: %d\n"%(self.t.shape[0])
		outFile.write(line)
		line = "numObservations: %d\n"%(self.t.shape[0] - numMasked)
		outFile.write(line)
		line = "meanFlux: %+17.16e\n"%(np.mean(self.y))
		outFile.write(line)
		line = "CadenceNum Mask t y yerr\n"
		outFile.write(line) 
		for i in xrange(self.Cadences.shape[0]-1):
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e\n"%(self.Cadences[i], self.Mask[i], self.t[i], self.y[i], self.yerr[i])
			outFile.write(line)
		line = "%d %1.0f %+17.16e %+17.16e %+17.16e"%(self.Cadences[self.Cadences.shape[0]-1], self.Mask[self.Cadences.shape[0]-1], self.t[self.Cadences.shape[0]-1], self.y[self.Cadences.shape[0]-1], self.yerr[self.Cadences.shape[0]-1])
		outFile.write(line)
		outFile.close()

	else:
		self.LCFile = self.WorkingDirectory + self.prefix + "_LC.dat"
		inFile = open(self.LCFile, 'rb')
		words = inFile.readline().rstrip('\n').split()
		LCHash = words[0]
		if (LCHash == self.ConfigFileHash):
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
		else:
			print "Hash mismatch! The ConfigFile %s in WorkingDirectory %s has changed and no longer matches that used to make the light curve. Exiting!"%(self.ConfigFile, self.WorkingDirectory)

	if self.plotLC == True:
		fig1 = plt.figure(1, figsize=(fwid, fhgt))
		ax1 = fig1.add_subplot(gs[:,:])
		ax1.ticklabel_format(useOffset = False)
		ax1.plot(self.t, self.y)
		ax1.set_xlabel(r'$t$ (d)')
		ax1.set_ylabel(r'Flux')
		if self.JPG == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.jpg" , dpi = dpi)
		if self.PDF == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.pdf" , dpi = dpi)
		if self.EPS == True:
			fig1.savefig(self.WorkingDirectory + self.prefix + "_LC.eps" , dpi = dpi)
		if self.showFig == True:
			plt.show()
	return 0'''