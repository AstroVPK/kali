#!/usr/bin/env python
"""	Module that defines base Task.

	For a demonstration of the module, please run the module as a command line program using 
	bash-prompt$ python task.py --help
	and
	bash-prompt$ python task.py $PWD/examples/taskTest taskTest01.ini
"""
import math as math
import cmath as cmath
import numpy as np
import copy as copy
import cffi as cffi
import inspect
import socket
HOST = socket.gethostname()
if HOST == 'dirac.physics.drexel.edu':
	import matplotlib
	matplotlib.use('Agg')
import sys as sys
import os as os
import time as time
import ConfigParser as CP
import argparse as AP
import hashlib as hashlib
import pdb

from bin._libcarma import ffi
import python.lc as lc

ffiObj = cffi.FFI()
C = ffi.dlopen("./bin/libcarma.so.1.0.0")
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class Task:
	"""	Base Task class. All other tasks inherit from Task.
	"""
	def __init__(self, WorkingDirectory, ConfigFile, TimeStr):
		"""	Initialize Task object.
		"""
		self.WorkingDirectory = WorkingDirectory
		self.ConfigFile = ConfigFile
		self.preprefix = ConfigFile.split(".")[0]
		self.PlotConfigFile = self.preprefix + ".plt"
		self.PlotConfigFileHash = self.getHash(self.WorkingDirectory + self.PlotConfigFile)
		self.PatternFile = self.preprefix + ".pat"
		self.PatternFileHash = self.getHash(self.WorkingDirectory + self.PatternFile)
		try:
			self.ConfigFileHash = self.getHash(self.WorkingDirectory + self.ConfigFile)
		except IOError as Err:
			print str(Err) + ". Exiting..."
			sys.exit(1)

		try:
			TestFile = open(WorkingDirectory + self.preprefix + '_' + TimeStr + '.log', 'r')
			TestFile.close()
			self.DateTime = TimeStr
			self.RunTime = None
			self.prefix = self.preprefix + '_' + self.DateTime
		except IOError as Err:
			self.DateTime = None
			self.RunTime = TimeStr
			self.prefix = self.preprefix + '_' + self.RunTime

		self.parser = CP.SafeConfigParser()
		self.parser.read(WorkingDirectory + self.ConfigFile)
		self.plotParser = CP.SafeConfigParser()
		self.plotParser.read(WorkingDirectory + self.PlotConfigFile)
		self.escChar = '#'
		self.LC = lc.LC()

	def strToBool(self, val):
		return val.lower() in ('yes', 'true', 't', '1')

	def formatFloat(self, val, formatStr = r'+3.2'):
		strVal = r'%' + formatStr + r'e'
		strVal = strVal%(val)
		frontVal = strVal[0:int(formatStr[1:2])+2]
		expLoc = strVal.find(r'e')
		expVal = strVal[expLoc+1:len(strVal)]
		if int(expVal) == 0:
			retVal = frontVal
		else:
			retVal = frontVal + r'\times 10^{' + expVal + r'}'
		return retVal

	def getHash(self, fullPathToFile):
		"""	Compute the hash value of HashFile
		"""
		hashFile = open(fullPathToFile, 'r')
		hashData = hashFile.read().replace('\n', '').replace(' ', '')
		hashObject = hashlib.sha512(hashData.encode())
		return hashObject.hexdigest()

	def _read_escChar(self):
		""" Attempts to set the escape charatcter to be used.
		"""
		try:
			self.escChar = self.parser.get('MISC', 'escChar')
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.escChar = '#'

	def _read_basicPlotOptions(self):
		"""	Attempts to read in the plot options to be used.
		"""
		try:
			self.makePlot = self.strToBool(self.plotParser.get('PLOT', 'makePlot'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.makePlot = False
		try:
			self.JPG = self.strToBool(self.plotParser.get('PLOT', 'JPG'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.JPG = False
		try:
			self.PDF = self.strToBool(self.plotParser.get('PLOT', 'PDF'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PDF = False
		try:
			self.EPS = self.strToBool(self.plotParser.get('PLOT', 'EPS'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.EPS = False
		try:
			self.PNG = self.strToBool(self.plotParser.get('PLOT', 'PNG'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.PNG = False
		try:
			self.showFig = self.strToBool(self.plotParser.get('PLOT', 'showFig'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.showFig = False

	def parseConfig(self):
		"""	Subclasses define function(s) that extract parameter values from the Config dict. This function 
			sequentially calls them.
		"""
		self.methodList = inspect.getmembers(self, predicate=inspect.ismethod)
		for method in self.methodList:
			if method[0][0:6] == '_read_':
				method[1]()

	def run(self):
		self.parseConfig()
		self.methodList = inspect.getmembers(self, predicate=inspect.ismethod)
		for method in self.methodList:
			if method[0][0:6] == '_make_':
				method[1]()

class SuppliedParametersTask(Task):
	def _read_00_CARMAProps(self):
		"""	Attempts to parse AR roots and MA coefficients.
		"""
		ARRoots = list()
		ARPoly = list()
		MACoefs = list()
		self.ARCoefs = list()
		self.ARRoots = list()

		doneReadingARRoots = False
		pRoot = 0
		while not doneReadingARRoots:
			try:
				ARRoots.append(complex(self.parser.get('C-ARMA', 'r_%d'%(pRoot + 1))))
				pRoot += 1
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingARRoots = True

		doneReadingARPoly = False
		pPoly = 0
		while not doneReadingARPoly:
			try:
				ARPoly.append(float(self.parser.get('C-ARMA', 'a_%d'%(pRoot + 1))))
				pPoly += 1
			except ValueError as Err:
				print str(Err) + '. All AR polynomial coefficients must be real!'
				sys.exit(1)
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingARPoly = True

		if (pRoot == pPoly):
			aPoly = np.polynomial.polynomial.polyfromroots(ARRoots)
			aPoly = aPoly.tolist()
			aPoly.reverse()
			aPoly.pop(0)
			aPoly = [coeff.real for coeff in aPoly]
			for ARCoef, aPolyCoef in zip(ARPoly, aPoly):
				if abs((ARCoef - aPolyCoef)/((ARCoef + aPolyCoef)/2.0)) > 1.0e-6:
					print 'ARRoots and ARPolynomial supplied are not equivalent!'
					sys.exit(1)
			self.p = pRoot
			self.ARRoots = np.array(ARRoots)
			self.ARCoefs = np.array(ARPoly)
		elif (pRoot == 0) and (pPoly > 0):
			self.p = pPoly
			self.ARCoefs = aPoly
			ARPoly = copy.deepcopy(self.ARCoefs)
			self.ARCoefs = np.array(self.ARCoefs)
			ARPoly.insert(0,1.0)
			self.ARRoots = np.roots(ARPoly)
			self.ARRoots = np.array(self.ARRoots)
		elif (pRoot > 0) and (pPoly == 0):
			self.p = pRoot
			self.ARRoots = ARRoots
			aPoly = np.polynomial.polynomial.polyfromroots(ARRoots)
			self.ARRoots = np.array(self.ARRoots)
			aPoly = aPoly.tolist()
			aPoly.reverse()
			aPoly.pop(0)
			aPoly = [coeff.real for coeff in aPoly]
			self.ARCoefs = copy.deepcopy(aPoly)
			self.ARCoefs = np.array(self.ARCoefs)
		else:
			print 'ARRoots and ARPolynomial supplied are not equivalent!'
			sys.exit(1)

		doneReadingMACoefs = False
		self.q = -1
		while not doneReadingMACoefs:
			try:
				MACoefs.append(float(self.parser.get('C-ARMA', 'b_%d'%(self.q + 1))))
				self.q += 1
			except (CP.NoOptionError, CP.NoSectionError) as Err:
				doneReadingMACoefs = True
		self.MACoefs = np.array(MACoefs)

		if self.p < 1:
			print 'No C-AR roots supplied!'
			sys.exit(1)

		if self.q < 0:
			print 'No C-MA co-efficients supplied!'
			sys.exit(1)

		if self.p <= self.q:
			print 'Too many C-MA co-efficients! Exiting...'
			sys.exit(1)

		Theta_cffi = ffiObj.new("double[%d]"%(self.p + self.q + 1))
		for i in xrange(self.p):
			Theta_cffi[i] = self.ARCoefs[i]
		for i in xrange(self.q + 1):
			Theta_cffi[self.p + i] = self.MACoefs[i]
		yORn = C._testSystem(self.LC.dt, self.p, self.q, Theta_cffi)
		if yORn == 0:
			print 'Bad C-ARMA Parameters!'
			sys.exit(1)

		self.ndims = self.p + self.q + 1

		try:
			self.numBurn = int(self.parser.get('C-ARMA', 'numBurn'))
		except (CP.NoOptionError, CP.NoSectionError) as Err:
			self.numBurn = 1000000
			print str(Err) + '. Using default numBurn = %d'%(self.numBurn)

	def _make_00_EqnString(self):
		"""	Attempts to construct a latex string consisting of the equation of the LC
		"""
		self.eqnStr = r'$'
		if self.p > 1:
			self.eqnStr += r'\mathrm{d}^{%d}F'%(self.p)
			for i in xrange(self.p - 2):
				self.eqnStr += (self.formatFloat(self.ARCoefs[i]) + r'\mathrm{d}^{%d}F'%(self.p - 1 - i))
			self.eqnStr += (self.formatFloat(self.ARCoefs[self.p - 2]) + r'\mathrm{d}F')
			self.eqnStr += (self.formatFloat(self.ARCoefs[self.p - 1]) + r'F=')
		elif self.p == 1:
			self.eqnStr += r'\mathrm{d}F'
			self.eqnStr += (self.formatFloat(self.ARCoefs[0]) + r'F=')
		self.eqnStr += (self.formatFloat(self.MACoefs[0]) + r'(\mathrm{d}W)')
		if self.q >= 1:
			self.eqnStr += (self.formatFloat(self.MACoefs[1]) + r'\mathrm{d}(\mathrm{d}W)')
		if self.q >= 2:
			for i in xrange(self.q - 1):
				self.eqnStr += (self.formatFloat(self.MACoefs[2 + i]) + r'\mathrm{d}^{%d}(\mathrm{d}W)'%(2 + i))
		self.eqnStr += r'$'