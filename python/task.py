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

import python.lc as lc

class Task:
	"""	Base Task class. All other tasks inherit from Task.
	"""
	def __init__(self, WorkingDirectory, ConfigFile, DateTime = None):
		"""	Initialize Task object.
		"""
		self.RunTime = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
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

		if DateTime:
			try:
				TestFile = open(WorkingDirectory + self.preprefix + '_' + DateTime + '_LC.dat', 'r')
				self.DateTime = DateTime
				self.prefix = ConfigFile.split(".")[0] + "_" + self.DateTime
			except IOError as Err:
				print str(Err) + ". Exiting..."
				sys.exit(1)
		else:
			self.DateTime = None
			self.prefix = ConfigFile.split(".")[0] + "_" + self.RunTime

		self.parser = CP.SafeConfigParser()
		self.parser.read(WorkingDirectory + self.ConfigFile)
		self.plotParser = CP.SafeConfigParser()
		self.plotParser.read(WorkingDirectory + self.PlotConfigFile)
		self.escChar = '#'
		self.LC = lc.LC()

	def strToBool(self, val):
		return val.lower() in ('yes', 'true', 't', '1')

	def formatFloat(self, val, formatStr = r'+4.3'):
		strVal = r'%' + formatStr + r'e'%(val)
		frontVal = strVal[0:float(formatStr[1:2])]
		expLoc = strVal.find(r'e')
		expVal = strVal[expLoc+1:-1]
		return r'$' + frontVal + r'\times 10^{' + expVal + r'}$'

	def getHash(self, fullPathToFile):
		"""	Compute the hash value of HashFile
		"""
		hashFile = open(fullPathToFile, 'r')
		hashData = hashFile.read().replace('\n', '').replace(' ', '')
		hashObject = hashlib.sha512(hashData.encode())
		return hashObject.hexdigest()

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