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

class Task:
	"""	Base Task class. All other tasks inherit from Task.
	"""
	def __init__(self, WorkingDirectory, ConfigFile, DateTime = None):
	"""	Initialize a Task object.
	"""
		self.RunTime = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
		self.WorkingDirectory = WorkingDirectory
		self.ConfigFile = ConfigFile
		self.preprefix = ConfigFile.split(".")[0]
		try:
			hashFile = open(self.WorkingDirectory + self.ConfigFile, 'r')
			hashData = hashFile.read().replace('\n', '').replace(' ', '')
			hashObject = hashlib.sha512(hashData.encode())
			self.ConfigFileHash = hashObject.hexdigest()
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
		self.parser.read(WorkingDirectory + ConfigFile)

	def getHash():
		"""	Compute the hash value of HashFile
		"""
		hashFile = open(self.WorkingDirectory + self.ConfigFile, 'r')
		hashData = hashFile.read().replace('\n', '').replace(' ', '')
		hashObject = hashlib.sha512(hashData.encode())
		return hashObject.hexdigest()

	def readConfig():
		"""	Read the Configfile into self.Config 
		"""
		self.Config = dict()
		self.Sections = self.parser.sections()
		self.Options = list()
		for Section in self.Sections:
			self.Options.append(self.parser.options(Section))
			for Option in Options[-1]:
				self.Config['%s %s'%(Section,Option)] = self.parser.get(Section, Option)

	def run(self):
		raise NotImplementedError