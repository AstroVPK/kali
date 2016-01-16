#!/usr/bin/env python
"""	Module that defines base LC.

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
print 'HOST: %s'%(str(HOST))
import sys as sys
import os as os
try: 
	os.environ['DISPLAY']
except KeyError as Err:
	print "No display environment! Using matplotlib backend 'Agg'"
	import matplotlib
	matplotlib.use('Agg')
import time as time
import ConfigParser as CP
import argparse as AP
import hashlib as hashlib
import pdb

class LC:
	def __init__(self):
		self.numCadences = 0
		self.dt = 0.0
		self.T = 0.0
		self.cadence = None
		self.mask = None
		self.t = None
		self.x = None
		self.y = None
		self.yerr = None
		self.t_incr = None
		self.intrinsicVar = 0.0
		self.noiseLvl = 0.0
		self.startT = 0.0
		self.tolIR = 0.0
		self.IR = False