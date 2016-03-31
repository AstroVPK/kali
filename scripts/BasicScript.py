#!/usr/bin/env python

import argparse as argparse
import math
import numpy as np
import scipy.stats as spstats
import sys
import socket
HOST = socket.gethostname()
print 'HOST: %s'%(str(HOST))
import os
import time
import pdb

from python.plotPSD import plotPSDTask
from python.makeMockLC import makeMockLCTask
from python.plotSuppliedLC import plotSuppliedLCTask
from python.fitCARMA import fitCARMATask

parser = argparse.ArgumentParser()
parser.add_argument("pwd", help = "Path to Working Directory")
parser.add_argument("cf", help = "Configuration File")
parser.add_argument("-o", "--old", help = "DateTime of run to be used")
parser.add_argument("-v", "--verbose", help = "Verbose T/F")
parser.add_argument("-p", "--prob", help = "Probability of retaining a point in lc")
parser.add_argument("-pS1", "--probSuper1", help = "Location of sinc peak in time for computing probability")
parser.add_argument("-pS2", "--probSuper2", help = "Divisor of total length of sinc")
args = parser.parse_args()

if args.old:
	try:
		TestFile = open(args.pwd + args.cf.split('.')[0] + '_' + args.old + '.lc', 'r')
		TestFile.close()
	except:
		print args.pwd + args.cf.split('.')[0] + '_' + args.old + '_LC.dat not found!'
		sys.exit(1)
	Stamp = args.old
else:
	TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
	Stamp = TimeStr

plotPSDTask(args.pwd, args.cf, Stamp).run()

makeMockLCTask(args.pwd, args.cf, Stamp).run()

fitCARMATask(args.pwd, args.cf, Stamp).run()