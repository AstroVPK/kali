#!/usr/bin/env python

import argparse as argparse
import numpy as np
import scipy.stats as spstats
import sys
import socket
HOST = socket.gethostname()
print 'HOST: %s'%(str(HOST))
import os
import time
import pdb

from python.plotPSDPeriodogram import plotPSDPeriodogramTask

parser = argparse.ArgumentParser(description = "Compute the PSD of a lightcurve specified by configuration file 'cf' and 'old' in working directory 'pwd' using C-ARMA parameter values ")
parser.add_argument("pwd", help = "Path to Working Directory")
parser.add_argument("cf", help = "Configuration File")
parser.add_argument("old", help = "DateTime of run to be used")
parser.add_argument("-p", "--p", help = "Order of C-AR")
parser.add_argument("-q", "--q", help = "Order of C-MA")
parser.add_argument("-theta", metavar = "Theta", type = float, nargs = "+", help = "List of p C-AR coefficients and q C-MA coefficients")
parser.add_argument("-v", "--verbose", help = "Verbose T/F")
args = parser.parse_args()

try:
	TestFile = open(args.pwd + args.cf.split('.')[0] + '.lc', 'r')
	TestFile.close()
except:
	print args.pwd + args.cf.split('.')[0] + '_' + args.old + '.lc not found!'
	sys.exit(1)
Stamp = args.old

if args.theta:
	if args.p:
		if args.q:
			if len(args.theta) != int(args.p) + int(args.q) + 1:
				print "Number of C-ARMA parameters supplied does not equal p + q + 1"
				sys.exit(1)
			else:
				plotPSDPeriodogramTask(args.pwd, args.cf, Stamp, *args.theta, p = args.p, q = args.q).run()
		else:
			print "Must supply q!"
			sys.exit(1)
	else:
		print "Must supply p!"
		sys.exit(1)
else:
	plotPSDPeriodogramTask(args.pwd, args.cf, Stamp).run()