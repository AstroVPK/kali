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

try:
	TestFile = open(args.pwd + 'Regular.lc', 'r')
	TestFile.close()
except IOError:
	cmd = 'cp %s%s %sRegular.lc'%(args.pwd, args.cf.split('.')[0] + '_' + Stamp + '.lc', args.pwd)
	os.system(cmd)

try:
	TestFile = open(args.pwd + 'Irregular.lc', 'r')
	TestFile.close()
	TestFile = open(args.pwd + 'Missing.lc', 'r')
	TestFile.close()
except IOError:
	RegularFile = 'Regular.lc'
	MissingFile = 'Missing.lc'
	IrregularFile = 'Irregular.lc'
	Regular = np.loadtxt(args.pwd + RegularFile, skiprows = 7)
	numCadences_Regular = Regular.shape[0]

	if args.probSuper1 and args.probSuper2 and args.prob == None:
		timePeriod = float(args.probSuper1)
		multiplier = float(args.probSuper2)
		ProbList = list()
		dt = np.median(Regular[1:,2] - Regular[:-1,2])
		for i in xrange(numCadences_Regular):
			p = abs(np.sinc((i*dt - timePeriod)/(numCadences_Regular*dt*multiplier)))
			ProbList.append(spstats.bernoulli.rvs(float(p), size = 1))

	if args.prob == None and args.probSuper1 == None and args.probSuper2 == None:
		args.prob  = 0.5
		if (float(args.prob) >= 1.0) or (float(args.prob) <= 0.0):
			raise RuntimeError('prob must be between 0.0 and 1.0')
		ProbList = spstats.bernoulli.rvs(float(args.prob), size = numCadences_Regular)

	numCadences_Missing = np.sum(ProbList)
	numCadences_Irregular = np.sum(ProbList)

	Missing = open(args.pwd + MissingFile, 'w')
	line = "#ConfigFileHash: %s\n"%('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a')
	Missing.write(line)
	line = "#SuppliedLCHash: %s\n"%('')
	Missing.write(line)
	line = "#numCadences: %d\n"%(numCadences_Regular)
	Missing.write(line)
	line = "#numObservations: %d\n"%(numCadences_Missing)
	Missing.write(line)
	line = "#meanFlux: %+17.16e\n"%(0.0)
	Missing.write(line)
	line = "#LnLike: %+17.16e\n"%(0.0)
	Missing.write(line)
	line = "#cadence mask t x y yerr\n"
	Missing.write(line)

	Irregular = open(args.pwd + IrregularFile, 'w')
	line = "#ConfigFileHash: %s\n"%('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a')
	Irregular.write(line)
	line = "#SuppliedLCHash: %s\n"%('')
	Irregular.write(line)
	line = "#numCadences: %d\n"%(numCadences_Irregular)
	Irregular.write(line)
	line = "#numObservations: %d\n"%(numCadences_Irregular)
	Irregular.write(line)
	line = "#meanFlux: %+17.16e\n"%(0.0)
	Irregular.write(line)
	line = "#LnLike: %+17.16e\n"%(0.0)
	Irregular.write(line)
	line = "#cadence mask t x y yerr\n"
	Irregular.write(line)
	
	IrregularCounter = 0
	for i in xrange(numCadences_Regular):
		if ProbList[i] == 1:
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(IrregularCounter, Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5])
			Irregular.write(line)
			IrregularCounter += 1
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(int(Regular[i,0]), Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5])
		else:
			line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(int(Regular[i,0]), 0.0, Regular[i,2], 0.0, 0.0, 1.3407807929942596e+154)
		Missing.write(line)
	
	Missing.close()
	Irregular.close()

try:
	TestFile = open(args.pwd + 'Irregular_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Irregular.ini', 'MMDDYYYYHHMMSS').run()
try:
	TestFile = open(args.pwd + 'Irregular_MMDDYYYYHHMMSS_CARMAResult.dat', 'r')
except IOError:
	fitCARMATask(args.pwd, 'Irregular.ini', 'MMDDYYYYHHMMSS').run()

try:
	TestFile = open(args.pwd + 'Missing_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Missing.ini', 'MMDDYYYYHHMMSS').run()
try:
	TestFile = open(args.pwd + 'Missing_MMDDYYYYHHMMSS_CARMAResult.dat', 'r')
except IOError:
	fitCARMATask(args.pwd, 'Missing.ini', 'MMDDYYYYHHMMSS').run()

try:
	TestFile = open(args.pwd + 'Regular_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Regular.ini', 'MMDDYYYYHHMMSS').run()
try:
	TestFile = open(args.pwd + 'Regular_MMDDYYYYHHMMSS_CARMAResult.dat', 'r')
except IOError:
	fitCARMATask(args.pwd, 'Regular.ini', 'MMDDYYYYHHMMSS').run()