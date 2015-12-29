#!/usr/bin/env python

import argparse as argparse
import sys
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
args = parser.parse_args()

try:
	TestFile = open(args.pwd + 'Regular_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Regular.ini', 'MMDDYYYYHHMMSS').run()
fitCARMATask(args.pwd, 'Regular.ini', 'MMDDYYYYHHMMSS').run()
try:
	TestFile = open(args.pwd + 'Irregular_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Irregular.ini', 'MMDDYYYYHHMMSS').run()
fitCARMATask(args.pwd, 'Irregular.ini', 'MMDDYYYYHHMMSS').run()
try:
	TestFile = open(args.pwd + 'Missing_MMDDYYYYHHMMSS.log', 'r')
	TestFile.close()
except IOError:
	plotSuppliedLCTask(args.pwd, 'Missing.ini', 'MMDDYYYYHHMMSS').run()
fitCARMATask(args.pwd, 'Missing.ini', 'MMDDYYYYHHMMSS').run()

if args.old:
	try:
		TestFile = open(args.pwd + args.cf.split('.')[0] + '_' + args.old + '.lc', 'r')
		TestFile.close()
	except:
		print args.pwd + args.cf.split('.')[0] + '_' + args.old + '_LC.dat not found!'
		sys.exit(1)
	plotPSDTask(args.pwd, args.cf, args.old).run()
	makeMockLCTask(args.pwd, args.cf, args.old).run()
else:
	TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
	plotPSDTask(args.pwd, args.cf, TimeStr).run()
	makeMockLCTask(args.pwd, args.cf, TimeStr).run()