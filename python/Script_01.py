import argparse as argparse
import sys
import time
import pdb

from python.plotPSD import plotPSDTask
from python.writeMockLC import writeMockLCTask

parser = argparse.ArgumentParser()
parser.add_argument("pwd", help = "Path to Working Directory")
parser.add_argument("cf", help = "Configuration File")
parser.add_argument("-o", "--old", help = "DateTime of run to be used")
parser.add_argument("-v", "--verbose", help = "Verbose T/F")
args = parser.parse_args()

if args.old:
	try:
		TestFile = open(args.pwd + args.cf.split('.')[0] + '_' + args.old + '_LC.dat', 'r')
		TestFile.close()
	except:
		print args.pwd + args.cf.split('.')[0] + '_' + args.old + '_LC.dat not found!'
		sys.exit(1)
	plotPSDTask(args.pwd, args.cf, args.old).run()
	writeMockLCTask(args.pwd, args.cf, args.old).run()
else:
	TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
	plotPSDTask(args.pwd, args.cf, TimeStr).run()
	writeMockLCTask(args.pwd, args.cf, TimeStr).run()