import argparse as argparse
import time
import pdb

from python.plotPSD import plotPSDTask
from python.writeMockLC import writeMockLCTask

parser = argparse.ArgumentParser()
parser.add_argument("--old", help = "DateTime of run to be used")
args = parser.parse_args()

if args.old:
	plotPSDTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/', 'Config.ini', args.old).run()
	writeMockLCTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/', 'Config.ini', args.old).run()
else:
	TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
	plotPSDTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/', 'Config.ini', TimeStr).run()
	writeMockLCTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/', 'Config.ini', TimeStr).run()