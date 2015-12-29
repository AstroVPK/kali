import argparse as argparse
import sys
import time
import pdb

from python.plotPSD import plotPSDTask

parser = argparse.ArgumentParser()
parser.add_argument("pwd", help = "Path to Working Directory")
parser.add_argument("cf", help = "Configuration File")
parser.add_argument("-o", "--old", help = "DateTime of run to be used")
parser.add_argument("-v", "--verbose", help = "Verbose T/F")
args = parser.parse_args()

TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
plotPSDTask(args.pwd, args.cf, TimeStr).run()