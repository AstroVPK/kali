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

from python.computeLnLike import computeLnLikeTask

parser = argparse.ArgumentParser()
parser.add_argument("pwd", help = "Path to Working Directory")
parser.add_argument("cf", help = "Configuration File")
parser.add_argument("old", help = "DateTime of run to be used")
parser.add_argument("-v", "--verbose", help = "Verbose T/F")
args = parser.parse_args()

try:
	TestFile = open(args.pwd + args.cf.split('.')[0] + '.lc', 'r')
	TestFile.close()
except:
	print args.pwd + args.cf.split('.')[0] + '_' + args.old + '.lc not found!'
	sys.exit(1)
Stamp = args.old
computeLnLikeTask(args.pwd, args.cf, Stamp).run()