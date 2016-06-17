import numpy as np
import argparse as argparse
import os
import sys
import pdb

try:
	import libcarma
	import sdss
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type = str, default = os.path.join(os.environ['LIBCARMA'],'examples/data'), help = r'Path to working directory')
parser.add_argument('-n', '--name', type = str, default = 'LightCurveSDSS_1.csv', help = r'SDSS Filename')
args = parser.parse_args()

sdssLC = sdss.sdssLC(name = args.name, band = 'g', pwd = args.pwd)

nt = libcarma.basicTask(3, 1)

Rho = np.array([-1.0/100.0, -1.0/55.0, -1.0/10.0, -1.0/25.0, 2.0e-08])
Theta = libcarma.coeffs(3, 1, Rho)

nt.set(sdssLC.dt, Theta)

print "logPrior: %+8.7e"%(nt.logPrior(sdssLC))

print "logLikelihood: %+8.7e"%(nt.logLikelihood(sdssLC))

print "logPosterior: %+8.7e"%(nt.logPosterior(sdssLC))