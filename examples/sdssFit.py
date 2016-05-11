import numpy as np
import math as math
import cmath as cmath
import psutil as psutil
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import argparse as argparse
import operator as operator
import warnings as warnings
import copy as copy
import time as time
import pdb
import os as os

import libcarma as libcarma
import util.mcmcviz as mcmcviz
import sdss as sdss
from util.mpl_settings import set_plot_params
import util.triangle as triangle

try: 
	os.environ['DISPLAY']
except KeyError as Err:
	warnings.warn('No display environment! Using matplotlib backend "Agg"')
	import matplotlib
	matplotlib.use('Agg')

try:
	import carmcmc as cmcmc
except ImportError:
	carma_pack = False
else:
	carma_pack = True

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type = str, default = '/home/vpk24/Documents', help = r'Path to working directory')
parser.add_argument('-n', '--name', type = str, default = 'LightCurveSDSS_1.csv', help = r'SDSS Filename')
parser.add_argument('-libcarmaChain', '--lC', type = str, default = 'libcarmaChain', help = r'libcarma Chain Filename')
parser.add_argument('-cmcmcChain', '--cC', type = str, default = 'cmcmcChain', help = r'carma_pack Chain Filename')
parser.add_argument('-nsteps', '--nsteps', type = int, default = 250, help = r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type = int, default = 25*psutil.cpu_count(logical = True), help = r'Number of walkers')
parser.add_argument('-pMax', '--pMax', type = int, default = 3, help = r'Maximum C-AR order')
parser.add_argument('-pMin', '--pMin', type = int, default = 1, help = r'Minimum C-AR order')
parser.add_argument('-qMax', '--qMax', type = int, default = 2, help = r'Maximum C-MA order')
parser.add_argument('-qMin', '--qMin', type = int, default = 0, help = r'Minimum C-MA order')
parser.add_argument('--plot', dest = 'plot', action = 'store_true', help = r'Show plot?')
parser.add_argument('--no-plot', dest = 'plot', action = 'store_false', help = r'Do not show plot?')
parser.set_defaults(plot = False)
parser.add_argument('-minT', '--minTimescale', type = float, default = 2.0, help = r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type = float, default = 0.5, help = r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type = float, default = 2.0, help = r'Maximum allowed sigma = maxSigma*var(lc)')
parser.add_argument('-sFac', '--scatterFactor', type = float, default = 10.0, help = r'Scatter factgor for starting locations of walkers pre-optimization')
parser.add_argument('--stop', dest = 'stop', action = 'store_true', help = r'Stop at end?')
parser.add_argument('--no-stop', dest = 'stop', action = 'store_false', help = r'Do not stop at end?')
parser.set_defaults(stop = False)
parser.add_argument('--save', dest = 'save', action = 'store_true', help = r'Save files?')
parser.add_argument('--no-save', dest = 'save', action = 'store_false', help = r'Do not save files?')
parser.set_defaults(save = False)
parser.add_argument('--log10', dest = 'log10', action = 'store_true', help = r'Compute distances in log space?')
parser.add_argument('--no-log10', dest = 'log10', action = 'store_false', help = r'Do not compute distances in log space?')
parser.set_defaults(log10 = False)
args = parser.parse_args()

if (args.qMax >= args.pMax):
	raise ValueError('pMax must be greater than qMax')
if (args.pMin < 1):
	raise ValueError('pMin must be greater than or equal to 1')
if (args.qMin < 0):
	raise ValueError('qMin must be greater than or equal to 0')

sdssLC = sdss.sdss_gLC(supplied = args.name, pwd = args.pwd)
sdssLC.minTimescale = args.minTimescale
sdssLC.maxTimescale = args.maxTimescale
sdssLC.maxSigma = args.maxSigma

taskDict = dict()
DICDict= dict()

for p in xrange(args.pMin, args.pMax + 1):
	for q in xrange(args.qMin, p):
		nt = libcarma.basicTask(p, q, nwalkers = args.nwalkers, nsteps = args.nsteps, scatterFactor = args.scatterFactor)

		minT = 5.0*sdssLC.dt*sdssLC.minTimescale
		maxT = 0.2*sdssLC.T*sdssLC.maxTimescale
		RhoGuess = -1.0/((maxT - minT)*np.random.random(p + q + 1) + minT)
		RhoGuess[-1] = 5.0e-2*np.std(sdssLC.y)
		GuessRAR, GuessIAR, GuessRMA, GuessIMA = libcarma.timescales(p, q, RhoGuess)
		TauGuess = np.array(sorted([i for i in GuessRAR]) + sorted([i for i in GuessIAR]) + sorted([i for i in GuessRMA]) + sorted([i for i in GuessIMA]) + [RhoGuess[-1]])
		print 'Tau Guess: %s'%(str(TauGuess))
		ThetaGuess = libcarma.coeffs(p, q, RhoGuess)

		print 'Starting libcarma fitting for p = %d and q = %d...'%(p, q)
		startLCARMA = time.time()
		nt.fit(sdssLC, ThetaGuess)
		stopLCARMA = time.time()
		timeLCARMA = stopLCARMA - startLCARMA
		print 'libcarma took %4.3f s = %4.3f min = %4.3f hrs'%(timeLCARMA, timeLCARMA/60.0, timeLCARMA/3600.0)

		Deviances = copy.copy(nt.LnPosterior[:,args.nsteps/2:]).reshape((-1))
		DIC = 0.5*math.pow(np.std(-2.0*Deviances),2.0) + np.mean(-2.0*Deviances)
		print 'C-ARMA(%d,%d) DIC: %+4.3e'%(p, q, DIC)
		DICDict['%d %d'%(p, q)] = DIC
		taskDict['%d %d'%(p, q)] = nt

sortedDICVals = sorted(DICDict.items(), key = operator.itemgetter(1))
pBest = int(sortedDICVals[0][0].split()[0])
qBest = int(sortedDICVals[0][0].split()[1])
print 'Best model is C-ARMA(%d,%d)'%(pBest, qBest)

bestTask = taskDict['%d %d'%(pBest, qBest)]
res = mcmcviz.vizWalkers(bestTask.Chain, bestTask.LnPosterior, 0, 1)

if args.stop:
	pdb.set_trace()
