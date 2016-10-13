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
import os as os
import sys as sys
import pdb

try:
    import kali.carma
    import kali.util.mcmcviz as mcmcviz
    import kali.s82
    from kali.util.mpl_settings import set_plot_params
    import kali.util.triangle as triangle
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

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
set_plot_params(useTex=True)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type=str, default=os.path.join(
    os.environ['KALI'], 'examples/data'), help=r'Path to working directory')
parser.add_argument('-n', '--name', type=str, default='rand', help=r'SDSS ID')
parser.add_argument('-b', '--band', type=str, default='g', help=r'SDSS bandpass')
parser.add_argument('-libcarmaChain', '--lC', type=str,
                    default='libcarmaChain', help=r'libcarma Chain Filename')
parser.add_argument('-cmcmcChain', '--cC', type=str,
                    default='cmcmcChain', help=r'carma_pack Chain Filename')
parser.add_argument('-nsteps', '--nsteps', type=int, default=250, help=r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type=int, default=25 *
                    psutil.cpu_count(logical=True), help=r'Number of walkers')
parser.add_argument('-pMax', '--pMax', type=int, default=1, help=r'Maximum C-AR order')
parser.add_argument('-pMin', '--pMin', type=int, default=1, help=r'Minimum C-AR order')
parser.add_argument('-qMax', '--qMax', type=int, default=-1, help=r'Maximum C-MA order')
parser.add_argument('-qMin', '--qMin', type=int, default=-1, help=r'Minimum C-MA order')
parser.add_argument('--plot', dest='plot', action='store_true', help=r'Show plot?')
parser.add_argument('--no-plot', dest='plot', action='store_false', help=r'Do not show plot?')
parser.set_defaults(plot=True)
parser.add_argument('-minT', '--minTimescale', type=float,
                    default=2.0, help=r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type=float,
                    default=0.5, help=r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type=float,
                    default=2.0, help=r'Maximum allowed sigma = maxSigma*var(lc)')
parser.add_argument('--stop', dest='stop', action='store_true', help=r'Stop at end?')
parser.add_argument('--no-stop', dest='stop', action='store_false', help=r'Do not stop at end?')
parser.set_defaults(stop=False)
parser.add_argument('--save', dest='save', action='store_true', help=r'Save files?')
parser.add_argument('--no-save', dest='save', action='store_false', help=r'Do not save files?')
parser.set_defaults(save=False)
parser.add_argument('--log10', dest='log10',
                    action='store_true', help=r'Compute distances in log space?')
parser.add_argument('--no-log10', dest='log10', action='store_false',
                    help=r'Do not compute distances in log space?')
parser.set_defaults(log10=False)
parser.add_argument('--viewer', dest='viewer', action='store_true', help=r'Visualize MCMC walkers')
parser.add_argument('--no-viewer', dest='viewer',
                    action='store_false', help=r'Do not visualize MCMC walkers')
parser.set_defaults(viewer=True)
args = parser.parse_args()

if (args.qMax >= args.pMax):
    raise ValueError('pMax must be greater than qMax')
if (args.qMax == -1):
    args.qMax = args.pMax - 1
if (args.qMin == -1):
    args.qMin = 0
if (args.pMin < 1):
    raise ValueError('pMin must be greater than or equal to 1')
if (args.qMin < 0):
    raise ValueError('qMin must be greater than or equal to 0')

sdssLC = kali.s82.sdssLC(name=args.name, band=args.band, pwd=args.pwd)
sdssLC.minTimescale = args.minTimescale
sdssLC.maxTimescale = args.maxTimescale
sdssLC.maxSigma = args.maxSigma

if args.plot:
    plt.figure(0, figsize=(fwid, fhgt))
    plt.errorbar(sdssLC.t, sdssLC.y, sdssLC.yerr, label=r'%s (%s-band)'%(args.name.split('.')[0], args.band),
                 fmt='.', capsize=0, color='#2ca25f', markeredgecolor='none', zorder=10)
    plt.xlabel(r'$t$ (MJD)')
    plt.ylabel(r'$F$ (Jy)')
    plt.title(r'Light curve')
    plt.legend()
    plt.show(False)

taskDict = dict()
DICDict = dict()
totalTime = 0.0

for p in xrange(args.pMin, args.pMax + 1):
    for q in xrange(args.qMin, min(p, args.qMax + 1)):
        nt = kali.carma.CARMATask(p, q, nwalkers=args.nwalkers, nsteps=args.nsteps)

        print 'Starting carma fitting for p = %d and q = %d...'%(p, q)
        startLCARMA = time.time()
        nt.fit(sdssLC)
        stopLCARMA = time.time()
        timeLCARMA = stopLCARMA - startLCARMA
        print 'carma took %4.3f s = %4.3f min = %4.3f hrs'%(timeLCARMA, timeLCARMA/60.0, timeLCARMA/3600.0)
        totalTime += timeLCARMA

        Deviances = copy.copy(nt.LnPosterior[:, args.nsteps/2:]).reshape((-1))
        DIC = 0.5*math.pow(np.nanstd(-2.0*Deviances), 2.0) + np.nanmean(-2.0*Deviances)
        print 'C-ARMA(%d,%d) DIC: %+4.3e'%(p, q, DIC)
        DICDict['%d %d'%(p, q)] = DIC
        taskDict['%d %d'%(p, q)] = nt
print 'Total time taken by carma is %4.3f s = %4.3f min = %4.3f hrs'%(totalTime, totalTime/60.0,
                                                                      totalTime/3600.0)

sortedDICVals = sorted(DICDict.items(), key=operator.itemgetter(1))
pBest = int(sortedDICVals[0][0].split()[0])
qBest = int(sortedDICVals[0][0].split()[1])
print 'Best model is C-ARMA(%d,%d)'%(pBest, qBest)

bestTask = taskDict['%d %d'%(pBest, qBest)]

if args.viewer:
    notDone = True
    while notDone:
        whatToView = -1
        while whatToView < 0 or whatToView > 3:
            whatToView = int(
                raw_input('View walkers in C-ARMA coefficients (0) or C-ARMA roots (1) or \
                C-ARMA timescales (2):'))
        pView = -1
        while pView < 1 or pView > args.pMax:
            pView = int(raw_input('C-AR model order:'))
        qView = -1
        while qView < 0 or qView >= pView:
            qView = int(raw_input('C-MA model order:'))

        dim1 = -1
        while dim1 < 0 or dim1 > pView + qView:
            dim1 = int(raw_input('1st Dimension to view:'))
        dim2 = -1
        while dim2 < 0 or dim2 > pView + qView or dim2 == dim1:
            dim2 = int(raw_input('2nd Dimension to view:'))

        if whatToView == 0:
            if dim1 < pView:
                dim1Name = r'$a_{%d}$'%(dim1)
            if dim1 >= pView and dim1 < pView + qView + 1:
                dim1Name = r'$b_{%d}$'%(dim1 - pView)
            if dim2 < pView:
                dim2Name = r'$a_{%d}$'%(dim2)
            if dim2 >= pView and dim2 < pView + qView + 1:
                dim2Name = r'$b_{%d}$'%(dim2 - pView)
            res = mcmcviz.vizWalkers(taskDict['%d %d'%(pView, qView)].Chain, taskDict[
                                     '%d %d'%(pView, qView)].LnPosterior, dim1, dim1Name, dim2, dim2Name)

        elif whatToView == 1:
            if dim1 < pView:
                dim1Name = r'$r_{%d}$'%(dim1)
            if dim1 >= pView and dim1 < pView + qView:
                dim1Name = r'$m_{%d}$'%(dim1 - pView)
            if dim1 == pView + qView:
                dim1Name = r'$\mathrm{Amp.}$'
            if dim2 < pView:
                dim2Name = r'$r_{%d}$'%(dim2)
            if dim2 >= pView and dim2 < pView + qView:
                dim2Name = r'$m_{%d}$'%(dim2 - pView)
            if dim2 == pView + qView:
                dim2Name = r'$\mathrm{Amp.}$'
            res = mcmcviz.vizWalkers(taskDict['%d %d'%(pView, qView)].rootChain, taskDict[
                                     '%d %d'%(pView, qView)].LnPosterior, dim1, dim1Name, dim2, dim2Name)

        else:
            if dim1 < pView + qView:
                dim1Name = r'$\tau_{%d}$'%(dim1)
            if dim1 == pView + qView:
                dim1Name = r'$\mathrm{Amp.}$'
            if dim2 < pView + qView:
                dim2Name = r'$\tau_{%d}$'%(dim2)
            if dim2 == pView + qView:
                dim2Name = r'$\mathrm{Amp.}$'
            res = mcmcviz.vizWalkers(taskDict['%d %d'%(pView, qView)].timescaleChain, taskDict[
                                     '%d %d'%(pView, qView)].LnPosterior, dim1, dim1Name, dim2, dim2Name)

        var = str(raw_input('Do you wish to view any more MCMC walkers? (y/n):')).lower()
        if var == 'n':
            notDone = False

Theta = bestTask.Chain[:, np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[
    0][0], np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[1][0]]
nt = kali.carma.CARMATask(pBest, qBest)
nt.set(sdssLC.dt, Theta)
nt.smooth(sdssLC)
sdssLC.plot()

plt.figure(1, figsize=(fwid, fhgt))
lagsEst, sfEst, sferrEst = sdssLC.sf()
lagsModel, sfModel = bestTask.sf(start=lagsEst[1], stop=lagsEst[-1], num=5000, spacing='log')
plt.loglog(lagsModel, sfModel, label=r'$SF(\delta t)$ (model)', color='#000000', zorder=5)
plt.errorbar(lagsEst, sfEst, sferrEst, label=r'$SF(\delta t)$ (est)',
             fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=0)
plt.xlabel(r'$\log_{10}\delta t$')
plt.ylabel(r'$\log_{10} SF$')
plt.legend(loc=2)
plt.show(True)

if args.stop:
    pdb.set_trace()
