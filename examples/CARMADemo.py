import numpy as np
import math as math
import time as time
import pdb as pdb
import sys as sys
import copy
import argparse as argparse
import sys as sys

import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import kali.carma
import kali.util.triangle as triangle

plt.ion()
parser = argparse.ArgumentParser()
parser.add_argument('--stop', dest='stop', action='store_true', help=r'Stop at end?')
parser.add_argument('--no-stop', dest='stop', action='store_false', help=r'Do not stop at end?')
parser.set_defaults(stop=False)
args = parser.parse_args()

tolIR = 1.0e-3
fracIntrinsicVar = 0.1
fracNoiseToSignal = 1.0e-18
maxSigma = 2.0
minTimescale = 2.0
maxTimescale = 0.5

p = 2
q = 1
ndims = p + q + 1
Rho = np.array([-(1.0/1.5)+0j, -(1.0/62.0)+0j, -(1.0/0.1725), 1.0e-9])
print "Rho: " + str(Rho)

Tau = kali.carma.timescales(p, q, Rho)
print "Tau: " + str(Tau)

Theta = kali.carma.coeffs(p, q, Rho)
print "Theta: " + str(Theta)

dt = 0.02
T = 1250.0

newTask = kali.carma.CARMATask(p, q, nwalkers=160, nsteps=200)

newTask.set(dt, Theta)

print "Sigma[0]: %e"%(math.sqrt(newTask.Sigma()[0, 0]))

Lags, ACVF = newTask.acvf(1.0e-4, 1.0e4, 10000, spacing='log')

newLC = newTask.simulate(T)

newTask.observe(newLC)

lnPrior = newTask.logPrior(newLC)

print 'The log prior is %e'%(lnPrior)

lnLikelihood = newTask.logLikelihood(newLC)

print 'The log likelihood is %e'%(lnLikelihood)

lnPosterior = newTask.logPosterior(newLC)

print 'The log posterior is %e'%(lnPosterior)

startT = time.time()
newTask.fit(newLC)
stopT = time.time()
print "Time taken: %+4.3e sec; %+4.3e min; %+4.3e hrs"%((stopT - startT), (stopT - startT)/60.0,
                                                        (stopT - startT)/3600.0)

maxLnPosterior = np.nanmax(newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps])
flatTimescaleChain = np.swapaxes(
    copy.deepcopy(newTask.timescaleChain[:, :, newTask.nsteps/2:]).reshape((ndims, -1), order='F'),
    axis1=0, axis2=1)
labelsList = [r'$\tau_{\mathrm{AR}, 1}$', r'$\tau_{\mathrm{AR}, 2}$',
              r'$\tau_{\mathrm{MA}, 1}$', r'$\mathrm{Amp.}$']
triangle.corner(flatTimescaleChain, labels=labelsList, show_titles=True, title_fmt=".2f",
                title_args={}, extents=None, truths=Tau, truth_color="#000000", scale_hist=False,
                quantiles=[], verbose=True, plot_contours=True, plot_datapoints=True,
                plot_contour_lines=False, fig=None, pcolor_cmap=colormap.Greys_r)

plt.show(True)

if args.stop:
    pdb.set_trace()

newLC.sampler = 'sincSampler'
sampledNewLC = newLC.sample(width=1.0)

lnPrior = newTask.logPrior(sampledNewLC)

print 'The log prior is %e'%(lnPrior)

lnLikelihood = newTask.logLikelihood(sampledNewLC)

print 'The log likelihood is %e'%(lnLikelihood)

lnPosterior = newTask.logPosterior(sampledNewLC)

print 'The log posterior is %e'%(lnPosterior)

startT = time.time()
newTask.fit(sampledNewLC)
stopT = time.time()
print "Time taken: %+4.3e sec; %+4.3e min; %+4.3e hrs"%((stopT - startT), (stopT - startT)/60.0,
                                                        (stopT - startT)/3600.0)

maxLnPosterior = np.nanmax(newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps])
flatTimescaleChain = np.swapaxes(
    copy.deepcopy(newTask.timescaleChain[:, :, newTask.nsteps/2:]).reshape((ndims, -1), order='F'),
    axis1=0, axis2=1)
labelsList = [r'$\tau_{\mathrm{AR}, 1}$', r'$\tau_{\mathrm{AR}, 2}$',
              r'$\tau_{\mathrm{MA}, 1}$', r'$\mathrm{Amp.}$']
triangle.corner(flatTimescaleChain, labels=labelsList, show_titles=True, title_fmt=".2f",
                title_args={}, extents=None, truths=Tau, truth_color="#000000", scale_hist=False,
                quantiles=[], verbose=True, plot_contours=True, plot_datapoints=True,
                plot_contour_lines=False, fig=None, pcolor_cmap=colormap.Greys_r)

plt.show(True)

if args.stop:
    pdb.set_trace()
