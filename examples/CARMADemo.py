import numpy as np
import math as math
import time as time
import pdb as pdb
import sys as sys
import argparse as argparse
import sys as sys

import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import kali.carma

plt.ion()
parser = argparse.ArgumentParser()
parser.add_argument("--stop", "-s", default=False, help="Enable pdb breakpoint at end?")
args = parser.parse_args()

tolIR = 1.0e-3
fracIntrinsicVar = 0.1
fracNoiseToSignal = 1.0e-18
maxSigma = 2.0
minTimescale = 2.0
maxTimescale = 0.5

p = 2
q = 1
Rho = np.array([-(1.0/7.50)+0j, -(1.0/30.0)+0j, -(1.0/5.0), 1.0e-9])
print "Rho: " + str(Rho)

Tau = kali.carma.timescales(p, q, Rho)
print "Tau: " + str(Tau)

Theta = kali.carma.coeffs(p, q, Rho)
print "Theta: " + str(Theta)

dt = 0.5
T = 1200.0

newTask = kali.carma.CARMATask(p, q, nwalkers=160, nsteps=250)

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

plt.figure(1)
plt.scatter(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps],
            newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps],
            c=maxLnPosterior - newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps],
            marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
plt.axhline(y=Theta[1], xmin=0.0, xmax=1.0)
plt.axvline(x=Theta[0], ymin=0.0, ymax=1.0)
plt.xlim([np.min(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps])])
plt.ylim([np.min(newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps])])
plt.xlabel(r'$a_{1}$')
plt.ylabel(r'$a_{2}$')

plt.figure(2)
plt.scatter(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps],
            newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps],
            c=maxLnPosterior - newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps],
            marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
plt.axhline(y=Theta[3], xmin=0.0, xmax=1.0)
plt.axvline(x=Theta[2], ymin=0.0, ymax=1.0)
plt.xlim([np.min(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps])])
plt.ylim([np.min(newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps])])
plt.xlabel(r'$b_{0}$')
plt.ylabel(r'$b_{1}$')

plt.show(True)

if args.stop:
    pdb.set_trace()

newLC.sampler = 'sincSampler'
sampledNewLC = newLC.sample()

lnPrior = newTask.logPrior(sampledNewLC)

print 'The log prior is %e'%(lnPrior)

lnLikelihood = newTask.logLikelihood(sampledNewLC)

print 'The log likelihood is %e'%(lnLikelihood)

lnPosterior = newTask.logPosterior(sampledNewLC)

print 'The log posterior is %e'%(lnPosterior)

startT = time.time()
newTask.fit(sampledNewLC)
stopT = time.time()

plt.figure(3)
plt.scatter(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps],
            newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps],
            c=maxLnPosterior - newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps],
            marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
plt.axhline(y=Theta[1], xmin=0.0, xmax=1.0)
plt.axvline(x=Theta[0], ymin=0.0, ymax=1.0)
plt.xlim([np.min(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[0, :, newTask.nsteps/2:newTask.nsteps])])
plt.ylim([np.min(newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[1, :, newTask.nsteps/2:newTask.nsteps])])
plt.xlabel(r'$a_{1}$')
plt.ylabel(r'$a_{2}$')

plt.figure(4)
plt.scatter(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps],
            newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps],
            c=maxLnPosterior - newTask.LnPosterior[:, newTask.nsteps/2:newTask.nsteps],
            marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
plt.axhline(y=Theta[3], xmin=0.0, xmax=1.0)
plt.axvline(x=Theta[2], ymin=0.0, ymax=1.0)
plt.xlim([np.min(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[2, :, newTask.nsteps/2:newTask.nsteps])])
plt.ylim([np.min(newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps]),
          np.max(newTask.Chain[3, :, newTask.nsteps/2:newTask.nsteps])])
plt.xlabel(r'$b_{0}$')
plt.ylabel(r'$b_{1}$')

plt.show(True)

if args.stop:
    pdb.set_trace()
