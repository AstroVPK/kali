import numpy as np
import math as math
import time as time
import pdb as pdb
import sys as sys
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import argparse as argparse
import sys as sys

import libcarma as libcarma

parser = argparse.ArgumentParser()
parser.add_argument("-pdb", "--pdb", default = False, help = "Enable pdb breakpoint at end?")
args = parser.parse_args()

tolIR = 1.0e-3
fracIntrinsicVar = 0.1
fracNoiseToSignal = 1.0e-18
maxSigma = 2.0
minTimescale = 2.0
maxTimescale = 0.5

p = 3
q = 1
Rho = np.array([-(1.0/12.0)+0j, -(1.0/7.50)+0j, -(1.0/30.0)+0j, -(1.0/5.0), 1.0e-9])
print "Rho: " + str(Rho)

Tau = libcarma.timescales(p, q, Rho)
print "Tau: " + str(Tau)

Theta = libcarma.coeffs(p, q, Rho)
print "Theta: " + str(Theta)

dt = 0.5
T = 1200.0

newTask = libcarma.basicTask(p, q, nwalkers = 160, nsteps = 250)

newTask.set(dt, Theta)

print "Sigma[0]: %e"%(math.sqrt(newTask.Sigma()[0,0]))

Lags, ACVF = newTask.acvf(1.0e-4, 1.0e4, 10000, spacing = 'log')

plt.figure(1)

plt.plot(np.log10(Lags), ACVF)

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

print "Time taken: %+4.3e sec; %+4.3e min; %+4.3e hrs"%((stopT - startT), (stopT - startT)/60.0, (stopT - startT)/3600.0)

plt.figure(2)
plt.plot(newLC.t, newLC.x, color = '#7570b3', zorder = 5, label = r'Intrinsic LC: $\ln \mathcal{L} = %+e$'%(lnLikelihood))
plt.errorbar(newLC.t, newLC.y, newLC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC: $\ln \mathcal{P} = %+e$'%(lnPosterior))
plt.legend()

plt.figure(3)
plt.scatter(newTask.Chain[0,:,newTask.nsteps/2:newTask.nsteps], newTask.Chain[1,:,newTask.nsteps/2:newTask.nsteps], c = np.max(newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps]) - newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)

plt.show()

if args.pdb:
	pdb.set_trace()