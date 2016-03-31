import numpy as np
import math
import time
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import libcarma
import rand
import CARMATask

T = 3500.0
dt = 0.1
numCadences = int(T/dt)
IR = False
tolIR = 1.0e-3
fracIntrinsicVar = 0.15
fracSignalToNoise = 1.0e-3
maxSigma = 2.0
#maxSigma = 1.0e2
minTimescale = 5.0e-1
maxTimescale = 2.0

r = libcarma.basicLC(numCadences, dt = dt, IR = IR, tolIR = tolIR, fracIntrinsicVar = fracIntrinsicVar, fracSignalToNoise = fracSignalToNoise, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale)

p = 3
q = 1

Theta = np.zeros(p + q + 1)
Theta[0] = 1.27
Theta[1] = 0.402
Theta[2] = 5.23e-3
Theta[3] = 7.0e-9
Theta[4] = 1.2e-9

newTask = CARMATask.CARMATask(p,q)

res = newTask.print_System(dt, Theta)

Sigma = np.zeros(p*p)

res = newTask.get_Sigma(dt, Theta, Sigma)

print '   Sigma[0]: %e'%(np.sqrt(Sigma[0]))

randSeeds = np.zeros(3, dtype = 'uint32')

res= rand.rdrand(randSeeds)

res = newTask.make_IntrinsicLC(dt, Theta, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, r.t, r.x, r.y, r.yerr, r.mask, randSeeds[0], randSeeds[1])

print 'LC Std Dev: %e'%(np.std(r.x))

meanFlux = newTask.get_meanFlux(dt, Theta, fracIntrinsicVar)

res = newTask.make_ObservedLC(dt, Theta, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, r.t, r.x, r.y, r.yerr, r.mask, randSeeds[0], randSeeds[1], randSeeds[2])

maxSigma *= np.std(r.y)
minTimescale *= dt
maxTimescale *= T

LnPrior = newTask.compute_LnPrior(dt, Theta, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, r.t, r.x, r.y, r.yerr, r.mask)

LnLikelihood = newTask.compute_LnLikelihood(dt, Theta, numCadences, IR, tolIR, r.t, r.x, r.y, r.yerr, r.mask)

LnPosterior = newTask.compute_LnPosterior(dt, Theta, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, r.t, r.x, r.y, r.yerr, r.mask)

plt.figure(1)
plt.plot(r.t, r.x + meanFlux, color = '#7570b3', zorder = 5, label = r'Intrinsic LC: $\ln \mathcal{L} = %+e$'%(LnLikelihood))
plt.errorbar(r.t, r.y + meanFlux, r.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC: $\ln \mathcal{P} = %+e$'%(LnPosterior))
plt.legend()

scatterFactor = 1.0e-1
nwalkers = 160
nsteps = 250
maxEvals = 1000
xTol = 0.005

mcmcSeeds = np.zeros(4, dtype = 'uint32')

res = rand.rdrand(mcmcSeeds)

xStart = np.zeros(p + q + 1)

xStart[0] = 1.4
xStart[1] = 0.5
xStart[2] = 4.75e-3
xStart[3] = 6.5e-9
xStart[4] = 1.0e-9

Chain = np.zeros((p + q + 1)*nwalkers*nsteps)
LnPosterior = np.zeros(nwalkers*nsteps)

print "Starting C-ARMA Model fit."

fitStart = time.time()
res = newTask.fit_CARMAModel(dt, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, r.t, r.x, r.y, r.yerr, r.mask, scatterFactor, nwalkers, nsteps, maxEvals, xTol, mcmcSeeds[0], mcmcSeeds[1], mcmcSeeds[2], mcmcSeeds[3], xStart, Chain, LnPosterior)
fitStop = time.time()

totalTime = fitStop - fitStart
print "Fitting C-ARMA Model took %f sec = %f min = %f hrs."%(totalTime, totalTime/60.0, totalTime/3600.0)

Chain_view = np.reshape(Chain, newshape = (p + q + 1, nwalkers, nsteps), order = 'F')
LnPosterior_view = np.reshape(LnPosterior, newshape = (nwalkers, nsteps), order = 'F')

plt.figure(2)
plt.scatter(Chain_view[0,:,nsteps/2:nsteps], Chain_view[1,:,nsteps/2:nsteps], c = np.max(LnPosterior_view[:,nsteps/2:nsteps]) - LnPosterior_view[:,nsteps/2:nsteps], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)
plt.show()

pdb.set_trace()