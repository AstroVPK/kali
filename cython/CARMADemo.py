import numpy as np
import math
import time
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import libcarma
import rand
import CARMATask

T = 500.0
dt = 0.1
numCadences = int(T/dt)
IR = False
tolIR = 1.0e-3
fracIntrinsicVar = 0.15
fracNoiseToSignal = 1.0e-1
maxSigma = 2.0
minTimescale = 5.0e-1
maxTimescale = 2.0

p = 3
q = 1

Theta = np.zeros(p + q + 1)
Theta[0] = 1.27
Theta[1] = 0.402
Theta[2] = 5.23e-3
Theta[3] = 7.0e-9
Theta[4] = 1.2e-9

newTask = libcarma.basicTask(p,q, nsteps = 5)

newTask.set(dt, Theta)

newLC = newTask.simulate(numCadences)

newTask.observe(newLC)

lnPrior = newTask.logPrior(newLC)

lnLikelihood = newTask.logLikelihood(newLC)

lnPosterior = newTask.logPosterior(newLC)

print 'The log prior is %e'%(lnPrior)

print 'The log likelihood is %e'%(lnLikelihood)

print 'The log posterior is %e'%(lnPosterior)

xStart = np.zeros(p + q + 1)
xStart[0] = 1.1
xStart[1] = 0.5
xStart[2] = 4.2e-3
xStart[3] = 6.0e-9
xStart[4] = 1.0e-9

newTask.fit(newLC, xStart)

plt.figure(1)
plt.plot(newLC.t, newLC.x, color = '#7570b3', zorder = 5, label = r'Intrinsic LC: $\ln \mathcal{L} = %+e$'%(lnLikelihood))
plt.errorbar(newLC.t, newLC.y, newLC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC: $\ln \mathcal{P} = %+e$'%(lnPosterior))
plt.legend()

plt.figure(2)
plt.scatter(newTask.Chain[0,:,newTask.nsteps/2:newTask.nsteps], newTask.Chain[1,:,newTask.nsteps/2:newTask.nsteps], c = np.max(newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps]) - newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)
plt.show()

pdb.set_trace()