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
fracIntrinsicVar = 0.1
fracNoiseToSignal = 1.0e-18
maxSigma = 2.0
minTimescale = 5.0e-1
maxTimescale = 2.0

p = 3
q = 1

Theta = np.zeros(p + q + 1)

r_1 = -0.73642081+0j
r_2 = -0.01357919+0j
r_3 = -0.52329875+0j
sigma = 7.0e-9
m_1 = -5.83333333

ARPoly = np.poly([r_1, r_2, r_3])
MAPoly = sigma/np.poly([m_1])

Theta[0] = ARPoly[1]
Theta[1] = ARPoly[2]
Theta[2] = ARPoly[3]
Theta[3] = MAPoly[0]
Theta[4] = MAPoly[1]

print "Theta: " + str(Theta)

newTask = libcarma.basicTask(p,q)#, nwalkers = 20, nsteps = 5)

newTask.set(dt, Theta)

print "Sigma[0]: %e"%(math.sqrt(newTask.Sigma()[0,0]))

newLC = newTask.simulate(numCadences)

newTask.observe(newLC)

lnPrior = newTask.logPrior(newLC)

lnLikelihood = newTask.logLikelihood(newLC)

lnPosterior = newTask.logPosterior(newLC)

print 'The log prior is %e'%(lnPrior)

print 'The log likelihood is %e'%(lnLikelihood)

print 'The log posterior is %e'%(lnPosterior)

xStart = np.zeros(p + q + 1) # This must be guessed but things are not too sensitive to the guess.

r_1_guess = -0.75+0j
r_2_guess = -0.01+0j
r_3_guess = -0.5+0j
sigma_guess = np.std(newLC.y)
m_1_guess = -5.8

ARPoly = np.poly([r_1_guess, r_2_guess, r_3_guess])
MAPoly = sigma_guess/np.poly([m_1_guess])

xStart[0] = ARPoly[1]
xStart[1] = ARPoly[2]
xStart[2] = ARPoly[3]
xStart[3] = MAPoly[0]
xStart[4] = MAPoly[1]

newTask.fit(newLC, xStart)

plt.figure(1)
plt.plot(newLC.t, newLC.x, color = '#7570b3', zorder = 5, label = r'Intrinsic LC: $\ln \mathcal{L} = %+e$'%(lnLikelihood))
plt.errorbar(newLC.t, newLC.y, newLC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC: $\ln \mathcal{P} = %+e$'%(lnPosterior))
plt.legend()

plt.figure(2)
plt.scatter(newTask.Chain[0,:,newTask.nsteps/2:newTask.nsteps], newTask.Chain[1,:,newTask.nsteps/2:newTask.nsteps], c = np.max(newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps]) - newTask.LnPosterior[:,newTask.nsteps/2:newTask.nsteps], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)

plt.show()

pdb.set_trace()