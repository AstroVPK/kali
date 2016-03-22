import numpy as np
import math
import pdb
import matplotlib.pyplot as plt

import lc
import rand
import CARMATask

T = 1000
dt = 0.1
numCadences = int(T/dt)
IR = False
tolIR = 1.0e-3
fracIntrinsicVar = 0.15
fracSignalToNoise = 0.01
maxSigma = 1.0e2
minTimescale = 5.0e-1
maxTimescale = 5.0

r = lc.basicLC(numCadences, dt = dt, IR = IR, tolIR = tolIR, fracIntrinsicVar = fracIntrinsicVar, fracSignalToNoise = fracSignalToNoise, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale)

p = 3
q = 1

Theta = np.zeros(p + q + 1)
Theta[0] = 1.27
Theta[1] = 0.402
Theta[2] = 5.23e-3
Theta[3] = 7.0e-9
Theta[4] = 1.2e-9

newTask = CARMATask.CARMATask(p,q)

res = newTask.printSystem(dt, Theta)

Sigma = np.zeros(p*p)

res = newTask.getSigma(dt, Theta, Sigma)

print 'Sigma[0]: %e'%(np.sqrt(Sigma[0]))

randSeeds = np.zeros(3, dtype = 'uint32')

res= rand.rdrand(randSeeds)

res = newTask.makeIntrinsicLC(dt, Theta, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, r.t, r.x, r.y, r.yerr, r.mask, randSeeds[0], randSeeds[1])

'''res = newTask.makeIntrinsicLC2(Theta, r, r.t, r.x, r.y, r.yerr, r.mask, randSeeds[0], randSeeds[1])'''

meanFlux = newTask.getMeanFlux(dt, Theta, fracIntrinsicVar)

res = newTask.makeObservedLC(dt, Theta, numCadences, IR, tolIR, fracIntrinsicVar, fracSignalToNoise, r.t, r.x, r.y, r.yerr, r.mask, randSeeds[0], randSeeds[1], randSeeds[2])

maxSigma *= np.std(r.y)
minTimescale *= dt
maxTimescale *= T

LnPrior = newTask.computeLnPrior(dt, Theta, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, r.t, r.x, r.y, r.yerr, r.mask)

LnLikelihood = newTask.computeLnLikelihood(dt, Theta, numCadences, IR, tolIR, r.t, r.x, r.y, r.yerr, r.mask)

LnPosterior = newTask.computeLnPosterior(dt, Theta, numCadences, IR, tolIR, maxSigma, minTimescale, maxTimescale, r.t, r.x, r.y, r.yerr, r.mask)

plt.figure(1)
plt.plot(r.t, r.x + meanFlux, color = '#7570b3', zorder = 5, label = r'Intrinsic LC: $\ln \mathcal{L} = %+e$'%(LnLikelihood))
plt.errorbar(r.t, r.y + meanFlux, r.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10, label = r'Observed LC: $\ln \mathcal{P} = %+e$'%(LnPosterior))
plt.legend()
plt.show()