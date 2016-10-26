import math
import numpy as np
import copy
import random
import psutil
import os
import sys
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import brewer2mpl

try:
    import kali.mbhbcarma
except ImportError:
    print 'Cannot import kali.mbhbcarma! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.carma
except ImportError:
    print 'Cannot import kali.carma! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

plt.ion()

BREAK = True
MULT = 50.0
BURNSEED = 731647386
DISTSEED = 219038190
NOISESEED = 87238923
SAMPLESEED = 36516342
ZSSEED = 384789247
WALKERSEED = 738472981
MOVESEED = 131343786
XSEED = 2348713647

EarthMass = 3.0025138e-12  # 10^6 MSun
SunMass = 1.0e-6  # 10^6 MSun
EarthOrbitRadius = 4.84814e-6  # AU
SunOrbitRadius = 4.84814e-6*(EarthMass/SunMass)  # AU
Period = 31557600.0/86164.090530833  # Day
EarthOrbitEccentricity = 0.0167
G = 6.67408e-11
c = 299792458.0
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
twoPi = 2.0*pi
Parsec = 3.0857e16
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day
SolarMass = 1.98855e30

DivergingList = ['BrBG', 'PRGn', 'PiYG', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral']

BASEPATH = '/home/vish/Documents/Research/MBHBCARMA/'

p = 1
q = 0
r = kali.mbhbcarma.MBHBCARMATask(p, q).r
PERIOD = 4.0*DayInYear
TAU = 0.4*DayInYear
dt = PERIOD/100.0
duration = PERIOD*100.0
N2S = 1.0e-18
NWALKERS = 2
NSTEPS = 10
rho_carma = np.array([-1.0/TAU, 1.0])
theta_carma = kali.carma.coeffs(p, q, rho_carma)
newTask_carma = kali.carma.CARMATask(p, q, nwalkers=NWALKERS, nsteps=NSTEPS)
res_carma = newTask_carma.set(dt, theta_carma)
newLC_carma = newTask_carma.simulate(duration=duration, fracNoiseToSignal=N2S, burnSeed=BURNSEED,
                                     distSeed=DISTSEED, noiseSeed=NOISESEED)
newTask_carma.observe(newLC_carma, noiseSeed=NOISESEED)
lnprior_carma = newTask_carma.logPrior(newLC_carma)
lnlikelihood_carma = newTask_carma.logLikelihood(newLC_carma)
lnposterior_carma = newTask_carma.logPosterior(newLC_carma)
print "     LnPrior (CARMA): %+e"%(lnprior_carma)
print "LnLikelihood (CARMA): %+e"%(lnlikelihood_carma)
print " LnPosterior (CARMA): %+e"%(lnposterior_carma)

theta_mbhbcarma = np.array([0.01, 0.02, PERIOD, 0.5, 30.0, 90.0, 10.0, newLC_carma.mean,
                            theta_carma[0], theta_carma[1]])
newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(p, q, nwalkers=NWALKERS, nsteps=NSTEPS)
res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
newLC_mbhbcarma = newTask_mbhbcarma.simulate(duration=duration, fracNoiseToSignal=N2S,
                                             burnSeed=BURNSEED, distSeed=DISTSEED, noiseSeed=NOISESEED)
newTask_mbhbcarma.observe(newLC_mbhbcarma, noiseSeed=NOISESEED)
lnprior_mbhbcarma = newTask_mbhbcarma.logPrior(newLC_mbhbcarma)
lnlikelihood_mbhbcarma = newTask_mbhbcarma.logLikelihood(newLC_mbhbcarma)
lnposterior_mbhbcarma = newTask_mbhbcarma.logPosterior(newLC_mbhbcarma)
print "     LnPrior (MBHBCARMA): %+e"%(lnprior_mbhbcarma)
print "LnLikelihood (MBHBCARMA): %+e"%(lnlikelihood_mbhbcarma)
print " LnPosterior (MBHBCARMA): %+e"%(lnposterior_mbhbcarma)
# pdb.set_trace()
