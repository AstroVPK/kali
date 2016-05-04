import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import pdb

import libcarma as libcarma
import sdss as sdss
from util.mpl_settings import set_plot_params
import util.triangle as triangle

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

P = 2
Q = 1
NSTEPS = 500

def timescales(p, q, Rho):
	imagPairs = 0
	for i in xrange(p):
		if Rho[i].imag != 0.0:
			imagPairs += 1
	numImag = imagPairs/2
	numReal = numImag + (p - imagPairs)
	decayTimescales = np.zeros(numReal)
	oscTimescales = np.zeros(numImag)
	realRoots = set(Rho[0:p].real)
	imagRoots = set(abs(Rho[0:p].imag)).difference(Set([0.0]))
	realAR = np.array([1.0/math.abs(x) for x in realRoots])
	imagAR = np.array([(2.0*math.pi)/math.abs(x) for x in imagRoots])
	imagPairs = 0
	for i in xrange(q):
		if Rho[i].imag != 0.0:
			imagPairs += 1
	numImag = imagPairs/2
	numReal = numImag + (q - imagPairs)
	decayTimescales = np.zeros(numReal)
	oscTimescales = np.zeros(numImag)
	realRoots = set(Rho[p:p + q].real)
	imagRoots = set(abs(Rho[p:p + q].imag)).difference(Set([0.0]))
	realMA = np.array([1.0/math.abs(x) for x in realRoots])
	imagMA = np.array([(2.0*math.pi)/math.abs(x) for x in imagRoots])
	return realAR, imagAR, realMA, imagMA

sdss0g = sdss.sdss_gLC(supplied = 'LightCurveSDSS_1.csv', pwd = '/home/vish/Desktop')
sdss0r = sdss.sdss_rLC(supplied = 'LightCurveSDSS_1.csv', pwd = '/home/vish/Desktop')

plt.figure(1, figsize = (fwid, fhgt))
plt.errorbar(sdss0g.t - sdss0g.startT, sdss0g.y, sdss0g.yerr, label = r'sdss-g', fmt = '.', capsize = 0, color = '#2ca25f', markeredgecolor = 'none', zorder = 10)
plt.errorbar(sdss0r.t - sdss0r.startT, sdss0r.y, sdss0r.yerr, label = r'sdss-r', fmt = '.', capsize = 0, color = '#feb24c', markeredgecolor = 'none', zorder = 10)
plt.xlabel(sdss0g.xunit)
plt.ylabel(sdss0g.yunit)
plt.legend()

Theta = np.array([0.725, 0.01, 7.0e-7, 1.2e-7])

ntg = libcarma.basicTask(P, Q, nsteps = NSTEPS)
ntr = libcarma.basicTask(P, Q, nsteps = NSTEPS)
ntg.set(sdss0g.dt, Theta)
ntr.set(sdss0r.dt, Theta)
ntg.fit(sdss0g, Theta)
ntr.fit(sdss0r, Theta)

fig2 = plt.figure(2, figsize = (fhgt*1.25, 2.25*fhgt))
gs = gridspec.GridSpec(225, 100)
ax1 = fig2.add_subplot(gs[0:99, :])
scatPlot1 = ax1.scatter(ntg.Chain[0,:,NSTEPS/2:], ntg.Chain[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
ax1.set_xlim(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(ntg.Chain[0,:,NSTEPS/2:]))
ax1.set_ylim(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(ntg.Chain[1,:,NSTEPS/2:]))
ax1.set_xlabel(r'$a_{1}$')
ax1.set_ylabel(r'$a_{2}$')
ax2 = fig2.add_subplot(gs[125:224, :])
scatPlot2 = ax2.scatter(ntg.Chain[2,:,NSTEPS/2:], ntg.Chain[3,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
ax2.set_xlim(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(ntg.Chain[2,:,NSTEPS/2:]))
ax2.set_ylim(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(ntg.Chain[3,:,NSTEPS/2:]))
ax2.set_xlabel(r'$b_{0}$')
ax2.set_ylabel(r'$b_{1}$')
cBar = plt.colorbar(scatPlot1, ax = [ax1, ax2], orientation = 'horizontal')#, ticks = cBarTicks, format = r'$\scriptstyle %2.1f$')
cBar.set_label(r'$\ln \mathcal{P}$')

fig3 = plt.figure(3, figsize = (fhgt*1.25, 2.25*fhgt))
gs = gridspec.GridSpec(225, 100)
ax1 = fig3.add_subplot(gs[0:99, :])
scatPlot1 = ax1.scatter(ntr.Chain[0,:,NSTEPS/2:], ntr.Chain[1,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
ax1.set_xlim(np.nanmin(ntr.Chain[0,:,NSTEPS/2:]), np.nanmax(ntr.Chain[0,:,NSTEPS/2:]))
ax1.set_ylim(np.nanmin(ntr.Chain[1,:,NSTEPS/2:]), np.nanmax(ntr.Chain[1,:,NSTEPS/2:]))
ax1.set_xlabel(r'$a_{1}$')
ax1.set_ylabel(r'$a_{2}$')
ax2 = fig3.add_subplot(gs[125:224, :])
scatPlot2 = ax2.scatter(ntr.Chain[2,:,NSTEPS/2:], ntr.Chain[3,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
ax2.set_xlim(np.nanmin(ntr.Chain[2,:,NSTEPS/2:]), np.nanmax(ntr.Chain[2,:,NSTEPS/2:]))
ax2.set_ylim(np.nanmin(ntr.Chain[3,:,NSTEPS/2:]), np.nanmax(ntr.Chain[3,:,NSTEPS/2:]))
ax2.set_xlabel(r'$b_{0}$')
ax2.set_ylabel(r'$b_{1}$')
cBar = plt.colorbar(scatPlot1, ax = [ax1, ax2], orientation = 'horizontal')#, ticks = cBarTicks, format = r'$\scriptstyle %2.1f$')
cBar.set_label(r'$\ln \mathcal{P}$')

plt.show()
pdb.set_trace()