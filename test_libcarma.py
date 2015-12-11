import math as math
import cmath as cmath
import numpy as np
import random as random
import cffi as cffi
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab
import time
import pdb

from _libcarma import ffi
from mpl_settings import *

goldenRatio=1.61803398875
fhgt=10.0
fwid=fhgt*goldenRatio
dpi = 300

AnnotateXXLarge = 72
AnnotateXLarge = 48
AnnotateLarge = 32
AnnotateMedium = 28
AnnotateSmall = 24
AnnotateXSmall = 20
AnnotateXXSmall = 16

LegendLarge = 24
LegendMedium = 20
LegendSmall = 16

LabelXLarge = 32
LabelLarge = 28
LabelMedium = 24
LabelSmall = 20
LabelXSmall = 16

AxisXXLarge = 32
AxisXLarge = 28
AxisLarge = 24
AxisMedium = 20
AxisSmall = 16
AxisXSmall = 12
AxisXXSmall = 8

normalFontSize=32
smallFontSize=24
footnoteFontSize=20
scriptFontSize=16
tinyFontSize=12

LabelSize = LabelXLarge
AxisSize = AxisLarge
AnnotateSize = AnnotateXLarge
AnnotateSizeAlt = AnnotateMedium
AnnotateSizeAltAlt = AnnotateLarge
LegendSize = LegendMedium

gs = gridspec.GridSpec(1000, 1000) 

set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=AxisMedium,useTex='True')

ffiObj = cffi.FFI()
C = ffi.dlopen("./libcarma.so")

dt = 0.02
p = 2
q = 1
ndims = p + q + 1
Theta = [0.75, 0.01, 7.0e-9, 1.2e-9]
numBurn = 1000000
numCadences = 60000
noiseSigma = 1.0e-18
startCadence = 0
burnSeed = 1311890535
distSeed = 2603023340
noiseSeed = 2410288857
cadence = np.array(numCadences*[0])
mask = np.array(numCadences*[0.0])
t = np.array(numCadences*[0.0])
y = np.array(numCadences*[0.0])
yerr = np.array(numCadences*[0.0])
nthreads = 8
nwalkers = 100
nsteps = 1000
maxEvals = 1000
xTol = 0.005
zSSeed = 2229588325
walkerSeed = 3767076656
moveSeed = 2867335446
xSeed = 1413995162
initSeed = 3684614774
Chain = np.zeros((nsteps,nwalkers,ndims))
LnLike = np.zeros((nsteps,nwalkers))

Theta_cffi = ffiObj.new("double[%d]"%(len(Theta)))
for i in xrange(len(Theta)):
	Theta_cffi[i] = Theta[i]
cadence_cffi = ffiObj.new("int[%d]"%(numCadences))
mask_cffi = ffiObj.new("double[%d]"%(numCadences))
t_cffi = ffiObj.new("double[%d]"%(numCadences))
y_cffi = ffiObj.new("double[%d]"%(numCadences))
yerr_cffi = ffiObj.new("double[%d]"%(numCadences))
for i in xrange(numCadences):
	cadence_cffi[i] = i
	mask_cffi[i] = 1.0
	t_cffi[i] = dt*i
	y_cffi[i] = 0.0
	yerr_cffi[i] = 0.0
Chain_cffi = ffiObj.new("double[%d]"%(ndims*nwalkers*nsteps))
for i in xrange(ndims*nwalkers*nsteps):
	Chain_cffi[i] = 0.0
LnLike_cffi = ffiObj.new("double[%d]"%(nwalkers*nsteps))
for i in xrange(nwalkers*nsteps):
	LnLike_cffi[i] = 0.0

makeLCStart = time.time()

LnLikeVal = C._makeMockLC(dt, p, q, Theta_cffi, numBurn, numCadences, noiseSigma, startCadence, burnSeed, distSeed, noiseSeed, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)

makeLCStop = time.time()

print "Time to make LC & compute LnLike: %f (s)"%(makeLCStop - makeLCStart)

computeLnLikeStart = time.time()

LnLikeValOther = C._computeLnLike(dt, p, q, Theta_cffi, numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi)

computeLnLikeStop = time.time()

print "LnLike: %+16.17e"%(LnLikeVal)

print "Time to compute LnLike: %f (s)"%(computeLnLikeStop - computeLnLikeStart)

for i in xrange(numCadences):
	cadence[i] = cadence_cffi[i]
	mask[i] = mask_cffi[i]
	t[i] = t_cffi[i]
	y[i] = y_cffi[i]
	yerr[i] = yerr_cffi[i]

fitCARMAStart = time.time()

C._fitCARMA(dt, p, q, numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi, nthreads, nwalkers, nsteps, maxEvals, xTol, zSSeed, walkerSeed, moveSeed, xSeed, initSeed, Chain_cffi, LnLike_cffi)

fitCARMAStop = time.time()

print "Time to fit C-ARMA model: %f (s)"%(fitCARMAStop - fitCARMAStart)

for stepNum in xrange(nsteps):
	for walkerNum in xrange(nwalkers):
		LnLike[stepNum, walkerNum] = LnLike_cffi[walkerNum + stepNum*nwalkers]
		for dimNum in xrange(ndims):
			Chain[stepNum, walkerNum, dimNum] = Chain_cffi[dimNum + walkerNum*ndims + stepNum*ndims*nwalkers]

plt.figure(1,figsize=(fwid,fhgt))
plt.errorbar(t,y,yerr,fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',zorder=10)
yMax=np.max(y[np.nonzero(y[:])])
yMin=np.min(y[np.nonzero(y[:])])
plt.ylabel(r'$F$ (arb units)')
plt.xlabel(r'$t$ (d)')
plt.xlim(t[0],t[-1])
plt.ylim(yMin,yMax)

plt.figure(2,figsize=(fwid,fhgt))
plt.scatter(Chain[int(nsteps/2.0):,:,0], Chain[int(nsteps/2.0):,:,1], c = np.max(LnLike[int(nsteps/2.0):,:]) - LnLike[int(nsteps/2.0):,:], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)
plt.xlim(np.min(Chain[int(nsteps/2.0):,:,0]),np.max(Chain[int(nsteps/2.0):,:,0]))
plt.xlim(np.min(Chain[int(nsteps/2.0):,:,1]),np.max(Chain[int(nsteps/2.0):,:,1]))
plt.xlabel(r'$a_{1}$')
plt.ylabel(r'$a_{2}$')

plt.figure(3,figsize=(fwid,fhgt))
plt.scatter(Chain[int(nsteps/2.0):,:,2], Chain[int(nsteps/2.0):,:,3], c = np.max(LnLike[int(nsteps/2.0):,:]) - LnLike[int(nsteps/2.0):,:], marker='.', cmap = colormap.gist_rainbow_r, linewidth = 0)
plt.xlim(np.min(Chain[int(nsteps/2.0):,:,2]),np.max(Chain[int(nsteps/2.0):,:,2]))
plt.xlim(np.min(Chain[int(nsteps/2.0):,:,3]),np.max(Chain[int(nsteps/2.0):,:,3]))
plt.xlabel(r'$b_{0}$')
plt.ylabel(r'$b_{1}$')

plt.show()

pdb.set_trace()