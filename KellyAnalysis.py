import math as math
import cmath as cmath
import numpy as np
import socket
HOST = socket.gethostname()
if HOST == 'dirac.physics.drexel.edu':
	import matplotlib
	matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab
import cPickle as cPickle
import pdb

from mpl_settings import *
import carmcmc as cmcmc

goldenRatio=1.61803398875
fhgt=10.0
fwid=fhgt*goldenRatio

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

pickledSamplePath = sys.argv[1]

numWalkers = 100
numSteps = 500

noiseLvl = 1.0e-16

sigmay = 1.2e-9

#ar_coefs = np.array([1.0,0.75,0.01])
#ar_roots = np.polynomial.polynomial.polyroots(ar_coefs)
ar_roots = np.array([-0.73642081+0j, -0.01357919+0j])

ma_coefs = np.array([1.0, 5.834])

sigsqr = sigmay ** 2 / cmcmc.carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)  # carma_variance computes the autcovariance function

dt = 0.02
T = 100.0
numCadences = int(T/dt)

t = np.arange(0.0, T, dt)

y = cmcmc.carma_process(t, sigsqr, ar_roots, ma_coefs=ma_coefs)
noise = np.random.normal(loc=0.0, scale=noiseLvl, size=numCadences)
y += noise

yerr = np.array(numCadences*[noiseLvl])

CARMA_Model = cmcmc.CarmaModel(t, y, yerr, p=2, q=1)
sample = CARMA_Model.run_mcmc(numWalkers*numSteps)

Sample = open(pickledSamplePath, 'wb')
cPickle.dump(sample, Sample, -1)
Sample.close()

'''fig1 = plt.figure(1, figsize=(fwid, fhgt))
ax1 = fig1.add_subplot(gs[:,:])
ax1.ticklabel_format(useOffset = False)
ax1.plot(times, y)
ax1.set_xlabel(r'$t$ (d)')
ax1.set_ylabel(r'Flux')

plt.show()'''
