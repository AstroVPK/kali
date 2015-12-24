import math as math
import cmath as cmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec, cm
import matplotlib.cm as colormap
import matplotlib.mlab as mlab
import pdb

from python.writeMockLC import writeMockLCTask
import python.util.triangle as triangle
from python.util.mpl_settings import *

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

gs = gridspec.GridSpec(1000, 2250) 

set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=AxisMedium,useTex='True')

newTask = writeMockLCTask()
newTask.parseConfig()
newTask.run()

fig1 = plt.figure(1, figsize=(fwid,fhgt))
ax1 = fig1.add_subplot(gs[:,:])
ax1.ticklabel_format(useOffset = False)
ax1.plot(newTask.t, newTask.y, color = '#7570b3', zorder = 5)
#ax1.errorbar(t, y, yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
yMax=np.max(newTask.y[np.nonzero(newTask.y[:])])
yMin=np.min(newTask.y[np.nonzero(newTask.y[:])])
ax1.set_ylabel(r'$F$ (arb units)')
ax1.set_xlabel(r'$t$ (d)')
ax1.set_xlim(newTask.t[0],newTask.t[-1])
ax1.set_ylim(yMin,yMax)

plt.show()
#pdb.set_trace()