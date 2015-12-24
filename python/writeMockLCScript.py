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

from python.lc import LC
from python.writeMockLC import writeMockLCTask
import python.util.triangle as triangle
from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendMedium']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')
gs = gridspec.GridSpec(1000, 2250) 

newTask = writeMockLCTask()
newTask.parseConfig()
newTask.run()

fig1 = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fhgt']))
ax1 = fig1.add_subplot(gs[:,:])
ax1.ticklabel_format(useOffset = False)
#ax1.plot(newTask.LC.t, newTask.LC.y, color = '#7570b3', zorder = 5)
ax1.errorbar(newTask.LC.t, newTask.LC.y, newTask.LC.yerr, fmt = '.', capsize = 0, color = '#d95f02', markeredgecolor = 'none', zorder = 10)
yMax=np.max(newTask.LC.y[np.nonzero(newTask.LC.y[:])])
yMin=np.min(newTask.LC.y[np.nonzero(newTask.LC.y[:])])
ax1.set_ylabel(r'$F$ (arb units)')
ax1.set_xlabel(r'$t$ (d)')
ax1.set_xlim(newTask.LC.t[0],newTask.LC.t[-1])
ax1.set_ylim(yMin,yMax)

plt.show()
#pdb.set_trace()