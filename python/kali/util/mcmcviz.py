import math as math
import cmath as cmath
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pdb

import kali.util.triangle as triangle

fwid = 16.0
fhgt = 10.0

# Data manipulation:


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    # Taken from a post by dpsanders at
    # http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3,
              alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Taken from a post by dpsanders at
    # http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def clear_frame(ax=None):
    # Taken from a post by Tony S Yu
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.itervalues():
        spine.set_visible(False)


def vizWalkers(Chain, LogPosterior, dim1, dim1Name, dim2, dim2Name):
    ndim = Chain.shape[0]
    nwalkers = Chain.shape[1]
    nsteps = Chain.shape[2]

    def init():
        for walkerNum in xrange(nwalkers):
            lineList[walkerNum].set_data([], [])
        step_text.set_text('stepNum: ')
        return lineList + [step_text]

    def animate(stepNum):
        for walkerNum in xrange(nwalkers):
            colorVal = scalarMap.to_rgba(LogPosterior[walkerNum, stepNum])
            lineList[walkerNum].set_data([Chain[dim1, walkerNum, stepNum], 0.0],
                                         [Chain[dim2, walkerNum, stepNum], 0.0])
            lineList[walkerNum].set_color(colorVal)
        step_text.set_text('stepNum: %d'%(stepNum))
        return lineList + [step_text]

    fig = plt.figure(1, figsize=(fhgt, fhgt))
    ax = fig.add_subplot(111)
    plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 0))
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.set_xlim(np.min(Chain[dim1, :, :]), np.max(Chain[dim1, :, :]))
    ax.set_ylim(np.min(Chain[dim2, :, :]), np.max(Chain[dim2, :, :]))
    ax.set_xlabel(dim1Name)
    ax.set_ylabel(dim2Name)
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    jet = cmx = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=np.nanmin(LogPosterior[:, :]), vmax=np.nanmax(LogPosterior[:, :]))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    lineList = list()
    for walkerNum in xrange(nwalkers):
        line, = ax.plot([], [], 'o', ms=10, color='#000000')
        lineList.append(line)
    plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nsteps, interval=41.6666666664,
                                   blit=True)

    # anim.save('/home/vish/Desktop/basicAnim.mp4', writer = 'mencoder', fps = 24,
    # extra_args = ['-vcodec', 'libx264'])

    plt.show()
    return 0


def vizTriangle(p, q, Chain, labelList, figTitle, doShow=False):
    ndims = Chain.shape[0]
    nwalkers = Chain.shape[1]
    nsteps = Chain.shape[2]

    flatChain = np.swapaxes(copy.copy(Chain[:, :, nsteps/2:]).reshape((ndims, -1), order='F'), axis1=0,
                            axis2=1)
    fig0, quantiles, qvalues = triangle.corner(flatChain, labels=labelList, fig_size=10.0, show_titles=True,
                                               fig_title=figTitle, title_args={'fontsize': 12},
                                               quantiles=[0.16, 0.5, 0.84], verbose=False,
                                               plot_contours=False, plot_datapoints=True,
                                               plot_contour_lines=False, pcolor_cmap=cm.gist_earth)
    if doShow:
        plt.show()
