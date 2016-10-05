#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner", "hist2d", "error_ellipse"]
__version__ = "0.1.1"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
    "Adrian Price-Whelan @adrn",
    "Brendon Brewer @eggplantbren",
    "Ekta Patel @ekta1224",
    "Emily Rice @emilurice",
    "Geoff Ryan @geoffryan",
    "Kyle Barbary @kbarbary",
    "Phil Marshall @drphilmarshall",
    "Pierre Gratier @pirg",
]

import math as math
import cmath as cmath
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.cm as cm
import pdb


def corner(xs, weights=None, labels=None, show_titles=False, title_fmt=".2f",
           title_args={}, extents=None, truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=[], verbose=True,
           plot_contours=True, plot_datapoints=True, plot_contour_lines=True, fig=None, pcolor_cmap=None,
           **kwargs):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like (nsamples, ndim)
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    weights : array_like (nsamples,)
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    labels : iterable (ndim,) (optional)
        A list of names for the dimensions.

    show_titles : bool (optional)
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string (optional)
        The format string for the quantiles given in titles.
        (default: `.2f`)

    title_args : dict (optional)
        Any extra keyword arguments to send to the `add_title` command.

    extents : iterable (ndim,) (optional)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds (extents) or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,) (optional)
        A list of reference values to indicate on the plots.

    truth_color : str (optional)
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool (optional)
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable (optional)
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool (optional)
        If true, print the values of the computed quantiles.

    plot_contours : bool (optional)
        Draw contours for dense regions of the plot.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    fig : matplotlib.Figure (optional)
        Overplot onto the provided figure object.

    """
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    # make border thicker
    mpl.rcParams['axes.linewidth'] = 1
    # make plotted lines thicker
    mpl.rcParams['lines.linewidth'] = 1
    # make fonts bigger
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.style'] = 'normal'  # 'normal', 'italic','oblique'
    mpl.rcParams['font.variant'] = 'normal'  # 'normal', 'small-caps'
    mpl.rcParams[
        'font.weight'] = 'normal'  # 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
    mpl.rcParams[
        'font.stretch'] = 'normal'  # ‘ultra-condensed’, ‘extra-condensed’, ‘condensed’, ‘semi-condensed’, ‘normal’, ‘semi-expanded’, ‘expanded’, ‘extra-expanded’, ‘ultra-expanded’
    mpl.rcParams[
        'font.size'] = 16  # ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    mpl.rcParams[
        'font.serif'] = ['Times', 'Times New Roman', 'Palatino', 'Bitstream Vera Serif', 'New Century Schoolbook',
                         'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Charter', 'serif']
    mpl.rcParams['font.sans-serif'] = ['Bitstream Vera Sans', 'Lucida Grande',
                                       'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
    mpl.rcParams['font.cursive'] = ['Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'cursive']
    mpl.rcParams['font.fantasy'] = ['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'fantasy']
    mpl.rcParams['font.monospace'] = ['Bitstream Vera Sans Mono', 'Andale Mono',
                                      'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace']
    mpl.rcParams['text.usetex'] = False
    # set math mode font properties
    mpl.rcParams['mathtext.cal'] = 'cursive'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['mathtext.tt'] = 'monospace'
    mpl.rcParams['mathtext.it'] = 'serif:italic'
    mpl.rcParams['mathtext.bf'] = 'serif:bold'
    mpl.rcParams['mathtext.sf'] = 'sans'
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Should be 'cm' (Computer Modern), 'stix','stixsans' or 'custom'
    mpl.rcParams['mathtext.fallback_to_cm'] = True  # When True, use symbols from the Computer Modern
                                 # fonts when a symbol can not be found in one of
                                 # the custom math fonts.

    mpl.rcParams['mathtext.default'] = 'rm'  # The default font to use for math.
                       # Can be any of the LaTeX font names, including
                       # the special name "regular" for the same font
                       # used in regular text.
    mpl.rcParams['pdf.fonttype'] = 42  # Force matplotlib to use Type42 (a.k.a. TrueType) fonts for .pdf
    mpl.rcParams['ps.fonttype'] = 42  # Force matplotlib to use Type42 (a.k.a. TrueType) fonts for .eps

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError('weights must be 1-D')
        if xs.shape[1] != weights.shape[0]:
            raise ValueError('lengths of weights must match number of samples')

    # backwards-compatibility
    plot_contours = kwargs.get("smooth", plot_contours)

    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    fig_size = kwargs.get('fig_size', 16.0)
    if fig is None:
        # fig, axes = pl.subplots(K, K, figsize=(dim, dim))
        fig, axes = pl.subplots(K, K, figsize=(fig_size, fig_size))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))
    fig_title = kwargs.get('fig_title', 'MCMC Chains')
    pl.suptitle(fig_title, **title_args)
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

        # Check for parameters that never change.
        m = np.array([e[0] == e[1] for e in extents], dtype=bool)
        if np.any(m):
            raise ValueError(("It looks like the parameter(s) in column(s) "
                              "{0} have no dynamic range. Please provide an "
                              "`extent` argument.")
                             .format(", ".join(map("{0}".format,
                                                   np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        for i in range(len(extents)):
            try:
                emin, emax = extents[i]
            except TypeError:
                q = [0.5 - 0.5*extents[i], 0.5 + 0.5*extents[i]]
                extents[i] = quantile(xs[i], q, weights=weights)

    allqvalues = list()
    for i, x in enumerate(xs):
        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.
        n, b, p = ax.hist(x, weights=weights, bins=kwargs.get("bins", 100),
                          range=extents[i], histtype="step",
                          color=kwargs.get("color", "k"))
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            allqvalues.append(qvalues)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=kwargs.get("color", "k"))

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

            if show_titles:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(x, quantiles,
                                            weights=weights)
                q_m, q_p = q_50-q_16, q_84-q_50

                # Format the quantile display.
                # fmt = "{{0:{0}}}".format(title_fmt).format
                # title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                # title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
                if labels is not None:
                    try:
                        numberLog10 = int(math.log10(q_50))
                        numberLog10 -= 1
                        if (numberLog10 != 0):
                            remainingPart = q_50/math.pow(10.0, numberLog10)
                            minusPart = q_m/math.pow(10.0, numberLog10)
                            plusPart = q_p/math.pow(10.0, numberLog10)
                            title = r"%s$ = %3.2f^{+%.2f}_{-%.2f} \times 10^{%d}$"%(
                                labels[i], remainingPart, plusPart, minusPart, numberLog10)
                        else:
                            title = r"%s$ = %3.2f^{+%.2f}_{-%.2f}$"%(labels[i], q_50, q_p, q_m)
                    except:
                        # print str(Err) + '. Using np.nan!'
                        title = str(np.nan)

                # Add in the column name if it's given.
                # if labels is not None:
                    # title = "{0} = {1}".format(labels[i], title)

                # Add the title to the axis.
                ax.set_title(title, **title_args)

        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(3))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]],
                   plot_contours=plot_contours,
                   plot_datapoints=plot_datapoints, plot_contour_lines=plot_contour_lines,
                   weights=weights, pcolor_cmap=pcolor_cmap, **kwargs)

            if truths is not None:
                ax.plot(truths[j], truths[i], "s", color=truth_color)
                ax.axvline(truths[j], color=truth_color)
                ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig, quantiles, allqvalues


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = pl.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def hist2d(x, y, plot_contour_lines, pcolor_cmap, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", pl.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 100)
    color = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 1)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1]*1.1, bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1]*1.1, bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                 weights=kwargs.get('weights', None))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    V = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        if plot_contours:
            ax.contourf(X1, Y1, H.T, [V[-1], H.max()], cmap=LinearSegmentedColormap.from_list(
                "cmap", ([1]*3, [1]*3), N=2), antialiased=True)

    if plot_contours:
        if pcolor_cmap is None:
            cmap = cmap
        else:
            cmap = pcolor_cmap
        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
        if (plot_contour_lines == True):
            ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
