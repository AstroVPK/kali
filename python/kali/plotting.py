'''
A small library of plotting functions
that allow a user to quickly look at
and asses various data structures throughout
the kali package.  Supported data structures
are the following:

    -lc
    -CARMATask

'''
import math
import cmath

import numpy as np
from scipy import stats as spstats
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

try:
	import rand, LCTools_cython, kali.sampler as sample, kali.kernel as kernel, \
	kali.util.mpl_settings as mpl_settings
except ImportError as e:
	print e
	sys.exit(1)

fhgt=10
fwid=16
mpl_settings.set_plot_params(useTex=True)
ln10=math.log10(10)

def setupFig(fig=-1, clearFig=True):

	newFig = plt.figure(fig, figsize=(fwid,fhgt))
	if clearFig: plt.clf()
	return newFig

def plotLC(LC, fig=-1, doShow=True, clearFig=True):
	"""DocString Here"""

	newFig = setupFig(fig, clearFig)
	
	if LC.isSmoothed:
		ts, xs, xerrs, cs= (LC.tSmooth, LC.xSmooth, LC.xerrSmooth, "#4daf4a")
		fc = "#ccebc5"

		plt.plot(ts, xs, color=c,
			marker='o', markeredgecolor='none', zorder=-5)		
		plt.fill_between(ts, xs-xerrs, xs+xerrs, facecolor=fc, alpha=0.5, zorder=-5)

	t, x, y, yerr, mask, c = (LC.t, LC.x, LC.y, LC.yerr, LC.mask, '#984ea3')
	tm, xm, ym, yerrm = map(lambda eks: eks[mask.astype(bool)], (t,x,y,yerr))

	if (np.sum(x)):
		if (not np.sum(y)):
			plt.plot(t, x, color=c, marker='o', \
				markeredgecolor="none", zorder=0)
		else:
			plt.plot(t, x-np.mean(x) + np.mean(ym), color=c, zorder=0)
	elif (np.sum(y)):
		l=r'%s (%s-band)'%(LC.name, LC.band)
		c = '#ff7f00'
		plt.errorbar(tm, ym, yerrm, label=l, fmt='o', capsize=0, color=c, 
			markeredgecolor="none", zorder=10)
	else:
		raise Exception("Bad LC")	

	plt.xlim(*t[[0,-1]])	
	plt.xlabel(LC.xunit)
	plt.ylabel(LC.yunit)
	plt.title(r'Light Curve')
	plt.legend()
	if doShow:
		plt.show(False)
	return newFig
	
def plotACVF(LC, fig=-2, newdt=None, doShow=True, clearFig=True):
	"""DocString Here"""

	newFig = setupFig(fig, clearFig)

	t, x, y, yerr, mask, c = (LC.t, LC.x, LC.y, LC.yerr, LC.mask, '#ff7f00')
	tm, xm, ym, yerrm = map(lambda eks: eks[mask.astype(bool)], (t,x,y,yerr))

	plt.plot(0.0,0.0)
	if (np.sum(y) != 0.0):
		acfres = LC.acvf(newdt)
		if (np.sum(acfres[1]) != 0.0):
			lagsE, acvfE, acvferrE = zip(*acfres)[0]
			plt.errorbar(*zip(*acfres)[0], label=r'obs. Autocovariance Function',
				fmt='o', capsize=0, color=c, markeredgecolor='none', zorder=10)
			for lagsE, acfE, acferrE in zip(*acfres)[1:]:
				if (acfE != 0.0):
					plt.errorbar(lagsE, acfE, acferrE, fmt='o', capsize=0, color=c,
						markeredgecolor='none', zorder=10)
			plt.xlim(*acfres[0][[1,-1]])
	plt.xlabel(r'$\delta t$')
	plt.ylabel(r'$ACF$')
	plt.title(r'AutoCorrelation Function')
	plt.legend(loc=3)
	if doShow:
		plt.show(False)
	return newFig






