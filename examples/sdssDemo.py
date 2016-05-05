import numpy as np
import math as math
import cmath as cmath
import psutil as psutil
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import argparse as argparse
import pdb
import os as os

import libcarma as libcarma
import sdss as sdss
from util.mpl_settings import set_plot_params
import util.triangle as triangle

try: 
	os.environ['DISPLAY']
except KeyError as Err:
	print "No display environment! Using matplotlib backend 'Agg'"
	import matplotlib
	matplotlib.use('Agg')

try:
	import carmcmc as cmcmc
except ImportError:
	carma_pack = False
else:
	carma_pack = True

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type = str, default = '/home/vpk24/Documents', help = r'Path to working directory')
parser.add_argument('-name', '--n', type = str, default = 'LightCurveSDSS_1.csv', help = r'SDSS Filename')
parser.add_argument('-nsteps', '--nsteps', type = int, default = 250, help = r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type = int, default = 25*psutil.cpu_count(logical = True), help = r'Number of walkers')
parser.add_argument('-p', '--p', type = int, default = 2, help = r'C-AR order')
parser.add_argument('-q', '--q', type = int, default = 1, help = r'C-MA order')
parser.add_argument('-g', '--g', dest = 'g', action = 'store_true', help = r'Analyze g-band LC')
parser.add_argument('-no-g', '--no-g', dest = 'g', action = 'store_false', help = r'Do not analyze g-band LC')
parser.set_defaults(g = True)
parser.add_argument('-r', '--r', dest = 'r', action = 'store_true', help = r'Analyze r-band LC')
parser.add_argument('-no-r', '--no-r', dest = 'r', action = 'store_false', help = r'Do not analyze r-band LC')
parser.set_defaults(r = False)
args = parser.parse_args()

P = args.p
Q = args.q
NSTEPS = args.nsteps
NWALKERS = args.nwalkers

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

if args.g:
	sdss0g = sdss.sdss_gLC(supplied = args.n, pwd = args.pwd)
if args.r:
	sdss0r = sdss.sdss_rLC(supplied = args.n, pwd = args.pwd)

if args.g or args.r:
	plt.figure(1, figsize = (fwid, fhgt))
	if args.g:
		plt.errorbar(sdss0g.t - sdss0g.startT, sdss0g.y, sdss0g.yerr, label = r'sdss-g', fmt = '.', capsize = 0, color = '#2ca25f', markeredgecolor = 'none', zorder = 10)
	if args.r:
		plt.errorbar(sdss0r.t - sdss0r.startT, sdss0r.y, sdss0r.yerr, label = r'sdss-r', fmt = '.', capsize = 0, color = '#feb24c', markeredgecolor = 'none', zorder = 10)
	plt.xlabel(sdss0g.xunit)
	plt.ylabel(sdss0g.yunit)
	plt.legend()

	Theta = np.array([0.725, 0.01, 7.0e-7, 1.2e-7])

	if args.g:
		ntg = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
		ntg.set(sdss0g.dt, Theta)
		ntg.fit(sdss0g, Theta)

	if args.r:
		ntr = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
		ntr.set(sdss0r.dt, Theta)
		ntr.fit(sdss0r, Theta)

	if carma_pack:
		NUMSAMPLES = NWALKERS*NSTEPS/2
		NBURNIN = NWALKERS*NSTEPS/2

		if args.g:
			carma_model_g = cmcmc.CarmaModel(sdss0g.t - sdss0g.startT, sdss0g.y, sdss0g.yerr, p = P, q = Q)  # create new CARMA process model
			carma_sample_g = carma_model_g.run_mcmc(NUMSAMPLES, nburnin = NBURNIN)
			ar_poly_g = carma_sample_g.get_samples('ar_coefs')
			ma_poly_g = carma_sample_g.get_samples('ma_coefs')
			sigma_g = carma_sample_g.get_samples('sigma')
			cmcmcLnPosterior_g = carma_sample_g.get_samples('logpost')
			numSamples = ar_poly_g.shape[0]
			cmcmcChain_g = np.zeros((P + Q + 1, numSamples))
			for i in xrange(numSamples):
				for j in xrange(P):
					cmcmcChain_g[j, i] = ar_poly_g[i,j + 1]
				for j in xrange(Q + 1):
					cmcmcChain_g[j + P, i] = ma_coefs_g[i,j]*sigma_g[i,0]

		if args.r:
			carma_model_r = cmcmc.CarmaModel(sdss0r.t, sdss0r.y, sdss0r.yerr, p = P, q = Q)
			carma_sample_r = carma_model_r.run_mcmc(NUMSAMPLES, nburnin = NBURNIN)
			ar_poly_r = carma_sample_r.get_samples('ar_coefs')
			ma_poly_r = carma_sample_r.get_samples('ma_coefs')
			sigma_r = carma_sample_r.get_samples('sigma')
			lnPosterior_r = carma_sample_r.get_samples('logpost')
			numSamples = ar_poly_g.shape[0]
			cmcmcChain_r = np.zeros((P + Q + 1, numSamples))
			for i in xrange(numSamples):
				for j in xrange(P):
					cmcmcChain_r[j, i] = ar_poly_r[i,j + 1]
				for j in xrange(Q + 1):
					cmcmcChain_r[j + P, i] = ma_coefs_r[i,j]*sigma_r[i,0]

	if args.g:
		fig2 = plt.figure(2, figsize = (fhgt, fhgt))
		plt.title(r'g-band C-AR Coeffs')
		scatPlot1 = plt.scatter(ntg.Chain[0,:,NSTEPS/2:], ntg.Chain[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		scatPlot1cmcmc = plt.scatter(cmcmcChain_g[0,:,:], cmcmcChain_g[1,:,:], c = cmcmcLnPosterior_g[:,:], marker = 'o', edgecolors = 'none')
		cBar1 = plt.colorbar(scatPlot1, orientation = 'horizontal')
		cBar1.set_label(r'$\ln \mathcal{P}$')
		plt.xlim(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(ntg.Chain[0,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(ntg.Chain[1,:,NSTEPS/2:]))
		plt.xlabel(r'$a_{1}$')
		plt.ylabel(r'$a_{2}$')

		fig3 = plt.figure(3, figsize = (fhgt, fhgt))
		plt.title(r'g-band C-MA Coeffs')
		scatPlot2 = plt.scatter(ntg.Chain[2,:,NSTEPS/2:], ntg.Chain[3,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		scatPlot2cmcmc = plt.scatter(cmcmcChain_g[2,:,:], cmcmcChain_g[3,:,:], c = cmcmcLnPosterior_g[:,:], marker = 'o', edgecolors = 'none')
		cBar2 = plt.colorbar(scatPlot2, orientation = 'horizontal')
		cBar2.set_label(r'$\ln \mathcal{P}$')
		plt.xlim(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(ntg.Chain[2,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(ntg.Chain[3,:,NSTEPS/2:]))
		plt.xlabel(r'$b_{0}$')
		plt.ylabel(r'$b_{1}$')

	if args.r:
		fig4 = plt.figure(4, figsize = (fhgt, fhgt))
		plt.title(r'r-band C-AR Coeffs')
		scatPlot3 = plt.scatter(ntr.Chain[0,:,NSTEPS/2:], ntr.Chain[1,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		scatPlot3cmcmc = plt.scatter(cmcmcChain_r[0,:,:], cmcmcChain_r[1,:,:], c = cmcmcLnPosterior_r[:,:], marker = 'o', edgecolors = 'none')
		cBar3 = plt.colorbar(scatPlot3, orientation = 'horizontal')
		cBar3.set_label(r'$\ln \mathcal{P}$')
		plt.xlim(np.nanmin(ntr.Chain[0,:,NSTEPS/2:]), np.nanmax(ntr.Chain[0,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntr.Chain[1,:,NSTEPS/2:]), np.nanmax(ntr.Chain[1,:,NSTEPS/2:]))
		plt.xlabel(r'$a_{1}$')
		plt.ylabel(r'$a_{2}$')

		fig5 = plt.figure(5, figsize = (fhgt, fhgt))
		plt.title(r'r-band C-MA Coeffs')
		scatPlot4 = plt.scatter(ntr.Chain[2,:,NSTEPS/2:], ntr.Chain[3,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		scatPlot4cmcmc = plt.scatter(cmcmcChain_r[2,:,:], cmcmcChain_r[3,:,:], c = cmcmcLnPosterior_r[:,:], marker = 'o', edgecolors = 'none')
		cBar4 = plt.colorbar(scatPlot4, orientation = 'horizontal')
		cBar4.set_label(r'$\ln \mathcal{P}$')
		plt.xlim(np.nanmin(ntr.Chain[2,:,NSTEPS/2:]), np.nanmax(ntr.Chain[2,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntr.Chain[3,:,NSTEPS/2:]), np.nanmax(ntr.Chain[3,:,NSTEPS/2:]))
		plt.xlabel(r'$b_{0}$')
		plt.ylabel(r'$b_{1}$')

	plt.show()

pdb.set_trace()