import numpy as np
import math as math
import cmath as cmath
import psutil as psutil
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import argparse as argparse
import warnings as warnings
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
parser.add_argument('-libcarmaChain', '--lC', type = str, default = 'libcarmaChain', help = r'libcarma Chain Filename')
parser.add_argument('-cmcmcChain', '--cC', type = str, default = 'cmcmcChain', help = r'carma_pack Chain Filename')
parser.add_argument('-nsteps', '--nsteps', type = int, default = 250, help = r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type = int, default = 25*psutil.cpu_count(logical = True), help = r'Number of walkers')
parser.add_argument('-p', '--p', type = int, default = 2, help = r'C-AR order')
parser.add_argument('-q', '--q', type = int, default = 1, help = r'C-MA order')
parser.add_argument('-g', '--g', dest = 'g', action = 'store_true', help = r'Analyze g-band LC')
parser.add_argument('-no-g', '--no-g', dest = 'g', action = 'store_false', help = r'Do not analyze g-band LC')
parser.set_defaults(g = True)
parser.add_argument('-r', '--r', dest = 'r', action = 'store_true', help = r'Analyze r-band LC')
parser.add_argument('-no-r', '--no-r', dest = 'r', action = 'store_false', help = r'Do not analyze r-band LC')
parser.set_defaults(r = True)
parser.add_argument('--plot', dest = 'plot', action = 'store_true', help = r'Show plot?')
parser.add_argument('--no-plot', dest = 'plot', action = 'store_false', help = r'Do not show plot?')
parser.set_defaults(plot = False)
parser.add_argument('-minT', '--minTimescale', type = float, default = 2.0, help = r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type = float, default = 0.5, help = r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type = float, default = 2.0, help = r'Maximum allowed sigma = maxSigma*var(lc)')
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
	imagRoots = set(abs(Rho[0:p].imag)).difference(set([0.0]))
	realAR = np.array([1.0/abs(x) for x in realRoots])
	imagAR = np.array([(2.0*math.pi)/abs(x) for x in imagRoots])
	imagPairs = 0
	for i in xrange(q):
		if Rho[i].imag != 0.0:
			imagPairs += 1
	numImag = imagPairs/2
	numReal = numImag + (q - imagPairs)
	decayTimescales = np.zeros(numReal)
	oscTimescales = np.zeros(numImag)
	realRoots = set(Rho[p:p + q].real)
	imagRoots = set(abs(Rho[p:p + q].imag)).difference(set([0.0]))
	realMA = np.array([1.0/abs(x) for x in realRoots])
	imagMA = np.array([(2.0*math.pi)/abs(x) for x in imagRoots])
	return realAR, imagAR, realMA, imagMA

if args.g or args.r:
	plt.figure(1, figsize = (fwid, fhgt))
	plt.xlabel('$t$ (MJD)')
	plt.ylabel('$F$ (Jy)')
	plt.legend()

	Theta = np.array([0.725, 0.01, 7.0e-7, 1.2e-7])

	if args.g:
		sdss0g = sdss.sdss_gLC(supplied = args.n, pwd = args.pwd)
		sdss0g.minTimescale = args.minTimescale
		sdss0g.maxTimescale = args.maxTimescale
		sdss0g.maxSigma = args.maxSigma

		plt.errorbar(sdss0g.t - sdss0g.startT, sdss0g.y, sdss0g.yerr, label = r'sdss-g', fmt = '.', capsize = 0, color = '#2ca25f', markeredgecolor = 'none', zorder = 10)
		fileName = args.n.split('.')[0] + '_' + args.lC + '_g.dat'
		libcarmaChain_g = os.path.join(args.pwd, fileName)
		try:
			chainFile = open(libcarmaChain_g, 'r')
		except IOError:
			chainFile = open(libcarmaChain_g, 'w')
			ntg = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
			ntg.set(sdss0g.dt, Theta)
			ntg.fit(sdss0g, Theta)
			line = '%d %d %d %d\n'%(P, Q, NWALKERS, NSTEPS)
			chainFile.write(line)
			for stepNum in xrange(NSTEPS):
				for walkerNum in xrange(NWALKERS):
					line = ''
					for dimNum in xrange(P + Q +1):
						line += '%+9.8e '%(ntg.Chain[dimNum, walkerNum, stepNum])
					line += '%+9.8e\n'%(ntg.LnPosterior[walkerNum, stepNum])
					chainFile.write(line)
					del line
		else:
			line = chainFile.readline()
			words = line.rstrip('\n').split()
			P = int(words[0])
			Q = int(words[1])
			NWALKERS = int(words[2])
			NSTEPS = int(words[3])
			ntg = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
			for stepNum in xrange(NSTEPS):
				for walkerNum in xrange(NWALKERS):
					line = chainFile.readline()
					words = line.rstrip('\n').split()
					for dimNum in xrange(P + Q + 1):
						ntg.Chain[dimNum, walkerNum, stepNum] = float(words[dimNum])
					ntg.LnPosterior[walkerNum, stepNum] = float(words[P + Q + 1])
		chainFile.close()

		fileName = args.n.split('.')[0] + '_' + args.cC + '_g.dat'
		cmcmcChain_g = os.path.join(args.pwd, fileName)
		try:
			chainFile = open(cmcmcChain_g, 'r')
		except IOError:
			NSAMPLES = NWALKERS*NSTEPS/2
			NBURNIN = NWALKERS*NSTEPS/2
			if carma_pack:
				chainFile = open(cmcmcChain_g, 'w')
				carma_model_g = cmcmc.CarmaModel(sdss0g.t - sdss0g.startT, sdss0g.y, sdss0g.yerr, p = P, q = Q)  # create new CARMA process model
				carma_sample_g = carma_model_g.run_mcmc(NSAMPLES, nburnin = NBURNIN)
				ar_poly_g = carma_sample_g.get_samples('ar_coefs')
				ma_poly_g = carma_sample_g.get_samples('ma_coefs')
				sigma_g = carma_sample_g.get_samples('sigma')
				logpost_g = carma_sample_g.get_samples('logpost')
				cmcmcChain_g = np.zeros((P + Q + 1, NSAMPLES))
				cmcmcLnPosterior_g = np.zeros(NSAMPLES)
				line = '%d %d %d\n'%(P, Q, NSAMPLES)
				chainFile.write(line)
				for sampleNum in xrange(NSAMPLES):
					for j in xrange(P):
						cmcmcChain_g[j, sampleNum] = ar_poly_g[sampleNum, j + 1]
					for j in xrange(Q + 1):
						cmcmcChain_g[j + P, sampleNum] = ma_poly_g[sampleNum, j]*sigma_g[sampleNum, 0]
					cmcmcLnPosterior_g[sampleNum] = logpost_g[sampleNum, 0]
					line = ''
					for dimNum in xrange(P + Q +1):
						line += '%+9.8e '%(cmcmcChain_g[dimNum, sampleNum])
					line += '%+9.8e\n'%(cmcmcLnPosterior_g[sampleNum])
					chainFile.write(line)
					del line
				chainFile.close()
				carma_pack_results_g = True
			else:
				warnings.warn('carma_pack not found \& pre-computed carma_pack chains not located. Not using carma_pack!')
				carma_pack_results_g = False
		else:
			line = chainFile.readline()
			words = line.rstrip('\n').split()
			P = int(words[0])
			Q = int(words[1])
			NSAMPLES = int(words[2])
			cmcmcChain_g = np.zeros((P + Q + 1, NSAMPLES))
			cmcmcLnPosterior_g = np.zeros(NSAMPLES)
			for sampleNum in xrange(NSAMPLES):
				line = chainFile.readline()
				words = line.rstrip('\n').split()
				for dimNum in xrange(P + Q + 1):
					cmcmcChain_g[dimNum, sampleNum] = float(words[dimNum])
				cmcmcLnPosterior_g[sampleNum] = float(words[P+Q+1])
			chainFile.close()
			carma_pack_results_g = True

		fig2 = plt.figure(2, figsize = (fhgt, fhgt))
		plt.title(r'g-band C-AR Coeffs')
		scatPlot1 = plt.scatter(ntg.Chain[0,:,NSTEPS/2:], ntg.Chain[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_g:
			scatPlot1cmcmc = plt.scatter(cmcmcChain_g[0,:], cmcmcChain_g[1,:], c = cmcmcLnPosterior_g[:], marker = 'o', edgecolors = 'none')
			plt.xlim(min(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[0,:])), max(np.nanmax(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[0,:])))
			plt.ylim(min(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[1,:])), max(np.nanmax(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[1,:])))
		else:
			plt.xlim(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(ntg.Chain[0,:,NSTEPS/2:]))
			plt.ylim(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(ntg.Chain[1,:,NSTEPS/2:]))
		cBar1 = plt.colorbar(scatPlot1, orientation = 'horizontal')
		cBar1.set_label(r'$\ln \mathcal{P}$')
		plt.xlabel(r'$a_{1}$')
		plt.ylabel(r'$a_{2}$')

		fig3 = plt.figure(3, figsize = (fhgt, fhgt))
		plt.title(r'g-band C-MA Coeffs')
		scatPlot2 = plt.scatter(ntg.Chain[2,:,NSTEPS/2:], ntg.Chain[3,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_g:
			scatPlot2cmcmc = plt.scatter(cmcmcChain_g[2,:], cmcmcChain_g[3,:], c = cmcmcLnPosterior_g[:], marker = 'o', edgecolors = 'none')
			plt.xlim(min(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[2,:])), max(np.nanmax(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[2,:])))
			plt.ylim(min(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[3,:])), max(np.nanmax(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[3,:])))
		else:
			plt.xlim(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(ntg.Chain[2,:,NSTEPS/2:]))
			plt.ylim(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(ntg.Chain[3,:,NSTEPS/2:]))
		cBar2 = plt.colorbar(scatPlot2, orientation = 'horizontal')
		cBar2.set_label(r'$\ln \mathcal{P}$')
		plt.xlabel(r'$b_{0}$')
		plt.ylabel(r'$b_{1}$')

		# Convert Theta -> Rho -> Tau
		lcarmaTau_g = np.zeros((P + Q + 1, NWALKERS, NSTEPS))
		for stepNum in xrange(NSTEPS):
			for walkerNum in xrange(NWALKERS):
				lcarmaRAR, lcarmaIAR, lcarmaRMA, lcarmaIMA = timescales(P, Q, ntg.rootChain[:, walkerNum, stepNum])
				lcarmaTau_g[:, walkerNum, stepNum] = np.array(sorted([i for i in lcarmaRAR]) + sorted([i for i in lcarmaIAR]) + sorted([i for i in lcarmaRMA]) + sorted([i for i in lcarmaIMA]) + [ntg.rootChain[P + Q, walkerNum, stepNum]])
		plt.figure(6, figsize = (fhgt, fhgt))
		plt.scatter(lcarmaTau_g[0,:,NSTEPS/2:], lcarmaTau_g[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_g:
			cmcmcRho_g = np.zeros((P + Q + 1, NSAMPLES))
			cmcmcTau_g = np.zeros((P + Q + 1, NSAMPLES))
			for sampleNum in xrange(NSAMPLES):
				cmcmcRho_g[:,sampleNum] = libcarma.roots(P, Q, cmcmcChain_g[:,sampleNum])
				cmcmcRAR, cmcmcIAR, cmcmcRMA, cmcmcIMA = timescales(P, Q, (cmcmcRho_g[:,sampleNum]))
				try:
					cmcmcTau_g[:,sampleNum] = np.array(sorted([i for i in cmcmcRAR]) + sorted([i for i in cmcmcIAR]) + sorted([i for i in cmcmcRMA]) + sorted([i for i in cmcmcIMA]) + [cmcmcRho_g[P + Q, sampleNum]])
				except ValueError: # Sometimes Kelly's roots are repeated!!! This should not be allowed!
					pass
			plt.scatter(cmcmcTau_g[0,:], cmcmcTau_g[1,:], c = cmcmcLnPosterior_g[:], marker = 'o', edgecolors = 'none')

	if args.r:
		sdss0r = sdss.sdss_rLC(supplied = args.n, pwd = args.pwd)
		sdss0r.minTimescale = args.minTimescale
		sdss0r.maxTimescale = args.maxTimescale
		sdss0r.maxSigma = args.maxSigma

		plt.errorbar(sdss0r.t - sdss0r.startT, sdss0r.y, sdss0r.yerr, label = r'sdss-r', fmt = '.', capsize = 0, color = '#feb24c', markeredgecolor = 'none', zorder = 10)
		fileName = args.n.split('.')[0] + '_' + args.lC + '_r.dat'
		libcarmaChain_r = os.path.join(args.pwd, fileName)
		try:
			chainFile = open(libcarmaChain_r, 'r')
		except IOError:
			NSAMPLES = NWALKERS*NSTEPS/2
			NBURNIN = NWALKERS*NSTEPS/2
			if carma_pack:
				chainFile = open(libcarmaChain_r, 'w')
				ntr = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
				ntr.set(sdss0r.dt, Theta)
				ntr.fit(sdss0r, Theta)
				line = '%d %d %d %d\n'%(P, Q, NWALKERS, NSTEPS)
				chainFile.write(line)
				for stepNum in xrange(NSTEPS):
					for walkerNum in xrange(NWALKERS):
						line = ''
						for dimNum in xrange(P + Q +1):
							line += '%+9.8e '%(ntr.Chain[dimNum, walkerNum, stepNum])
						line += '%+9.8e\n'%(ntr.LnPosterior[walkerNum, stepNum])
						chainFile.write(line)
						del line
			chainFile.close()
		else:
			line = chainFile.readline()
			words = line.rstrip('\n').split()
			P = int(words[0])
			Q = int(words[1])
			NWALKERS = int(words[2])
			NSTEPS = int(words[3])
			ntr = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
			for stepNum in xrange(NSTEPS):
				for walkerNum in xrange(NWALKERS):
					line = chainFile.readline()
					words = line.rstrip('\n').split()
					for dimNum in xrange(P + Q + 1):
						ntr.Chain[dimNum, walkerNum, stepNum] = float(words[dimNum])
					ntr.LnPosterior[walkerNum, stepNum] = float(words[P + Q + 1])

		fileName = args.n.split('.')[0] + '_' + args.cC + '_r.dat'
		cmcmcChain_r = os.path.join(args.pwd, fileName)
		try:
			chainFile = open(cmcmcChain_r, 'r')
		except IOError:
			NSAMPLES = NWALKERS*NSTEPS/2
			NBURNIN = NWALKERS*NSTEPS/2
			if carma_pack:
				chainFile = open(cmcmcChain_r, 'w')
				carma_model_r = cmcmc.CarmaModel(sdss0r.t - sdss0r.startT, sdss0r.y, sdss0r.yerr, p = P, q = Q)  # create new CARMA process model
				carma_sample_r = carma_model_r.run_mcmc(NSAMPLES, nburnin = NBURNIN)
				ar_poly_r = carma_sample_r.get_samples('ar_coefs')
				ma_poly_r = carma_sample_r.get_samples('ma_coefs')
				sigma_r = carma_sample_r.get_samples('sigma')
				logpost_r = carma_sample_r.get_samples('logpost')
				cmcmcChain_r = np.zeros((P + Q + 1, NSAMPLES))
				cmcmcLnPosterior_r = np.zeros(NSAMPLES)
				line = '%d %d %d\n'%(P, Q, NSAMPLES)
				chainFile.write(line)
				for sampleNum in xrange(NSAMPLES):
					for j in xrange(P):
						cmcmcChain_r[j, sampleNum] = ar_poly_r[sampleNum, j + 1]
					for j in xrange(Q + 1):
						cmcmcChain_r[j + P, sampleNum] = ma_poly_r[sampleNum, j]*sigma_r[sampleNum, 0]
					cmcmcLnPosterior_r[sampleNum] = logpost_r[sampleNum, 0]
					line = ''
					for dimNum in xrange(P + Q +1):
						line += '%+9.8e '%(cmcmcChain_r[dimNum, sampleNum])
					line += '%+9.8e\n'%(cmcmcLnPosterior_r[sampleNum])
					chainFile.write(line)
					del line
				chainFile.close()
				carma_pack_results_r = True
			else:
				warnings.warn('carma_pack not found \& pre-computed carma_pack chains not located. Not using carma_pack!')
				carma_pack_results_r = False
		else:
			line = chainFile.readline()
			words = line.rstrip('\n').split()
			P = int(words[0])
			Q = int(words[1])
			NSAMPLES = int(words[2])
			cmcmcChain_r = np.zeros((P + Q + 1, NSAMPLES))
			cmcmcLnPosterior_r = np.zeros(NSAMPLES)
			for sampleNum in xrange(NSAMPLES):
				line = chainFile.readline()
				words = line.rstrip('\n').split()
				for dimNum in xrange(P + Q + 1):
					cmcmcChain_r[dimNum, sampleNum] = float(words[dimNum])
				cmcmcLnPosterior_r[sampleNum] = float(words[P + Q + 1])
			chainFile.close()
			carma_pack_results_r = True

		fig4 = plt.figure(4, figsize = (fhgt, fhgt))
		plt.title(r'r-band C-AR Coeffs')
		scatPlot3 = plt.scatter(ntr.Chain[0,:,NSTEPS/2:], ntr.Chain[1,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_r:
			scatPlot3cmcmc = plt.scatter(cmcmcChain_r[0,:], cmcmcChain_r[1,:], c = cmcmcLnPosterior_r[:], marker = 'o', edgecolors = 'none')
			plt.xlim(min(np.nanmin(ntr.Chain[0,:,NSTEPS/2:]), np.nanmin(cmcmcChain_r[0,:])), max(np.nanmax(ntr.Chain[0,:,NSTEPS/2:]), np.nanmax(cmcmcChain_r[0,:])))
			plt.ylim(min(np.nanmin(ntr.Chain[1,:,NSTEPS/2:]), np.nanmin(cmcmcChain_r[1,:])), max(np.nanmax(ntr.Chain[1,:,NSTEPS/2:]), np.nanmax(cmcmcChain_r[1,:])))
		else:
			plt.xlim(np.nanmin(ntr.Chain[0,:,NSTEPS/2:]), np.nanmax(ntr.Chain[0,:,NSTEPS/2:]))
			plt.ylim(np.nanmin(ntr.Chain[1,:,NSTEPS/2:]), np.nanmax(ntr.Chain[1,:,NSTEPS/2:]))
		cBar3 = plt.colorbar(scatPlot3, orientation = 'horizontal')
		cBar3.set_label(r'$\ln \mathcal{P}$')
		plt.xlabel(r'$a_{1}$')
		plt.ylabel(r'$a_{2}$')

		fig5 = plt.figure(5, figsize = (fhgt, fhgt))
		plt.title(r'r-band C-MA Coeffs')
		scatPlot4 = plt.scatter(ntr.Chain[2,:,NSTEPS/2:], ntr.Chain[3,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_r:
			scatPlot4cmcmc = plt.scatter(cmcmcChain_r[2,:], cmcmcChain_r[3,:], c = cmcmcLnPosterior_r[:], marker = 'o', edgecolors = 'none')
			plt.xlim(min(np.nanmin(ntr.Chain[2,:,NSTEPS/2:]), np.nanmin(cmcmcChain_r[2,:])), max(np.nanmax(ntr.Chain[2,:,NSTEPS/2:]), np.nanmax(cmcmcChain_r[2,:])))
			plt.ylim(min(np.nanmin(ntr.Chain[3,:,NSTEPS/2:]), np.nanmin(cmcmcChain_r[3,:])), max(np.nanmax(ntr.Chain[3,:,NSTEPS/2:]), np.nanmax(cmcmcChain_r[3,:])))
		else:
			plt.xlim(np.nanmin(ntr.Chain[2,:,NSTEPS/2:]), np.nanmax(ntr.Chain[2,:,NSTEPS/2:]))
			plt.ylim(np.nanmin(ntr.Chain[3,:,NSTEPS/2:]), np.nanmax(ntr.Chain[3,:,NSTEPS/2:]))
		cBar4 = plt.colorbar(scatPlot4, orientation = 'horizontal')
		cBar4.set_label(r'$\ln \mathcal{P}$')
		plt.xlabel(r'$b_{0}$')
		plt.ylabel(r'$b_{1}$')

		# Convert Theta -> Rho -> Tau
		lcarmaTau_r = np.zeros((P + Q + 1, NWALKERS, NSTEPS))
		for stepNum in xrange(NSTEPS):
			for walkerNum in xrange(NWALKERS):
				lcarmaRAR, lcarmaIAR, lcarmaRMA, lcarmaIMA = timescales(P, Q, ntr.rootChain[:, walkerNum, stepNum])
				lcarmaTau_r[:, walkerNum, stepNum] = np.array(sorted([i for i in lcarmaRAR]) + sorted([i for i in lcarmaIAR]) + sorted([i for i in lcarmaRMA]) + sorted([i for i in lcarmaIMA]) + [ntr.rootChain[P + Q, walkerNum, stepNum]])
		plt.figure(6, figsize = (fhgt, fhgt))
		plt.scatter(lcarmaTau_r[0,:,NSTEPS/2:], lcarmaTau_r[1,:,NSTEPS/2:], c = ntr.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')
		if carma_pack_results_r:
			cmcmcRho_r = np.zeros((P + Q + 1, NSAMPLES))
			cmcmcTau_r = np.zeros((P + Q + 1, NSAMPLES))
			for sampleNum in xrange(NSAMPLES):
				cmcmcRho_r[:,sampleNum] = libcarma.roots(P, Q, cmcmcChain_r[:,sampleNum])
				cmcmcRAR, cmcmcIAR, cmcmcRMA, cmcmcIMA = timescales(P, Q, (cmcmcRho_r[:,sampleNum]))
				try:
					cmcmcTau_r[:,sampleNum] = np.array(sorted([i for i in cmcmcRAR]) + sorted([i for i in cmcmcIAR]) + sorted([i for i in cmcmcRMA]) + sorted([i for i in cmcmcIMA]) + [cmcmcRho_r[P + Q, sampleNum]])
				except ValueError: # Sometimes Kelly's roots are repeated!!! This should not be allowed!
					pass
			plt.scatter(cmcmcTau_r[0,:], cmcmcTau_r[1,:], c = cmcmcLnPosterior_r[:], marker = 'o', edgecolors = 'none')

	if args.plot:
		plt.show()

pdb.set_trace()