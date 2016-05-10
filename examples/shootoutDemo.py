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
parser.add_argument('--plot', dest = 'plot', action = 'store_true', help = r'Show plot?')
parser.add_argument('--no-plot', dest = 'plot', action = 'store_false', help = r'Do not show plot?')
parser.set_defaults(plot = False)
parser.add_argument('-minT', '--minTimescale', type = float, default = 2.0, help = r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type = float, default = 0.5, help = r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type = float, default = 2.0, help = r'Maximum allowed sigma = maxSigma*var(lc)')
parser.add_argument('-sFac', '--scatterFactor', type = float, default = 0.1, help = r'Scatter factgor for starting locations of walkers pre-optimization')
parser.add_argument('--stop', dest = 'stop', action = 'store_true', help = r'Stop at end?')
parser.add_argument('--no-stop', dest = 'stop', action = 'store_false', help = r'Do not stop at end?')
parser.set_defaults(stop = False)
parser.add_argument('--save', dest = 'save', action = 'store_true', help = r'Save files?')
parser.add_argument('--no-save', dest = 'save', action = 'store_false', help = r'Do not save files?')
parser.set_defaults(save = False)
args = parser.parse_args()

P = args.p
Q = args.q
NSTEPS = args.nsteps
NWALKERS = args.nwalkers

sdss0g = sdss.sdss_gLC(supplied = args.n, pwd = args.pwd)
sdss0g.minTimescale = args.minTimescale
sdss0g.maxTimescale = args.maxTimescale
sdss0g.maxSigma = args.maxSigma

filename_mock = args.n.split('.')[0] + '_' + args.lC + '_mock.dat'
file_mock = os.path.join(args.pwd, filename_mock)
try:
	mockFile = open(file_mock, 'r')
except IOError:
	minT = sdss0g.dt*sdss0g.minTimescale*10.0
	maxT = sdss0g.T*sdss0g.maxTimescale*0.1
	RhoMock = -1.0/((maxT - minT)*np.random.random(P + Q + 1) + minT)
	RhoMock[-1] = 6.0e-2*np.std(sdss0g.y)
	MockRAR, MockIAR, MockRMA, MockIMA = libcarma.timescales(P, Q, RhoMock)
	TauMock = np.array(sorted([i for i in MockRAR]) + sorted([i for i in MockIAR]) + sorted([i for i in MockRMA]) + sorted([i for i in MockIMA]) + [RhoMock[-1]])
	ThetaMock = libcarma.coeffs(P, Q, RhoMock)
	
	newTask = libcarma.basicTask(P,Q)
	newTask.set(sdss0g.dt, ThetaMock)
	sdss_NtS = np.median(sdss0g.yerr/sdss0g.y)
	sdss_iV = np.std(sdss0g.y)/np.mean(sdss0g.y)
	premock_sdss0g = newTask.simulate(1.1*sdss0g.T, fracIntrinsicVar = sdss_iV, fracNoiseToSignal = sdss_NtS)
	newTask.observe(premock_sdss0g)
	premock_sdss0g.sampler = 'matchSampler'
	mock_sdss0g = premock_sdss0g.sample(timestamps = sdss0g.t)
	if args.save:
		with open(file_mock, 'w') as mockFile:
			line = 'ThetaMock: '
			for i in xrange(ThetaMock.shape[0]):
				line += '%+8.7e '%(ThetaMock[i])
			line += '\n'
			mockFile.write(line)
			line = 'RhoMock: '
			for i in xrange(RhoMock.shape[0]):
				line += '%+8.7e '%(RhoMock[i])
			line += '\n'
			mockFile.write(line)
			line = 'TauMock: '
			for i in xrange(TauMock.shape[0]):
				line += '%+8.7e '%(TauMock[i])
			line += '\n'
			mockFile.write(line)
			for i in xrange(len(mock_sdss0g)):
				line = '%+8.7e %+8.7e %+8.7e %+8.7e %+8.7e\n'%(mock_sdss0g.t[i], mock_sdss0g.x[i], mock_sdss0g.y[i], mock_sdss0g.yerr[i], mock_sdss0g.mask[i])
				mockFile.write(line)
else:
	with open(file_mock, 'r') as mockFile:
		words = mockFile.readline().rstrip('\n').split()
		ThetaMock = list()
		for i in xrange(1, len(words)):
			ThetaMock.append(float(words[i]))
		ThetaMock = np.array(ThetaMock)
		words = mockFile.readline().rstrip('\n').split()
		RhoMock = list()
		for i in xrange(1, len(words)):
			RhoMock.append(float(words[i]))
		RhoMock = np.array(RhoMock)
		words = mockFile.readline().rstrip('\n').split()
		TauMock = list()
		for i in xrange(1, len(words)):
			TauMock.append(float(words[i]))
		TauMock = np.array(TauMock)
		mock_sdss0gt = list()
		mock_sdss0gx = list()
		mock_sdss0gy = list()
		mock_sdss0gyerr = list()
		mock_sdss0gmask = list()
		for line in mockFile:
			words = line.rstrip('\n').split()
			mock_sdss0gt.append(float(words[0]))
			mock_sdss0gx.append(float(words[1]))
			mock_sdss0gy.append(float(words[2]))
			mock_sdss0gyerr.append(float(words[3]))
			mock_sdss0gmask.append(float(words[4]))
		mock_sdss0gt = np.array(mock_sdss0gt)
		mock_sdss0gx = np.array(mock_sdss0gx)
		mock_sdss0gy = np.array(mock_sdss0gy)
		mock_sdss0gyerr = np.array(mock_sdss0gyerr)
		mock_sdss0gmask = np.array(mock_sdss0gmask)
		dt = float(np.min(mock_sdss0gt[1:] - mock_sdss0gt[:-1]))
		mock_sdss0g = libcarma.basicLC(mock_sdss0gt.shape[0], dt)
		mock_sdss0g.t = mock_sdss0gt
		mock_sdss0g.x = mock_sdss0gx
		mock_sdss0g.y = mock_sdss0gy
		mock_sdss0g.yerr = mock_sdss0gyerr
		mock_sdss0g.mask = mock_sdss0gmask
		mock_sdss0g.dt = dt
		mock_sdss0g.T = mock_sdss0gt[-1] - mock_sdss0gt[0]

if args.plot:
	plt.figure(1, figsize = (fwid, fhgt))
	plt.errorbar(sdss0g.t, sdss0g.y, sdss0g.yerr, label = r'sdss-g', fmt = '.', capsize = 0, color = '#2ca25f', markeredgecolor = 'none', zorder = 10)
	plt.errorbar(mock_sdss0g.t, mock_sdss0g.y, mock_sdss0g.yerr, label = r'mock sdss-g', fmt = '.', capsize = 0, color = '#000000', markeredgecolor = 'none', zorder = 10)
	plt.xlabel('$t$ (MJD)')
	plt.ylabel('$F$ (Jy)')
	plt.title(r'Light curve')
	plt.legend()

fileName = args.n.split('.')[0] + '_' + args.lC + '_g.dat'
libcarmaChainFilePath = os.path.join(args.pwd, fileName)
try:
	chainFile = open(libcarmaChainFilePath, 'r')
except IOError:
	ntg = libcarma.basicTask(P, Q, nwalkers = NWALKERS, nsteps = NSTEPS)
	ntg.scatterFactor = args.scatterFactor
	minT = sdss0g.dt*sdss0g.minTimescale
	maxT = sdss0g.T*sdss0g.maxTimescale
	RhoGuess = -1.0/((maxT - minT)*np.random.random(P + Q + 1) + minT)
	RhoGuess[-1] = 6.0e-2*np.std(sdss0g.y)
	GuessRAR, GuessIAR, GuessRMA, GuessIMA = libcarma.timescales(P, Q, RhoGuess)
	TauGuess = np.array(sorted([i for i in GuessRAR]) + sorted([i for i in GuessIAR]) + sorted([i for i in GuessRMA]) + sorted([i for i in GuessIMA]) + [RhoGuess[-1]])
	ThetaGuess = libcarma.coeffs(P, Q, RhoGuess)
	ntg.set(mock_sdss0g.dt, ThetaGuess)
	ntg.fit(mock_sdss0g, ThetaGuess)
	if args.save:
		with open(libcarmaChainFilePath, 'w') as chainFile:
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
cmcmcChainFilePath = os.path.join(args.pwd, fileName)
try:
	chainFile = open(cmcmcChainFilePath, 'r')
except IOError:
	NSAMPLES = NWALKERS*NSTEPS/2
	NBURNIN = NWALKERS*NSTEPS/2
	if carma_pack:
		carma_model_g = cmcmc.CarmaModel(mock_sdss0g.t, mock_sdss0g.y, mock_sdss0g.yerr, p = P, q = Q)  # create new CARMA process model
		carma_sample_g = carma_model_g.run_mcmc(NSAMPLES, nburnin = NBURNIN)
		ar_poly_g = carma_sample_g.get_samples('ar_coefs')
		ma_poly_g = carma_sample_g.get_samples('ma_coefs')
		sigma_g = carma_sample_g.get_samples('sigma')
		logpost_g = carma_sample_g.get_samples('logpost')
		cmcmcChain_g = np.zeros((P + Q + 1, NSAMPLES))
		cmcmcLnPosterior_g = np.zeros(NSAMPLES)
		if args.save:
			with open(cmcmcChainFilePath, 'w') as chainFile:
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

lcarmaMedianThetaDist = 0.0
lcarmaMedianThetaLoc = np.zeros(P + Q + 1)
for i in xrange(P + Q + 1):
	lcarmaMedianThetaLoc[i] = np.median(ntg.Chain[i,:,NSTEPS/2:])
	lcarmaMedianThetaDelta = (lcarmaMedianThetaLoc[i] - ThetaMock[i])/ThetaMock[i]
	lcarmaMedianThetaDist += math.pow(lcarmaMedianThetaDelta, 2.0)
lcarmaMedianThetaDist = math.sqrt(lcarmaMedianThetaDist)
lcarmaMedianThetaDist /= (P + Q + 1)
print 'lcarma Median Fractional Theta Dist Per Param: %+4.3e'%(lcarmaMedianThetaDist)

if carma_pack_results_g:
	lcmcmcMedianThetaDist = 0.0
	lcmcmcMedianThetaLoc = np.zeros(P + Q + 1)
	for i in xrange(P + Q + 1):
		lcmcmcMedianThetaLoc[i] = np.median(cmcmcChain_g[i,:])
		lcmcmcMedianThetaDelta = (lcmcmcMedianThetaLoc[i] - ThetaMock[i])/ThetaMock[i]
		lcmcmcMedianThetaDist += math.pow(lcmcmcMedianThetaDelta, 2.0)
	lcmcmcMedianThetaDist = math.sqrt(lcmcmcMedianThetaDist)
	lcmcmcMedianThetaDist /= (P + Q + 1)
	print 'lcmcmc Median Fractional Theta Dist Per Param: %+4.3e'%(lcmcmcMedianThetaDist)

lcarmaMLEThetaDist = 0.0
bestWalker = np.where(ntg.LnPosterior[:,NSTEPS/2:] == np.max(ntg.LnPosterior[:,NSTEPS/2:]))[0][0]
bestStep = np.where(ntg.LnPosterior[:,NSTEPS/2:] == np.max(ntg.LnPosterior[:,NSTEPS/2:]))[1][0] + NSTEPS/2
lcarmaMLEThetaLoc = np.zeros(P + Q + 1)
for i in xrange(P + Q + 1):
	lcarmaMLEThetaLoc[i] = ntg.Chain[i,bestWalker,bestStep]
	lcarmaMLEThetaDelta = (lcarmaMLEThetaLoc[i] - ThetaMock[i])/ThetaMock[i]
	lcarmaMLEThetaDist += math.pow(lcarmaMLEThetaDelta, 2.0)
lcarmaMLEThetaDist = math.sqrt(lcarmaMLEThetaDist)
lcarmaMLEThetaDist /= (P + Q + 1)
print 'lcarma MLE Fractional Theta Dist Per Param: %+4.3e'%(lcarmaMLEThetaDist)

if carma_pack_results_g:
	lcmcmcMLEThetaDist = 0.0
	bestSample = np.where(ntg.LnPosterior[:,NSTEPS/2:] == np.max(ntg.LnPosterior[:,NSTEPS/2:]))[0][0] ## CHANGE THIS SO IT WORKS!!!!
	lcmcmcMLEThetaLoc = np.zeros(P + Q + 1)
	for i in xrange(P + Q + 1):
		lcmcmcMLEThetaLoc[i] = cmcmcChain_g[i,bestSample]
		lcmcmcMLEThetaDelta = (lcmcmcMLEThetaLoc[i] - ThetaMock[i])/ThetaMock[i]
		lcmcmcMLEThetaDist += math.pow(lcmcmcMLEThetaDelta, 2.0)
	lcmcmcMLEThetaDist = math.sqrt(lcmcmcMLEThetaDist)
	lcmcmcMLEThetaDist /= (P + Q + 1)
	print 'lcmcmc MLE Fractional Theta Dist Per Param: %+4.3e'%(lcmcmcMLEThetaDist)

if args.plot:
	fig2 = plt.figure(2, figsize = (fhgt, fhgt))
	plt.title(r'g-band C-AR Coeffs')
	scatPlot1 = plt.scatter(ntg.Chain[0,:,NSTEPS/2:], ntg.Chain[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')

	plt.axvline(x = ThetaMock[0], ymin = 0, ymax = 1, color = '#999999', linestyle = 'solid', label = r'True')
	plt.axhline(y = ThetaMock[1], xmin = 0, xmax = 1, color = '#999999')

	plt.axvline(x = lcarmaMedianThetaLoc[0], ymin = 0, ymax = 1, color = '#ef8a62', linestyle = 'dashed', label = r'lcarma Median')
	plt.axhline(y = lcarmaMedianThetaLoc[1], xmin = 0, xmax = 1, color = '#ef8a62', linestyle = 'dashed')

	plt.axvline(x = lcarmaMLEThetaLoc[0], ymin = 0, ymax = 1, color = '#67a9cf', linestyle = 'dashed', label = r'lcarma MLE')
	plt.axhline(y = lcarmaMLEThetaLoc[1], xmin = 0, xmax = 1, color = '#67a9cf', linestyle = 'dashed')

	if carma_pack_results_g:
		scatPlot1cmcmc = plt.scatter(cmcmcChain_g[0,:], cmcmcChain_g[1,:], c = cmcmcLnPosterior_g[:], marker = 'o', edgecolors = 'none')

		plt.axvline(x = lcmcmcMedianThetaLoc[0], ymin = 0, ymax = 1, color = '#ef8a62', linestyle = 'dotted', label = r'lcmcmc Median')
		plt.axhline(y = lcmcmcMedianThetaLoc[1], xmin = 0, xmax = 1, color = '#ef8a62', linestyle = 'dotted')

		plt.axvline(x = lcmcmcMLEThetaLoc[0], ymin = 0, ymax = 1, color = '#67a9cf', linestyle = 'dotted', label = r'lcmcmc MLE')
		plt.axhline(y = lcmcmcMLEThetaLoc[1], xmin = 0, xmax = 1, color = '#67a9cf', linestyle = 'dotted')

		plt.xlim(min(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[0,:])), max(np.nanmax(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[0,:])))
		plt.ylim(min(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[1,:])), max(np.nanmax(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[1,:])))
	else:
		plt.xlim(np.nanmin(ntg.Chain[0,:,NSTEPS/2:]), np.nanmax(ntg.Chain[0,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntg.Chain[1,:,NSTEPS/2:]), np.nanmax(ntg.Chain[1,:,NSTEPS/2:]))
	plt.legend()
	cBar1 = plt.colorbar(scatPlot1, orientation = 'horizontal')
	cBar1.set_label(r'$\ln \mathcal{P}$')
	plt.xlabel(r'$a_{1}$')
	plt.ylabel(r'$a_{2}$')

	fig3 = plt.figure(3, figsize = (fhgt, fhgt))
	plt.title(r'g-band C-MA Coeffs')
	scatPlot2 = plt.scatter(ntg.Chain[2,:,NSTEPS/2:], ntg.Chain[3,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], marker = 'o', edgecolors = 'none')

	plt.axvline(x = ThetaMock[2], ymin = 0, ymax = 1, color = '#999999', linestyle = 'solid', label = r'True')
	plt.axhline(y = ThetaMock[3], xmin = 0, xmax = 1, color = '#999999')

	plt.axvline(x = lcarmaMedianThetaLoc[2], ymin = 0, ymax = 1, color = '#ef8a62', linestyle = 'dashed', label = r'lcarma Median')
	plt.axhline(y = lcarmaMedianThetaLoc[3], xmin = 0, xmax = 1, color = '#ef8a62', linestyle = 'dashed')

	plt.axvline(x = lcarmaMLEThetaLoc[2], ymin = 0, ymax = 1, color = '#67a9cf', linestyle = 'dashed', label = r'lcarma MLE')
	plt.axhline(y = lcarmaMLEThetaLoc[3], xmin = 0, xmax = 1, color = '#67a9cf', linestyle = 'dashed')

	if carma_pack_results_g:
		scatPlot2cmcmc = plt.scatter(cmcmcChain_g[2,:], cmcmcChain_g[3,:], c = cmcmcLnPosterior_g[:], marker = 'o', edgecolors = 'none')

		plt.axvline(x = lcmcmcMedianThetaLoc[2], ymin = 0, ymax = 1, color = '#ef8a62', linestyle = 'dotted', label = r'lcmcmc Median')
		plt.axhline(y = lcmcmcMedianThetaLoc[3], xmin = 0, xmax = 1, color = '#ef8a62', linestyle = 'dotted')

		plt.axvline(x = lcmcmcMLEThetaLoc[2], ymin = 0, ymax = 1, color = '#67a9cf', linestyle = 'dotted', label = r'lcmcmc MLE')
		plt.axhline(y = lcmcmcMLEThetaLoc[3], xmin = 0, xmax = 1, color = '#67a9cf', linestyle = 'dotted')


		plt.xlim(min(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[2,:])), max(np.nanmax(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[2,:])))
		plt.ylim(min(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmin(cmcmcChain_g[3,:])), max(np.nanmax(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(cmcmcChain_g[3,:])))
	else:
		plt.xlim(np.nanmin(ntg.Chain[2,:,NSTEPS/2:]), np.nanmax(ntg.Chain[2,:,NSTEPS/2:]))
		plt.ylim(np.nanmin(ntg.Chain[3,:,NSTEPS/2:]), np.nanmax(ntg.Chain[3,:,NSTEPS/2:]))
	plt.legend()
	cBar2 = plt.colorbar(scatPlot2, orientation = 'horizontal')
	cBar2.set_label(r'$\ln \mathcal{P}$')
	plt.xlabel(r'$b_{0}$')
	plt.ylabel(r'$b_{1}$')

# Convert Theta -> Rho -> Tau
lcarmaTau_g = np.zeros((P + Q + 1, NWALKERS, NSTEPS))
for stepNum in xrange(NSTEPS):
	for walkerNum in xrange(NWALKERS):
		lcarmaRAR, lcarmaIAR, lcarmaRMA, lcarmaIMA = libcarma.timescales(P, Q, ntg.rootChain[:, walkerNum, stepNum])
		lcarmaTau_g[:, walkerNum, stepNum] = np.array(sorted([i for i in lcarmaRAR]) + sorted([i for i in lcarmaIAR]) + sorted([i for i in lcarmaRMA]) + sorted([i for i in lcarmaIMA]) + [ntg.rootChain[P + Q, walkerNum, stepNum]])

if args.plot:
	plt.figure(6, figsize = (fhgt, fhgt))
	plt.title(r'g-band C-AR Timescales')
	plt.scatter(lcarmaTau_g[0,:,NSTEPS/2:], lcarmaTau_g[1,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], cmap = cm.Blues, marker = 'o', edgecolors = 'none', zorder = 5)
	plt.axvline(x = TauMock[0], ymin = 0, ymax = 1, color = '#e34a33')
	plt.axhline(y = TauMock[1], xmin = 0, xmax = 1, color = '#e34a33')
	plt.xlabel(r'$\tau_{\mathrm{AR},0}$ (d)')
	plt.ylabel(r'$\tau_{\mathrm{AR},1}$ (d)')
	
	plt.figure(7, figsize = (fhgt, fhgt))
	plt.title(r'g-band C-MA Timescales')
	plt.scatter(lcarmaTau_g[2,:,NSTEPS/2:], lcarmaTau_g[3,:,NSTEPS/2:], c = ntg.LnPosterior[:,NSTEPS/2:], cmap = cm.Blues, marker = 'o', edgecolors = 'none', zorder = 5)
	plt.axvline(x = TauMock[2], ymin = 0, ymax = 1, color = '#e34a33')
	plt.axhline(y = TauMock[3], xmin = 0, xmax = 1, color = '#e34a33')
	plt.xlabel(r'$\tau_{\mathrm{MA},0}$ (d)')
	plt.ylabel(r'$A_{\mathrm{MA}}$ (Jy)')

if carma_pack_results_g:
	cmcmcRho_g = np.zeros((P + Q + 1, NSAMPLES))
	cmcmcTau_g = np.zeros((P + Q + 1, NSAMPLES))
	for sampleNum in xrange(NSAMPLES):
		cmcmcRho_g[:,sampleNum] = libcarma.roots(P, Q, cmcmcChain_g[:,sampleNum])
		cmcmcRAR, cmcmcIAR, cmcmcRMA, cmcmcIMA = libcarma.timescales(P, Q, (cmcmcRho_g[:,sampleNum]))
		try:
			cmcmcTau_g[:,sampleNum] = np.array(sorted([i for i in cmcmcRAR]) + sorted([i for i in cmcmcIAR]) + sorted([i for i in cmcmcRMA]) + sorted([i for i in cmcmcIMA]) + [cmcmcRho_g[P + Q, sampleNum]])
		except ValueError: # Sometimes Kelly's roots are repeated!!! This should not be allowed!
			pass

	if args.plot:
		plt.figure(6)
		plt.scatter(cmcmcTau_g[0,:], cmcmcTau_g[1,:], c = cmcmcLnPosterior_g[:], cmap = cm.Reds, marker = 'o', edgecolors = 'none', zorder = 0)
		plt.xlim(min(np.min(cmcmcTau_g[0,:]), np.min(lcarmaTau_g[0,:,NSTEPS/2:])), max(np.max(cmcmcTau_g[0,:]), np.max(lcarmaTau_g[0,:,NSTEPS/2:])))
		plt.ylim(min(np.min(cmcmcTau_g[1,:]), np.min(lcarmaTau_g[1,:,NSTEPS/2:])), max(np.max(cmcmcTau_g[1,:]), np.max(lcarmaTau_g[1,:,NSTEPS/2:])))
		plt.tight_layout()
	
		plt.figure(7)
		plt.scatter(cmcmcTau_g[2,:], cmcmcTau_g[3,:], c = cmcmcLnPosterior_g[:], cmap = cm.Reds, marker = 'o', edgecolors = 'none', zorder = 0)
		#plt.xlim(min(np.min(cmcmcTau_g[2,:]), np.min(lcarmaTau_g[2,:,NSTEPS/2:])), max(np.max(cmcmcTau_g[2,:]), np.max(lcarmaTau_g[2,:,NSTEPS/2:])))
		plt.xlim(np.min(lcarmaTau_g[2,:,NSTEPS/2:]), np.max(lcarmaTau_g[2,:,NSTEPS/2:]))
		plt.ylim(min(np.min(cmcmcTau_g[3,:]), np.min(lcarmaTau_g[3,:,NSTEPS/2:])), max(np.max(cmcmcTau_g[3,:]), np.max(lcarmaTau_g[3,:,NSTEPS/2:])))
		plt.tight_layout()

if args.plot:
	plt.show()

if args.stop:
	pdb.set_trace()