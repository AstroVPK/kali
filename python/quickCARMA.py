#!/usr/bin/env python
"""	Module that defines quickCARMA.

	Quickly run CARMA analysis on a lightcurve provided through code
	
	Test this module by running this script

"""
import math, cmath, time, sys, os, psutil, multiprocessing, pdb, operator
import numpy as np
import copy as copy
import random as random
import ConfigParser as CP
import argparse as AP
import cffi as cffi
try: 
	os.environ['DISPLAY']
except KeyError as Err:
	print "No display environment! Using matplotlib backend 'Agg'"
	import matplotlib
	matplotlib.use('Agg')

from bin._libcarma import ffi
from python.lc import LC

ffiObj = cffi.FFI()
try:
	libcarmaPath = str(os.environ['LIBCARMA'])
except KeyError as Err:
	print str(Err) + '. Exiting....'
	sys.exit(1)

C = ffi.dlopen(libcarmaPath + '/bin/libcarma.so.1.0.0')
new_uint = ffiObj.new_allocator(alloc = C._malloc_uint, free = C._free_uint)
new_int = ffiObj.new_allocator(alloc = C._malloc_int, free = C._free_int)
new_double = ffiObj.new_allocator(alloc = C._malloc_double, free = C._free_double)

class CARMAResultObj(object):

	def __init__(self, **kwargs):

		self.__dict__.update(kwargs)

	def _writeMCMC(self):
		"""	Write the Chain and LnLike out
		"""
		chainFile = open('test' + '_%d'%(self.p) + '_%d'%(self.q) + '.mcmc', 'w')
		line = "nsteps: %d\n"%(self.nsteps)
		chainFile.write(line)
		line = "nwalkers: %d\n"%(self.nwalkers)
		chainFile.write(line)
		line = "ndim: %d; p: %d; q: %d\n"%(self.ndims, self.p, self.q)
		chainFile.write(line)

		for stepNum in xrange(self.nsteps):
			for walkerNum in xrange(self.nwalkers):
				line = "stepNum: %d; walkerNum: %d; "%(stepNum, walkerNum)
				for pNum in xrange(self.p):
					line += "%+17.16e "%(self.Chain[stepNum,walkerNum,pNum])
				for qNum in xrange(self.p, self.p+self.q + 1):
					line += "%+17.16e "%(self.Chain[stepNum,walkerNum,qNum])
				line += "%+17.16e\n"%(self.LnLike[stepNum,walkerNum])
				chainFile.write(line)
		chainFile.close()

	def getArCoefs(self):

		return self.Chain[:,:,:self.p].reshape(-1, self.p)

	def getMaCoefs(self):
		
		return self.Chain[:,:,self.p:self.p+self.q+1].reshape(-1, self.q+1)

	def getLnLike(self):

		return self.LnLike.reshape(-1, 1)

	def test(self):
		print "Testing In Progress, Keep Out..."
		from matplotlib.pyplot import subplots, show
		fig, ax = subplots(1,1)
		ar = self.getArCoefs()
		x = ar[:,0]
		y = ar[:,1]
		ll = self.getLnLike()
		z = ll[:,0]
		best = z.argmin()
		print "AR Coefs:", ar[z.argmax()]
		print "Original Coefs:", 0.001, -0.75
		ax.scatter(x, y, c=z)
		show()

class quickCARMA(object):
	"""	Fit a C-ARMA model to the supplied LC.
	"""

	def __init__(self, t, y, yerr, mask = None, tolIR = 1.0e-3):
		'''Setup the class by building an LC object to work with'''
		assert len(t) == len(y) == len(yerr), "Arrays must be the same size"

		'''First, we'll just plug in the supplied arguments'''	
		self.LC = LC()
		self.LC.numCadences = len(t)
		self.LC.t = np.array(t)
		self.LC.y = np.array(y)
		self.LC.x = np.array(y) #Careful Here, Not Sure if this is best, look at quickCARMAFit ffi seclarations, might possibly skip assigning LC.x and just use an array of zeros for the ffi array
		self.LC.yerr = np.array(yerr)
		self.LC.meanFlux = np.mean(y)
		self.LC.cadence = np.arange(self.LC.numCadences)
		self.LC.tolIR = tolIR	
		self.LC.numObservations = self.LC.numCadences
	
		'''Now for the implicit attributes'''
		self.LC.T = self.LC.t[-1] - self.LC.t[0]	
		self.LC.t_incr = self.LC.t[1:] - self.LC.t[:-1]
		self.LC.dt = np.median(self.LC.t_incr)
	
		if mask is not None: #Was a mask provided or are we assuming everything is unmasked?
			assert len(mask) == len(self)
			self.LC.mask = np.array(mask)
		else:
			self.LC.mask = np.ones(len(self))

		self._setIR() #Is this LC irregularly sampled?



	def __len__(self): #This is for the sake of brevity really

		return self.LC.numCadences	

	def _setIR(self):
		'''Find out if the Light Curve is Irregularly Sampled'''
		#I promise the math works out the same
		if (np.absolute(2.0*(1 - 2*self.LC.dt/(self.LC.dt+self.LC.t_incr))) > self.LC.tolIR).any():
			self.LC.IR = True
		else:
			self.LC.IR = False

	def _checkPQ(self):
		"""	Check to see if supplied pMax etc... are valid."""
		if self.pMax < self.pMin:
			a = self.pMin
			self.pMin = self.pMax
			self.pMax = self.a
		try:
			qMaxInt = int(self.qMax)
			if qMaxInt < self.qMin:
				a = self.qMin
				self.qMin = self.qMax
				self.qMax = self.a
			if qMaxInt >= self.pMax:
				print 'Invalid pMax and qMax!'
				sys.exit(1)
		except ValueError as Err:
			pass

	def rdrand(self, nrands, rands):
		yORn = C._getRandoms(nrands, rands)
		for i in xrange(nrands):
			if rands[i] == 0:
				rands[i] = np.random.randint(1,4294967296)
		return yORn

	def quickCARMAFit(self, p, q, nsteps = 10, nwalkers = 4, sigmaFactor = 1e-2, maxSigmaFactor = 1e2, minTimescaleFactor = 1e-2, maxTimescaleFactor = 1e2, scatterFactor = 1e-1, nthreads = None, maxEvals = 1000, xTol = 160):
		'''Run CARMA on the lightcurve with the supplied arguments'''
		self.p, self.q = (p, q)
		ndims = self.p + self.q + 1	

		#setting up initial parameters
		if nthreads is None: nthreads = psutil.cpu_count(logical = False)
		sigma = np.std(self.LC.y) - np.median(self.LC.yerr) * sigmaFactor
		maxSigma = maxSigmaFactor*np.std(self.LC.y)
		minTimescale = minTimescaleFactor*min(self.LC.t_incr)
		maxTimescale = maxTimescaleFactor*self.LC.T

		randList = np.unique(-1.0*np.random.random_sample(1000)) #unique random numbers
		#pRoots: AR Roots
		#qRoots: MA Roots
		pRoots = np.random.choice(randList,self.p)		
		qRoots = np.random.choice(randList,self.q)		

		#aPoly: Auto-Regressive Polynomial
		#bPoly: Moving-Average Polynonmial
		aPoly, bPoly = map(np.polynomial.polynomial.polyfromroots, (pRoots, qRoots))
		aPoly = aPoly[:0:-1].real #Eliminate Constant Term
		divisor = bPoly[0].real
		bPoly = sigma*bPoly.real/divisor	
		xStart = np.concatenate((aPoly, bPoly))
	
		#Get all of the cffi/ffi stuff declared
		randomSeeds = ffiObj.new('unsigned int[4]')	
		cadence_cffi = ffiObj.new('int[]', self.LC.cadence.tolist())
		mask_cffi = ffiObj.new('double[]', self.LC.mask.tolist())
		t_cffi = ffiObj.new('double[]', self.LC.t.tolist())
		x_cffi = ffiObj.new('double[]', self.LC.x.tolist())
		y_cffi = ffiObj.new('double[]', self.LC.y.tolist())
		yerr_cffi = ffiObj.new('double[]', self.LC.yerr.tolist())
		Chain_cffi = ffiObj.new('double[]',nsteps*nwalkers*ndims)
		LnLike_cffi = ffiObj.new('double[]', nsteps*nwalkers)
		xStart_cffi = ffiObj.new('double[]', xStart.tolist())

		yORn = self.rdrand(4, randomSeeds)
		IR = 1 if self.LC.IR else 0
	
		#RUN CARMA!	
		C._fitCARMA(self.LC.dt, self.p, self.q, IR, self.LC.tolIR, scatterFactor, self.LC.numCadences, cadence_cffi, mask_cffi, t_cffi, y_cffi, yerr_cffi, maxSigma, minTimescale, maxTimescale, nthreads, nwalkers, nsteps, maxEvals, xTol, randomSeeds[0], randomSeeds[1], randomSeeds[2], randomSeeds[3], xStart_cffi, Chain_cffi, LnLike_cffi)

		#Convert back to usable numpy arrays 
		LnLike = np.array(map(None, LnLike_cffi)).reshape(nsteps, nwalkers)
		Deviances = -2.0*np.array(map(None, LnLike_cffi)).reshape(nsteps, nwalkers)
		Chain = np.array(map(None, Chain_cffi)).reshape(nsteps, nwalkers, ndims)

		#write out to a result object and return it so we can get our data back
		result = CARMAResultObj(p = self.p, q = self.q, lc = self.LC, nsteps = nsteps, nwalkers = nwalkers, ndims = ndims, Chain = Chain, LnLike = LnLike, Deviances = Deviances, sigmaFactor = sigmaFactor, maxSigmaFactor = maxSigmaFactor, minTimescaleFactor = minTimescaleFactor, maxTimescaleFactor = maxTimescaleFactor, scatterFactor = scatterFactor, nthreads = nthreads, maxEvals = maxEvals, xTol = xTol)
		
		return result

	def _make_00_runMCMC(self):
		self.dictDIC = dict()
		self.ResultFile = self.WorkingDirectory + self.prefix + '_CARMAResult.dat'
		try:
			Result = open(self.ResultFile,'w')
		except IOError as Err:
			print str(Err) + '. Exiting...'
		Result.close()
		if self.qMax.replace(' ','') == 'p-1':
			for p in xrange(self.pMin, self.pMax + 1):
				for q in xrange(self.qMin, p):
					self.p = p
					self.q = q
					self._doCARMAFit();
		else:
			self.qMax = int(self.qMax)
			for p in xrange(self.pMin, self.pMax + 1):
				for q in xrange(self.qMin, self.qMax + 1):
					if q >= p:
						break
					else:
						self.p = p
						self.q = q
						self._doCARMAFit()
		try:
			Result = open(self.ResultFile, 'a')
		except IOError as Err:
			print str(Err) + '. Exiting...'
		self.sortedDICVals = sorted(self.dictDIC.items(), key = operator.itemgetter(1))
		self.pBest = int(self.sortedDICVals[0][0].split()[0])
		self.qBest = int(self.sortedDICVals[0][0].split()[1])
		line = "Model Appropriateness (in descending order of appropriateness) & Relative Likelihood (i.e. Relative Likelihood of Minimal Infrmation Loss)\n"
		Result.write(line)
		line = "Model       DIC Value    Relative Likelihood\n"
		Result.write(line)
		line = "-----       ---------    -------------------\n"
		Result.write(line)
		for i in range(len(self.sortedDICVals)):
			RelProbOfMinInfLoss = 100.0*math.exp(0.5*(float(self.sortedDICVals[0][1]) - float(self.sortedDICVals[i][1])))
			line = '{:>4}   {:> 13.3f}    {:> 18.2f}%\n'.format(self.sortedDICVals[i][0], float(self.sortedDICVals[i][1]), RelProbOfMinInfLoss)
			Result.write(line);

		Result.close()

def test():

	import JacksTools.jio as jio
	data = jio.load('../examples/Demo00/Config_03302016140027.lc', delimiter = ' ')
	y = data['y']
	yerr = data['yerr']
	t = data['t']

	qCARMA = quickCARMA(t, y, yerr)
	result = qCARMA.quickCARMAFit(2, 0, nsteps = 1000, nwalkers = 4, nthreads = 2)
	result.test()
	pdb.set_trace()

if __name__ == '__main__':

	test()


