#!/usr/bin/env python
"""	Module to write MCMC chains produced by Kelly's C-ARMA code in a format that Kasliwal's C-ARMA code can 
	then plot etc...

	Standard Usage
	'myMachine$ python writeKellyChains.py pickledSampleFile chainFilePath chainFilePrefix'
	Alternatively, one may import this module and call the function writeChains directly.

	Functions
	1. loadSample(pickledSampleFile): Read in given pickled file pickledSampleFile of MCMC samples. Returns 
	sample
	2. writeChains(sample, chainFilePath, chainFilePrefix): Write out given chain 'sample' obtained using 
	Kelly's C-ARMA code to file chainFilePath + chainFilePrefix + '_numP' + '_numQ.dat' in a format that 
	Kasliwal's C-ARMA library understands.
"""

import math as math
import cmath as cmath
import numpy as np
import cPickle as cP
import sys
import pdb

import carmcmc

def loadSample(pickledSampleFile):
	"""Read in a given pickled 'sample' file

	pickledSamplePath: Full path to pickled sample file output using Kelly's C-ARMA code.

	Assuming that using Kelly's C-ARMA code, we have executed
	>>> CARMA_Model = carmcmc.CarmaModel(t, y, yerr)
	>>> MLE, pqlist, AICc_list = CARMA_Model.choose_order(pmax, njobs=16)
	>>> sample = CARMA_Model.run_mcmc(nwalkers)
	where (t, y, yerr) is the lightcurve of the object that we are interested in. 'sample' is the MCMC chain 
	produced by Kelly's C-ARMA code. To save time and not re-run the Kelly C-ARMA code, we can save 'sample' 
	by pickling it using something like
	>>> Sample = open(pickledSamplePath, 'wb')
	>>> cPickle.dump(sample, Sample, -1)
	>>> Sample.close()
	This code requires the 'pickledSamplePath' that 'sample' is dumped to.
	"""
	samplesFile = open(pickledSampleFile,'rb')
	sample = cP.load(samplesFile)
	samplesFile.close()
	return sample

def writeChains(sample, chainFilePath, chainFilePrefix):
	"""	Write out a given chain 'sample' obtained using Kelly's C-ARMA code in a format that Kasliwal's C-ARMA
	library understands.

	sample: MCMC chain produced by Kelly's code.
	chainFilePath: Full path that the chain should be written out to (excluding the filename).
	chainFilePrefix: Prefix to use in chain file name. The full name used will be '<prefix>_numP_numQ.dat'

	Assuming that using Kelly's C-ARMA code, we have executed
	>>> CARMA_Model = carmcmc.CarmaModel(t, y, yerr)
	>>> MLE, pqlist, AICc_list = CARMA_Model.choose_order(pmax, njobs=16)
	>>> sample = CARMA_Model.run_mcmc(nwalkers)
	where (t, y, yerr) is the lightcurve of the object that we are interested in. 'sample' is the MCMC chain 
	produced by Kelly's C-ARMA code. We wish to write this 'sample' in a format that Kasliwal's C-ARMA 
	utilities can understand.
	"""
	ar_poly = sample.get_samples('ar_coefs')
	ma_coefs = sample.get_samples('ma_coefs')
	sigma = sample.get_samples('sigma')
	loglike = -1.0*sample.get_samples('loglik')
	ma_poly = ma_coefs*sigma
	numWalkers = ar_poly.shape[0]
	numSteps = 1
	numP = ar_poly.shape[1] - 1
	numQ = ma_poly.shape[1]
	numDims = numP + numQ
	chainFile = open(chainFilePath + chainFilePrefix + '_%d'%(numP) + '_%d'%(numQ - 1) + '.dat', 'w')
	line = "nsteps: %d\n"%(numSteps)
	chainFile.write(line)
	line = "nwalkers: %d\n"%(numWalkers)
	chainFile.write(line)
	line = "ndim: %d\n"%(numDims)
	chainFile.write(line)

	for i in xrange(numWalkers):
		line = "stepNum: %d; walkerNum: %d; "%(0, i)
		for j in xrange(numP):
			line += "%+17.16e "%(ar_poly[i, j + 1])
		for j in xrange(numQ):
			line += "%+17.16e "%(ma_poly[i, j])
		line += "%+17.16e\n"%(loglike[i])
		chainFile.write(line)
	chainFile.close()
	return 1

if __name__ == "__main__":
	pickledSampleFile = sys.argv[1]
	chainFilePath = sys.argv[2]
	chainFilePrefix = sys.argv[3]
	sample = loadSample(pickledSampleFile)
	writeChains(sample, chainFilePath, chainFilePrefix)

