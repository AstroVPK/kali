#!/usr/bin/env python

import argparse as argparse, shutil, os, sys, math, socket, time, pdb
import numpy as np, scipy.stats as spstats
HOST = socket.gethostname()
print 'HOST: %s'%(str(HOST))

from python.plotPSD import plotPSDTask
from python.makeMockLC import makeMockLCTask
from python.plotSuppliedLC import plotSuppliedLCTask
from python.fitCARMA import fitCARMATask

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("pwd", help = "Path to Working Directory")
	parser.add_argument("cf", help = "Configuration File")
	parser.add_argument("-o", "--old", help = "DateTime of run to be used")
	parser.add_argument("-v", "--verbose", help = "Verbose T/F")
	parser.add_argument("-p", "--prob", help = "Probability of retaining a point in lc")
	parser.add_argument("-pS1", "--probSuper1", help = "Location of sinc peak in time for computing probability")
	parser.add_argument("-pS2", "--probSuper2", help = "Divisor of total length of sinc")
	args = parser.parse_args()

	TestFile = lambda file_name: os.path.isfile(os.path.join(args.pwd, file_name))
	'''Check if a file is contained in the working directory'''

	if args.old:
		
		if not TestFile(args.cf.split('.')[0] + '_' + args.old + '.lc'):
			print args.pwd + args.cf.split('.')[0] + '_' + args.old + '_LC.dat not found!'
			sys.exit(1)
		Stamp = args.old
	else:
		TimeStr = time.strftime("%m%d%Y") + time.strftime("%H%M%S")
		Stamp = TimeStr
	plotPSDTask(args.pwd, args.cf, Stamp).run()
	makeMockLCTask(args.pwd, args.cf, Stamp).run()

	if not TestFile('Regular.lc'):
		src = os.path.join(args.pwd, ''.join((args.cf.split('.')[0],'_',Stamp,'.lc')))
		dst = os.path.join(args.pwd,'Regular.lc')
		shutil.copy(src, dst)

	if not (TestFile('Irregular.lc') and TestFile('Missing.lc')):

		RegularFile = 'Regular.lc'
		MissingFile = 'Missing.lc'
		IrregularFile = 'Irregular.lc'
		Regular = np.loadtxt(os.path.join(args.pwd, RegularFile), skiprows = 7)
		numCadences_Regular = Regular.shape[0]

		if args.probSuper1 and args.probSuper2 and args.prob is None:
			timePeriod = float(args.probSuper1)
			multiplier = float(args.probSuper2)
			dt = np.median(Regular[1:,2] - Regular[:-1,2])
			ProbList = np.array([spstats.bernoulli.rvs(abs(np.sinc((i*dt - timePeriod)/(numCadences_Regular*dt*multiplier))), size = 1) for i in xrange(numCadences_Regular)])

		if args.prob is None and args.probSuper1 is None and args.probSuper2 is None:
			args.prob  = 0.5
		if args.prob:
			if (float(args.prob) >= 1.0) or (float(args.prob) <= 0.0):
				raise RuntimeError('prob must be between 0.0 and 1.0')
			ProbList = spstats.bernoulli.rvs(float(args.prob), size = numCadences_Regular)

		numCadences_Missing = np.sum(ProbList)
		numCadences_Irregular = np.sum(ProbList)

		Mlines = [] #Prepare the lines for MissingFile
		Mlines.append("#ConfigFileHash: %s" % ('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a'))
		Mlines.append("#SuppliedLCHash: %s" % (''))
		Mlines.append("#numCadences: %d" % (numCadences_Regular))
		Mlines.append("#numObservations: %d" % (numCadences_Missing))
		Mlines.append("#meanFlux: %+17.16e" % (0.0))
		Mlines.append("#LnLike: %+17.16e" % (0.0))
		Mlines.append("#cadence mask t x y yerr")

		Ilines = [] #Prepare the lines for Irregular File
		Ilines.append("#ConfigFileHash: %s" % ('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a'))
		Ilines.append("#SuppliedLCHash: %s" % (''))
		Ilines.append("#numCadences: %d" % (numCadences_Irregular))
		Ilines.append("#numObservations: %d" % (numCadences_Irregular))
		Ilines.append("#meanFlux: %+17.16e" % (0.0))
		Ilines.append("#LnLike: %+17.16e" % (0.0))
		Ilines.append("#cadence mask t x y yerr")
		
		IrregularCounter = iter(xrange(numCadences_Regular))
		for i in xrange(numCadences_Regular):
			if ProbList[i] == 1:
				Ilines.append("%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e" % (IrregularCounter.next(), Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5]))
				Mlines.append("%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e" % (int(Regular[i,0]), Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5]))
			else:
				Mlines.append("%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e" % (int(Regular[i,0]), 0.0, Regular[i,2], 0.0, 0.0, 1.3407807929942596e+154))
		
		with open(os.path.join(args.pwd, MissingFile), 'w') as Missing:
			Missing.write('\n'.join(Mlines))
		with open(os.path.join(args.pwd, IrregularFile), 'w') as Irregular:
			Irregular.write('\n'.join(Ilines))		

	TimeStr = 'MMDDYYYYHHMMSS'
	for tag in ['Irregular','Missing','Regular']:

		if not TestFile(TimeStr.join(('%s_' % tag, '.log'))):
			plotSuppliedLCTask(args.pwd, '.'.join((tag,'ini')), TimeStr).run()
		if not TestFile(TimeStr.join(('%s_' % tag, '_CARMAResults.dat'))):
			fitCARMATask(args.pwd, '.'.join((tag, 'ini')), TimeStr).run()

if __name__ == '__main__':
	main()

