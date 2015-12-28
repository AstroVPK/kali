import math as math
import cmath as cmath
import numpy as np
from scipy.stats import bernoulli
import pdb

Path = '/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/'
RegularFile = 'Regular.dat'
MissingFile = 'Missing.dat'
IrregularFile = 'Irregular.dat'

Regular = np.loadtxt(Path + RegularFile, skiprows = 7)
numCadences_Regular = Regular.shape[0]

Prob = 0.1
ProbList = bernoulli.rvs(Prob, size = numCadences_Regular)
numCadences_Missing = np.sum(ProbList)
numCadences_Irregular = np.sum(ProbList)

Missing = open(Path + MissingFile, 'w')
line = "#ConfigFileHash: %s\n"%('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a')
Missing.write(line)
line = "#SuppliedLCHash: %s\n"%('')
Missing.write(line)
line = "#numCadences: %d\n"%(numCadences_Regular)
Missing.write(line)
line = "#numObservations: %d\n"%(numCadences_Missing)
Missing.write(line)
line = "#meanFlux: %+17.16e\n"%(0.0)
Missing.write(line)
line = "#LnLike: %+17.16e\n"%(0.0)
Missing.write(line)
line = "#cadence mask t x y yerr\n"
Missing.write(line)

Irregular = open(Path + IrregularFile, 'w')
line = "#ConfigFileHash: %s\n"%('e338349c2ce27cd3daa690704386d14c6299d410efe52e3df9c5e1ca75c8347d32782aa7e289514b95cc8901ad3a88b87cb56e1925392968d4471fb480e1e37a')
Irregular.write(line)
line = "#SuppliedLCHash: %s\n"%('')
Irregular.write(line)
line = "#numCadences: %d\n"%(numCadences_Irregular)
Irregular.write(line)
line = "#numObservations: %d\n"%(numCadences_Irregular)
Irregular.write(line)
line = "#meanFlux: %+17.16e\n"%(0.0)
Irregular.write(line)
line = "#LnLike: %+17.16e\n"%(0.0)
Irregular.write(line)
line = "#cadence mask t x y yerr\n"
Irregular.write(line)

IrregularCounter = 0
for i in xrange(numCadences_Regular):
	if ProbList[i] == 1:
		line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(IrregularCounter, Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5])
		Irregular.write(line)
		IrregularCounter += 1
		line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(int(Regular[i,0]), Regular[i,1], Regular[i,2], Regular[i,3], Regular[i,4], Regular[i,5])
	else:
		line = "%d %1.0f %+17.16e %+17.16e %+17.16e %+17.16e\n"%(int(Regular[i,0]), 0.0, Regular[i,2], 0.0, 0.0, 1.3407807929942596e+154)
	Missing.write(line)

Missing.close()
Irregular.close()