import numpy as np
import os
import pdb

import libcarma
import k2

LC = k2.k2LC(name = 'ktwo212141173-c05_llc.csv', band = 'Kepler', pwd = os.path.join(os.environ['LIBCARMA'],'examples/data'), lctype = 'cal')

nt = libcarma.basicTask(1, 0)
nt.set(LC.dt, libcarma.coeffs(1, 0, np.array([-1.0/20, 100.0])))

print "LnLikelihood: %+4.3e"%(nt.logLikelihood(LC))