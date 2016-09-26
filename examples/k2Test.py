import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
plt.ion()

import libcarma
import k2

lc212141173 = k2.k2LC(name = '212141173', campaign = 'c05', band = 'Kepler', pwd = os.path.join(os.environ['KALI'],'examples/data'), processing = 'vj')

nt = libcarma.basicTask(1, 0)
nt.set(lc212141173.dt, libcarma.coeffs(1, 0, np.array([-1.0/20, 100.0])))

print "LnLikelihood: %+4.3e"%(nt.logLikelihood(lc212141173))

lc212141173.plot()
lc212141173.plotacvf()
lc212141173.plotacf()
lc212141173.plotsf()

pdb.set_trace()
