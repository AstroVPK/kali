import math
import numpy as np
import copy
import unittest
import random
import psutil
import sys
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as colormap

try:
    import kali.carma
except ImportError:
    print 'Cannot import kali.carma! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.s82
except ImportError:
    print 'Cannot import kali.s82! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

plt.ion()
skipWorking = False
BURNSEED = 731647386
DISTSEED = 219038190
NOISESEED = 87238923
SAMPLESEED = 36516342
ZSSEED = 384789247
WALKERSEED = 738472981
MOVESEED = 131343786
XSEED = 2348713647

@unittest.skipIf(skipWorking, 'Works!')
class TestSmooth(unittest.TestCase):
    def setUp(self):
        self.p = 2
        self.q = 1
        self.nWalkers = psutil.cpu_count(logical=True)
        self.nSteps = 200
        self.dt = 0.01
        self.T = 1000.0
        self.sincWidth = 2.0
        self.sincCenter = self.T/2.0
        self.newTask = kali.carma.CARMATask(self.p, self.q, nwalkers=self.nWalkers, nsteps=self.nSteps)

    def tearDown(self):
        del self.newTask

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        newLC = kali.s82.sdssLC(name='rand', band=r'r')
        self.newTask.fit(newLC)
        self.newTask.set(newLC.dt, self.newTask.bestTheta)
        self.newTask.smooth(newLC, stopT=1.25*newLC.t[-1])
        self.assertNotEqual(newLC.tSmooth[-1], newLC.t[-1])

if __name__ == "__main__":
    unittest.main()
