import math
import cmath
import numpy as np
import unittest
import time
import pdb

import matplotlib.pyplot as plt

import kali.s82
import kali.k2
import kali.crts
import kali.mbhb
import kali.carma


plt.ion()
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day
BURNSEED = 731647386
DISTSEED = 219038190
NOISESEED = 87238923


class TestSamplers(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.T = 1000.0
        self.TAR1 = 10.0
        self.Amp = 1.0
        self.rho = np.array([-1.0/self.TAR1, self.Amp])
        self.theta = kali.carma.coeffs(1, 0, self.rho)
        self.nt = kali.carma.CARMATask(1, 0)
        self.nt.set(self.dt, self.theta)
        self.ntMock = kali.carma.CARMATask(1, 0)
        self.ntMock.set(self.dt, self.theta)

    def tearDown(self):
        del self.nt
        del self.theta

    def test_sincSampler(self):
        nl = self.nt.simulate(duration=self.T)
        self.nt.observe(nl)
        nl.sampler = 'sincSampler'
        nlMock = nl.sample()
        startNT = time.time()
        self.nt.fit(nl)
        stopNT = time.time()
        durNT = stopNT - startNT
        startNTMOCK = time.time()
        self.ntMock.fit(nlMock)
        stopNTMOCK = time.time()
        durNTMOCK = stopNTMOCK - startNTMOCK
        self.assertLessEqual(durNTMOCK*5.0, durNT)

if __name__ == "__main__":
    unittest.main()
