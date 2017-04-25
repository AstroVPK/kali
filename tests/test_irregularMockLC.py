import math
import cmath
import numpy as np
import unittest
import pdb

import matplotlib.pyplot as plt

import kali.mbhb
import kali.carma


plt.ion()
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day
BURNSEED = 731647386
DISTSEED = 219038190
NOISESEED = 87238923


class TestIrregularMBHB(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.eccentricity = 0.6
        self.omega1_1 = 0.0
        self.inclination = 90.0
        self.tau = 0.0
        self.flux = 100.0
        self.theta = np.array([
            self.a1, self.a2, self.period, self.eccentricity, self.omega1_1, self.inclination, self.tau,
            self.flux])
        self.nt = kali.mbhb.MBHBTask()
        self.nt.set(self.dt, self.theta)

    def tearDown(self):
        del self.nt
        del self.theta

    def test_irregularLC(self):
        nl1 = self.nt.simulate(duration=self.period*10.0)
        tInOff = np.random.normal(0.0, (nl1.t[1] - nl1.t[0])/10.0, nl1.t.shape[0])
        tIn = nl1.t + tInOff
        nl2 = self.nt.simulate(tIn=tIn)
        self.assertNotEqual(np.mean(nl1.t - nl2.t), 0.0)
        self.assertNotEqual(np.mean(nl2.t), 0.0)


class TestIrregularCARMA(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.TAR1 = 75.0
        self.TAR2 = 10.0
        self.TMA1 = 2.0
        self.Amp = 1.0
        self.rho = np.array([
            -1.0/self.TAR1, -1.0/self.TAR2, -1.0/self.TMA1, self.Amp])
        self.theta = kali.carma.coeffs(2, 1, self.rho)
        self.nt = kali.carma.CARMATask(2, 1)
        self.nt.set(self.dt, self.theta)
        self.nl1 = self.nt.simulate(duration=300.0, burnSeed=BURNSEED, distSeed=DISTSEED)

    def tearDown(self):
        del self.nl1
        del self.nt
        del self.rho
        del self.theta

    def test_irregularSimulate(self):
        tInOff = np.random.normal(0.0, (self.nl1.t[1] - self.nl1.t[0])/10.0, self.nl1.t.shape[0])
        tIn = self.nl1.t + tInOff
        nl2 = self.nt.simulate(tIn=tIn, burnSeed=BURNSEED, distSeed=DISTSEED)
        self.assertNotEqual(np.mean(self.nl1.t - nl2.t), 0.0)
        self.assertNotEqual(np.mean(nl2.t), 0.0)

    def test_irregularExtend(self):
        nl1End = math.ceil(self.nl1.t[-1])
        tInOff = np.random.normal(0.0, (self.nl1.t[1] - self.nl1.t[0])/10.0, self.nl1.t.shape[0])
        tIn = np.require(np.zeros(self.nl1.t.shape[0]), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(self.nl1.t.shape[0]):
            tIn[i] = i*self.nl1.dt + nl1End + tInOff[i]
        self.nt.extend(self.nl1, tIn=tIn, distSeed=DISTSEED, noiseSeed=NOISESEED)
        self.assertNotEqual(np.mean(self.nl1.t[0:self.nl1.t.shape[0]/2]),
                            np.mean(self.nl1.t[self.nl1.t.shape[0]/2:]))

if __name__ == "__main__":
    unittest.main()
