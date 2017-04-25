import math
import cmath
import numpy as np
import unittest
import os
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

kalidir = os.environ['KALI']

class TestCopyMockLC(unittest.TestCase):

    def setUp(self):
        self.dt = 1.0
        self.T = 100.0
        self.TAR1 = 10.0
        self.Amp = 1.0
        self.rho = np.array([-1.0/self.TAR1, self.Amp])
        self.theta = kali.carma.coeffs(1, 0, self.rho)
        self.nt = kali.carma.CARMATask(1, 0)
        self.nt.set(self.dt, self.theta)

    def tearDown(self):
        del self.nt
        del self.theta

    def test_copyRegular(self):
        nl = self.nt.simulate(duration=self.T)
        self.nt.observe(nl)
        nlCopy = nl.copy()
        nl.t[0] = 100.0
        self.assertNotEqual(nl.t[0], nlCopy.t[0])

    def test_copyIrregular(self):
        tInOff = np.random.normal(0.0, self.dt/10.0, int(self.T/self.dt))
        tIn = np.require(np.zeros(int(self.T/self.dt)), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(int(self.T/self.dt)):
            tIn[i] = i*self.dt + tInOff[i]
        nl = self.nt.simulate(tIn=tIn)
        self.nt.observe(nl)
        nlCopy = nl.copy()
        nl.t[0] = 100.0
        self.assertNotEqual(nl.t[0], nlCopy.t[0])


@unittest.skipUnless(kali.s82.ONLINE, 's82 server offline! Skipping test...')
class TestCopySDSSLC(unittest.TestCase):

    def test_copy(self):
        nl = kali.s82.sdssLC(name='rand', band='g')
        nlCopy = nl.copy()
        nl.t[0] = 100.0
        self.assertNotEqual(nl.t[0], nlCopy.t[0])


class TestCopyK2LC(unittest.TestCase):

    def test_copy(self):
        nl = kali.k2.k2LC(name='205905563', band='Kep', campaign='c03', path=os.path.join(kalidir, 'examples',
                                                                                          'data'))
        nlCopy = nl.copy()
        nl.t[0] = 100.0
        self.assertNotEqual(nl.t[0], nlCopy.t[0])


class TestCopyCRTSLC(unittest.TestCase):

    def test_copy(self):
        nl = kali.crts.crtsLC(name='OJ287', band='V', path = os.path.join(kalidir, 'examples', 'data'))
        nlCopy = nl.copy()
        nl.t[0] = 100.0
        self.assertNotEqual(nl.t[0], nlCopy.t[0])

if __name__ == "__main__":
    unittest.main()
