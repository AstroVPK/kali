import math
import cmath
import numpy as np
import unittest
import pdb

import matplotlib.pyplot as plt

import kali.mbhb


plt.ion()
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day


class TestIrregularLC(unittest.TestCase):

    def setUp(self):
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
        self.nt.set(self.theta)

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
