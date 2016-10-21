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
    import kali.mbhbcarma
except ImportError:
    print 'Cannot import kali.mbhbcarma! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.carma
except ImportError:
    print 'Cannot import kali.carma! kali is not setup. Setup kali by sourcing bin/setup.sh'
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

EarthMass = 3.0025138e-12  # 10^6 MSun
SunMass = 1.0e-6  # 10^6 MSun
EarthOrbitRadius = 4.84814e-6  # AU
SunOrbitRadius = 4.84814e-6*(EarthMass/SunMass)  # AU
Period = 31557600.0/86164.090530833  # Day
EarthOrbitEccentricity = 0.0167


@unittest.skipIf(skipWorking, 'Works!')
class TestConversions10(unittest.TestCase):

    def setUp(self):
        self.p = 1
        self.q = 0
        self.r = kali.mbhbcarma.MBHBCARMATask(self.p, self.q).r

    def tearDown(self):
        pass

    def test_coeffs(self):
        rho_carma = np.array([-1.0/50.0, 1.0])
        theta_carma = kali.carma.coeffs(self.p, self.q, rho_carma)
        rho_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                  0.0, 0.0, 0.0, 100.0, -1.0/50.0, 1.0])
        theta_mbhbcarma = kali.mbhbcarma.coeffs(self.p, self.q, rho_mbhbcarma)
        np.testing.assert_array_almost_equal(rho_mbhbcarma[0:self.r], theta_mbhbcarma[0: self.r])
        self.assertAlmostEqual(theta_carma[0], theta_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(theta_carma[1], theta_mbhbcarma[self.r + 1])

    def test_timescales(self):
        rho_carma = np.array([-1.0/50.0, 1.0])
        tau_carma = kali.carma.timescales(self.p, self.q, rho_carma)
        rho_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                  0.0, 0.0, 0.0, 100.0, -1.0/50.0, 1.0])
        tau_mbhbcarma = kali.mbhbcarma.timescales(self.p, self.q, rho_mbhbcarma)
        np.testing.assert_array_almost_equal(rho_mbhbcarma[0:self.r], tau_mbhbcarma[0: self.r])
        self.assertAlmostEqual(tau_carma[0], tau_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(tau_carma[1], tau_mbhbcarma[self.r + 1])

    def test_roots(self):
        theta_carma = np.array([0.02, 0.2])
        rho_carma = kali.carma.roots(self.p, self.q, theta_carma)
        theta_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                    0.0, 0.0, 0.0, 100.0, 0.02, 0.2])
        rho_mbhbcarma = kali.mbhbcarma.roots(self.p, self.q, theta_mbhbcarma)
        np.testing.assert_array_almost_equal(theta_mbhbcarma[0:self.r], rho_mbhbcarma[0: self.r])
        self.assertAlmostEqual(rho_carma[0], rho_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(rho_carma[1], rho_mbhbcarma[self.r + 1])


@unittest.skipIf(skipWorking, 'Works!')
class TestCoeffs21(unittest.TestCase):

    def setUp(self):
        self.r = 8
        self.p = 2
        self.q = 1

    def tearDown(self):
        pass

    def test_coeffs(self):
        rho_carma = np.array([-1.0/50.0, -1.0/26, -1.0/2.5, 1.0])
        theta_carma = kali.carma.coeffs(self.p, self.q, rho_carma)
        rho_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                  0.0, 0.0, 0.0, 100.0, -1.0/50.0, -1.0/26, -1.0/2.5, 1.0])
        theta_mbhbcarma = kali.mbhbcarma.coeffs(self.p, self.q, rho_mbhbcarma)
        np.testing.assert_array_almost_equal(rho_mbhbcarma[0:self.r], theta_mbhbcarma[0: self.r])
        self.assertAlmostEqual(theta_carma[0], theta_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(theta_carma[1], theta_mbhbcarma[self.r + 1])
        self.assertAlmostEqual(theta_carma[2], theta_mbhbcarma[self.r + 2])
        self.assertAlmostEqual(theta_carma[3], theta_mbhbcarma[self.r + 3])

    def test_timescales(self):
        rho_carma = np.array([-1.0/50.0, -1.0/26, -1.0/2.5, 1.0])
        tau_carma = kali.carma.timescales(self.p, self.q, rho_carma)
        rho_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                  0.0, 0.0, 0.0, 100.0, -1.0/50.0, -1.0/26, -1.0/2.5, 1.0])
        tau_mbhbcarma = kali.mbhbcarma.timescales(self.p, self.q, rho_mbhbcarma)
        np.testing.assert_array_almost_equal(rho_mbhbcarma[0:self.r], tau_mbhbcarma[0: self.r])
        self.assertAlmostEqual(tau_carma[0], tau_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(tau_carma[1], tau_mbhbcarma[self.r + 1])
        self.assertAlmostEqual(tau_carma[2], tau_mbhbcarma[self.r + 2])
        self.assertAlmostEqual(tau_carma[3], tau_mbhbcarma[self.r + 3])

    def test_roots(self):
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        rho_carma = kali.carma.roots(self.p, self.q, theta_carma)
        theta_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                    0.0, 0.0, 0.0, 100.0, 0.05846154, 0.00076923, 0.009461, 0.0236525])
        rho_mbhbcarma = kali.mbhbcarma.roots(self.p, self.q, theta_mbhbcarma)
        np.testing.assert_array_almost_equal(theta_mbhbcarma[0:self.r], rho_mbhbcarma[0: self.r])
        self.assertAlmostEqual(rho_carma[0], rho_mbhbcarma[self.r + 0])
        self.assertAlmostEqual(rho_carma[1], rho_mbhbcarma[self.r + 1])
        self.assertAlmostEqual(rho_carma[2], rho_mbhbcarma[self.r + 2])
        self.assertAlmostEqual(rho_carma[3], rho_mbhbcarma[self.r + 3])


@unittest.skipIf(skipWorking, 'Works!')
class TestMakeTask(unittest.TestCase):

    def setUp(self):
        self.p = 3
        self.q = 2

    def tearDown(self):
        pass

    def test_createNewTask(self):
        newTask = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        self.assertEqual(newTask.r, 8, "newTask.r is not working!")
        self.assertEqual(newTask.p, self.p, "newTask.p is not working!")
        self.assertEqual(newTask.q, self.q, "newTask.q is not working!")
        self.assertEqual(newTask, eval(repr(newTask)))


if __name__ == "__main__":
    unittest.main()
