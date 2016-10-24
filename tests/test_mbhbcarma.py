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
G = 6.67408e-11
c = 299792458.0
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
twoPi = 2.0*pi
Parsec = 3.0857e16
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day
SolarMass = 1.98855e30


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
        self.p = 2
        self.q = 1
        self.r = kali.mbhbcarma.MBHBCARMATask(self.p, self.q).r

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
        self.p = 2
        self.q = 1
        self.r = kali.mbhbcarma.MBHBCARMATask(self.p, self.q).r

    def tearDown(self):
        pass

    def test_createNewTask(self):
        newTask = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        self.assertEqual(newTask.r, 8, "newTask.r is not working!")
        self.assertEqual(newTask.p, self.p, "newTask.p is not working!")
        self.assertEqual(newTask.q, self.q, "newTask.q is not working!")
        self.assertEqual(newTask, eval(repr(newTask)))

    def test_changeExistingTask(self):
        pNew = 4
        qNew = 2
        newTask = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        newTask.reset(p=pNew, q=qNew)
        self.assertEqual(newTask.r, 8, "newTask.r is not working!")
        self.assertEqual(newTask.p, pNew, "newTask.p is not working!")
        self.assertEqual(newTask.q, qNew, "newTask.q is not working!")

    def test_checkTask(self):
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.check(theta_carma)
        theta_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                    0.0, 0.0, 0.0, 100.0, 0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.check(theta_mbhbcarma)
        self.assertEqual(res_carma, res_mbhbcarma)

    def test_setTask(self):
        dt = 0.1
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.set(dt, theta_carma)
        theta_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                    0.0, 0.0, 0.0, 100.0, 0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
        self.assertEqual(res_carma, res_mbhbcarma)

    def test_miscProps(self):
        dt = 0.1
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.set(dt, theta_carma)
        theta_mbhbcarma = np.array([SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity,
                                    0.0, 0.0, 0.0, 100.0, 0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
        self.assertAlmostEqual(newTask_carma.dt(), newTask_mbhbcarma.dt())
        np.testing.assert_array_almost_equal(newTask_carma.Theta(), newTask_mbhbcarma.Theta()[self.r:])
        np.testing.assert_array_equal(newTask_carma.list(), newTask_mbhbcarma.list())
        np.testing.assert_array_almost_equal(newTask_carma.Sigma(), newTask_mbhbcarma.Sigma())
        np.testing.assert_array_almost_equal(newTask_carma.X(), newTask_mbhbcarma.X())
        np.testing.assert_array_almost_equal(newTask_carma.P(), newTask_mbhbcarma.P())

    def test_simulateLC(self):
        dt = 0.1
        N2S = 1.0e-18
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.set(dt, theta_carma)
        newLC_carma = newTask_carma.simulate(duration=2000.0, fracNoiseToSignal=N2S, burnSeed=BURNSEED,
                                             distSeed=DISTSEED, noiseSeed=NOISESEED)
        newTask_carma.observe(newLC_carma, noiseSeed=NOISESEED)
        theta_mbhbcarma = np.array([0.01, 0.02, 3.0*DayInYear, 0.1, 0.0, 90.0, 0.0, newLC_carma.mean,
                                    0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
        newLC_mbhbcarma = newTask_mbhbcarma.simulate(duration=2000.0, fracNoiseToSignal=N2S,
                                                     burnSeed=BURNSEED, distSeed=DISTSEED,
                                                     noiseSeed=NOISESEED)
        newTask_mbhbcarma.observe(newLC_mbhbcarma, noiseSeed=NOISESEED)
        lcRatio = newLC_mbhbcarma.y/newLC_carma.y
        self.assertNotEqual(np.mean(lcRatio), 0.0)

    def test_logPrior(self):
        dt = 0.1
        N2S = 1.0e-18
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.set(dt, theta_carma)
        newLC_carma = newTask_carma.simulate(duration=2000.0, fracNoiseToSignal=N2S, burnSeed=BURNSEED,
                                             distSeed=DISTSEED, noiseSeed=NOISESEED)
        newTask_carma.observe(newLC_carma, noiseSeed=NOISESEED)
        logPrior_carma = newTask_carma.logPrior(newLC_carma)
        self.assertEqual(logPrior_carma, 0.0)
        theta_mbhbcarma = np.array([0.01, 0.02, 3.0*DayInYear, 0.1, 0.0, 90.0, 0.0, newLC_carma.mean,
                                    0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
        newLC_mbhbcarma = newTask_mbhbcarma.simulate(duration=2000.0, fracNoiseToSignal=N2S,
                                                     burnSeed=BURNSEED, distSeed=DISTSEED,
                                                     noiseSeed=NOISESEED)
        newTask_mbhbcarma.observe(newLC_mbhbcarma, noiseSeed=NOISESEED)
        logPrior_mbhbcarma = newTask_mbhbcarma.logPrior(newLC_mbhbcarma)
        self.assertEqual(logPrior_mbhbcarma, 0.0)

    def test_logLikelihood(self):
        dt = 0.1
        N2S = 1.0e-18
        theta_carma = np.array([0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_carma = kali.carma.CARMATask(self.p, self.q)
        res_carma = newTask_carma.set(dt, theta_carma)
        newLC_carma = newTask_carma.simulate(duration=2000.0, fracNoiseToSignal=N2S, burnSeed=BURNSEED,
                                             distSeed=DISTSEED, noiseSeed=NOISESEED)
        newTask_carma.observe(newLC_carma, noiseSeed=NOISESEED)
        logLikelihood_carma = newTask_carma.logLikelihood(newLC_carma)
        self.assertNotEqual(logLikelihood_carma, 0.0)
        theta_mbhbcarma = np.array([0.01, 0.02, 3.0*DayInYear, 0.1, 0.0, 90.0, 0.0, newLC_carma.mean,
                                    0.05846154, 0.00076923, 0.009461, 0.0236525])
        newTask_mbhbcarma = kali.mbhbcarma.MBHBCARMATask(self.p, self.q)
        res_mbhbcarma = newTask_mbhbcarma.set(dt, theta_mbhbcarma)
        newLC_mbhbcarma = newTask_mbhbcarma.simulate(duration=2000.0, fracNoiseToSignal=N2S,
                                                     burnSeed=BURNSEED, distSeed=DISTSEED,
                                                     noiseSeed=NOISESEED)
        newTask_mbhbcarma.observe(newLC_mbhbcarma, noiseSeed=NOISESEED)
        logLikelihood_mbhbcarma = newTask_mbhbcarma.logLikelihood(newLC_mbhbcarma)
        self.assertNotEqual(logLikelihood_mbhbcarma, 0.0)

if __name__ == "__main__":
    unittest.main()
