import math
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt
import pdb

plt.ion()

try:
    import mbhb
except ImportError:
    print 'mbhbs is not setup. Setup mbhb by sourcing bin/setup.sh'
    sys.exit(1)

G = 6.67408e-11
c = 299792458.0
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
twoPi = 2.0*pi
Parsec = 3.0857e16
Day = 86164.090530833
Year = 31557600.0
DayInYear = Year/Day
SolarMass = 1.98855e30

skipLnLikelihood = False
skipWorking = False
doPlot = False


class TestPeriod(unittest.TestCase):

    def test_period(self):
        nt = mbhb.MBHBTask()
        EarthMass = 3.0025138e-12  # 10^6 MSun
        SunMass = 1.0e-6  # 10^6 MSun
        EarthOrbitRadius = 4.84814e-6  # AU
        SunOrbitRadius = 4.84814e-6*(EarthMass/SunMass)  # AU
        Period = 31557600.0/86164.090530833  # Day
        EarthOrbitEccentricity = 0.0167
        Theta = np.array(
            [SunOrbitRadius, EarthOrbitRadius, Period, EarthOrbitEccentricity, 0.0, 0.0, 0.0, 100.0])
        res = nt.set(Theta)
        self.assertEqual(res, 0)
        self.assertAlmostEqual(nt.m1(), SunMass, 3)
        self.assertAlmostEqual(nt.m2(), EarthMass, 3)


class TestNoInclination(unittest.TestCase):

    def setUp(self):
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.eccentricity = 0.0
        self.omega1_1 = 0.0
        self.omega1_2 = 180.0
        self.inclination = 0.0
        self.tau = 0.0
        self.flux = 100.0
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_1, self.inclination, self.tau,
                self.flux])
        self.Theta2 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_2, self.inclination, self.tau,
                self.flux])
        self.nt1 = mbhb.MBHBTask()
        self.nt2 = mbhb.MBHBTask()
        self.nt1.set(self.Theta1)
        self.nt2.set(self.Theta2)

    def tearDown(self):
        del self.Theta1
        del self.Theta2
        del self.nt1
        del self.nt2

    def test_masses(self):
        self.assertEqual(self.nt1.q(), (self.a1/self.a2))

    @unittest.skipIf(skipWorking, 'Works!')
    def test_beamingFactor(self):
        self.nt1(0.0)  # We start at Day 0
        bF1 = self.nt1.beamingFactor1()  # Get the beaming factor

        self.nt1(self.nt1.Day)  # Now step to Day 1
        bF2 = self.nt1.beamingFactor1()  # Get the beaming factor again
        self.assertEqual(bF1, bF2)  # Since the orbital inclination is zero, these should be the same.

    @unittest.skipIf(skipWorking, 'Works!')
    def test_lightCurve(self):
        nl1 = self.nt1.simulate(self.nt1.period()*10.0)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0)
        self.assertEqual(len(np.where((nl1.x - nl2.x) != 0.0)[0].tolist()), 0)  # Both have constant beaming


class TestInclinationNoNoise(unittest.TestCase):

    def setUp(self):
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.eccentricity = 0.0
        self.omega1_1 = 0.0
        self.omega1_2 = 180.0
        self.inclination = 90.0
        self.tau = 0.0
        self.flux = 100.0
        nstepsVal = 1
        nwalkersVal = 8
        maxEvalsVal = 1000
        xTolVal = 1.0e-2
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_1, self.inclination, self.tau,
                self.flux])
        self.Theta2 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_2, self.inclination, self.tau,
                self.flux])
        self.nt1 = mbhb.MBHBTask()
        self.nt2 = mbhb.MBHBTask()
        self.nt1.set(self.Theta1)
        self.nt2.set(self.Theta2)

    def tearDown(self):
        del self.Theta1
        del self.Theta2
        del self.nt1
        del self.nt2

    @unittest.skipIf(skipWorking, 'Works!')
    def test_masses(self):
        self.assertEqual(self.nt1.q(), (self.a1/self.a2))

    @unittest.skipIf(skipWorking, 'Works!')
    def test_beamingFactor(self):
        self.nt1(0.0)  # We start at Day 0
        bF1 = self.nt1.beamingFactor1()  # Get the beaming factor
        self.nt1(self.nt1.Day)  # Now step to Day 1
        bF2 = self.nt1.beamingFactor1()  # Get the beaming factor again
        self.assertNotEqual(bF1, bF2)  # Since the orbital inclination is 90, these should not be the same.

    @unittest.skipIf(skipLnLikelihood or skipWorking, 'Skipping LnLikelihood tests')
    def test_lightCurve(self):
        n2s = 1.0e-18
        nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal=n2s)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal=n2s)
        self.nt1.observe(nl1)
        self.nt2.observe(nl2)

        LnLike11 = self.nt1.logLikelihood(nl1)
        LnLike22 = self.nt2.logLikelihood(nl2)
        LnLike12 = self.nt1.logLikelihood(nl2)
        LnLike21 = self.nt2.logLikelihood(nl1)

        self.assertGreater(LnLike11, LnLike12)
        self.assertGreater(LnLike11, LnLike21)

        self.assertGreater(LnLike22, LnLike12)
        self.assertGreater(LnLike22, LnLike21)


class TestInclinationNoise(unittest.TestCase):

    def setUp(self):
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.eccentricity = 0.0
        self.omega1_1 = 0.0
        self.omega1_2 = 180.0
        self.inclination = 90.0
        self.tau = 0.0
        self.flux = 100.0
        nstepsVal = 1
        nwalkersVal = 8
        maxEvalsVal = 1000
        xTolVal = 1.0e-2
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_1, self.inclination, self.tau,
                self.flux])
        self.Theta2 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1_2, self.inclination, self.tau,
                self.flux])
        self.nt1 = mbhb.MBHBTask()
        self.nt2 = mbhb.MBHBTask()
        self.nt1.set(self.Theta1)
        self.nt2.set(self.Theta2)

    def tearDown(self):
        del self.Theta1
        del self.Theta2
        del self.nt1
        del self.nt2

    @unittest.skipIf(skipWorking, 'Works!')
    def test_masses(self):
        self.assertEqual(self.nt1.q(), (self.a1/self.a2))

    @unittest.skipIf(skipWorking, 'Works!')
    def test_beamingFactor(self):
        self.nt1(0.0)  # We start at Day 0
        bF1 = self.nt1.beamingFactor1()  # Get the beaming factor
        self.nt1(self.nt1.Day)  # Now step to Day 1
        bF2 = self.nt1.beamingFactor1()  # Get the beaming factor again
        self.assertNotEqual(bF1, bF2)  # Since the orbital inclination is zero, these should be the same.

    @unittest.skipIf(skipLnLikelihood or skipWorking, 'Skipping LnLikelihood tests')
    def test_lightCurveVLowNoise(self):
        n2s = 1.0e-4
        nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal=n2s)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal=n2s)
        self.nt1.observe(nl1)
        self.nt2.observe(nl2)

        LnLike11 = self.nt1.logLikelihood(nl1)
        LnLike22 = self.nt2.logLikelihood(nl2)
        LnLike12 = self.nt1.logLikelihood(nl2)
        LnLike21 = self.nt2.logLikelihood(nl1)

        self.assertGreater(LnLike11, LnLike12)
        self.assertGreater(LnLike11, LnLike21)

        self.assertGreater(LnLike22, LnLike12)
        self.assertGreater(LnLike22, LnLike21)

    @unittest.skipIf(skipLnLikelihood or skipWorking, 'Skipping LnLikelihood tests')
    def test_lightCurveLowNoise(self):
        n2s = 1.0e-3
        nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal=n2s)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal=n2s)
        self.nt1.observe(nl1)
        self.nt2.observe(nl2)

        LnLike11 = self.nt1.logLikelihood(nl1)
        LnLike22 = self.nt2.logLikelihood(nl2)
        LnLike12 = self.nt1.logLikelihood(nl2)
        LnLike21 = self.nt2.logLikelihood(nl1)

        self.assertGreater(LnLike11, LnLike12)
        self.assertGreater(LnLike11, LnLike21)

        self.assertGreater(LnLike22, LnLike12)
        self.assertGreater(LnLike22, LnLike21)

    @unittest.skipIf(skipLnLikelihood or skipWorking, 'Skipping LnLikelihood tests')
    def test_lightCurveNoise(self):
        n2s = 1.0e-2
        nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal=n2s)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal=n2s)
        self.nt1.observe(nl1)
        self.nt2.observe(nl2)

        LnLike11 = self.nt1.logLikelihood(nl1)
        LnLike22 = self.nt2.logLikelihood(nl2)
        LnLike12 = self.nt1.logLikelihood(nl2)
        LnLike21 = self.nt2.logLikelihood(nl1)

        self.assertGreater(LnLike11, LnLike12)
        self.assertGreater(LnLike11, LnLike21)

        self.assertGreater(LnLike22, LnLike12)
        self.assertGreater(LnLike22, LnLike21)

    @unittest.skipIf(skipLnLikelihood or skipWorking, 'Skipping LnLikelihood tests')
    def test_lightCurveHighNoise(self):
        n2s = 5.0e-2
        nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal=n2s)
        nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal=n2s)
        self.nt1.observe(nl1)
        self.nt2.observe(nl2)

        LnLike11 = self.nt1.logLikelihood(nl1)
        LnLike22 = self.nt2.logLikelihood(nl2)
        LnLike12 = self.nt1.logLikelihood(nl2)
        LnLike21 = self.nt2.logLikelihood(nl1)

        self.assertGreater(LnLike11, LnLike12)
        self.assertGreater(LnLike11, LnLike21)

        self.assertGreater(LnLike22, LnLike12)
        self.assertGreater(LnLike22, LnLike21)


class TestEstimate(unittest.TestCase):

    def setUp(self):
        self.n2s = 1.0e-3
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.nt1 = mbhb.MBHBTask()

    def tearDown(self):
        del self.nt1

    def checkAsserts(self, fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst):
        self.assertAlmostEqual(math.fabs((fluxEst - self.flux)/self.flux), 0.0, places=1)
        self.assertAlmostEqual(math.fabs((periodEst - self.nt1.period())/self.nt1.period()), 0.0, places=1)
        self.assertAlmostEqual(math.fabs((eccentricityEst - self.eccentricity)/self.eccentricity), 0.0,
                               delta=0.25)
        self.assertAlmostEqual(math.fabs((
            math.cos(omega1Est*(math.pi/180.0)) - math.cos(self.omega1*(math.pi/180.0)))/math.cos(self.omega1*(math.pi/180.0))),
            0.0, delta=0.1)
        self.assertAlmostEqual(math.fabs((tauEst - self.tau)/self.tau), 0.0, delta=periodEst/10.0)
        self.assertAlmostEqual(
            math.fabs((a2sinInclinationEst - self.nt1.a2()*math.sin(self.inclination*(math.pi/180.0)))/(self.nt1.a2()*math.sin(self.inclination*(math.pi/180.0)))),
            0.0, delta=1.0)

    @unittest.skipIf(skipWorking, 'Works!')
    def test_estimates1(self):
        self.eccentricity = 0.4
        self.omega1 = 15.0
        self.inclination = 90.0
        self.tau = 135.0
        self.flux = 100.0
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1, self.inclination, self.tau,
                self.flux])
        self.nt1.set(self.Theta1)
        nl1 = self.nt1.simulate(self.period*10.0, fracNoiseToSignal=self.n2s)
        self.nt1.observe(nl1)
        fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst = self.nt1.estimate(nl1)
        self.checkAsserts(fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst)

    @unittest.skipIf(skipWorking, 'Works!')
    def test_estimates2(self):
        self.eccentricity = 0.5
        self.omega1 = 15.0
        self.inclination = 90.0
        self.tau = 135.0
        self.flux = 100.0
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1, self.inclination, self.tau,
                self.flux])
        self.nt1.set(self.Theta1)
        nl1 = self.nt1.simulate(self.period*10.0, fracNoiseToSignal=self.n2s)
        self.nt1.observe(nl1)
        fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst = self.nt1.estimate(nl1)
        self.checkAsserts(fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst)

    @unittest.skipIf(skipWorking, 'Works!')
    def test_estimates3(self):
        self.eccentricity = 0.6
        self.omega1 = 15.0
        self.inclination = 90.0
        self.tau = 135.0
        self.flux = 100.0
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1, self.inclination, self.tau,
                self.flux])
        self.nt1.set(self.Theta1)
        nl1 = self.nt1.simulate(self.period*10.0, fracNoiseToSignal=self.n2s)
        self.nt1.observe(nl1)
        fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst = self.nt1.estimate(nl1)
        self.checkAsserts(fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst)

    @unittest.skipIf(skipWorking, 'Works!')
    def test_estimates4(self):
        self.eccentricity = 0.6
        self.omega1 = 15.0
        self.inclination = 90.0
        self.tau = 135.0
        self.flux = 100.0
        self.Theta1 = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1, self.inclination, self.tau,
                self.flux])
        self.nt1.set(self.Theta1)
        for i in xrange(10):
            nl1 = self.nt1.simulate(self.period*10.0, fracNoiseToSignal=self.n2s)
            self.nt1.observe(nl1)
            fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2SinInclinationEst = self.nt1.estimate(nl1)
            a1Guess, a2Guess, inclinationGuess = self.nt1.guess(a2SinInclinationEst)
            ntEst = mbhb.MBHBTask()
            ThetaEst = np.array(
                [a1Guess, a2Guess, periodEst, eccentricityEst, omega1Est, inclinationGuess, tauEst,
                    fluxEst])
            res = ntEst.set(ThetaEst)
            self.assertGreaterEqual(self.nt1.logPosterior(nl1), ntEst.logPosterior(nl1))
            nlEst = ntEst.simulate(periodEst*10.0, fracNoiseToSignal=self.n2s)
            ntEst.observe(nlEst)
            self.assertGreaterEqual(ntEst.logPosterior(nlEst), self.nt1.logPosterior(nlEst))


class TestFit(unittest.TestCase):
    def setUp(self):
        self.n2s = 1.0e-3
        self.nsteps = 2000
        self.a1 = 0.01
        self.a2 = 0.02
        self.period = 10.0*DayInYear
        self.eccentricity = 0.5
        self.omega1 = 30.0
        self.inclination = 90.0
        self.tau = 0.0
        self.flux = 100.0
        self.Theta = np.array(
            [self.a1, self.a2, self.period, self.eccentricity, self.omega1, self.inclination, self.tau,
                self.flux])
        self.nt1 = mbhb.MBHBTask()
        self.nt1.set(self.Theta)
        self.nl1 = self.nt1.simulate(self.period*5.0, fracNoiseToSignal=self.n2s)
        self.nt1.observe(self.nl1)

    def tearDown(self):
        del self.Theta
        del self.nt1
        del self.nl1

    # @unittest.skipIf(skipWorking, 'Works!')
    def test_fit(self):
        ntFit = mbhb.MBHBTask(nsteps=self.nsteps)
        ntFit.fit(self.nl1)

        if doPlot:
            plt.figure(0)
            plt.scatter(ntFit.Chain[0, :, self.nsteps/2:], ntFit.Chain[1, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.a1)
            plt.axhline(self.a2)
            plt.xlabel(r'$a_{1}$ (pc)')
            plt.ylabel(r'$a_{2}$ (pc)')

            plt.figure(1)
            plt.scatter(ntFit.Chain[1, :, self.nsteps/2:], ntFit.Chain[2, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.a2)
            plt.axhline(self.period)
            plt.xlabel(r'$a_{2}$ (pc)')
            plt.ylabel(r'$T$ (d)')

            plt.figure(2)
            plt.scatter(ntFit.Chain[2, :, self.nsteps/2:], ntFit.Chain[3, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.period)
            plt.axhline(self.eccentricity)
            plt.xlabel(r'$T$ (d)')
            plt.ylabel(r'$e$')

            plt.figure(3)
            plt.scatter(ntFit.Chain[3, :, self.nsteps/2:], ntFit.Chain[4, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.eccentricity)
            plt.axhline(self.omega1)
            plt.xlabel(r'$e$')
            plt.ylabel(r'$\omega_{1}$')

            plt.figure(4)
            plt.scatter(ntFit.Chain[4, :, self.nsteps/2:], ntFit.Chain[5, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.omega1)
            plt.axhline(self.inclination)
            plt.xlabel(r'$\omega_{1}$')
            plt.ylabel(r'$i$')

            plt.figure(5)
            plt.scatter(ntFit.Chain[5, :, self.nsteps/2:], ntFit.Chain[6, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.inclination)
            plt.axhline(self.tau)
            plt.xlabel(r'$i$')
            plt.ylabel(r'$\tau$ (d)')

            plt.figure(6)
            plt.scatter(ntFit.Chain[6, :, self.nsteps/2:], ntFit.Chain[7, :, self.nsteps/2:],
                        c=ntFit.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.axvline(self.tau)
            plt.axhline(self.flux)
            plt.xlabel(r'$\tau$ (d)')
            plt.ylabel(r'$F_{\mathrm{intrin}}$ (Jy)')

            pdb.set_trace()

if __name__ == "__main__":
    unittest.main()
