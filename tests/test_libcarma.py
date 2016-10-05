import math
import numpy as np
import copy
import unittest
import random
import psutil
import pdb

try:
    import libcarma
except ImportError:
    print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
    sys.exit(1)

skipWorking = False

@unittest.skipIf(skipWorking, 'Works!')
class TestCoeffs(unittest.TestCase):
    def test_coeffs(self):
        for p in xrange(1, 10):
            for q in xrange(0, p):
                oldRho = np.zeros(p + q + 1)
                for i in xrange(p + q):
                    oldRho[i] = -1.0/random.uniform(1.0, 100.0)
                oldRho[p + q] = 1.0
                oldTheta = libcarma._old_coeffs(p, q, oldRho)
                dt = 1.0
                nt = libcarma.basicTask(p, q)
                nt.set(dt, oldTheta)
                sigma00 = nt.Sigma()[0, 0]
                newRho = copy.copy(oldRho)
                newRho[p + q] = math.sqrt(sigma00)
                newTheta = libcarma.coeffs(p, q, newRho)
                for i in xrange(p + q + 1):
                    self.assertAlmostEqual(oldTheta[i], newTheta[i])

@unittest.skipIf(skipWorking, 'Works!')
class TestFitCARMA10(unittest.TestCase):
    def setUp(self):
        self.p = 1
        self.q = 0
        self.nWalkers = 25*psutil.cpu_count(logical = True)
        self.nSteps = 250
        self.newTask = libcarma.basicTask(self.p, self.q, nwalkers = self.nWalkers, nsteps = self.nSteps)

    def tearDown(self):
        del self.newTask

    def run_test(self, N2S):
        builtInTAR1 = random.uniform(10.0, 25.0)
        builtInAmp = 1.0
        dt = 0.1
        T = 1000.0
        Rho = np.array([-1.0/builtInTAR1, builtInAmp])
        Theta = libcarma.coeffs(self.p, self.q, Rho)
        self.newTask.set(dt, Theta)
        newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
        self.newTask.observe(newLC)
        self.newTask.fit(newLC)
        recoveredTAR1Mean = np.mean(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Mean), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMean), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Mean) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3 # LSST-ish
        self.run_test(N2S)

@unittest.skipIf(skipWorking, 'Works!')
class TestFitCARMA20(unittest.TestCase):
    def setUp(self):
        self.p = 2
        self.q = 0
        self.nWalkers = 25*psutil.cpu_count(logical = True)
        self.nSteps = 250
        self.newTask = libcarma.basicTask(self.p, self.q, nwalkers = self.nWalkers, nsteps = self.nSteps)

    def tearDown(self):
        del self.newTask

    def run_test(self, N2S):
        builtInTAR1 = random.uniform(10.0, 25.0)
        builtInTAR2 = random.uniform(builtInTAR1, 100.0)
        builtInAmp = 1.0
        dt = 0.1
        T = 1000.0
        Rho = np.array([-1.0/builtInTAR1, -1.0/builtInTAR2, builtInAmp])
        Theta = libcarma.coeffs(self.p, self.q, Rho)
        self.newTask.set(dt, Theta)
        newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
        self.newTask.observe(newLC)
        self.newTask.fit(newLC)
        recoveredTAR1Mean = np.mean(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredTAR2Mean = np.mean(self.newTask.timescaleChain[1,:, self.nSteps/2:])
        recoveredTAR2Std = np.std(self.newTask.timescaleChain[1,:, self.nSteps/2:])
        recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Mean), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInTAR2 - recoveredTAR2Mean), 5.0*recoveredTAR2Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMean), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Mean) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInTAR2 - recoveredTAR2Mean) < 5.0*recoveredTAR2Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3 # LSST-ish
        self.run_test(N2S)

@unittest.skipIf(skipWorking, 'Works!')
class TestFitCARMA21(unittest.TestCase):
    def setUp(self):
        self.p = 2
        self.q = 1
        self.nWalkers = 25*psutil.cpu_count(logical = True)
        self.nSteps = 250
        self.newTask = libcarma.basicTask(self.p, self.q, nwalkers = self.nWalkers, nsteps = self.nSteps)

    def tearDown(self):
        del self.newTask

    def run_test(self, N2S):
        builtInTAR1 = random.uniform(10.0, 25.0)
        builtInTAR2 = random.uniform(builtInTAR1, 100.0)
        builtInTMA1 = random.uniform(1.0, 10.0)
        builtInAmp = 1.0
        dt = 0.1
        T = 1000.0
        Rho = np.array([-1.0/builtInTAR1, -1.0/builtInTAR2, -1.0/builtInTMA1, builtInAmp])
        Theta = libcarma.coeffs(self.p, self.q, Rho)
        self.newTask.set(dt, Theta)
        newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
        self.newTask.observe(newLC)
        self.newTask.fit(newLC)
        recoveredTAR1Mean = np.mean(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0,:, self.nSteps/2:])
        recoveredTAR2Mean = np.mean(self.newTask.timescaleChain[1,:, self.nSteps/2:])
        recoveredTAR2Std = np.std(self.newTask.timescaleChain[1,:, self.nSteps/2:])
        recoveredTMA1Mean = np.mean(self.newTask.timescaleChain[2,:, self.nSteps/2:])
        recoveredTMA1Std = np.std(self.newTask.timescaleChain[2,:, self.nSteps/2:])
        recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Mean), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInTAR2 - recoveredTAR2Mean), 5.0*recoveredTAR2Std)
        print '%e %e'%(math.fabs(builtInTMA1 - recoveredTMA1Mean), 5.0*recoveredTMA1Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMean), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Mean) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInTAR2 - recoveredTAR2Mean) < 5.0*recoveredTAR2Std)
        self.assertTrue(math.fabs(builtInTMA1 - recoveredTMA1Mean) < 5.0*recoveredTMA1Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3 # LSST-ish
        self.run_test(N2S)

if __name__ == "__main__":
    unittest.main()
