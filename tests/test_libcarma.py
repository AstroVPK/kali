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

skipWorking = True

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
				sigma00 = nt.Sigma()[0,0]
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
		self.nWalkers = psutil.cpu_count(logical = True)
		self.nSteps = 1000
		self.newTask = libcarma.basicTask(self.p, self.q, nwalkers = self.nWalkers, nsteps = self.nSteps)

	def tearDown(self):
		del self.newTask

	def test_noiselessrecovery(self):
		N2S = 1.0e-18
		builtInT1 = random.uniform(10.0, 100.0)
		builtInAmp = 1.0
		dt = (1.0/100.0)*builtInT1
		T = 100.0*builtInT1
		Rho = np.array([-1.0/builtInT1, builtInAmp])
		Theta = libcarma.coeffs(self.p, self.q, Rho)
		self.newTask.set(dt, Theta)
		newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
		self.newTask.observe(newLC)
		self.newTask.fit(newLC)
		recoveredT1Mean = np.mean(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT1Std = np.std(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		self.assertTrue(math.fabs(builtInT1 - recoveredT1Mean) < 3.0*recoveredT1Std)
		self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 3.0*recoveredAmpStd)

	def test_noiselyrecovery(self):
		N2S = 1.0e-3 # LSST-ish
		builtInT1 = random.uniform(10.0, 100.0)
		builtInAmp = 1.0
		dt = (1.0/100.0)*builtInT1
		T = 100.0*builtInT1
		Rho = np.array([-1.0/builtInT1, builtInAmp])
		Theta = libcarma.coeffs(self.p, self.q, Rho)
		self.newTask.set(dt, Theta)
		newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
		self.newTask.observe(newLC)
		self.newTask.fit(newLC)
		recoveredT1Mean = np.mean(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT1Std = np.std(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		self.assertTrue(math.fabs(builtInT1 - recoveredT1Mean) < 5.0*recoveredT1Std)
		self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 5.0*recoveredAmpStd)

class TestFitCARMA20(unittest.TestCase):
	def setUp(self):
		self.p = 2
		self.q = 0
		self.nWalkers = psutil.cpu_count(logical = True)
		self.nSteps = 1000
		self.newTask = libcarma.basicTask(self.p, self.q, nwalkers = self.nWalkers, nsteps = self.nSteps)

	def tearDown(self):
		del self.newTask

	def test_noiselessrecovery(self):
		N2S = 1.0e-18
		builtInT1 = random.uniform(10.0, 100.0)
		builtInT2 = random.uniform(builtInT1, 500.0)
		builtInAmp = 1.0
		dt = (1.0/100.0)*builtInT1
		T = 100.0*builtInT2
		Rho = np.array([-1.0/builtInT1, -1.0/builtInT2, builtInAmp])
		Theta = libcarma.coeffs(self.p, self.q, Rho)
		self.newTask.set(dt, Theta)
		newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
		self.newTask.observe(newLC)
		self.newTask.fit(newLC)
		recoveredT1Mean = np.mean(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT1Std = np.std(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT2Mean = np.mean(self.newTask.timescaleChain[1,:,self.nSteps/2:])
		recoveredT2Std = np.std(self.newTask.timescaleChain[1,:,self.nSteps/2:])
		recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		self.assertTrue(math.fabs(builtInT1 - recoveredT1Mean) < 3.0*recoveredT1Std)
		self.assertTrue(math.fabs(builtInT2 - recoveredT2Mean) < 3.0*recoveredT2Std)
		self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 3.0*recoveredAmpStd)

	def test_noiselyrecovery(self):
		N2S = 1.0e-3 # LSST-ish
		builtInT1 = random.uniform(10.0, 100.0)
		builtInT2 = random.uniform(builtInT1, 500.0)
		builtInAmp = 1.0
		dt = (1.0/100.0)*builtInT1
		T = 100.0*builtInT2
		Rho = np.array([-1.0/builtInT1, -1.0/builtInT2, builtInAmp])
		Theta = libcarma.coeffs(self.p, self.q, Rho)
		self.newTask.set(dt, Theta)
		newLC = self.newTask.simulate(T, fracNoiseToSignal = N2S)
		self.newTask.observe(newLC)
		self.newTask.fit(newLC)
		recoveredT1Mean = np.mean(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT1Std = np.std(self.newTask.timescaleChain[0,:,self.nSteps/2:])
		recoveredT2Mean = np.mean(self.newTask.timescaleChain[1,:,self.nSteps/2:])
		recoveredT2Std = np.std(self.newTask.timescaleChain[1,:,self.nSteps/2:])
		recoveredAmpMean = np.mean(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		recoveredAmpStd = np.std(self.newTask.timescaleChain[-1,:,self.nSteps/2:])
		self.assertTrue(math.fabs(builtInT1 - recoveredT1Mean) < 3.0*recoveredT1Std)
		self.assertTrue(math.fabs(builtInT2 - recoveredT2Mean) < 3.0*recoveredT2Std)
		self.assertTrue(math.fabs(builtInAmp - recoveredAmpMean) < 3.0*recoveredAmpStd)

if __name__ == "__main__":
	unittest.main()