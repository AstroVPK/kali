import math
import numpy as np
import unittest
import matplotlib.pyplot as plt

try:
	import libbsmbh
except ImportError:
	print 'libbsmbh is not setup. Setup libbsmbh by sourcing bin/setup.sh'
	sys.exit(1)

'''class TestPeriod(unittest.TestCase):
	def test_period(self):
		nt = libbsmbh.binarySMBHTask()
		Theta = np.array([0.0005, 75.0, 10.0, 0.75, 0.0, 0.0, 0.0, 100.0])
		res = nt.set(Theta)
		self.assertEqual(res, 0)
		self.assertAlmostEqual(nt.period(), 332.855455695)

class TestNoInclination(unittest.TestCase):

	def setUp(self):
		self.rPeriTot = 0.001
		self.m1 = 75.0
		self.m2 = 10.0
		self.eccentricity = 0.0
		self.omega1_1 = 0.0
		self.omega1_2 = 180.0
		self.inclination = 0.0
		self.tau = 0.0
		self.flux = 100.0
		self.Theta1 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.Theta2 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.nt1 = libbsmbh.binarySMBHTask()
		self.nt2 = libbsmbh.binarySMBHTask()
		self.nt1.set(self.Theta1)
		self.nt2.set(self.Theta2)

	def tearDown(self):
		del self.Theta1
		del self.Theta2
		del self.nt1
		del self.nt2

	def test_a(self):
		self.assertEqual(self.nt1.rPeri(), self.rPeriTot)

	def test_beamingFactor(self):
		self.nt1(0.0) # We start at Day 0
		bF1 = self.nt1.beamingFactor1() # Get the beaming factor

		self.nt1(self.nt1.Day) # Now step to Day 1
		bF2 = self.nt1.beamingFactor1() # Get the beaming factor again
		self.assertEqual(bF1, bF2) # Since the orbital inclination is zero, these should be the same.

	def test_lightCurve(self):
		nl1 = self.nt1.simulate(self.nt1.period()*10.0)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0)
		self.assertEqual(len(np.where((nl1.x - nl2.x) != 0.0)[0].tolist()), 0) # Both have constant beaming

class TestInclinationNoNoise(unittest.TestCase):

	def setUp(self):
		self.rPeriTot = 0.001
		self.m1 = 75.0
		self.m2 = 10.0
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
		self.Theta1 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.Theta2 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.nt1 = libbsmbh.binarySMBHTask()
		self.nt2 = libbsmbh.binarySMBHTask()
		self.nt1.set(self.Theta1)
		self.nt2.set(self.Theta2)

	def tearDown(self):
		del self.Theta1
		del self.Theta2
		del self.nt1
		del self.nt2

	def test_a(self):
		self.assertEqual(self.nt1.rPeri(), self.rPeriTot)

	def test_beamingFactor(self):
		self.nt1(0.0) # We start at Day 0
		bF1 = self.nt1.beamingFactor1() # Get the beaming factor

		self.nt1(self.nt1.Day) # Now step to Day 1
		bF2 = self.nt1.beamingFactor1() # Get the beaming factor again
		self.assertNotEqual(bF1, bF2) # Since the orbital inclination is zero, these should be the same.

	def test_lightCurve(self):
		n2s = 1.0e-18
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal = n2s)
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
		self.rPeriTot = 0.001
		self.m1 = 75.0
		self.m2 = 10.0
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
		self.Theta1 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.Theta2 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.nt1 = libbsmbh.binarySMBHTask()
		self.nt2 = libbsmbh.binarySMBHTask()
		self.nt1.set(self.Theta1)
		self.nt2.set(self.Theta2)

	def tearDown(self):
		del self.Theta1
		del self.Theta2
		del self.nt1
		del self.nt2

	def test_a(self):
		self.assertEqual(self.nt1.rPeri(), self.rPeriTot)

	def test_beamingFactor(self):
		self.nt1(0.0) # We start at Day 0
		bF1 = self.nt1.beamingFactor1() # Get the beaming factor

		self.nt1(self.nt1.Day) # Now step to Day 1
		bF2 = self.nt1.beamingFactor1() # Get the beaming factor again
		self.assertNotEqual(bF1, bF2) # Since the orbital inclination is zero, these should be the same.

	def test_lightCurveVLowNoise(self):
		n2s = 1.0e-4
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal = n2s)
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

	def test_lightCurveLowNoise(self):
		n2s = 1.0e-3
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal = n2s)
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

	def test_lightCurveNoise(self):
		n2s = 1.0e-2
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal = n2s)
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

	def test_lightCurveHighNoise(self):
		n2s = 5.0e-2
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		nl2 = self.nt2.simulate(self.nt2.period()*10.0, fracNoiseToSignal = n2s)
		self.nt1.observe(nl1)
		self.nt2.observe(nl2)

		LnLike11 = self.nt1.logLikelihood(nl1)
		LnLike22 = self.nt2.logLikelihood(nl2)
		LnLike12 = self.nt1.logLikelihood(nl2)
		LnLike21 = self.nt2.logLikelihood(nl1)

		self.assertGreater(LnLike11, LnLike12)
		self.assertGreater(LnLike11, LnLike21)

		self.assertGreater(LnLike22, LnLike12)
		self.assertGreater(LnLike22, LnLike21)'''

class TestFitNoNoise(unittest.TestCase):

	def setUp(self):
		self.rPeriTot = 0.001
		self.m1 = 75.0
		self.m2 = 10.0
		self.eccentricity = 0.5
		self.omega1_1 = 30.0
		self.omega1_2 = 30.0
		self.inclination = 90.0
		self.tau = 0.0
		self.flux = 100.0
		nstepsVal = 1
		nwalkersVal = 200
		maxEvalsVal = 10000
		xTolVal = 1.0e-2
		self.Theta1 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_1, self.inclination, self.tau, self.flux])
		self.Theta2 = np.array([self.rPeriTot, self.m1, self.m2, self.eccentricity, self.omega1_2, self.inclination, self.tau, self.flux])
		self.nt1 = libbsmbh.binarySMBHTask()
		self.nt2 = libbsmbh.binarySMBHTask(nwalkers = nwalkersVal, nsteps = nstepsVal, maxEvals = maxEvalsVal, xTol = xTolVal)
		self.nt1.set(self.Theta1)
		self.nt2.set(self.Theta2)

	def tearDown(self):
		del self.Theta1
		del self.Theta2
		del self.nt1
		del self.nt2

	def test_lightCurveFit(self):
		n2s = 1.0e-3
		nl1 = self.nt1.simulate(self.nt1.period()*10.0, fracNoiseToSignal = n2s)
		#nl1 = self.nt1.simulate(0.4, dt = 0.2, fracNoiseToSignal = n2s)
		self.nt1.observe(nl1)
		#self.nt1.logLikelihood(nl1)
		#import pdb; pdb.set_trace()

		import pdb; pdb.set_trace()
		self.nt2.fit(nl1)
		import pdb; pdb.set_trace()
		walkerBest = np.where(np.nanmax(self.nt2.LnPosterior[:,:]) == self.nt2.LnPosterior[:,:])[0][0]
		stepBest = np.where(np.nanmax(self.nt2.LnPosterior[:,:]) == self.nt2.LnPosterior[:,:])[1][0]
		ThetaBest = self.nt2.Chain[:,walkerBest,stepBest]
		ntBest = libbsmbh.binarySMBHTask()
		ntBest.set(ThetaBest)
		import pdb; pdb.set_trace()
		nlBest = ntBest.simulate(self.nt1.period()*10.0, dt = nl1.dt, fracNoiseToSignal = n2s)
		plt.ion()
		nl1.plot()
		nlBest.plot()
		plt.show()

		import pdb; pdb.set_trace()

if __name__ == "__main__":
	unittest.main()