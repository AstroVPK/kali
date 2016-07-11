import math
import numpy as np
import copy
import unittest
import random
import pdb

try:
	import libcarma
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

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

if __name__ == "__main__":
	unittest.main()