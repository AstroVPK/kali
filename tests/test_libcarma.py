import math
import numpy as np
import unittest
import matplotlib.pyplot as plt

try:
	import libcarma
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

class TestCoeffs(unittest.TestCase):
	def test_coeffs(self):
		Rho = np.array([-1.0/10.0, 1.0])
		Theta = libcarma.coeffs(1, 0, Rho)
		self.assertAlmostEqual(-1.0*Theta[0], Rho[0])

if __name__ == "__main__":
	unittest.main()