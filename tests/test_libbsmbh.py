import math
import numpy as np
import unittest
import matplotlib.pyplot as plt

try:
	import libbsmbh
except ImportError:
	print 'libbsmbh is not setup. Setup libbsmbh by sourcing bin/setup.sh'
	sys.exit(1)

class TestPeriod(unittest.TestCase):
	def test_period(self):
		nt = libbsmbh.binarySMBHTask()
		Theta = np.array([0.0005, 75.0, 10.0, 0.75, 0.0, 0.0, 0.0, 100.0, 0.5])
		res = nt.set(Theta)
		self.assertEqual(res, 0)
		self.assertAlmostEqual(nt.period(), 332.855455695)

if __name__ == "__main__":
	unittest.main()