import math
import numpy as np
import unittest
import sys
import pdb

try:
    import kali.carma
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


class TestComputeLnLikeCARMA10(unittest.TestCase):

    def setUp(self):
        self.p = 1
        self.q = 0
        self.nt = kali.carma.CARMATask(self.p, self.q)
        self.dt = 1.0
        self.TAR1 = 5.0
        self.Amp = 1.0
        self.rho = np.array([-1.0/self.TAR1, self.Amp])
        self.theta = kali.carma.coeffs(self.p, self.q, self.rho)
        self.nt.set(self.dt, self.theta)

    def tearDown(self):
        del self.nt

    def test_makeLC(self):
        nl = self.nt.simulate(10.0)
        self.nt.observe(nl)
        LnLike = self.nt.logLikelihood(nl)
        self.assertFalse(np.isinf(LnLike))

if __name__ == "__main__":
    unittest.main()
