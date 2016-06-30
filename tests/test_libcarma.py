import math
import numpy as np
import matplotlib.pyplot as plt

try:
	import libcarma
except ImportError:
	print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
	sys.exit(1)

def test_coeffs():
	Rho = np.array([-1.0/10.0, 1.0])
	Theta = libcarma.coeffs(1, 0, Rho)
	assert -1.0*Theta[0] == Rho[0]