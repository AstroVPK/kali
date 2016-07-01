import math
import numpy as np
import matplotlib.pyplot as plt

try:
	import libbsmbh
except ImportError:
	print 'libbsmbh is not setup. Setup libbsmbh by sourcing bin/setup.sh'
	sys.exit(1)

def test_MakeModel():
	nt = libbsmbh.binarySMBHTask()
	Theta = np.array([0.0005, 75.0, 10.0, 0.75, 0.0, 0.0, 0.0, 100.0, 0.5])
	res = nt.set(Theta)
	assert res == 0, 'Bad parameter checking!'
	assert nt.period() == 332.855455695, 'Incorrect orbital period!'