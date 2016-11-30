import math
import numpy as np
import copy
import unittest
import random
import psutil
import os
import sys
import cPickle as pickle
import tempfile
import shutil
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as colormap

try:
    import kali.carma
except ImportError:
    print 'Cannot import kali.carma! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.s82
except ImportError:
    print 'Cannot import kali.s82! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

plt.ion()
skipWorking = False
BURNSEED = 731647386
DISTSEED = 219038190
NOISESEED = 87238923
SAMPLESEED = 36516342
ZSSEED = 384789247
WALKERSEED = 738472981
MOVESEED = 131343786
XSEED = 2348713647


@unittest.skipIf(skipWorking, 'Works!')
class TestPickleLC(unittest.TestCase):
    def setUp(self):
        self.dirpath = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def test_pickleLightcurve(self):
        aNewSDSSlc = kali.s82.sdssLC(name='', band='r')
        lcName = aNewSDSSlc.name
        pickle.dump(aNewSDSSlc, open(os.path.join(self.dirpath, '%s.pkl'%(aNewSDSSlc.name)), 'wb'))
        del aNewSDSSlc
        aNewSDSSlcReborn = pickle.load(open(os.path.join(self.dirpath, '%s.pkl'%(lcName)), 'rb'))
        aNewSDSSlcDoppelganger = kali.s82.sdssLC(name='%s'%(lcName), band='r')
        np.testing.assert_array_equal(aNewSDSSlcReborn.t, aNewSDSSlcDoppelganger.t)


@unittest.skipIf(skipWorking, 'Works!')
class TestPickleTask(unittest.TestCase):
    def setUp(self):
        self.dirpath = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def test_pickleLightcurve(self):
        aNewSDSSlc = kali.s82.sdssLC(name='', band='r')
        aNewTask = kali.carma.CARMATask(1, 0)
        aNewTask.fit(aNewSDSSlc)
        aNewTask.bestTau
        pickle.dump(aNewTask, open(os.path.join(self.dirpath, 'Task.pkl'), 'wb'))
        aNewTaskReborn = pickle.load(open(os.path.join(self.dirpath, 'Task.pkl'), 'rb'))
        np.testing.assert_array_equal(aNewTaskReborn.timescaleChain, aNewTask.timescaleChain)


if __name__ == "__main__":
    unittest.main()
