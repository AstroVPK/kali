import unittest

try:
    import kali.s82
except ImportError:
    print 'Could not import kali.s82! kali may not be setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.k2
except ImportError:
    print 'Could not import kali.k2! kali may not be setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.fitter
except ImportError:
    print 'Could not import kali.fitter! kali may not be setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


class TestFitter(unittest.TestCase):

    def setUp(self):
        self.lcs = list()
        names = ['210113.15-005120.3',
                 '221550.17-005641.8']
        bands = ['r', 'i']
        for name in names:
            for band in bands:
                newlc = kali.s82.sdssLC(name=name, band=band)
                newlc.minTimescale = 2.0*newlc.mediandt
                self.lcs.append(newlc)

        self.models = list(['carma(1,)', 'carma(2,)', 'mbhbcarma(1,)'])

    def tearDown(self):
        del self.lcs
        del self.models

    def test_one(self):
        newFitter = kali.fitter.fitter(self.lcs, models=self.models, nsteps=1000)
        newFitter.rerun = True
        newFitter.run()

if __name__ == "__main__":
    unittest.main()
