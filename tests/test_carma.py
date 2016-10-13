import math
import numpy as np
import copy
import unittest
import random
import psutil
import sys
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as colormap

try:
    import kali.carma
except ImportError:
    print 'Cannot import kali.carma! kali is not setup. Setup kali by sourcing bin/setup.sh'
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
class TestFitCARMA10(unittest.TestCase):
    def setUp(self):
        self.p = 1
        self.q = 0
        self.nWalkers = 25*psutil.cpu_count(logical=True)
        self.nSteps = 200
        self.dt = 0.01
        self.T = 1000.0
        self.sincWidth = 2.0
        self.sincCenter = self.T/2.0
        self.newTask = kali.carma.CARMATask(self.p, self.q, nwalkers=self.nWalkers, nsteps=self.nSteps)

    def tearDown(self):
        del self.newTask

    def make_plots(self, Theta):
        maxLnPosterior = np.nanmax(self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps])
        plt.figure(1)
        plt.scatter(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    c=maxLnPosterior - self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps],
                    marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
        plt.axhline(y=Theta[1], xmin=0.0, xmax=1.0)
        plt.axvline(x=Theta[0], ymin=0.0, ymax=1.0)
        plt.xlim([np.min(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.ylim([np.min(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.xlabel(r'$a_{1}$')
        plt.ylabel(r'$b_{0}$')
        plt.show(False)

    def run_test(self, N2S):
        builtInTAR1 = 62.0  # random.uniform(10.0, 25.0)
        builtInAmp = 1.0
        Rho = np.array([-1.0/builtInTAR1, builtInAmp])
        Theta = kali.carma.coeffs(self.p, self.q, Rho)
        self.newTask.set(self.dt, Theta)
        newLC = self.newTask.simulate(duration=self.T, fracNoiseToSignal=N2S,
                                      burnSeed=BURNSEED, distSeed=DISTSEED, noiseSeed=NOISESEED)
        self.newTask.observe(newLC, noiseSeed=NOISESEED)
        newLC.sampler = 'sincSampler'
        sampledLC = newLC.sample(width=self.sincWidth, center=self.sincCenter, sampleSeed=SAMPLESEED)
        self.newTask.fit(sampledLC, zSSeed=ZSSEED, walkerSeed=WALKERSEED, moveSeed=MOVESEED, xSeed=XSEED)
        recoveredTAR1Median = np.median(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredAmpMedian = np.median(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Median), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMedian), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Median) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMedian) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3  # LSST-ish
        self.run_test(N2S)


@unittest.skipIf(skipWorking, 'Works!')
class TestFitCARMA20(unittest.TestCase):
    def setUp(self):
        self.p = 2
        self.q = 0
        self.nWalkers = 25*psutil.cpu_count(logical=True)
        self.nSteps = 200
        self.dt = 0.01
        self.T = 1000.0
        self.sincWidth = 2.0
        self.sincCenter = self.T/2.0
        self.newTask = kali.carma.CARMATask(self.p, self.q, nwalkers=self.nWalkers, nsteps=self.nSteps)

    def tearDown(self):
        del self.newTask

    def make_plots(self, Theta):
        maxLnPosterior = np.nanmax(self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps])
        plt.figure(1)
        plt.scatter(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    c=maxLnPosterior - self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps],
                    marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
        plt.axhline(y=Theta[1], xmin=0.0, xmax=1.0)
        plt.axvline(x=Theta[0], ymin=0.0, ymax=1.0)
        plt.xlim([np.min(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.ylim([np.min(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.xlabel(r'$a_{1}$')
        plt.ylabel(r'$a_{2}$')
        plt.figure(2)
        plt.scatter(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    c=maxLnPosterior - self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps],
                    marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
        plt.axhline(y=Theta[2], xmin=0.0, xmax=1.0)
        plt.axvline(x=Theta[1], ymin=0.0, ymax=1.0)
        plt.xlim([np.min(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.ylim([np.min(self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.xlabel(r'$a_{1}$')
        plt.ylabel(r'$b_{0}$')
        plt.show(False)

    def run_test(self, N2S):
        builtInTAR1 = 1.5  # random.uniform(10.0, 25.0)
        builtInTAR2 = 62.0  # random.uniform(50.0, 100.0)
        builtInAmp = 1.0
        Rho = np.array([-1.0/builtInTAR1, -1.0/builtInTAR2, builtInAmp])
        Theta = kali.carma.coeffs(self.p, self.q, Rho)
        self.newTask.set(self.dt, Theta)
        newLC = self.newTask.simulate(duration=self.T, fracNoiseToSignal=N2S,
                                      burnSeed=BURNSEED, distSeed=DISTSEED, noiseSeed=NOISESEED)
        self.newTask.observe(newLC, noiseSeed=NOISESEED)
        newLC.sampler = 'sincSampler'
        sampledLC = newLC.sample(width=self.sincWidth, center=self.sincCenter, sampleSeed=SAMPLESEED)
        self.newTask.fit(sampledLC, zSSeed=ZSSEED, walkerSeed=WALKERSEED, moveSeed=MOVESEED, xSeed=XSEED)
        recoveredTAR1Median = np.median(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredTAR2Median = np.median(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        recoveredTAR2Std = np.std(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        recoveredAmpMedian = np.median(self.newTask.timescaleChain[2, :, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[2, :, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Median), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInTAR2 - recoveredTAR2Median), 5.0*recoveredTAR2Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMedian), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Median) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInTAR2 - recoveredTAR2Median) < 5.0*recoveredTAR2Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMedian) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3  # LSST-ish
        self.run_test(N2S)


@unittest.skipIf(skipWorking, 'Works!')
class TestFitCARMA21(unittest.TestCase):
    def setUp(self):
        self.p = 2
        self.q = 1
        self.nWalkers = 25*psutil.cpu_count(logical=True)
        self.nSteps = 200
        self.dt = 0.01
        self.T = 1000.0
        self.sincWidth = 2.0
        self.sincCenter = self.T/2.0
        self.newTask = kali.carma.CARMATask(self.p, self.q, nwalkers=self.nWalkers, nsteps=self.nSteps)

    def tearDown(self):
        del self.newTask

    def make_plots(self, Theta):
        maxLnPosterior = np.nanmax(self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps])
        plt.figure(1)
        plt.scatter(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    c=maxLnPosterior - self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps],
                    marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
        plt.axhline(y=Theta[1], xmin=0.0, xmax=1.0)
        plt.axvline(x=Theta[0], ymin=0.0, ymax=1.0)
        plt.xlim([np.min(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[0, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.ylim([np.min(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[1, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.xlabel(r'$a_{1}$')
        plt.ylabel(r'$a_{2}$')
        plt.figure(2)
        plt.scatter(self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    self.newTask.Chain[3, :, self.newTask.nsteps/2:self.newTask.nsteps],
                    c=maxLnPosterior - self.newTask.LnPosterior[:, self.newTask.nsteps/2:self.newTask.nsteps],
                    marker='.', cmap=colormap.gist_rainbow_r, linewidth=0)
        plt.axhline(y=Theta[3], xmin=0.0, xmax=1.0)
        plt.axvline(x=Theta[2], ymin=0.0, ymax=1.0)
        plt.xlim([np.min(self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[2, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.ylim([np.min(self.newTask.Chain[3, :, self.newTask.nsteps/2:self.newTask.nsteps]),
                  np.max(self.newTask.Chain[3, :, self.newTask.nsteps/2:self.newTask.nsteps])])
        plt.xlabel(r'$b_{0}$')
        plt.ylabel(r'$b_{1}$')
        plt.show(False)

    def run_test(self, N2S):
        builtInTAR1 = 1.5  # random.uniform(10.0, 25.0)
        builtInTAR2 = 62.0  # random.uniform(50, 100.0)
        builtInTMA1 = 0.1725  # random.uniform(1.0, 5.0)
        builtInAmp = 1.0
        Rho = np.array([-1.0/builtInTAR1, -1.0/builtInTAR2, -1.0/builtInTMA1, builtInAmp])
        Theta = kali.carma.coeffs(self.p, self.q, Rho)
        self.newTask.set(self.dt, Theta)
        newLC = self.newTask.simulate(duration=self.T, fracNoiseToSignal=N2S,
                                      burnSeed=BURNSEED, distSeed=DISTSEED, noiseSeed=NOISESEED)
        self.newTask.observe(newLC, noiseSeed=NOISESEED)
        newLC.sampler = 'sincSampler'
        sampledLC = newLC.sample(width=self.sincWidth, center=self.sincCenter, sampleSeed=SAMPLESEED)
        res = self.newTask.fit(sampledLC,
                               zSSeed=ZSSEED, walkerSeed=WALKERSEED, moveSeed=MOVESEED, xSeed=XSEED)
        recoveredTAR1Median = np.median(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredTAR1Std = np.std(self.newTask.timescaleChain[0, :, self.nSteps/2:])
        recoveredTAR2Median = np.median(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        recoveredTAR2Std = np.std(self.newTask.timescaleChain[1, :, self.nSteps/2:])
        recoveredTMA1Median = np.median(self.newTask.timescaleChain[2, :, self.nSteps/2:])
        recoveredTMA1Std = np.std(self.newTask.timescaleChain[2, :, self.nSteps/2:])
        recoveredAmpMedian = np.median(self.newTask.timescaleChain[3, :, self.nSteps/2:])
        recoveredAmpStd = np.std(self.newTask.timescaleChain[3, :, self.nSteps/2:])
        print '%e %e'%(math.fabs(builtInTAR1 - recoveredTAR1Median), 5.0*recoveredTAR1Std)
        print '%e %e'%(math.fabs(builtInTAR2 - recoveredTAR2Median), 5.0*recoveredTAR2Std)
        print '%e %e'%(math.fabs(builtInTMA1 - recoveredTMA1Median), 5.0*recoveredTMA1Std)
        print '%e %e'%(math.fabs(builtInAmp - recoveredAmpMedian), 5.0*recoveredAmpStd)
        self.assertTrue(math.fabs(builtInTAR1 - recoveredTAR1Median) < 5.0*recoveredTAR1Std)
        self.assertTrue(math.fabs(builtInTAR2 - recoveredTAR2Median) < 5.0*recoveredTAR2Std)
        self.assertTrue(math.fabs(builtInTMA1 - recoveredTMA1Median) < 5.0*recoveredTMA1Std)
        self.assertTrue(math.fabs(builtInAmp - recoveredAmpMedian) < 5.0*recoveredAmpStd)

    def test_noiselessrecovery(self):
        N2S = 1.0e-18
        self.run_test(N2S)

    def test_noiselyrecovery(self):
        N2S = 1.0e-3  # LSST-ish
        self.run_test(N2S)

if __name__ == "__main__":
    unittest.main()
