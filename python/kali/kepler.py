import math as math
import numpy as np
import urllib
import os as os
import sys as sys
import warnings
import fitsio
from fitsio import FITS, FITSHDR
import subprocess
import argparse
import pdb

from astropy import units
from astropy.coordinates import SkyCoord

try:
    os.environ['DISPLAY']
except KeyError as Err:
    warnings.warn('No display environment! Using matplotlib backend "Agg"')
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import kali.lc
except ImportError:
    print('kali is not setup. Setup kali by sourcing bin/setup.sh')
    sys.exit(1)


class keplerLC(kali.lc.lc):
    window = 5
    Day = 86164.090530833
    integrationTime = 6.019802903
    readTime = 0.5189485261
    numIntegrationsSC = 9
    numIntegrationsLC = 270
    samplingIntervalSC = (integrationTime + readTime)*numIntegrationsSC/Day
    samplingIntervalLC = (integrationTime + readTime)*numIntegrationsLC/Day

    def _estimate_deltaT(self):
        self.deltaT = self.samplingIntervalLC/(1.0 + self.z)

    def _ingest_raw_lc(self):
        self.pre_t = np.array(self.numPts*[0.0])
        self.pre_y = np.array(self.numPts*[0.0])
        self.pre_mask = np.array(self.numPts*[0.0])
        for i in range(self.numPts):
            self.pre_t[i] = i*self.deltaT
        for i in range(self.rawNumPts):
            t_idx = int(round(self.rawData[i, 0]/self.deltaT))
            self.pre_t[t_idx] = self.rawData[i, 0]
            self.pre_y[t_idx] = self.rawData[i, 1]
            self.pre_mask[t_idx] = 1.0
        for i in range(self.numPts):
            if self.pre_mask[i] == 0.0:
                self.pre_y[i] = np.nan

    def _estimate_lc(self, window=5):
        for pt in range(self.numCadences):
            self.t[pt] = self.pre_t[pt+self.window]
            self.y[pt] = self.pre_y[pt+self.window]
            self.yerr[pt] = np.nanstd(self.pre_y[pt: pt+2*self.window+1])
            self.mask[pt] = self.pre_mask[pt+self.window]

    def read(self, name, band=None, path=None, ancillary=None, **kwargs):
        self.z = kwargs.get('z', 0.0)
        fileName = 'lcout_' + name + '.dat'
        if path is None:
            try:
                self.path = os.environ['KEPLERDATADIR']
            except KeyError:
                raise KeyError('Environment variable "KEPLERDATADIR" not set! Please set "KEPLERDATADIR" to point \
                where all KEPLER data lives first...')
        else:
            self.path = path
        filePath = os.path.join(self.path, fileName)

        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'Kep')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$t$~(MJD)'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'$F$~($\mathrm{e^{-}}$)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

        self.rawData = np.loadtxt(filePath)
        self.rawNumPts = self.rawData.shape[0]

        self._estimate_deltaT()
        self.numPts = int(math.ceil((self.rawData[-1, 0] - self.rawData[0, 0])/self.deltaT)) + 1
        self.numCadences = self.numPts - 2*self.window
        self._ingest_raw_lc()

        self.t = np.array(self.numCadences*[0.0])
        self.x = np.array(self.numCadences*[0.0])
        self.y = np.array(self.numCadences*[0.0])
        self.yerr = np.array(self.numCadences*[0.0])
        self.mask = np.array(self.numCadences*[0.0])

        self._estimate_lc()

        self.t = np.require(self.t, requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(self.x, requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(self.y, requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(self.yerr, requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(self.mask, requirements=['F', 'A', 'W', 'O', 'E'])


    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--ID', type=str, default='211991001', help=r'EPIC ID')
    parser.add_argument('-z', '--z', type=float, default='0.3056', help=r'object redshift')
    parser.add_argument('-p', '--processing', type=str,
                        default='k2sff', help=r'sap/pdcsap/k2sff/k2sc/k2varcat etc...')
    parser.add_argument('-c', '--campaign', type=str, default='c05', help=r'Campaign')
    parser.add_argument('-goid', '--goID', type=str,
                        default='Edelson, Wehrle, Carini, Olling', help=r'Guest Observer ID')
    parser.add_argument('-gopi', '--goPI', type=str,
                        default='GO5038, GO5053, GO5056, GO5096', help=r'Guest Observer PI')
    args = parser.parse_args()

    LC = k2LC(name=args.ID, band='Kep', z=args.z, processing=args.processing,
              campaign=args.campaign, goid=args.goID, gopi=args.goPI)

    LC.plot()
    LC.plotacf()
    LC.plotsf()
    plt.show()
