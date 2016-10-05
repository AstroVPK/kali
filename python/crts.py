import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import subprocess
import re
import argparse
import matplotlib.pyplot as plt
import pdb

try:
    import libcarma as libcarma
except ImportError:
    print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
    sys.exit(1)


class crtsLC(libcarma.basicLC):

    def read(self, name, band=r'V', path=None, **kwargs):
        if path is None:
            try:
                path = os.environ['CRTSDATADIR']
            except KeyError:
                raise KeyError(
                    'Environment variable "CRTSDATADIR" not set! Please set "CRTSDATADIR" to point where all CRTS data should live first...')
        self.z = kwargs.get('z', 0.0)
        extension = kwargs.get('extension', '.txt')
        source = kwargs.get('source', 'batch')
        fullPath = os.path.join(path, name + extension)
        with open(fullPath, 'rb') as fileOpen:
            allLines = fileOpen.readlines()
        allLines = [line.rstrip('\n') for line in allLines]
        masterID = list()
        Mag = list()
        Magerr = list()
        Flux = list()
        Fluxerr = list()
        RA = list()
        Dec = list()
        MJD = list()
        for i in xrange(1, len(allLines)-1):
            splitLine = re.split(r'[ ,|\t]+', allLines[i])
            if int(splitLine[6]) == 0:
                masterID.append(int(float(splitLine[0])))
                Mag.append(float(splitLine[1]))
                Magerr.append(float(splitLine[2]))
                flux, fluxerr = libcarma.pogsonFlux(float(splitLine[1]), float(splitLine[2]))
                Flux.append(flux)
                Fluxerr.append(fluxerr)
                RA.append(float(splitLine[3]))
                Dec.append(float(splitLine[4]))
                MJD.append(float(splitLine[5]))
        self._numCadences = len(MJD)
        zipped = sorted(zip(MJD, masterID, Mag, Magerr, Flux, Fluxerr, RA, Dec))
        MJD, masterID, Mag, Magerr, Flux, Fluxerr, RA, Dec = zip(*zipped)
        self.startT = MJD[0]
        MJD = [(mjd - MJD[0])/(1.0 + self.z) for mjd in MJD]
        if self.z == 0.0:
            self.z = r'NO REDSHIFT'

        self.mask = np.require(np.array(self._numCadences*[1.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.array(MJD), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.array(Flux), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.array(Fluxerr), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.array(self._numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mag = np.require(np.array(Mag), requirements=['F', 'A', 'W', 'O', 'E'])
        self.magerr = np.require(np.array(Magerr), requirements=['F', 'A', 'W', 'O', 'E'])
        self.RA = np.require(np.array(RA), requirements=['F', 'A', 'W', 'O', 'E'])
        self.Dec = np.require(np.array(Dec), requirements=['F', 'A', 'W', 'O', 'E'])
        self.masterID = np.require(np.array(masterID), requirements=['F', 'A', 'W', 'O', 'E'])
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

        self._computedCadenceNum = -1
        self._tolIR = 1.0e-3
        self._fracIntrinsicVar = 0.0
        self._fracNoiseToSignal = 0.0
        self._maxSigma = 2.0
        self._minTimescale = 2.0
        self._maxTimescale = 0.5
        self._pSim = 0
        self._qSim = 0
        self._pComp = 0
        self._qComp = 0
        self._isSmoothed = False  # Has the LC been smoothed?
        self._dtSmooth = 0.0
        self._isRegular = False
        self.XSim = np.require(np.zeros(self.pSim), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # State of light curve at last timestamp
        self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Uncertainty in state of light curve at last timestamp.
        self.XComp = np.require(np.zeros(self.pComp), requirements=[
                                'F', 'A', 'W', 'O', 'E'])  # State of light curve at last timestamp
        self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=[
                                'F', 'A', 'W', 'O', 'E'])  # Uncertainty in state of light curve at last timestamp.
        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'V')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$d$'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'$F$ (Jy)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

        count = int(np.sum(self.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(self.numCadences):
            y_meanSum += self.mask[i]*self.y[i]
            yerr_meanSum += self.mask[i]*self.yerr[i]
        if count > 0.0:
            self._mean = y_meanSum/count
            self._meanerr = yerr_meanSum/count
        else:
            self._mean = 0.0
            self._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(self.numCadences):
            y_stdSum += math.pow(self.mask[i]*self.y[i] - self._mean, 2.0)
            yerr_stdSum += math.pow(self.mask[i]*self.yerr[i] - self._meanerr, 2.0)
        if count > 0.0:
            self._std = math.sqrt(y_stdSum/count)
            self._stderr = math.sqrt(yerr_stdSum/count)
        else:
            self._std = 0.0
            self._stderr = 0.0

    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(r'-n', r'--name', type=str, default=r'PG1302102', help=r'Object name')
    parser.add_argument(r'-z', r'--z', type=float, default=0.2784, help=r'Object redshift')
    # parser.add_argument(r'-n', r'--name', type = str, default = r'OH287', help = r'Object name')
    # parser.add_argument(r'-z', r'--redShift', type = float, default = 0.305, help = r'Object redshift')
    args = parser.parse_args()

    LC = crtsLC(name=args.name, band='V', z=args.z)

    LC.plot()
    LC.plotacf()
    LC.plotsf()
    plt.show(False)
