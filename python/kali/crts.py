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
    import kali.lc
    import kali.carma
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


class crtsLC(kali.lc.lc):

    def read(self, name, band=r'V', path=None, **kwargs):
        if path is None:
            try:
                self.path = os.environ['CRTSDATADIR']
            except KeyError:
                raise KeyError(
                    'Environment variable "CRTSDATADIR" not set! Please set "CRTSDATADIR" to point where all \
                    CRTS data should live first...')
        else:
            self.path = path
        self.z = kwargs.get('z', 0.0)
        extension = kwargs.get('extension', '.txt')
        source = kwargs.get('source', 'batch')
        fullPath = os.path.join(self.path, name + extension)
        with open(fullPath, 'rb') as fileOpen:
            allLines = fileOpen.readlines()
        allLines = [line.rstrip('\n') for line in allLines]
        masterID = list()
        Mag = list()
        MagErr = list()
        Flux = list()
        FluxErr = list()
        RA = list()
        Dec = list()
        MJD = list()
        for i in xrange(1, len(allLines)-1):
            splitLine = re.split(r'[ ,|\t]+', allLines[i])
            if int(splitLine[6]) == 0:
                masterID.append(int(float(splitLine[0])))
                Mag.append(float(splitLine[1]))
                MagErr.append(float(splitLine[2]))
                flux, fluxerr = kali.carma.pogsonFlux(float(splitLine[1]), float(splitLine[2]))
                Flux.append(flux)
                FluxErr.append(fluxerr)
                RA.append(float(splitLine[3]))
                Dec.append(float(splitLine[4]))
                MJD.append(float(splitLine[5]))
        zipped = sorted(zip(MJD, masterID, Mag, MagErr, Flux, FluxErr, RA, Dec))
        MJD, masterID, Mag, MagErr, Flux, FluxErr, RA, Dec = zip(*zipped)
        oldNumCadences = len(MJD)
        intMJD = np.array(sorted(list(set([int(MJD[i]//1) for i in xrange(len(MJD))]))))
        self.numCadences = len(intMJD)
        self.t = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mag = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.magerr = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.RA = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.Dec = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.array(self.numCadences*[1.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self.counts = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])

        for i in xrange(oldNumCadences):
            index = np.where(int(MJD[i]//1) == intMJD)[0][0]
            self.t[index] += MJD[i]
            self.mag[index] += Mag[i]
            # self.magerr[index] += MagErr[i]
            self.y[index] += Flux[i]
            # self.yerr[index] += FluxErr[i]
            self.RA[index] += RA[i]
            self.Dec[index] += Dec[i]
            self.counts[index] += 1.0

        MagCont = list()
        FluxCont = list()
        for i in xrange(self.numCadences):
            self.t[i] = self.t[i]/self.counts[i]
            self.mag[i] = self.mag[i]/self.counts[i]
            self.y[i] = self.y[i]/self.counts[i]
            self.RA[i] = self.RA[i]/self.counts[i]
            self.Dec[i] = self.Dec[i]/self.counts[i]
            del MagCont[:]
            del FluxCont[:]
            for j in xrange(oldNumCadences):
                if int(MJD[j]//1) == intMJD[i]:
                    MagCont.append(Mag[j])
                    FluxCont.append(Flux[j])
            self.magerr[i] = np.std(np.array(MagCont), ddof=1)
            self.yerr[i] = np.std(np.array(FluxCont), ddof=1)

        SNRat = np.zeros(self.numCadences)
        for i in xrange(self.numCadences):
            if self.yerr[i] != 0.0:
                SNRat[i] = self.y[i]/self.yerr[i]
            else:
                SNRat[i] = np.nan
        meanSNRat = np.nanmedian(SNRat)

        for i in xrange(self.numCadences):
            if self.yerr[i] == 0.0 or np.isnan(self.yerr[i]):
                self.yerr[i] = self.y[i]/meanSNRat  # Ugly hack in case the fluxes are all actaully the same.

        self.startT = self.t[0]/(1.0 + self.z)
        self.t = (self.t - self.startT)/(1.0 + self.z)
        self.x = np.require(np.array(self.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'V')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$t_{\mathrm{rest}}$ (d)'  # Unit in which time is measured.
        self._yunit = r'$F$ (Jy)'  # Unit in which the flux is measured.

    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(r'-n', r'--name', type=str, default=r'PG1302-102', help=r'Object name')
    parser.add_argument(r'-z', r'--z', type=float, default=0.2784, help=r'Object redshift')
    # parser.add_argument(r'-n', r'--name', type = str, default = r'OH287', help = r'Object name')
    # parser.add_argument(r'-z', r'--redShift', type = float, default = 0.305, help = r'Object redshift')
    args = parser.parse_args()

    LC = crtsLC(name=args.name, band='V', z=args.z)

    LC.plot()
    plt.show()
