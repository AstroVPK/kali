import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import fitsio
from fitsio import FITS, FITSHDR
import subprocess
import argparse
import matplotlib.pyplot as plt
import pdb

try:
    import kali.lc
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


class k2LC(kali.lc.basicLC):

    sap = ['sap', 'raw', 'uncal', 'un-cal', 'uncalibrated', 'un-calibrated']
    pdcsap = ['pdcsap', 'mast', 'cal', 'calib', 'calibrated']
    k2sff = ['k2sff', 'vj', 'vanderburg', 'vanderburgjohnson', 'vanderburg-johnson']
    k2sc = ['k2sc', 'aigrain']
    k2varcat = ['k2varcat', 'armstrong']
    everest = ['everest', 'luger']

    def _getCanonicalFileName(self, name, campaign, processing):
        fileName = ''
        if processing in self.sap or processing in self.pdcsap:
            fileName = ''.join(['ktwo', name, '-', campaign, '_llc.dat'])
        elif (processing in self.k2sff or processing in self.k2sc or processing in self.k2varcat or
              processing in self.everest):
            if processing in self.k2sff:
                fileName = ''.join(['hlsp_k2sff_k2_lightcurve_', name, '-', campaign, '_kepler_v1_llc.dat'])
            elif processing in self.k2sc:
                fileName = ''.join(['hlsp_k2sc_k2_llc_', name, '-', campaign, '_kepler_v1_lc.dat'])
            elif processing in self.k2varcat:
                fileName = ''.join(
                    ['hlsp_k2varcat_k2_lightcurve_', name, '-', campaign, '_kepler_v2_llc.dat'])
            elif processing in self.everest:
                fileName = ''.join(['hlsp_everest_k2_llc_', name, '-', campaign, '_kepler_v1.0_lc.dat'])
            else:
                raise ValueError('Unrecognized k2LC type')
        else:
            raise ValueError('Unrecognized k2LC type')
        return fileName

    def _getMAST(self, name, campaign, path, goid, gopi):
        baseURL = 'http://archive.stsci.edu/pub/k2/lightcurves'
        recordFile = 'k2List.dat'

        fileName = self._getCanonicalFileName(name, campaign, 'mast')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = ''.join([filePath[0:-3], 'fits'])

        if not os.path.isfile(filePathFits):
            recordFilePath = os.path.join(path, recordFile)
            with open(recordFilePath, 'a') as record:
                record.write('%s %s %s %s\n'%(name, campaign, goid, gopi))
            camp = ''.join(['c', str(int(campaign[1:]))])
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:6], '000'])
            fullURL = '/'.join([baseURL, camp, name1Dir, name2Dir, fileNameFits])
            result = urllib.urlretrieve(fullURL, filePathFits)

    def _getHLSP(self, name, campaign, path):
        baseURL = 'http://archive.stsci.edu/missions/hlsp'

        fileName = self._getCanonicalFileName(name, campaign, 'k2sff')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = name[4:]
            fullURL = '/'.join([baseURL, 'k2sff', campaign, name1Dir, name2Dir, fileNameFits])
            try:
                ret = urllib2.urlopen(fullURL)
            except urllib2.HTTPError:
                pass
            else:
                result = urllib.urlretrieve(fullURL, filePathFits)
        if not os.path.isfile(filePath) and os.path.isfile(filePathFits):
            subprocess.call(['topcat', '-stilts', 'tcopy', 'in=%s' %
                            (filePathFits), 'ofmt=ascii', 'out=%s'%(filePath)])

        fileName = self._getCanonicalFileName(name, campaign, 'k2sc')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            fullURL = '/'.join([baseURL, 'k2sc', campaign, name1Dir, fileNameFits])
            try:
                ret = urllib2.urlopen(fullURL)
            except urllib2.HTTPError:
                pass
            else:
                result = urllib.urlretrieve(fullURL, filePathFits)
        if not os.path.isfile(filePath) and os.path.isfile(filePathFits):
            subprocess.call(['topcat', '-stilts', 'tcopy', 'in=%s' %
                            (filePathFits), 'ofmt=ascii', 'out=%s'%(filePath)])

        fileName = self._getCanonicalFileName(name, campaign, 'k2varcat')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:6], '000'])
            fullURL = '/'.join([baseURL, 'k2varcat', campaign, name1Dir, name2Dir, fileNameFits])
            try:
                ret = urllib2.urlopen(fullURL)
            except urllib2.HTTPError:
                pass
            else:
                result = urllib.urlretrieve(fullURL, filePathFits)
        if not os.path.isfile(filePath) and os.path.isfile(filePathFits):
            subprocess.call(['topcat', '-stilts', 'tcopy', 'in=%s' %
                            (filePathFits), 'ofmt=ascii', 'out=%s'%(filePath)])

        fileName = self._getCanonicalFileName(name, campaign, 'everest')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:]])
            fullURL = '/'.join([baseURL, 'everest', campaign, name1Dir, name2Dir, fileNameFits])
            try:
                ret = urllib2.urlopen(fullURL)
            except urllib2.HTTPError:
                pass
            else:
                result = urllib.urlretrieve(fullURL, filePathFits)
        if not os.path.isfile(filePath) and os.path.isfile(filePathFits):
            subprocess.call(['topcat', '-stilts', 'tcopy', 'in=%s' %
                            (filePathFits), 'ofmt=ascii', 'out=%s'%(filePath)])

    def _readMAST(self, name, campaign, path, processing):
        fileName = self._getCanonicalFileName(name, campaign, processing)
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePathFits = os.path.join(path, fileNameFits)
        dataInFile = fitsio.read(filePathFits)
        self._numCadences = dataInFile.shape[0]
        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = dataInFile[lineNum][0]
            startTCandNext = dataInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(self.numCadences):
            dataLine = dataInFile[i]
            self.cadence[i] = int(dataLine[2])
            if dataLine[9] == 0:
                self.t[i] = float(dataLine[0]) - self.startT
                if processing in self.sap:
                    if not np.isnan(dataLine[3]) and not np.isnan(dataLine[4]):
                        self.y[i] = float(dataLine[3])
                        self.yerr[i] = float(dataLine[4])
                        self.mask[i] = 1.0
                    else:
                        self.y[i] = 0.0
                        self.yerr[i] = math.sqrt(sys.float_info[0])
                        self.mask[i] = 0.0
                elif processing in self.pdcsap:
                    if not np.isnan(dataLine[7]) and not np.isnan(dataLine[8]):
                        self.y[i] = float(dataLine[7])
                        self.yerr[i] = float(dataLine[8])
                        self.mask[i] = 1.0
                    else:
                        self.y[i] = 0.0
                        self.yerr[i] = math.sqrt(sys.float_info[0])
                        self.mask[i] = 0.0
                else:
                    raise ValueError('Unrecognized k2LC type')
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                self.yerr[i] = math.sqrt(sys.float_info[0])
                self.mask[i] = 0.0
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

    def _readK2SFF(self, name, campaign, path, processing):
        fileNameMAST = self._getCanonicalFileName(name, campaign, 'mast')
        fileNameMASTFits = ''.join([fileNameMAST[0:-3], 'fits'])
        filePathMASTFits = os.path.join(path, fileNameMASTFits)
        MASTInFile = fitsio.read(filePathMASTFits)
        self._numCadences = MASTInFile.shape[0]
        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = MASTInFile[lineNum][0]
            startTCandNext = MASTInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(self.numCadences):
            dataLine = MASTInFile[i]
            self.cadence[i] = int(dataLine[2])
            self.yerr[i] = math.sqrt(sys.float_info[0])
            if dataLine[9] == 0:
                self.t[i] = float(dataLine[0]) - self.startT
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                else:
                    self.t[i] = self.t[i - 1] + self.dt
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

        fileName = self._getCanonicalFileName(name, campaign, 'k2sff')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePathFits = os.path.join(path, fileNameFits)
        dataInFile = fitsio.read(filePathFits)
        for i in xrange(dataInFile.shape[0]):
            dataLine = dataInFile[i]
            cadNum = int(dataLine[5])
            index = np.where(self.cadence == cadNum)[0][0]
            self.y[index] = float(dataLine[2])
            self.mask[index] = 1.0

        valSum = 0.0
        countSum = 0.0
        for i in xrange(self.numCadences - 1):
            valSum += self.mask[i + 1]*self.mask[i]*math.pow((self.y[i + 1] - self.y[i]), 2.0)
            countSum += self.mask[i + 1]*self.mask[i]
        noise = math.sqrt(valSum/countSum)
        for i in xrange(self.numCadences):
            if self.mask[i] == 1.0:
                self.yerr[i] = noise

    def _readK2SC(self, name, campaign, path, processing):
        fileNameMAST = self._getCanonicalFileName(name, campaign, 'mast')
        fileNameMASTFits = ''.join([fileNameMAST[0:-3], 'fits'])
        filePathMASTFits = os.path.join(path, fileNameMASTFits)
        MASTInFile = fitsio.read(filePathMASTFits)
        self._numCadences = MASTInFile.shape[0]
        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = MASTInFile[lineNum][0]
            startTCandNext = MASTInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(self.numCadences):
            dataLine = MASTInFile[i]
            self.cadence[i] = int(dataLine[2])
            self.yerr[i] = math.sqrt(sys.float_info[0])
            if dataLine[9] == 0:
                self.t[i] = float(dataLine[0]) - self.startT
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                else:
                    self.t[i] = self.t[i - 1] + self.dt
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

        fileName = self._getCanonicalFileName(name, campaign, 'k2sc')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePathFits = os.path.join(path, fileNameFits)
        dataInFile = fitsio.read(filePathFits)
        for i in xrange(dataInFile.shape[0]):
            dataLine = dataInFile[i]
            if dataLine[7] == 0:
                time = float(dataLine[0]) - self.startT
                if not np.isnan(time):
                    index = np.where(self.t == time)[0][0]
                    self.y[index] = float(dataLine[8])
                    self.yerr[index] = float(dataLine[6])
                    self.mask[index] = 1.0
                else:
                    pass

    def _readK2VARCAT(self, name, campaign, path, processing):
        fileNameMAST = self._getCanonicalFileName(name, campaign, 'mast')
        fileNameMASTFits = ''.join([fileNameMAST[0:-3], 'fits'])
        filePathMASTFits = os.path.join(path, fileNameMASTFits)
        MASTInFile = fitsio.read(filePathMASTFits)
        self._numCadences = MASTInFile.shape[0]
        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = MASTInFile[lineNum][0]
            startTCandNext = MASTInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(self.numCadences):
            dataLine = MASTInFile[i]
            self.cadence[i] = int(dataLine[2])
            self.yerr[i] = math.sqrt(sys.float_info[0])
            if dataLine[9] == 0:
                self.t[i] = float(dataLine[0]) - self.startT
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                else:
                    self.t[i] = self.t[i - 1] + self.dt
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

        fileName = self._getCanonicalFileName(name, campaign, 'k2varcat')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePathFits = os.path.join(path, fileNameFits)
        try:
            dataInFile = fitsio.read(filePathFits)
        except IOError as Err:
            pass
        else:
            for i in xrange(dataInFile.shape[0]):
                dataLine = dataInFile[i]
                time = float(dataLine[0]) - self.startT
                if not np.isnan(time):
                    index = np.where(self.t == time)[0][0]
                    self.y[index] = float(dataLine[3])
                    self.yerr[index] = float(dataLine[4])
                    self.mask[index] = 1.0
                else:
                    pass

    def _readEVEREST(self, name, campaign, path, processing):
        fileNameMAST = self._getCanonicalFileName(name, campaign, 'mast')
        fileNameMASTFits = ''.join([fileNameMAST[0:-3], 'fits'])
        filePathMASTFits = os.path.join(path, fileNameMASTFits)
        MASTInFile = fitsio.read(filePathMASTFits)
        self._numCadences = MASTInFile.shape[0]
        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = MASTInFile[lineNum][0]
            startTCandNext = MASTInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(self.numCadences):
            dataLine = MASTInFile[i]
            self.cadence[i] = int(dataLine[2])
            self.yerr[i] = math.sqrt(sys.float_info[0])
            if dataLine[9] == 0:
                self.t[i] = float(dataLine[0]) - self.startT
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                else:
                    self.t[i] = self.t[i - 1] + self.dt
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

        fileName = self._getCanonicalFileName(name, campaign, 'everest')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePathFits = os.path.join(path, fileNameFits)
        try:
            dataInFile = fitsio.read(filePathFits)
        except IOError as Err:
            pass
        else:
            for i in xrange(dataInFile.shape[0]):
                dataLine = dataInFile[i]
                time = float(dataLine[0]) - self.startT
                if not np.isnan(time):
                    index = np.where(self.t == time)[0][0]
                    if not np.isnan(float(dataLine[1])):
                        self.y[index] = float(dataLine[1])
                        self.mask[index] = 1.0
                else:
                    pass
        valSum = 0.0
        countSum = 0.0
        for i in xrange(self.numCadences - 1):
            valSum += self.mask[i + 1]*self.mask[i]*math.pow((self.y[i + 1] - self.y[i]), 2.0)
            countSum += self.mask[i + 1]*self.mask[i]
        noise = math.sqrt(valSum/countSum)
        for i in xrange(self.numCadences):
            if self.mask[i] == 1.0:
                self.yerr[i] = noise

    def read(self, name, band=None, path=None, **kwargs):
        self.z = kwargs.get('z', 0.0)
        self.processing = kwargs.get('processing', 'k2sff').lower()
        self.campaign = kwargs.get('campaign', 'c05').lower()
        fileName = self._getCanonicalFileName(name, self.campaign, self.processing)
        self.goid = kwargs.get('goid', '').lower()
        self.gopi = kwargs.get('gopi', '').lower()
        if path is None:
            try:
                path = os.environ['K2DATADIR']
            except KeyError:
                raise KeyError('Environment variable "K2DATADIR" not set! Please set "K2DATADIR" to point \
                where all K2 data should live first...')
        filePath = os.path.join(path, fileName)

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
        self._isRegular = True
        self.XSim = np.require(np.zeros(self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])
        self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])
        self.XComp = np.require(np.zeros(self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])
        self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])
        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'Kep')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$d$'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        # self._yunit = r'who the f*** knows?' ## Unit in which the flux is measured (eg Wm^{-2} etc...).
        self._yunit = r'$F$'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

        self._getMAST(name, self.campaign, path, self.goid, self.gopi)
        self._getHLSP(name, self.campaign, path)

        if self.processing in self.sap or self.processing in self.pdcsap:
            self._readMAST(name, self.campaign, path, self.processing)
        elif self.processing in self.k2sff:
            self._readK2SFF(name, self.campaign, path, self.processing)
        elif self.processing in self.k2sc:
            self._readK2SC(name, self.campaign, path, self.processing)
        elif self.processing in self.k2varcat:
            self._readK2VARCAT(name, self.campaign, path, self.processing)
        elif self.processing in self.everest:
            self._readEVEREST(name, self.campaign, path, self.processing)
        else:
            raise ValueError('Processing not found!')

        for i in xrange(self._numCadences):
            self.t[i] = self.t[i]/(1.0 + self.z)

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
