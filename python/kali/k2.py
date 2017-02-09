import math as math
import numpy as np
import urllib
import urllib2
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
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


class k2LC(kali.lc.lc):

    sap = ['sap', 'raw', 'uncal', 'un-cal', 'uncalibrated', 'un-calibrated']
    pdcsap = ['pdcsap', 'mast', 'cal', 'calib', 'calibrated']
    k2sff = ['k2sff', 'vj', 'v-j', 'vanderburg', 'vanderburgjohnson', 'vanderburg-johnson']
    k2sc = ['k2sc', 'aigrain']
    k2varcat = ['k2varcat', 'armstrong']
    everest = ['everest', 'luger']

    def _getCanonicalFileName(self, name, campaign, processing):
        fileName = ''
        if campaign == 'c10':
            campaign = 'c102'
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

    def _fetchFromURL(self, fullURL):
        ret = None
        try:
            ret = urllib2.urlopen(fullURL)
        except (urllib2.HTTPError, urllib2.URLError) as err:
                warnings.warn('Could not fetch %s'%(fullURL), UserWarning)
                if isinstance(err, urllib2.HTTPError):
                    pass
                else:
                    try:
                        retGOOGLE = urllib2.urlopen('http://www.google.com')
                    except (urllib2.HTTPError, urllib2.URLError) as errGOOGLE:
                        raise urllib2.URLError('Could not reach www.google.com -\
                        check your internet connection')
                    else:
                        try:
                            retSTSCI = urllib2.urlopen('http://www.stsci.edu')
                        except (urllib2.HTTPError, urllib2.URLError) as errSTSCI:
                            raise urllib2.URLError('stsci.edu may be down.')
                        else:
                            raise ValueError('Invalid name & campaign combination!')
        return ret

    def _getMAST(self, name, campaign, path, goid, gopi):
        baseURL = 'http://archive.stsci.edu/pub/k2/lightcurves'
        mastRet = False
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
            if camp == 'c10':
                camp = 'c102'
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:6], '000'])
            fullURL = '/'.join([baseURL, camp, name1Dir, name2Dir, fileNameFits])
            ret = self._fetchFromURL(fullURL)
            if ret is not None:
                result = urllib.urlretrieve(fullURL, filePathFits)
                mastRet = True
            else:
                raise ValueError('Invalid name & campaign combination!')
        else:
            mastRet = True
        return mastRet

    def _getHLSP(self, name, campaign, path):
        baseURL = 'http://archive.stsci.edu/missions/hlsp'
        hlspRet = [False, False, False, False]

        fileName = self._getCanonicalFileName(name, campaign, 'k2sff')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = name[4:]
            fullURL = '/'.join([baseURL, 'k2sff', campaign, name1Dir, name2Dir, fileNameFits])
            ret = self._fetchFromURL(fullURL)
            if ret is not None:
                result = urllib.urlretrieve(fullURL, filePathFits)
                hlspRet[0] = True
            else:
                pass
        else:
            hlspRet[0] = True

        fileName = self._getCanonicalFileName(name, campaign, 'k2sc')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            fullURL = '/'.join([baseURL, 'k2sc', campaign, name1Dir, fileNameFits])
            ret = self._fetchFromURL(fullURL)
            if ret is not None:
                result = urllib.urlretrieve(fullURL, filePathFits)
                hlspRet[1] = True
            else:
                pass
        else:
            hlspRet[1] = True

        fileName = self._getCanonicalFileName(name, campaign, 'k2varcat')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:6], '000'])
            fullURL = '/'.join([baseURL, 'k2varcat', campaign, name1Dir, name2Dir, fileNameFits])
            ret = self._fetchFromURL(fullURL)
            if ret is not None:
                result = urllib.urlretrieve(fullURL, filePathFits)
                hlspRet[2] = True
            else:
                pass
        else:
            hlspRet[2] = True

        fileName = self._getCanonicalFileName(name, campaign, 'everest')
        fileNameFits = ''.join([fileName[0:-3], 'fits'])
        filePath = os.path.join(path, fileName)
        filePathFits = os.path.join(path, fileNameFits)
        if not os.path.isfile(filePathFits):
            name1Dir = ''.join([name[0:4], '00000'])
            name2Dir = ''.join([name[4:]])
            fullURL = '/'.join([baseURL, 'everest', campaign, name1Dir, name2Dir, fileNameFits])
            ret = self._fetchFromURL(fullURL)
            if ret is not None:
                result = urllib.urlretrieve(fullURL, filePathFits)
                hlspRet[3] = True
            else:
                pass
        else:
            hlspRet[3] = True

        return hlspRet

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
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
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
                self.terr[i] = float(dataLine[1])
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
                    self.terr[i] = float(dataLine[1])
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                    self.terr[i] = 0.0
                self.yerr[i] = math.sqrt(sys.float_info[0])
                self.mask[i] = 0.0

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
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
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
                self.terr[i] = float(dataLine[1])
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                    self.terr[i] = float(dataLine[1])
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                    self.terr[i] = 0.0

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
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
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
                self.terr[i] = float(dataLine[1])
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                    self.terr[i] = float(dataLine[1])
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                    self.terr[i] = 0.0

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
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
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
                self.terr[i] = float(dataLine[1])
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                    self.terr[i] = float(dataLine[1])
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                    self.terr[i] = 0.0

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
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
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
                self.terr[i] = float(dataLine[1])
            else:
                if not np.isnan(dataLine[0]):
                    self.t[i] = float(dataLine[0]) - self.startT
                    self.terr[i] = float(dataLine[1])
                else:
                    self.t[i] = self.t[i - 1] + self.dt
                    self.terr[i] = 0.0

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
        if self.campaign[0] != 'c':
            if int(self.campaign) < 10:
                self.campaign = 'c0' + str(int(self.campaign))
            else:
                self.campaign = 'c' + self.campaign
        fileName = self._getCanonicalFileName(name, self.campaign, self.processing)
        self.goid = kwargs.get('goid', '').lower()
        self.gopi = kwargs.get('gopi', '').lower()
        if path is None:
            try:
                self.path = os.environ['K2DATADIR']
            except KeyError:
                raise KeyError('Environment variable "K2DATADIR" not set! Please set "K2DATADIR" to point \
                where all K2 data should live first...')
        else:
            self.path = path
        filePath = os.path.join(self.path, fileName)

        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'Kep')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$t$~(MJD)'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'$F$~($\mathrm{e^{-}}$)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

        mastRet = self._getMAST(name, self.campaign, self.path, self.goid, self.gopi)
        hlspRet = self._getHLSP(name, self.campaign, self.path)

        if self.processing in self.sap or self.processing in self.pdcsap:
            if mastRet:
                self._readMAST(name, self.campaign, self.path, self.processing)
            else:
                raise ValueError('MAST light curve not found!')
        elif self.processing in self.k2sff:
            if hlspRet[0]:
                self._readK2SFF(name, self.campaign, self.path, self.processing)
            else:
                raise ValueError('Vanderburg-Johnson light curve not found!')
        elif self.processing in self.k2sc:
            if hlspRet[1]:
                self._readK2SC(name, self.campaign, self.path, self.processing)
            else:
                raise ValueError('Aigrain Lightcurve not found!')
        elif self.processing in self.k2varcat:
            if hlspRet[2]:
                self._readK2VARCAT(name, self.campaign, self.path, self.processing)
            else:
                raise ValueError('Armstrong light curve not found!')
        elif self.processing in self.everest:
            if hlspRet[3]:
                self._readEVEREST(name, self.campaign, self.path, self.processing)
            else:
                raise ValueError('Luger light curve not found!')
        else:
            raise ValueError('Processing not understood!')

        for i in xrange(self._numCadences):
            self.t[i] = self.t[i]/(1.0 + self.z)
        coords = self._getCoordinates()
        if coords:
            self.coordinates = SkyCoord(self._getCoordinates(),
                                        unit=(units.hourangle, units.deg), frame='icrs')

    def _getCoordinates(self):
        """!
        \brief Query K2 EPIC ID to get ra dec etc...
        """
        url = 'http://archive.stsci.edu/k2/epic/search.php?action=Search&id=%s&outputformat=CSV'%(
            self.name)
        try:
            lines = urllib.urlopen(url)
        except (urllib2.HTTPError, urllib2.URLError, IOError) as err:
            print 'Could not reach %s -\
            %s may not be functioning at this time. Error %s'%(url, url, err)
            sys.exit(-1)
        data = {}
        counter = 0
        lineList = list()
        for line in lines:
            lineList.append(line.rstrip('\n'))
        if lineList[0] == 'target not resolved, continue':
            warnings.warn('Unable to fetch cordinates! - %s returns: %s'%(url, lineList[0]), UserWarning)
            coord_str = None
        else:
            vals = lineList[2].split(',')
            coord_str = vals[1] + ' ' + vals[2]
        return coord_str

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
