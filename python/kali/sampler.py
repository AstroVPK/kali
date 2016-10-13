#!/usr/bin/env python
"""	Module to downsample light curves.
"""

import numpy as np
import math as math
import scipy.stats as spstats
import cmath as cmath
import operator as operator
import sys as sys
import abc as abc
import psutil as psutil
import types as types
import os as os
import reprlib as reprlib
import copy as copy
from scipy.interpolate import UnivariateSpline
import warnings as warnings
import matplotlib.pyplot as plt
import pdb as pdb

try:
    import rand
except ImportError:
    print 'Cannot import rand! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)
try:
    import kali.lc
except ImportError:
    print 'Cannot import kali.lc! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)
try:
    from kali.util.mpl_settings import set_plot_params
except ImportError:
    print 'Cannot import kali.util.mpl_settings! kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex=True)

ln10 = math.log(10)


class sampler(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lcObj):
        """!
        \brief Initialize the sampler.

        """
        if isinstance(lcObj, kali.lc.lc):
            self.min_dt = np.min(lcObj.t[1:] - lcObj.t[:-1])
            self.max_T = lcObj.t[-1] - lcObj.t[0]
            self.lcObj = lcObj

    @abc.abstractmethod
    def sample(self, **kwargs):
        raise NotImplemented


class jumpSampler(sampler):

    def sample(self, **kwargs):
        returnLC = self.lcObj.copy()
        jumpVal = kwargs.get('jump', 1)
        newNumCadences = int(self.lcObj.numCadences/float(jumpVal))
        tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(newNumCadences):
            tNew[i] = self.lcObj.t[jumpVal*i]
            xNew[i] = self.lcObj.x[jumpVal*i]
            yNew[i] = self.lcObj.y[jumpVal*i]
            yerrNew[i] = self.lcObj.yerr[jumpVal*i]
            maskNew[i] = self.lcObj.mask[jumpVal*i]
        returnLC.t = tNew
        returnLC.x = xNew
        returnLC.y = yNew
        returnLC.yerr = yerrNew
        returnLC.mask = maskNew
        returnLC._numCadences = newNumCadences
        returnLC._checkIsRegular()
        returnLC._times()
        returnLC._statistics()
        return returnLC


class bernoulliSampler(sampler):

    def sample(self, **kwargs):
        returnLC = self.lcObj.copy()
        probVal = kwargs.get('probability', 1.0)
        sampleSeedVal = kwargs.get('sampleSeed', rand.rdrand(np.array([0], dtype='uint32')))
        np.random.seed(seed=sampleSeedVal)
        keepArray = spstats.bernoulli.rvs(probVal, size=self.lcObj.numCadences)
        newNumCadences = np.sum(keepArray)
        tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        counter = 0
        for i in xrange(self.lcObj.numCadences):
            if keepArray[i] == 1:
                tNew[counter] = self.lcObj.t[i]
                xNew[counter] = self.lcObj.x[i]
                yNew[counter] = self.lcObj.y[i]
                yerrNew[counter] = self.lcObj.yerr[i]
                maskNew[counter] = self.lcObj.mask[i]
                counter += 1
        returnLC.t = tNew
        returnLC.x = xNew
        returnLC.y = yNew
        returnLC.yerr = yerrNew
        returnLC.mask = maskNew
        returnLC._numCadences = newNumCadences
        returnLC._checkIsRegular()
        returnLC._times()
        returnLC._statistics()
        return returnLC


class matchSampler(sampler):

    def sample(self, **kwargs):
        returnLC = self.lcObj.copy()
        timeStamps = kwargs.get('timestamps', None)
        timeStampDeltas = timeStamps[1:] - timeStamps[:-1]
        SDSSLength = timeStamps[-1] - timeStamps[0]
        minDelta = np.min(timeStampDeltas)
        if minDelta < self.lcObj.dt:
            raise ValueError('Insufficiently dense sampling!')
        if SDSSLength > self.lcObj.T:
            raise ValueError('Insufficiently long lc!')
        newNumCadences = timeStamps.shape[0]
        tNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        xNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        yerrNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        maskNew = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(newNumCadences):
            index = np.where(self.lcObj.t > timeStamps[i])[0][0]
            tNew[i] = self.lcObj.t[index]
            xNew[i] = self.lcObj.x[index]
            yNew[i] = self.lcObj.y[index]
            yerrNew[i] = self.lcObj.yerr[index]
            maskNew[i] = self.lcObj.mask[index]
        returnLC.t = tNew
        returnLC.x = xNew
        returnLC.y = yNew
        returnLC.yerr = yerrNew
        returnLC.mask = maskNew
        returnLC._numCadences = newNumCadences
        returnLC._checkIsRegular()
        returnLC._times()
        returnLC._statistics()
        return returnLC


class sincSampler(sampler):

    def normalizedSincSq(self, widthVal, centerVal, t):
        if ((t - centerVal)/widthVal) == 0:
            val = 1.0
        else:
            val = math.fabs(math.sin(
                math.pi*((t - centerVal)/widthVal))/(math.pi*((t - centerVal)/widthVal)))
        return val

    def sample(self, **kwargs):
        returnLC = self.lcObj.copy()
        widthVal = kwargs.get('width', returnLC.T/10.0)
        centerVal = kwargs.get('center', returnLC.T/2.0)
        sampleSeedVal = kwargs.get('sampleSeed', rand.rdrand(np.array([0], dtype='uint32')))
        np.random.seed(seed=sampleSeedVal)
        del returnLC.t
        del returnLC.x
        del returnLC.y
        del returnLC.yerr
        del returnLC.mask
        tList = list()
        xList = list()
        yList = list()
        yerrList = list()
        maskList = list()
        for i in xrange(self.lcObj.numCadences):
            keepYN = np.random.binomial(1, self.normalizedSincSq(widthVal, centerVal, self.lcObj.t[i]))
            if keepYN == 1:
                tList.append(self.lcObj.t[i])
                xList.append(self.lcObj.x[i])
                yList.append(self.lcObj.y[i])
                yerrList.append(self.lcObj.yerr[i])
                maskList.append(self.lcObj.mask[i])
        newNumCadences = len(tList)
        returnLC.t = np.require(np.array(tList), requirements=['F', 'A', 'W', 'O', 'E'])
        returnLC.x = np.require(np.array(xList), requirements=['F', 'A', 'W', 'O', 'E'])
        returnLC.y = np.require(np.array(yList), requirements=['F', 'A', 'W', 'O', 'E'])
        returnLC.yerr = np.require(np.array(yerrList), requirements=['F', 'A', 'W', 'O', 'E'])
        returnLC.mask = np.require(np.array(maskList), requirements=['F', 'A', 'W', 'O', 'E'])
        returnLC._numCadences = newNumCadences
        returnLC._checkIsRegular()
        returnLC._times()
        returnLC._statistics()
        return returnLC
