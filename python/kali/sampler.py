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
    import kali.lc
    from kali.util.mpl_settings import set_plot_params
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
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
        if isinstance(lcObj, kali.lc.basicLC):
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
        returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
        returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
        returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
        returnLC._mindt = float(np.nanmin(returnLC.t))
        returnLC._maxdt = float(np.nanmax(returnLC.t))
        count = int(np.sum(returnLC.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_meanSum += returnLC.mask[i]*returnLC.y[i]
            yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
        if count > 0.0:
            returnLC._mean = y_meanSum/count
            returnLC._meanerr = yerr_meanSum/count
        else:
            returnLC._mean = 0.0
            returnLC._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
            yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
        if count > 0.0:
            returnLC._std = math.sqrt(y_stdSum/count)
            returnLC._stderr = math.sqrt(yerr_stdSum/count)
        else:
            returnLC._std = 0.0
            returnLC._stderr = 0.0
        return returnLC


class bernoulliSampler(sampler):

    def sample(self, **kwargs):
        returnLC = self.lcObj.copy()
        probVal = kwargs.get('probability', 1.0)
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
        returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
        returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
        returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
        returnLC._mindt = float(np.nanmin(returnLC.t))
        returnLC._maxdt = float(np.nanmax(returnLC.t))
        count = int(np.sum(returnLC.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_meanSum += returnLC.mask[i]*returnLC.y[i]
            yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
        if count > 0.0:
            returnLC._mean = y_meanSum/count
            returnLC._meanerr = yerr_meanSum/count
        else:
            returnLC._mean = 0.0
            returnLC._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
            yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
        if count > 0.0:
            returnLC._std = math.sqrt(y_stdSum/count)
            returnLC._stderr = math.sqrt(yerr_stdSum/count)
        else:
            returnLC._std = 0.0
            returnLC._stderr = 0.0
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
        returnLC._T = float(returnLC.t[-1] - returnLC.t[0])
        returnLC._dt = float(returnLC.t[1] - returnLC.t[0])
        returnLC._meandt = float(np.nanmean(returnLC.t[1:] - returnLC.t[:-1]))
        returnLC._mindt = float(np.nanmin(returnLC.t))
        returnLC._maxdt = float(np.nanmax(returnLC.t))
        count = int(np.sum(returnLC.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_meanSum += returnLC.mask[i]*returnLC.y[i]
            yerr_meanSum += returnLC.mask[i]*returnLC.yerr[i]
        if count > 0.0:
            returnLC._mean = y_meanSum/count
            returnLC._meanerr = yerr_meanSum/count
        else:
            returnLC._mean = 0.0
            returnLC._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(returnLC.numCadences):
            y_stdSum += math.pow(returnLC.mask[i]*returnLC.y[i] - returnLC._mean, 2.0)
            yerr_stdSum += math.pow(returnLC.mask[i]*returnLC.yerr[i] - returnLC._meanerr, 2.0)
        if count > 0.0:
            returnLC._std = math.sqrt(y_stdSum/count)
            returnLC._stderr = math.sqrt(yerr_stdSum/count)
        else:
            returnLC._std = 0.0
            returnLC._stderr = 0.0
        return returnLC
