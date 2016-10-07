#!/usr/bin/env python
"""	Module to perform basic C-ARMA modelling.
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
    import rand as rand
    import CARMATask_cython as CARMATask_cython
    import kali.sampler
    from kali.util.mpl_settings import set_plot_params
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex=True)

ln10 = math.log(10)


class epoch(object):

    """!
    \anchor epoch_

    \brief Class to hold individual epochs of a light curve.

    We wish to hold individual epochs in a light curve in an organized manner. This class lets us examine
    individual epochs and check for equality with other epochs. Two epochs are equal iff they have the same
    timestamp. Later on, we will implement some sort of unit system for the quantities (i.e. is the tiumestamp
    in sec, min, day, MJD etc...?)
    """

    def __init__(self, t, x, y, yerr, mask):
        """!
        \brief Initialize the epoch.

        Non-keyword arguments
        \param[in] t:       Timestamp.
        \param[in] x:       Intrinsic Flux i.e. the theoretical value of the underlying flux in the absence of
                            measurement error.
        \param[in] y:       Observed Flux i.e. the observed value of the flux given a noise-to-signal level.
        \param[in] yerr:    Error in Observed Flux i.e. the measurement error in the flux.
        \param[in] mask:    Mask value at this epoch. 0.0 means that this epoch has a missing observation. 1.0
                            means that the observation exists.
        """
        self.t = t  # Timestamp
        self.x = x  # Intrinsic flux
        self.y = y  # Observed flux
        self.yerr = yerr  # Error in Observed Flux
        self.mask = mask  # Mask value at this epoch

    def __repr__(self):
        """!
        \brief Return a representation of the epoch such that eval(repr(someEpoch)) == someEpoch is True.
        """
        return u"libcarma.epoch(%f, %f, %f, %f, %f)"%(self.t, self.x, self.y, self.yerr, self.mask)

    def __str__(self):
        """!
        \brief Return a human readable representation of the epoch.
        """
        if self.mask == 1.0:
            return r't = %f MJD; intrinsic flux = %+f; observed flux = %+f; \
                     observed flux error = %+f'%(self.t, self.x, self.y, self.yerr)
        else:
            return r't = %f MJD; no data!'%(self.t)

    def __eq__(self, other):
        """!
        \brief Check for equality.

        Check for equality. Two epochs are equal iff the timestamps are equal.

        Non-keyword arguments
        \param[in] other: Another epoch or subclass of epoch.
        """
        if isinstance(other, type(self)):
            return self.t == other.t
        return False

    def __neq__(self, other):
        """!
        \brief Check for inequality.

        Check for inequality. Two epochs are not equal iff the timestamps are not equal.

        Non-keyword arguments
        \param[in] other: Another epoch or subclass of epoch.
        """
        if self == other:
            return False
        else:
            return True


class lc(object):

    """!
    \anchor lc_

    \brief Class to hold light curve.

    ABC to model a light curve. Light curve objects consist of a number of properties and numpy arrays to hold
    the list of t, x, y, yerr, and mask.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, numCadences=None, dt=None, meandt=None, mindt=None, dtSmooth=None, name=None,
                 band=None, xunit=None, yunit=None, tolIR=1.0e-3, fracIntrinsicVar=0.15,
                 fracNoiseToSignal=0.001, maxSigma=2.0, minTimescale=2.0, maxTimescale=0.5, pSim=0, qSim=0,
                 pComp=0, qComp=0, sampler=None, path=None, **kwargs):
        """!
        \brief Initialize a new light curve

        The constructor assumes that the light curve to be constructed is regular. There is no option to
        construct irregular light curves. Irregular light can be obtained by reading in a supplied irregular
        light curve. The constructor takes an optional keyword argument (supplied = <light curve file>) that
        is read in using the read method. This supplied light curve can be irregular. Typically, the supplied
        light curve is irregular either because the user created it that way, or because the survey that
        produced it sampled the sky at irregular intervals.

        Non-keyword arguments
        \param[in] numCadences:         The number of cadences in the light curve.
        \param[in] p          :         The order of the C-ARMA model used.
        \param[in] q          :         The order of the C-ARMA model used.

        Keyword arguments
        \param[in] dt:                  The spacing between cadences.
        \param[in] dt:                  The spacing between cadences after smoothing.
        \param[in] name:                The name of the light curve (usually the object's name).
        \param[in] band:                The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        \param[in] xunit                Unit in which time is measured (eg. s, sec, seconds etc...).
        \param[in] yunit                Unit in which the flux is measured (eg Wm^{-2} etc...).
        \param[in] tolIR:               The tolerance level at which a given step in the lightcurve should be
                                        considered irregular for the purpose of solving the C-ARMA model. The
                                        C-ARMA model needs to be re-solved if
                                        abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR
                                        where t_incr is the new increment in time and dt is the previous
                                        increment in time. If IR  == False, this parameter is not used.
        \param[in] fracIntrinsicVar:    The fractional variability of the source i.e.
                                        fracIntrinsicVar = sqrt(Sigma[0,0])/mean_flux.
        \param[in] fracNoiseToSignal:   The fractional noise level i.e. fracNoiseToSignal = sigma_noise/flux.
                                        We assume that this level is fixed. In the future, we may wish to make
                                        this value flux dependent to make the noise model more realistic.
        \param[in] maxSigma:            The maximum allowed value of sqrt(Sigma[0,0]) = maxSigma*stddev(y)
                                        when fitting a C-ARMA model. Note that if the observed light curve is
                                        shorter than the de-correlation timescale, stddev(y) may be much
                                        smaller than sqrt(Sigma[0,0]) and hence maxSigma should be made larger
                                        in such cases.
        \param[in] minTimescale:        The shortest allowed timescale = minTimescale*dt. Note that if the
                                        observed light curve is very sparsely sampled, dt may be much larger
                                        than the actaul minimum timescale present and hence minTimescale
                                        should be made smaller in such cases.
        \param[in] maxTimescale:        The longest allowed timescale = maxTimescale*T. Note that if the
                                        observed light curve is shorter than the longest timescale present, T
                                        may be much smaller than the longest timescale and hence maxTimescale
                                        should be made larger in such cases.
        \param[in] supplied:            Reference for supplied light curve. Since this class is an ABC,
                                        individual subclasses must implement a read method and the format
                                        expected for supplied (i.e. full path or name etc...) will be
                                        determined by the subclass.
        \param[in] path:                Reference for supplied light curve. Since this class is an ABC,
                                        individual subclasses must implement a read method and the format
                                        expected for supplied (i.e. full path or name etc...) will be
                                        determined by the subclass.
        """
        if name is not None and band is not None:
            self.read(name=name, band=band, path=path, **kwargs)
        else:
            self._numCadences = numCadences     # The number of cadences in the light curve. This is not the
            # same thing as the number of actual observations as we can have missing observations.
            self._simulatedCadenceNum = -1      # How many cadences have already been simulated.
            self._observedCadenceNum = -1   # How many cadences have already been observed.
            self._computedCadenceNum = -1   # How many cadences have been LnLikelihood'd already.
            self._pSim = pSim  # C-ARMA model used to simulate the LC.
            self._qSim = qSim  # C-ARMA model used to simulate the LC.
            self._pComp = pComp  # C-ARMA model used to simulate the LC.
            self._qComp = qComp  # C-ARMA model used to simulate the LC.
            self._isSmoothed = False  # Has the LC been smoothed?
            self._dtSmooth = 0.0
            self.t = np.require(np.zeros(self.numCadences), requirements=[
                                'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self.x = np.require(np.zeros(self.numCadences), requirements=[
                                'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self.y = np.require(np.zeros(self.numCadences), requirements=[
                                'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed fluxes.
            self.yerr = np.require(np.zeros(self.numCadences), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed flux errors.
            self.mask = np.require(np.zeros(self.numCadences), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
            self.XSim = np.require(np.zeros(self.pSim), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])  # State of light curve at last timestamp
            self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])    # Uncertainty in state of light curve at last
            # timestamp.
            self.XComp = np.require(np.zeros(self.pComp), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # State of light curve at last timestamp
            self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Uncertainty in state of light curve at last
            # timestamp.
            self._name = str(name)  # The name of the light curve (usually the object's name).
            self._band = str(band)  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
            if str(xunit)[0] != '$':
                self._xunit = r'$' + \
                    str(xunit) + '$'  # Unit in which time is measured (eg. s, sec, seconds etc...).
            else:
                self._xunit = str(xunit)
            if str(yunit)[0] != '$':
                self._yunit = r'$' + \
                    str(yunit) + '$'  # Unit in which the flux is measured (eg Wm^{-2} etc...).
            else:
                self._yunit = str(yunit)
            self._tolIR = tolIR  # Tolerance on the irregularity. If IR == False, this parameter is not used.
            # Otherwise, a timestep is irregular iff abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where
            # t_incr is the new increment in time and dt is the previous increment in time.
            self._fracIntrinsicVar = fracIntrinsicVar
            self._fracNoiseToSignal = fracNoiseToSignal
            self._maxSigma = maxSigma
            self._minTimescale = minTimescale
            self._maxTimescale = maxTimescale
            for i in xrange(self._numCadences):
                self.t[i] = i*dt
                self.mask[i] = 1.0
            self._isRegular = True
            self._dt = float(self.t[1] - self.t[0])
            self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
            self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
            self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
            self._T = float(self.t[-1] - self.t[0])
        self._lcCython = CARMATask_cython.lc(
            self.t, self.x, self.y, self.yerr, self.mask, self.XSim, self.PSim, self.XComp, self.PComp,
            dt=self._dt, meandt=self._meandt, mindt=self._mindt, maxdt=self._maxdt, dtSmooth=self._dtSmooth,
            tolIR=self._tolIR, fracIntrinsicVar=self._fracIntrinsicVar,
            fracNoiseToSignal=self._fracNoiseToSignal, maxSigma=self._maxSigma,
            minTimescale=self._minTimescale, maxTimescale=self._maxTimescale)
        if sampler is not None:
            self._sampler = sampler(self)
        else:
            self._sampler = None

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

    @property
    def numCadences(self):
        return self._numCadences

    @numCadences.setter
    def numCadences(self, value):
        self._lcCython.numCadences = value
        self._numCadences = value

    @property
    def simulatedCadenceNum(self):
        return self._simulatedCadenceNum

    @property
    def observedCadenceNum(self):
        return self._observedCadenceNum

    @property
    def computedCadenceNum(self):
        return self._computedCadenceNum

    @property
    def isRegular(self):
        return self._isRegular

    @property
    def isSmoothed(self):
        return self._isSmoothed

    @property
    def pSim(self):
        return self._pSim

    @pSim.setter
    def pSim(self, value):
        if value != self.pSim:
            newXSim = np.require(np.zeros(value), requirements=['F', 'A', 'W', 'O', 'E'])
            newPSim = np.require(np.zeros(value**2), requirements=['F', 'A', 'W', 'O', 'E'])
            large_number = math.sqrt(sys.float_info[0])
            if self.pSim > 0:
                if value > self.pSim:
                    iterMax = self.pSim
                elif value < self.pSim:
                    iterMax = value
                for i in xrange(iterMax):
                    newXSim[i] = self.XSim[i]
                    for j in xrange(iterMax):
                        newPSim[i + j*value] = self.PSim[i + j*self.pSim]
            elif self.pSim == 0:
                for i in xrange(value):
                    newXSim[i] = 0.0
                    for j in xrange(value):
                        newPSim[i + j*value] = 0.0
            self._pSim = value
            self.XSim = newXSim
            self.PSim = newPSim
        else:
            pass

    @property
    def pComp(self):
        return self._pComp

    @pComp.setter
    def pComp(self, value):
        if value != self.pComp:
            newXComp = np.require(np.zeros(value), requirements=['F', 'A', 'W', 'O', 'E'])
            newPComp = np.require(np.zeros(value**2), requirements=['F', 'A', 'W', 'O', 'E'])
            large_number = math.sqrt(sys.float_info[0])
            if self.pComp > 0:
                if value > self.pComp:
                    iterMax = self.pComp
                elif value < self.pComp:
                    iterMax = value
                for i in xrange(iterMax):
                    newXComp[i] = self.XComp[i]
                    for j in xrange(iterMax):
                        newPComp[i + j*value] = self.PComp[i + j*self.pComp]
            elif self.pComp == 0.0:
                for i in xrange(value):
                    newXComp[i] = 0.0
                    for j in xrange(value):
                        newPComp[i + j*value] = 0.0
            self._pComp = value
            self.XComp = newXComp
            self.PComp = newPComp
        else:
            pass

    @property
    def qSim(self):
        return self._qSim

    @qSim.setter
    def qSim(self, value):
        self._qSim = value

    @property
    def qComp(self):
        return self._qComp

    @qComp.setter
    def qComp(self, value):
        self._qComp = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._lcCython.dt = value
        self._dt = value

    @property
    def meandt(self):
        return self._meandt

    @meandt.setter
    def meandt(self, value):
        self._lcCython.meandt = value
        self._meandt = value

    @property
    def mindt(self):
        return self._mindt

    @mindt.setter
    def mindt(self, value):
        self._lcCython.mindt = value
        self._mindt = value

    @property
    def maxdt(self):
        return self._maxdt

    @maxdt.setter
    def maxdt(self, value):
        self._lcCython.maxdt = value
        self._maxdt = value

    @property
    def dtSmooth(self):
        return self._dtSmooth

    @dtSmooth.setter
    def dtSmooth(self, value):
        self._lcCython.dtSmooth = value
        self._dtSmooth = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def band(self):
        return self._band

    @band.setter
    def band(self, value):
        self._band = str(value)

    @property
    def xunit(self):
        return self._xunit

    @xunit.setter
    def xunit(self, value):
        if str(value)[0] != r'$':
            self._xunit = r'$' + str(value) + r'$'
        else:
            self._xunit = str(value)

    @property
    def yunit(self):
        return self._yunit

    @yunit.setter
    def yunit(self, value):
        if str(value)[0] != r'$':
            self._yunit = r'$' + str(value) + r'$'
        else:
            self._yunit = str(value)

    @property
    def tolIR(self):
        return self._tolIR

    @tolIR.setter
    def tolIR(self, value):
        self._tolIR = value
        self._lcCython.tolIR = value

    @property
    def fracIntrinsicVar(self):
        return self._fracIntrinsicVar

    @fracIntrinsicVar.setter
    def fracIntrinsicVar(self, value):
        self._fracIntrinsicVar = value
        self._lcCython.fracIntrinsicVar = value

    @property
    def fracNoiseToSignal(self):
        return self._fracNoiseToSignal

    @fracNoiseToSignal.setter
    def fracNoiseToSignal(self, value):
        self._fracNoiseToSignal = value
        self._lcCython.fracNoiseToSignal = value

    @property
    def maxSigma(self):
        return self._maxSigma

    @maxSigma.setter
    def maxSigma(self, value):
        self._maxSigma = value
        self._lcCython.maxSigma = value

    @property
    def minTimescale(self):
        return self._minTimescale

    @minTimescale.setter
    def minTimescale(self, value):
        self._minTimescale = value
        self._lcCython.minTimescale = value

    @property
    def maxTimescale(self):
        return self._maxTimescale

    @maxTimescale.setter
    def maxTimescale(self, value):
        self._maxTimescale = value
        self._lcCython.maxTimescale = value

    @property
    def sampler(self):
        return str(self._sampler)

    @sampler.setter
    def sampler(self, value):
        self._sampler = eval('kali.sampler.' + str(value).split('.')[-1])(self)

    @property
    def mean(self):
        """!
        \brief Mean of the observations y.
        """
        return self._mean

    @property
    def meanerr(self):
        """!
        \brief Mean of the observation errors yerr.
        """
        return self._meanerr

    @property
    def std(self):
        """!
        \brief Standard deviation of the observations y.
        """
        return self._std

    @property
    def stderr(self):
        """!
        \brief Standard deviation of the observation errors yerr.
        """
        return self._stderr

    def __len__(self):
        return self._numCadences

    def __repr__(self):
        """!
        \brief Return a representation of the lc such that eval(repr(someLC)) == someLC is True.
        """
        return u"libcarma.lc(%d, %f, %s, %s, %s, %s, %f, %f, %f, %f, %f, %f)"%(self._numCadences, self._dt,
                                                                               self._name, self._band,
                                                                               self._xunit, self._yunit,
                                                                               self._tolIR,
                                                                               self._fracIntrinsicVar,
                                                                               self._fracNoiseToSignal,
                                                                               self._maxSigma,
                                                                               self._minTimescale,
                                                                               self._maxTimescale)

    def __str__(self):
        """!
        \brief Return a human readable representation of the light curve.
        """
        line = ''
        line += '                                             Name: %s\n'%(self._name)
        line += '                                             Band: %s\n'%(self._band)
        line += '                                        Time Unit: %s\n'%(self._xunit)
        line += '                                        Flux Unit: %s\n'%(self._yunit)
        line += '                                      numCadences: %d\n'%(self._numCadences)
        line += '                                               dt: %e\n'%(self._dt)
        line += '                                                T: %e\n'%(self._T)
        line += '                                     mean of flux: %e\n'%(self._mean)
        line += '                           std. deviation of flux: %e\n'%(self._std)
        line += '                               mean of flux error: %e\n'%(self._meanerr)
        line += '                     std. deviation of flux error: %e\n'%(self._stderr)
        line += '               tolIR (Tolerance for irregularity): %e\n'%(self._tolIR)
        line += 'fracIntrinsicVar (Intrinsic variability fraction): %e\n'%(self._fracIntrinsicVar)
        line += '     fracNoiseToSignal (Noise to signal fraction): %e\n'%(self._fracNoiseToSignal)
        line += '      maxSigma (Maximum allowed sigma multiplier): %e\n'%(self._maxSigma)
        line += '  minTimescale (Minimum allowed timescale factor): %e\n'%(self._minTimescale)
        line += '  maxTimescale (Maximum allowed timescale factor): %e\n'%(self._maxTimescale)
        line += '\n'
        epochline = ''
        for i in xrange(self._numCadences - 1):
            epochline += str(self[i])
            epochline += '\n'
        epochline += str(self[self._numCadences - 1])
        line += reprlib.repr(epochline)
        return line

    def __eq__(self, other):
        """!
        \brief Check for equality.

        Check for equality. Two light curves are equal only iff all thier attributes are the same.

        Non-keyword arguments
        \param[in] other: Another lc or subclass of lc.
        """
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False

    def __neq__(self, other):
        """!
        \brief Check for inequality.

        Check for inequality. Two light curves are in-equal only iff all thier attributes are not the same.

        Non-keyword arguments
        \param[in] other: Another lc or subclass of lc.
        """
        if self == other:
            return False
        else:
            return True

    def __getitem__(self, key):
        """!
        \brief Return an epoch.

        Return an epoch corresponding to the index.
        """
        return epoch(self.t[key], self.x[key], self.y[key], self.yerr[key], self.mask[key])

    def __setitem__(self, key, val):
        """!
        \brief Set an epoch.

        Set the epoch corresponding to the provided index using the values from the epoch val.
        """
        if isinstance(val, epoch):
            self.t[key] = val.t
            self.x[key] = val.x
            self.y[key] = val.y
            self.yerr[key] = val.yerr
            self.mask[key] = val.mask
        self._mean = np.mean(self.y)
        self._std = np.std(self.y)
        self._meanerr = np.mean(self.yerr)
        self._stderr = np.std(self.yerr)

    def __iter__(self):
        """!
        \brief Return a light curve iterator.

        Return a light curve iterator object making light curves iterable.
        """
        return lcIterator(self.t, self.x, self.y, self.yerr, self.mask)

    @abc.abstractmethod
    def copy(self):
        """!
        \brief Return a copy

        Return a (deep) copy of the object.
        """
        raise NotImplementedError(r'Override copy by subclassing lc!')

    def __invert__(self):
        """!
        \brief Return the lc without the mean.

        Return a new lc with the mean removed.
        """
        lccopy = self.copy()
        lccopy.x -= lccopy._mean
        lccopy.y -= lccopy._mean

    def __pos__(self):
        """!
        \brief Return + light curve.

        Return + light curve i.e. do nothing but just return a deepcopy of the object.
        """
        return self.copy()

    def __neg__(self):
        """!
        \brief Invert the light curve.

        Return a light curve with the delta fluxes flipped in sign.
        """
        lccopy = self.copy()
        lccopy.x = -1.0*(self.x - np.mean(self.x)) + np.mean(self.x)
        lccopy.y = -1.0*(self.y - self._mean) + self._mean
        lccopy._mean = np.mean(lccopy.y)
        lccopy._std = np.std(lccopy.y)
        return lccopy

    def __abs__(self):
        """!
        \brief Abs the light curve.

        Return a light curve with the abs of the delta fluxes.
        """
        lccopy = self.copy()
        lccopy.x = np.abs(self.x - np.mean(self.x)) + np.mean(self.x)
        lccopy.y = np.abs(self.y - self._mean) + self._mean
        lccopy._mean = np.mean(lccopy.y)
        lccopy._std = np.std(lccopy.y)
        return lccopy

    def __add__(self, other):
        """!
        \brief Add.

        Add another light curve or scalar to the light curve.
        """
        lccopy = self.copy()
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            lccopy.x += other
            lccopy.y += other
            lccopy._mean = np.mean(lccopy.y)
            lccopy._std = np.std(lccopy.y)
        elif isinstance(other, lc):
            if other.numCadences == self.numCadences:
                lccopy.x += other.x
                lccopy.y += other.y
                lccopy.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
                lccopy._mean = np.mean(lccopy.y)
                lccopy._std = np.std(lccopy.y)
                lccopy._mean = np.mean(lccopy.yerr)
                lccopy._stderr = np.std(lccopy.yerr)
            else:
                raise ValueError('Light curves have un-equal length')
        else:
            raise NotImplemented
        return lccopy

    def __radd__(self, other):
        """!
        \brief Add.

        Add a light curve to a scalar.
        """
        return self + other

    def __sub__(self, other):
        """!
        \brief Subtract.

        Subtract another light curve or scalar from the light curve.
        """
        return self + (- other)

    def __rsub__(self, other):
        """!
        \brief Subtract.

        Subtract a light curve from a scalar .
        """
        return self + (- other)

    def __iadd__(self, other):
        """!
        \brief Inplace add.

        Inplace add another light curve or scalar to the light curve.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            self.x += other
            self.y += other
            self._mean += other
        elif isinstance(other, lc):
            if other.numCadences == self.numCadences:
                self.x += other.x
                self.y += other.y
                self.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
                self._mean = np.mean(self.y)
                self._std = np.std(self.y)
                self._mean = np.mean(self.yerr)
                self._stderr = np.std(self.yerr)
            else:
                raise ValueError('Light curves have un-equal length')
        return self

    def __isub__(self, other):
        """!
        \brief Inplace subtract.

        Inplace subtract another light curve or scalar from the light curve.
        """
        return self.iadd(- other)

    def __mul__(self, other):
        """!
        \brief Multiply.

        Multiply the light curve by a scalar.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            if isinstance(other, complex):
                other = complex(other)
            else:
                other = float(other)
            lccopy = self.copy()
            lccopy.x *= other
            lccopy.y *= other
            lccopy.yerr *= other
            lccopy._mean += other
            lccopy._std *= other
            lccopy._meanerr *= other
            lccopy._stderr *= other
            return lccopy
        else:
            raise NotImplemented

    def __rmul__(self, other):
        """!
        \brief Multiply.

        Multiply a scalar by the light curve.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            if isinstance(other, complex):
                other = complex(other)
            else:
                other = float(other)
            return self*other
        else:
            raise NotImplemented

    def __div__(self, other):
        """!
        \brief Divide.

        Divide the light curve by a scalar.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            if isinstance(other, complex):
                other = complex(other)
            else:
                other = float(other)
            return self*(1.0/other)
        else:
            raise NotImplemented

    def __rdiv__(self, other):
        """!
        \brief Divide  - not defined & not implemented.

        Divide a scalar by the light curve - not defined & not implemented.
        """
        raise NotImplemented

    def __imul__(self, other):
        """!
        \brief Inplace multiply.

        Inplace multiply a light curve by a scalar.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            if isinstance(other, complex):
                other = complex(other)
            else:
                other = float(other)
            self.x *= other
            self.y *= other
            self.yerr *= other
            self._mean += other
            self._std *= other
            self._meanerr *= other
            self._stderr *= other
            return self
        else:
            raise NotImplemented

    def __idiv__(self, other):
        """!
        \brief Inplace divide.

        Inplace divide a light curve by a scalar.
        """
        if (isinstance(other, int) or isinstance(other, int) or isinstance(other, float) or
                isinstance(other, complex)):
            if isinstance(other, complex):
                other = complex(other)
            else:
                other = float(other)
            self.x *= (1.0/other)
            self.y *= (1.0/other)
            self.yerr *= (1.0/other)
            self._mean += (1.0/other)
            self._std *= (1.0/other)
            self._meanerr *= (1.0/other)
            self._stderr *= (1.0/other)
            return self
        else:
            raise NotImplemented

    @abc.abstractmethod
    def read(self, name, band, path=os.environ['PWD'], **kwargs):
        """!
        \brief Read the light curve from disk.

        Not implemented!
        """
        raise NotImplementedError(r'Override read by subclassing lc!')

    @abc.abstractmethod
    def write(self, name, band, path=os.environ['PWD'], **kwargs):
        """!
        \brief Write the light curve to disk.

        Not implemented
        """
        raise NotImplementedError(r'Override write by subclassing lc!')

    def regularize(self, newdt=None):
        """!
        \brief Re-sample the light curve on a grid of spacing newdt

        Creates a new LC on gridding newdt and copies in the required points.
        """
        if not self.isRegular:
            if not newdt:
                newdt = self.mindt/10.0
            if newdt > self.mindt:
                raise ValueError('newdt cannot be greater than mindt')
            newLC = self.copy()
            newLC.dt = newdt
            newLC.meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
            newLC.mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
            newLC.maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
            newLC.meandt = float(self.t[-1] - self.t[0])
            newLC.numCadences = int(math.ceil(self.T/newLC.dt))
            del newLC.t
            del newLC.x
            del newLC.y
            del newLC.yerr
            del newLC.mask
            newLC.t = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            newLC.x = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            newLC.y = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed fluxes.
            newLC.yerr = np.require(np.zeros(newLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed flux errors.
            newLC.mask = np.require(np.zeros(newLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
            for i in xrange(newLC.numCadences):
                newLC.t[i] = i*newLC.dt + self.t[0]
            for i in xrange(self.numCadences):
                tOff = (self.t[i] - self.t[0])
                index = int(math.floor(tOff/newLC.dt))
                newLC.x[index] = self.x[i]
                newLC.y[index] = self.y[i]
                newLC.yerr[index] = self.yerr[i]
                newLC.mask[index] = 1.0
            count = int(np.sum(newLC.mask[i]))
            y_meanSum = 0.0
            yerr_meanSum = 0.0
            for i in xrange(newLC.numCadences):
                y_meanSum += newLC.mask[i]*newLC.y[i]
                yerr_meanSum += newLC.mask[i]*newLC.yerr[i]
            if count > 0:
                newLC._mean = y_meanSum/count
                newLC._meanerr = yerr_meanSum/count
            y_stdSum = 0.0
            yerr_stdSum = 0.0
            for i in xrange(newLC.numCadences):
                y_stdSum += math.pow(newLC.mask[i]*newLC.y[i] - newLC._mean, 2.0)
                yerr_stdSum += math.pow(newLC.mask[i]*newLC.yerr[i] - newLC._meanerr, 2.0)
            if count > 0:
                newLC._std = math.sqrt(y_stdSum/count)
                newLC._stderr = math.sqrt(yerr_stdSum/count)

            return newLC
        else:
            return self

    def sample(self, **kwargs):
        return self._sampler.sample(**kwargs)

    def acvf(self, newdt=None):
        if hasattr(self, '_acvflags') and hasattr(self, '_acvf') and hasattr(self, '_acvferr'):
            return self._acvflags, self._acvf, self._acvferr
        else:
            if not self.isRegular:
                useLC = self.regularize(newdt)
            else:
                useLC = self
            self._acvflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._acvf = np.require(np.zeros(useLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._acvferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                       'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            useLC._lcCython.compute_ACVF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
                                         useLC.y, useLC.yerr, useLC.mask, self._acvflags, self._acvf,
                                         self._acvferr)
            return self._acvflags, self._acvf, self._acvferr

    def acf(self, newdt=None):
        if hasattr(self, '_acflags') and hasattr(self, '_acf') and hasattr(self, '_acferr'):
            return self._acflags, self._acf, self._acferr
        else:
            if not self.isRegular:
                useLC = self.regularize(newdt)
            else:
                useLC = self
            self._acflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                       'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._acf = np.require(np.zeros(useLC.numCadences), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._acferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                      'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            useLC._lcCython.compute_ACF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
                                        useLC.y, useLC.yerr, useLC.mask, self._acflags, self._acf,
                                        self._acferr)
            return self._acflags, self._acf, self._acferr

    def dacf(self, nbins=None):
        if hasattr(self, '_dacflags') and hasattr(self, '_dacf') and hasattr(self, '_dacferr'):
            return self._dacflags, self._dacf, self._dacferr
        else:
            if nbins is None:
                nbins = int(self.numCadences/10)
            self._dacflags = np.require(np.linspace(start=0.0, stop=self.T, num=nbins), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._dacf = np.require(np.zeros(self._dacflags.shape[0]), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._dacferr = np.require(np.zeros(self._dacflags.shape[0]), requirements=[
                                       'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._lcCython.compute_DACF(self.numCadences, self.dt, self.t, self.x, self.y,
                                        self.yerr, self.mask, nbins, self._dacflags, self._dacf,
                                        self._dacferr)
            return self._dacflags, self._dacf, self._dacferr

    def sf(self, newdt=None):
        if hasattr(self, '_sflags') and hasattr(self, '_sf') and hasattr(self, '_sferr'):
            return self._sflags, self._sf, self._sferr
        else:
            if not self.isRegular:
                useLC = self.regularize(newdt)
            else:
                useLC = self
            self._sflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                      'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._sf = np.require(np.zeros(useLC.numCadences), requirements=[
                                  'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._sferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                     'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            useLC._lcCython.compute_SF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
                                       useLC.y, useLC.yerr, useLC.mask, self._sflags, self._sf, self._sferr)
            return self._sflags, self._sf, self._sferr

    def plot(self, fig=-1, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if (np.sum(self.x) != 0.0) and (np.sum(self.y) == 0.0):
            plt.plot(self.t, self.x, color='#984ea3', zorder=0)
            plt.plot(self.t, self.x, color='#984ea3', marker='o', markeredgecolor='none', zorder=0)
        if (np.sum(self.x) == 0.0) and (np.sum(self.y) != 0.0):
            plt.errorbar(
                self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]],
                self.yerr[np.where(self.mask == 1.0)[0]], label=r'%s (%s-band)'%(self.name, self.band),
                fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
            plt.xlim(self.t[0], self.t[-1])
        if (np.sum(self.x) != 0.0) and (np.sum(self.y) != 0.0):
            plt.plot(self.t, self.x - np.mean(self.x) + np.mean(
                self.y[np.where(self.mask == 1.0)[0]]), color='#984ea3', zorder=0)
            plt.plot(self.t, self.x - np.mean(self.x) + np.mean(
                self.y[np.where(self.mask == 1.0)[0]]), color='#984ea3', marker='o', markeredgecolor='none',
                zorder=0)
            plt.errorbar(
                self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]],
                self.yerr[np.where(self.mask == 1.0)[0]], label=r'%s (%s-band)'%(self.name, self.band),
                fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
        if self.isSmoothed:
            plt.plot(self.tSmooth, self.xSmooth, color='#4daf4a',
                     marker='o', markeredgecolor='none', zorder=-5)
            plt.plot(self.tSmooth, self.xSmooth, color='#4daf4a', zorder=-5)
            plt.fill_between(self.tSmooth, self.xSmooth - self.xerrSmooth, self.xSmooth +
                             self.xerrSmooth, facecolor='#ccebc5', alpha=0.5, zorder=-5)
        plt.xlim(self.t[0], self.t[-1])
        plt.xlabel(self.xunit)
        plt.ylabel(self.yunit)
        plt.title(r'Light curve')
        plt.legend()
        if doShow:
            plt.show(False)
        return newFig

    def plotacvf(self, fig=-2, newdt=None, doShow=False):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        plt.plot(0.0, 0.0)
        if np.sum(self.y) != 0.0:
            lagsE, acvfE, acvferrE = self.acvf(newdt)
            if np.sum(acvfE) != 0.0:
                plt.errorbar(lagsE[0], acvfE[0], acvferrE[0], label=r'obs. Autocovariance Function',
                             fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
                for i in xrange(1, lagsE.shape[0]):
                    if acvfE[i] != 0.0:
                        plt.errorbar(lagsE[i], acvfE[i], acvferrE[i], fmt='o', capsize=0, color='#ff7f00',
                                     markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$ACVF$')
        plt.title(r'AutoCovariance Function')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig

    def plotacf(self, fig=-3, newdt=None, doShow=False):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        plt.plot(0.0, 0.0)
        if np.sum(self.y) != 0.0:
            lagsE, acfE, acferrE = self.acf(newdt)
            if np.sum(acfE) != 0.0:
                plt.errorbar(lagsE[0], acfE[0], acferrE[0], label=r'obs. Autocorrelation Function',
                             fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
                for i in xrange(1, lagsE.shape[0]):
                    if acfE[i] != 0.0:
                        plt.errorbar(lagsE[i], acfE[i], acferrE[i], fmt='o', capsize=0, color='#ff7f00',
                                     markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$ACF$')
        plt.title(r'AutoCorrelation Function')
        plt.legend(loc=3)
        plt.ylim(-1.0, 1.0)
        if doShow:
            plt.show(False)
        return newFig

    def plotdacf(self, fig=-4, numBins=None, doShow=False):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if np.sum(self.y) != 0.0:
            lagsE, dacfE, dacferrE = self.dacf(nbins=numBins)
            if np.sum(dacfE) != 0.0:
                plt.errorbar(
                    lagsE[0], dacfE[0], dacferrE[0], label=r'obs. Discrete Autocorrelation Function', fmt='o',
                    capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
                for i in xrange(0, lagsE.shape[0]):
                    if dacfE[i] != 0.0:
                        plt.errorbar(lagsE[i], dacfE[i], dacferrE[i], fmt='o', capsize=0, color='#ff7f00',
                                     markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$DACF$')
        plt.title(r'Discrete Autocorrelation Function')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig

    def plotsf(self, fig=-5, newdt=None, doShow=False):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        ln10 = math.log(10.0)
        if np.sum(self.y) != 0.0:
            lagsE, sfE, sferrE = self.sf(newdt)
            if np.sum(sfE) != 0.0:
                i = 1
                for i in xrange(1, lagsE.shape[0]):
                    if sfE[i] != 0.0:
                        break
                plt.errorbar(math.log10(lagsE[i]), math.log10(sfE[i]), math.fabs(
                    sferrE[i]/(ln10*sfE[i])), label=r'obs. Structure Function', fmt='o', capsize=0,
                    color='#ff7f00', markeredgecolor='none', zorder=10)
                startI = i
                for i in xrange(i+1, lagsE.shape[0]):
                    if sfE[i] != 0.0:
                        plt.errorbar(
                            math.log10(lagsE[i]), math.log10(sfE[i]), math.fabs(sferrE[i]/(ln10*sfE[i])),
                            fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=10)
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\log SF$')
        plt.title(r'Structure Function')
        plt.legend(loc=2)
        if doShow:
            plt.show(False)
        return newFig

    def spline(self, ptFactor=10, degree=3):
        unObsUncertVal = math.sqrt(sys.float_info[0])
        newLC = self.copy()
        del newLC.t
        del newLC.x
        del newLC.y
        del newLC.yerr
        del newLC.mask
        newLC.numCadences = self.numCadences*ptFactor
        newLC.T = self.T
        newLC.dt = newLC.T/newLC.numCadences
        newLC.t = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.x = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.y = np.require(np.zeros(newLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.yerr = np.require(
            np.array(newLC.numCadences*[unObsUncertVal]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.mask = np.require(np.array(newLC.numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        spl = UnivariateSpline(
            self.t[np.where(self.mask == 1.0)], self.y[np.where(self.mask == 1.0)],
            1.0/self.yerr[np.where(self.mask == 1.0)], k=degree, check_finite=True)
        for i in xrange(newLC.numCadences):
            newLC.t[i] = self.t[0] + i*newLC.dt
            newLC.x[i] = spl(newLC.t[i])
        return newLC

    def fold(self, foldPeriod, tStart=None):
        if tStart is None:
            tStart = self.t[0]
        numFolds = int(math.floor(self.T/foldPeriod))
        newLC = self.copy()
        del newLC.t
        del newLC.x
        del newLC.y
        del newLC.yerr
        del newLC.mask
        tList = list()
        xList = list()
        yList = list()
        yerrList = list()
        maskList = list()
        for i in xrange(self.numCadences):
            tList.append((self.t[i] - tStart)%foldPeriod)
            xList.append(self.x[i])
            yList.append(self.y[i])
            yerrList.append(self.yerr[i])
            maskList.append(self.mask[i])
        sortedLists = zip(*sorted(zip(tList, xList, yList, yerrList, maskList), key=operator.itemgetter(0)))
        newLC.t = np.require(np.array(sortedLists[0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.x = np.require(np.array(sortedLists[1]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.y = np.require(np.array(sortedLists[2]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.yerr = np.require(np.array(sortedLists[3]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.mask = np.require(np.array(sortedLists[4]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.dt = float(newLC.t[1] - newLC.t[0])
        newLC.T = float(newLC.t[-1] - newLC.t[0])
        return newLC

    def bin(self, binRatio=10):
        newLC = self.copy()
        newLC.numCadences = self.numCadences/binRatio
        del newLC.t
        del newLC.x
        del newLC.y
        del newLC.yerr
        del newLC.mask
        tList = list()
        xList = list()
        yList = list()
        yerrList = list()
        maskList = list()
        for i in xrange(newLC.numCadences):
            tList.append(np.mean(self.t[i*binRatio:(i + 1)*binRatio]))
            xList.append(np.mean(self.x[i*binRatio:(i + 1)*binRatio]))
            yList.append(np.mean(self.y[i*binRatio:(i + 1)*binRatio]))
            yerrList.append(math.sqrt(np.mean(np.power(self.yerr[i*binRatio:(i + 1)*binRatio], 2.0))))
            maskList.append(1.0)
        sortedLists = zip(*sorted(zip(tList, xList, yList, yerrList, maskList), key=operator.itemgetter(0)))
        newLC.t = np.require(np.array(sortedLists[0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.x = np.require(np.array(sortedLists[1]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.y = np.require(np.array(sortedLists[2]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.yerr = np.require(np.array(sortedLists[3]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.mask = np.require(np.array(sortedLists[4]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.dt = float(newLC.t[1] - newLC.t[0])
        newLC.T = float(newLC.t[-1] - newLC.t[0])
        return newLC


class lcIterator(object):

    def __init__(self, t, x, y, yerr, mask):
        self.t = t
        self.x = x
        self.y = y
        self.yerr = yerr
        self.mask = mask
        self.index = 0

    def next(self):
        """!
        \brief Return the next epoch.

        To make light curves iterable, return the next epoch.
        """
        try:
            nextEpoch = epoch(self.t[self.index], self.x[self.index], self.y[
                              self.index], self.yerr[self.index], self.mask[self.index])
        except IndexError:
            raise StopIteration
        self.index += 1
        return nextEpoch

    def __iter__(self):
        return self


class basicLC(lc):

    def copy(self):
        lccopy = basicLC(
            self.numCadences, dt=self.dt, meandt=self.meandt, mindt=self.mindt, maxdt=self.maxdt,
            dtSmooth=self.dtSmooth, name=None, band=self.band, xunit=self.xunit, yunit=self.yunit,
            tolIR=self.tolIR, fracIntrinsicVar=self.fracIntrinsicVar,
            fracNoiseToSignal=self.fracNoiseToSignal, maxSigma=self.maxSigma, minTimescale=self.minTimescale,
            maxTimescale=self.maxTimescale)
        lccopy.t = np.copy(self.t)
        lccopy.x = np.copy(self.x)
        lccopy.y = np.copy(self.y)
        lccopy.yerr = np.copy(self.yerr)
        lccopy.mask = np.copy(self.mask)
        lccopy.pSim = np.copy(self.pSim)
        lccopy.qSim = np.copy(self.qSim)
        lccopy.pComp = np.copy(self.pComp)
        lccopy.qComp = np.copy(self.qComp)

        count = int(np.sum(lccopy.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(lccopy.numCadences):
            y_meanSum += lccopy.mask[i]*lccopy.y[i]
            yerr_meanSum += lccopy.mask[i]*lccopy.yerr[i]
        if count > 0.0:
            lccopy._mean = y_meanSum/count
            lccopy._meanerr = yerr_meanSum/count
        else:
            lccopy._mean = 0.0
            lccopy._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(lccopy.numCadences):
            y_stdSum += math.pow(lccopy.mask[i]*lccopy.y[i] - lccopy._mean, 2.0)
            yerr_stdSum += math.pow(lccopy.mask[i]*lccopy.yerr[i] - lccopy._meanerr, 2.0)
        if count > 0.0:
            lccopy._std = math.sqrt(y_stdSum/count)
            lccopy._stderr = math.sqrt(yerr_stdSum/count)
        else:
            lccopy._std = 0.0
            lccopy._stderr = 0.0

        return lccopy

    def read(self, name=None, band=None, pwd=None, **kwargs):
        pass

    def write(self, name=None, band=None, pwd=None, **kwargs):
        pass


class externalLC(basicLC):

    def _checkIsRegular(self):
        self._isRegular = True
        for i in xrange(1, self.numCadences):
            t_incr = self.t[i] - self.t[i-1]
            fracChange = abs((t_incr - self.dt)/((t_incr + self.dt)/2.0))
            if fracChange > self._tolIR:
                self._isRegular = False
                break

    def read(self, name, band, path=None, **kwargs):
        self._name = name
        self._band = band
        self._path = path
        t = kwargs.get('t')
        if t is not None:
            self.t = np.require(t, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument t!')
        self._numCadences = self.t.shape[0]
        self.x = np.require(
            kwargs.get('x', np.zeros(self.numCadences)), requirements=['F', 'A', 'W', 'O', 'E'])
        y = kwargs.get('y')
        if y is not None:
            self.y = np.require(y, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument y!')
        yerr = kwargs.get('yerr')
        if yerr is not None:
            self.yerr = np.require(yerr, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise Keyerror('Must supply key-word argument yerr!')
        mask = kwargs.get('mask')
        if mask is not None:
            self.mask = np.require(mask, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise Keyerror('Must supply key-word argument mask!')

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
        self.XSim = np.require(np.zeros(self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])  # State of light
        # curve at last timestamp
        # Uncertainty in state of light curve at last timestamp
        self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])
        self.XComp = np.require(np.zeros(self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])  # State of
        # light curve at last timestamp
        # Uncertainty in state of light curve at last timestamp.
        self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])
        self._xunit = r'$t$ (d)'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'who the f*** knows?'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

        self._startT = float(self.t[0])
        self.t -= self._startT
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])
        self._checkIsRegular()

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