#!/usr/bin/env python
"""	Module that defines light curve objects.
"""
import math
import cmath
import numpy as np
import operator
import sys
import abc
import psutil
import types
import os
import reprlib
import copy
import warnings
import pdb as pdb

import scipy.stats as spstats
from scipy.interpolate import UnivariateSpline
import gatspy.periodic

import gatspy.periodic
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
    import rand
    import LCTools_cython
    import kali.sampler
    import kali.kernel
    from kali.util.mpl_settings import set_plot_params
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex=True)
COLORX = r'#984ea3'
COLORY = r'#ff7f00'
COLORS = [r'#4daf4a', r'#ccebc5']
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

    def __init__(self, name=None, band=None, path=None, **kwargs):
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
        self.read(name=name, band=band, path=path, **kwargs)
        self._simulatedCadenceNum = -1      # How many cadences have already been simulated.
        self._observedCadenceNum = -1   # How many cadences have already been observed.
        self._computedCadenceNum = -1   # How many cadences have been LnLikelihood'd already.
        self._pSim = kwargs.get('pSim', 0)  # C-ARMA model used to simulate the LC.
        self._qSim = kwargs.get('qSim', 0)  # C-ARMA model used to simulate the LC.
        self._pComp = kwargs.get('pComp', 0)  # C-ARMA model used to simulate the LC.
        self._qComp = kwargs.get('qComp', 0)  # C-ARMA model used to simulate the LC.
        self.XSim = np.require(np.zeros(self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])
        self.PSim = np.require(np.zeros(self.pSim*self.pSim), requirements=['F', 'A', 'W', 'O', 'E'])
        self.XComp = np.require(np.zeros(self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])
        self.PComp = np.require(np.zeros(self.pComp*self.pComp), requirements=['F', 'A', 'W', 'O', 'E'])
        self._tolIR = kwargs.get('tolIR', 1.0e-3)  # Tolerance on the irregularity. If IR == False, this
        # parameter is not used. Otherwise, a timestep is irregular iff
        # abs((t_incr - dt)/((t_incr + dt)/2.0)) > tolIR where t_incr is the new increment in time and dt is
        # the previous increment in time.
        self._fracIntrinsicVar = kwargs.get('fracIntrinsicVar', 1.5e-2)
        self._fracNoiseToSignal = kwargs.get('fracNoiseToSignal', 1.0e-3)
        self._maxSigma = kwargs.get('maxSigma', 2.0)
        self._minTimescale = kwargs.get('minTimescale', 2.0)
        self._maxTimescale = kwargs.get('maxTimescale', 0.5)
        self._sampler = kwargs.get('sampler', 'sincSampler')
        self._times()
        self._checkIsRegular()
        self._statistics()
        self._isSmoothed = False  # Has the LC been smoothed?
        self._dtSmooth = kwargs.get('dtSmooth', self.mindt/10.0)
        if not hasattr(self, 'coordinates'):
            self.coordinates = kwargs.get('coordinates')
        '''try:
            self._catalogue()
        except Exception as Err:
            self.simbad = Err
            self.ned = Err
            self.vizier = Err
            self.sdss = Err'''

    '''def _catalogue(self):
        if self.coordinates is not None:
            try:
                self.simbad = Simbad.query_region(self.coordinates, radius=5*units.arcsec)
            except Exception as Err:
                self.simbad = Err
            try:
                self.ned = Ned.query_region(self.coordinates, radius=5*units.arcsec)
            except Exception as Err:
                self.ned = Err
            try:
                self.vizier = Vizier.query_region(self.coordinates, radius=5*units.arcsec)
            except Exception as Err:
                self.vizier = Err
            try:
                self.sdss = SDSS.query_region(self.coordinates, radius=5*units.arcsec)
            except Exception as Err:
                self.sdss = Err
        else:
            try:
                self.ned = Ned.query_object(self.name)
            except Exception as Err:
                self.ned = Err'''

    @property
    def numCadences(self):
        return self._numCadences

    @numCadences.setter
    def numCadences(self, value):
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
        self._dt = value

    @property
    def meandt(self):
        return self._meandt

    @meandt.setter
    def meandt(self, value):
        self._meandt = value

    @property
    def mindt(self):
        return self._mindt

    @mindt.setter
    def mindt(self, value):
        self._mindt = value

    @property
    def maxdt(self):
        return self._maxdt

    @maxdt.setter
    def maxdt(self, value):
        self._maxdt = value

    @property
    def dtSmooth(self):
        return self._dtSmooth

    @dtSmooth.setter
    def dtSmooth(self, value):
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
        self._xunit = str(value)

    @property
    def yunit(self):
        return self._yunit

    @yunit.setter
    def yunit(self, value):
        self._yunit = str(value)

    @property
    def tolIR(self):
        return self._tolIR

    @tolIR.setter
    def tolIR(self, value):
        self._tolIR = value

    @property
    def fracIntrinsicVar(self):
        return self._fracIntrinsicVar

    @fracIntrinsicVar.setter
    def fracIntrinsicVar(self, value):
        self._fracIntrinsicVar = value

    @property
    def fracNoiseToSignal(self):
        return self._fracNoiseToSignal

    @fracNoiseToSignal.setter
    def fracNoiseToSignal(self, value):
        self._fracNoiseToSignal = value

    @property
    def maxSigma(self):
        return self._maxSigma

    @maxSigma.setter
    def maxSigma(self, value):
        self._maxSigma = value

    @property
    def minTimescale(self):
        return self._minTimescale

    @minTimescale.setter
    def minTimescale(self, value):
        self._minTimescale = value

    @property
    def maxTimescale(self):
        return self._maxTimescale

    @maxTimescale.setter
    def maxTimescale(self, value):
        self._maxTimescale = value

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

    def _times(self):
        self._dt = float(self.t[1] - self.t[0])
        self._mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
        self._maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
        self._meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
        self._T = float(self.t[-1] - self.t[0])

    def _statistics(self):
        """!
        \brief Set the four basic statistics - mean, std, meanerr, & stderr
        """
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
        self._statistics()

    def __iter__(self):
        """!
        \brief Return a light curve iterator.

        Return a light curve iterator object making light curves iterable.
        """
        return lcIterator(self.t, self.x, self.y, self.yerr, self.mask)

    def __copy__(self):
        """!
        \brief Return a shallow copy of self
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """!
        \brief Return a deep copy of self
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        """!
        \brief Type-saver function to prevent the user from having to call copy.deepcopy
        """
        return copy.deepcopy(self)

    def _checkIsRegular(self):
        """!
        \brief Set the isRegular flag if the lc is regular to within the specified tolerence.
        """
        self._isRegular = True
        for i in xrange(1, self.numCadences):
            t_incr = self.t[i] - self.t[i-1]
            fracChange = abs((t_incr - self.dt)/((t_incr + self.dt)/2.0))
            if fracChange > self._tolIR:
                self._isRegular = False
                break

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
        lccopy.y = -1.0*(self.y - self.mean) + self.mean
        lccopy._statistics()
        return lccopy

    def __abs__(self):
        """!
        \brief Abs the light curve.

        Return a light curve with the abs of the delta fluxes.
        """
        lccopy = self.copy()
        lccopy.x = np.abs(self.x - np.mean(self.x)) + np.mean(self.x)
        lccopy.y = np.abs(self.y - self.mean) + self.mean
        lccopy._statistics()
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
            lccopy._statistics()
        elif isinstance(other, lc):
            if other.numCadences == self.numCadences:
                lccopy.x += other.x
                lccopy.y += other.y
                lccopy.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
                lccopy._statistics()
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
            self._statistics()
        elif isinstance(other, lc):
            if other.numCadences == self.numCadences:
                self.x += other.x
                self.y += other.y
                self.yerr = np.sqrt(np.power(self.yerr, 2.0) + np.power(other.yerr, 2.0))
                self._statistics()
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
            lccopy._statistics()
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
            self._statistics()
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
            self._statistics()
            return self
        else:
            raise NotImplemented

    def regularize(self, newdt=None):
        """!
        \brief Re-sample the light curve on a grid of spacing newdt

        Creates a new LC on gridding newdt and copies in the required points.
        """
        if not self.isRegular:
            if not newdt:
                if hasattr(self, 'terr'):
                    newdt = np.mean(self.terr)
                else:
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
            newLC._statistics()
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
                useLC = self.regularize(newdt=newdt)
            else:
                useLC = self
            self._acvflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._acvf = np.require(np.zeros(useLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._acvferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                       'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            LCTools_cython.compute_ACVF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
                                        useLC.y, useLC.yerr, useLC.mask, self._acvflags, self._acvf,
                                        self._acvferr)
            return self._acvflags, self._acvf, self._acvferr

    def acf(self, newdt=None):
        if hasattr(self, '_acflags') and hasattr(self, '_acf') and hasattr(self, '_acferr'):
            return self._acflags, self._acf, self._acferr
        else:
            if not self.isRegular:
                useLC = self.regularize(newdt=newdt)
            else:
                useLC = self
            self._acflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                       'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._acf = np.require(np.zeros(useLC.numCadences), requirements=[
                                   'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._acferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                      'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            LCTools_cython.compute_ACF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
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
            LCTools_cython.compute_DACF(self.numCadences, self.dt, self.t, self.x, self.y,
                                        self.yerr, self.mask, nbins, self._dacflags, self._dacf,
                                        self._dacferr)
            return self._dacflags, self._dacf, self._dacferr

    def sf(self, newdt=None):
        if hasattr(self, '_sflags') and hasattr(self, '_sf') and hasattr(self, '_sferr'):
            return self._sflags, self._sf, self._sferr
        else:
            if not self.isRegular:
                useLC = self.regularize(newdt=newdt)
            else:
                useLC = self
            self._sflags = np.require(np.zeros(useLC.numCadences), requirements=[
                                      'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            self._sf = np.require(np.zeros(useLC.numCadences), requirements=[
                                  'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            self._sferr = np.require(np.zeros(useLC.numCadences), requirements=[
                                     'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            LCTools_cython.compute_SF(useLC.numCadences, useLC.dt, useLC.t, useLC.x,
                                      useLC.y, useLC.yerr, useLC.mask, self._sflags, self._sf, self._sferr)
            return self._sflags, self._sf, self._sferr

    def periodogram(self):
        if (hasattr(self, '_periodogramfreqs') and
                hasattr(self, '_periodogram') and
                hasattr(self, '_periodogramerr')):
            return self._periodogramfreqs, self._periodogram, self._periodogramerr
        else:
            if self.numCadences > 50:
                model = gatspy.periodic.LombScargleFast()
            else:
                model = gatspy.periodic.LombScargle()
            model.optimizer.set(quiet=True, period_range=(2.0*self.meandt, self.T))
            model.fit(self.t,
                      self.y,
                      self.yerr)
            periodogramlags, self._periodogram = model.periodogram_auto()
            self._periodogramfreqs = np.require(1.0/np.array(periodogramlags),
                                                requirements=['F', 'A', 'W', 'O', 'E'])
            self._periodogram = np.require(np.array(self._periodogram),
                                           requirements=['F', 'A', 'W', 'O', 'E'])
            self._periodogramerr = np.require(np.array(self._periodogram.shape[0]*[0.0]),
                                              requirements=['F', 'A', 'W', 'O', 'E'])
            for i in xrange(self._periodogram.shape[0]):
                if self._periodogram[i] <= 0.0:
                    self._periodogram[i] = np.nan
            return self._periodogramfreqs, self._periodogram, self._periodogramerr

    def plot(self, fig=-1, doShow=False, clearFig=True, colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if (np.sum(self.x) != 0.0) and (np.sum(self.y) == 0.0):
            plt.plot(self.t, self.x, color=colorx, zorder=0)
            plt.plot(self.t, self.x, color=colorx, marker='o', markeredgecolor='none', zorder=0)
        if (np.sum(self.x) == 0.0) and (np.sum(self.y) != 0.0):
            plt.errorbar(
                self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]],
                self.yerr[np.where(self.mask == 1.0)[0]], label=r'%s (%s-band)'%(self.name, self.band),
                fmt='o', capsize=0, color=colory, markeredgecolor='none', zorder=10)
        if (np.sum(self.x) != 0.0) and (np.sum(self.y) != 0.0):
            plt.plot(self.t, self.x - np.mean(self.x) + np.mean(
                self.y[np.where(self.mask == 1.0)[0]]), color=colorx, zorder=0)
            plt.plot(self.t, self.x - np.mean(self.x) + np.mean(
                self.y[np.where(self.mask == 1.0)[0]]), color=colorx, marker='o', markeredgecolor='none',
                zorder=0)
            plt.errorbar(
                self.t[np.where(self.mask == 1.0)[0]], self.y[np.where(self.mask == 1.0)[0]],
                self.yerr[np.where(self.mask == 1.0)[0]], label=r'%s (%s-band)'%(self.name, self.band),
                fmt='o', capsize=0, color=colory, markeredgecolor='none', zorder=10)
        if self.isSmoothed:
            plt.plot(self.tSmooth, self.xSmooth, color=colors[0],
                     marker='o', markeredgecolor='none', zorder=-5)
            plt.plot(self.tSmooth, self.xSmooth, color=colors[0], zorder=-5)
            plt.fill_between(self.tSmooth, self.xSmooth - self.xerrSmooth, self.xSmooth +
                             self.xerrSmooth, facecolor=colors[1], alpha=0.5, zorder=-5)
        if hasattr(self, 'tSmooth'):
            plt.xlim(self.tSmooth[0], self.tSmooth[-1])
        else:
            plt.xlim(self.t[0], self.t[-1])
        plt.xlabel(self.xunit)
        plt.ylabel(self.yunit)
        plt.title(r'Light curve')
        plt.legend()
        if doShow:
            plt.show(False)
        return newFig

    def plotacvf(self, fig=-2, newdt=None, doShow=False, clearFig=True,
                 colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        plt.plot(0.0, 0.0)
        if np.sum(self.y) != 0.0:
            lagsE, acvfE, acvferrE = self.acvf(newdt=newdt)
            if np.sum(acvfE) != 0.0:
                plt.errorbar(lagsE[0], acvfE[0], acvferrE[0],
                             label=r'obs. Autocovariance Function', fmt='o', capsize=0,
                             color=colory, markeredgecolor='none', zorder=10)
                plt.errorbar(lagsE[1:], acvfE[1:], acvferrE[1:],
                             fmt='o', capsize=0, color=colory, markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[0], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\gamma(\delta t)$')
        plt.title(r'AutoCovariance Function')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig

    def plotacf(self, fig=-3, newdt=None, doShow=False, clearFig=True,
                colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        plt.plot(0.0, 0.0)
        if np.sum(self.y) != 0.0:
            lagsE, acfE, acferrE = self.acf(newdt=newdt)
            if np.sum(acfE) != 0.0:
                plt.errorbar(lagsE[0], acfE[0], acferrE[0],
                             label=r'obs. Autocorrelation Function', fmt='o', capsize=0,
                             color=colory, markeredgecolor='none', zorder=10)
                plt.errorbar(lagsE[1:], acfE[1:], acferrE[1:],
                             fmt='o', capsize=0, color=colory, markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[0], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\rho(\delta t)$')
        plt.title(r'AutoCorrelation Function')
        plt.legend(loc=3)
        plt.ylim(-1.0, 1.0)
        if doShow:
            plt.show(False)
        return newFig

    def plotdacf(self, fig=-4, numBins=None, doShow=False, clearFig=True,
                 colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if np.sum(self.y) != 0.0:
            lagsE, dacfE, dacferrE = self.dacf(nbins=numBins)
            if np.sum(dacfE) != 0.0:
                plt.errorbar(
                    lagsE[0], dacfE[0], dacferrE[0], label=r'obs. Discrete Autocorrelation Function', fmt='o',
                    capsize=0, color=colory, markeredgecolor='none', zorder=10)
                for i in xrange(0, lagsE.shape[0]):
                    if dacfE[i] != 0.0:
                        plt.errorbar(lagsE[i], dacfE[i], dacferrE[i], fmt='o', capsize=0, color=colory,
                                     markeredgecolor='none', zorder=10)
                plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$DACF$')
        plt.title(r'Discrete Autocorrelation Function')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig

    def plotsf(self, fig=-5, newdt=None, doShow=False, clearFig=True,
               colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        ln10 = math.log(10.0)
        xmin = 0.0
        xmax = -1.0*sys.float_info[0]
        ymin = 1.0*sys.float_info[0]
        ymax = -1.0*sys.float_info[0]
        if np.sum(self.y) != 0.0:
            lagsE, sfE, sferrE = self.sf(newdt)
            if np.sum(sfE) != 0.0:
                i = 1
                for i in xrange(1, lagsE.shape[0]):
                    if sfE[i] != 0.0:
                        break
                xmin = math.log10(lagsE[i])
                if math.log10(lagsE[i]) > xmax:
                    xmax = math.log10(lagsE[i])
                if math.log10(sfE[i]) - math.fabs(sferrE[i]/(ln10*sfE[i])) < ymin:
                    ymin = math.log10(sfE[i]) - math.fabs(sferrE[i]/(ln10*sfE[i]))
                if math.log10(sfE[i]) + math.fabs(sferrE[i]/(ln10*sfE[i])) > ymax:
                    ymax = math.log10(sfE[i]) + math.fabs(sferrE[i]/(ln10*sfE[i]))
                plt.errorbar(math.log10(lagsE[i]), math.log10(sfE[i]), math.fabs(
                    sferrE[i]/(ln10*sfE[i])), label=r'obs. Structure Function', fmt='o', capsize=0,
                    color=colory, markeredgecolor='none', zorder=10)
                startI = i
                for i in xrange(i+1, lagsE.shape[0]):
                    if sfE[i] != 0.0:
                        if math.log10(lagsE[i]) > xmax:
                            xmax = math.log10(lagsE[i])
                        if math.log10(sfE[i]) < ymin:
                            ymin = math.log10(sfE[i])
                        if math.log10(sfE[i]) > ymax:
                            ymax = math.log10(sfE[i])
                        plt.errorbar(
                            math.log10(lagsE[i]), math.log10(sfE[i]), math.fabs(sferrE[i]/(ln10*sfE[i])),
                            fmt='o', capsize=0, color=colory, markeredgecolor='none', zorder=10)
            plt.xlim(xmin, xmax)
            plt.ylim(1.1*ymin, 1.1*ymax)
        plt.xlabel(r'$\log \delta t$')
        plt.ylabel(r'$\log SF$')
        plt.title(r'Structure Function')
        plt.legend(loc=2)
        if doShow:
            plt.show(False)
        return newFig

    def plotperiodogram(self, fig=-6, newdt=None, doShow=False, clearFig=True,
                        colorx=COLORX, colory=COLORY, colors=COLORS):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if np.sum(self.y) != 0.0:
            freqsE, periodogramE, periodogramerrE = self.periodogram()
            plt.loglog(freqsE, periodogramE, color=colory, zorder=-5)
            plt.fill_between(freqsE, periodogramE - periodogramerrE,
                             periodogramE + periodogramerrE, facecolor=colory, alpha=0.5, zorder=-5)
        plt.xlabel(r'$\log \nu$')
        plt.ylabel(r'$\log P$')
        plt.title(r'Periodogram')
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

    @property
    def period(self):
        if hasattr(self, '_period'):
            return self._period
        else:
            if self.numCadences > 50:
                model = gatspy.periodic.LombScargleFast(
                    optimizer_kwds={"quiet": True}).fit(self.t, self.y, self.yerr)
            else:
                model = gatspy.periodic.LombScargle(
                    optimizer_kwds={"quiet": True}).fit(self.t, self.y, self.yerr)
            periods, power = model.periodogram_auto(nyquist_factor=self.numCadences)
            model.optimizer.period_range = (2.0*np.mean(self.t[1:] - self.t[:-1]), self.T)
            self._period = model.best_period
            return self._period

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

    def interpolate(self, dt=None, a=None):
        if a is None:
            a = int(math.ceil(self.maxdt*9))
        if dt is not None:
            if dt < self.meandt:
                raise ValueError('Can\'t interpolate finer than the nyquist rate!')
        else:
            dt = self.meandt
        newLC = self.copy()
        del newLC.t
        del newLC.x
        del newLC.y
        del newLC.yerr
        del newLC.mask
        numCadences = int(round(float(self.T/dt)))
        newLC.t = np.require(np.array(numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.x = np.require(np.array(numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.y = np.require(np.array(numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.yerr = np.require(np.array(numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        newLC.mask = np.require(np.array(numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(numCadences):
            newLC.t[i] = self.t[0] + i*dt
            newLC.mask[i] = 1.0
        newLC.numCadences = numCadences
        newLC.dt = float(newLC.t[1] - newLC.t[0])
        newLC.mindt = float(np.nanmin(newLC.t[1:] - newLC.t[:-1]))
        newLC.maxdt = float(np.nanmax(newLC.t[1:] - newLC.t[:-1]))
        newLC.meandt = float(np.nanmean(newLC.t[1:] - newLC.t[:-1]))
        newLC.T = float(newLC.t[-1] - newLC.t[0])
        convKernel = kali.kernel.Lanczos(a)
        convKernel(self, newLC)
        newLC._statistics()
        return newLC

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


class mockLC(lc):

    def read(self, name=None, band=None, path=None, **kwargs):
        self.name = name
        self.band = band
        if path is None:
            try:
                self.path = os.environ['DATADIR']
            except KeyError:
                # raise KeyError('Environment variable "DATADIR" not set! Please set "DATADIR" to point where
                # all SDSS S82 data should live first...')
                self.path = os.environ['HOME']
        else:
            elf.path = path
        numCadences = kwargs.get('numCadences')  # The number of cadences in the light curve. This is
        # not the same thing as the number of actual observations as we can have missing observations.
        deltaT = kwargs.get('deltaT')
        tIn = kwargs.get('tIn')
        if numCadences is not None and tIn is None:
            if deltaT is not None:
                self.numCadences = numCadences
                self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
                self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
                self.startT = kwargs.get('startT', 0.0)
                for i in xrange(self._numCadences):
                    self.t[i] = i*deltaT
                    self.mask[i] = 1.0
            else:
                raise ValueError('Must supply deltaT if numCadences is supplied!')
        elif tIn is not None and numCadences is None and deltaT is None:
            self.t = np.require(np.array(tIn), requirements=['F', 'A', 'W', 'O', 'E'])
            self.startT = self.t[0]
            self.t = self.t - self.startT
            self.numCadences = self.t.shape[0]
            self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise ValueError('Error! Supply either numCadences & dt or tIn')
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.xunit = kwargs.get('xunit', r'$t$')
        self.yunit = kwargs.get('yunit', r'$F$')

    def write(self, name=None, band=None, pwd=None, **kwargs):
        pass


class externalLC(lc):

    def read(self, name, band, path=None, **kwargs):
        self.name = name
        self.band = band
        if path is None:
            try:
                self.path = os.environ['DATADIR']
            except KeyError:
                # raise KeyError('Environment variable "DATADIR" not set! Please set "DATADIR" to point where
                # all SDSS S82 data should live first...')
                self.path = os.environ['HOME']
        else:
            self.path = path
        t = kwargs.get('tIn')
        if t is not None:
            self.t = np.require(t, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument t!')
        self.startT = self.t[0]
        self.t = self.t - self.startT
        self._numCadences = self.t.shape[0]
        self.x = np.require(
            kwargs.get('x', np.zeros(self.numCadences)), requirements=['F', 'A', 'W', 'O', 'E'])
        y = kwargs.get('yIn')
        if y is not None:
            self.y = np.require(y, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument y!')
        yerr = kwargs.get('yerrIn')
        if yerr is not None:
            self.yerr = np.require(yerr, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise Keyerror('Must supply key-word argument yerr!')
        mask = kwargs.get('maskIn')
        if mask is not None:
            self.mask = np.require(mask, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise Keyerror('Must supply key-word argument mask!')
        self.xunit = kwargs.get('xunit', r'$t$')
        self.yunit = kwargs.get('yunit', r'$F$')
