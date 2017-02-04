#!/usr/bin/env python
"""	Module to model MBHBs.
"""

import numpy as np
import math as math
import scipy.stats as spstats
import scipy.signal.spectral
from scipy.interpolate import UnivariateSpline
import cmath as cmath
import random
import sys as sys
import abc as abc
import psutil as psutil
import types as types
import os as os
import reprlib as reprlib
import copy as copy
import warnings as warnings
import matplotlib.pyplot as plt
import pdb as pdb

import multi_key_dict
import gatspy.periodic
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from matplotlib import cm

try:
    import kali.lc
    import rand as rand
    import MBHBTask_cython as MBHBTask_cython
    from util.mpl_settings import set_plot_params
    import kali.util.classproperty
    import kali.util.triangle
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


def d2r(degree):
    return degree*(math.pi/180.0)


def r2d(radian):
    return radian*(180.0/math.pi)


def _f7(seq):
    """http://tinyurl.com/angxm5"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class MBHBTask(object):

    _type = 'kali.mbhb'
    _name = 'kali.MBHBTask()'
    _r = 8
    G = 6.67408e-11
    c = 299792458.0
    pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
    twoPi = 2.0*pi
    fourPi = 4.0*pi
    fourPiSq = 4.0*math.pow(pi, 2.0)
    Parsec = 3.0857e16
    Day = 86164.090530833
    Year = 31557600.0
    SolarMass = 1.98855e30
    kms2ms = 1.0e3
    SolarMassPerCubicParsec = SolarMass/math.pow(Parsec, 3.0)
    sigmaStars = 200.0
    rhoStars = 1000.0
    H = 16.0

    _dict = multi_key_dict.multi_key_dict()
    _dict[r'$a_{1}$ (pc)', 0, r'0', r'a1', r'a_1', r'$a_{1}$', r'$a_{1}~\mathrm{(pc)}$',
          'semi-major axis 1', 'semimajor axis 1'] = 0
    _dict[r'$a_{2}$ (pc)', 1, r'1', r'a2', r'a_2', r'$a_{2}$', r'$a_{2}~\mathrm{(pc)}$',
          'semi-major axis 2', 'semimajor axis 2'] = 1
    _dict[r'$T$ (d)', 2, r'2', r'T', r'$T$', r'$T~\mathrm{(d)}$', 'period'] = 2
    _dict[r'$e$', 3, r'3', r'e', 'ellipticity', 'eccentricity'] = 3
    _dict[r'$\Omega$ (deg.)', 4, r'4', r'$\Omega~\mathrm{(deg.)}$', r'Omega', r'omega'] = 4
    _dict[r'$i$ (deg.)', 5, r'5', r'$i~\mathrm{(deg.)}$', r'Inclination', r'inclination'] = 5
    _dict[r'$\tau$ (d)', 6, r'6', r'$\tau~\mathrm{(d)}$', r'Tau', r'tau'] = 6
    _dict[r'$F$', 7, r'7', r'Flux', r'flux'] = 7

    def __init__(self, p=0, q=0, nthreads=psutil.cpu_count(logical=True),
                 nwalkers=25*psutil.cpu_count(logical=True),
                 nsteps=250, maxEvals=10000, xTol=0.01, mcmcA=2.0):
        try:
            assert p == 0, r'p must be 0'
            assert q == 0, r'q must be 0'
            assert nthreads > 0, r'nthreads must be greater than 0'
            assert isinstance(nthreads, int), r'nthreads must be an integer'
            assert nwalkers > 0, r'nwalkers must be greater than 0'
            assert isinstance(nwalkers, int), r'nwalkers must be an integer'
            assert nsteps > 0, r'nsteps must be greater than 0'
            assert isinstance(nsteps, int), r'nsteps must be an integer'
            assert maxEvals > 0, r'maxEvals must be greater than 0'
            assert isinstance(maxEvals, int), r'maxEvals must be an integer'
            assert xTol > 0.0, r'xTol must be greater than 0'
            assert isinstance(xTol, float), r'xTol must be a float'
            self._p = p
            self._q = q
            self._ndims = self.r
            self._nthreads = nthreads
            self._nwalkers = nwalkers
            self._nsteps = nsteps
            self._maxEvals = maxEvals
            self._xTol = xTol
            self._mcmcA = mcmcA
            self._Chain = np.require(
                np.zeros(self._ndims*self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnPrior = np.require(
                np.zeros(self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnLikelihood = np.require(
                np.zeros(self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
            self._taskCython = MBHBTask_cython.MBHBTask_cython(self._nthreads)
            self._pDIC = None
            self._dic = None
        except AssertionError as err:
            raise AttributeError(str(err))

    def __getstate__(self):
        """!
        \brief Trim the c-pointer from the task and return the remaining __dict__ for pickling.
        """
        state = copy.copy(self.__dict__)
        del state['_taskCython']
        return state

    def __setstate__(self, state):
        """!
        \brief Restore an un-pickled task completely by setting the c-pointer to the right Cython class.
        """
        self.__dict__ = copy.copy(state)
        self._taskCython = MBHBTask_cython.MBHBTask_cython(self._nthreads)

    @kali.util.classproperty.ClassProperty
    @classmethod
    def type(self):
        return self._type

    @kali.util.classproperty.ClassProperty
    @classmethod
    def name(self):
        return self._name

    @kali.util.classproperty.ClassProperty
    @classmethod
    def r(self):
        return self._r

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def id(self):
        return self.type + '.' + str(self.p) + '.' + str(self.q)

    @property
    def nthreads(self):
        return self._nthreads

    @property
    def ndims(self):
        return self._ndims

    @property
    def nwalkers(self):
        return self._nwalkers

    @nwalkers.setter
    def nwalkers(self, value):
        try:
            assert value >= 0, r'nwalkers must be greater than or equal to 0'
            assert isinstance(value, int), r'nwalkers must be an integer'
            self._nwalkers = value
            self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps),
                                     requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnPrior = np.require(np.zeros(self._nwalkers*self._nsteps),
                                       requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnLikelihood = np.require(np.zeros(self._nwalkers*self._nsteps),
                                            requirements=['F', 'A', 'W', 'O', 'E'])
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def nsteps(self):
        return self._nsteps

    @nsteps.setter
    def nsteps(self, value):
        try:
            assert value >= 0, r'nsteps must be greater than or equal to 0'
            assert isinstance(value, int), r'nsteps must be an integer'
            self._nsteps = value
            self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps),
                                     requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnPrior = np.require(np.zeros(self._nwalkers*self._nsteps),
                                       requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnLikelihood = np.require(np.zeros(self._nwalkers*self._nsteps),
                                            requirements=['F', 'A', 'W', 'O', 'E'])
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def maxEvals(self):
        return self._maxEvals

    @maxEvals.setter
    def maxEvals(self, value):
        try:
            assert value >= 0, r'maxEvals must be greater than or equal to 0'
            assert isinstance(value, int), r'maxEvals must be an integer'
            self._maxEvals = value
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def xTol(self):
        return self._xTol

    @xTol.setter
    def xTol(self, value):
        try:
            assert value >= 0, r'xTol must be greater than or equal to 0'
            assert isinstance(value, float), r'xTol must be a float'
            self._xTol = value
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def mcmcA(self):
        return self._mcmcA

    @mcmcA.setter
    def mcmcA(self, value):
        try:
            assert value >= 0, r'mcmcA must be greater than or equal to 0.0'
            assert isinstance(value, float), r'mcmcA must be a float'
            self._mcmcA = value
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def Chain(self):
        return np.reshape(self._Chain, newshape=(self._ndims, self._nwalkers, self._nsteps), order='F')

    @property
    def rootChain(self):
        if hasattr(self, '_rootChain'):
            return self._rootChain
        else:
            self._rootChain = self.Chain
        return self._rootChain

    @property
    def timescaleChain(self):
        if hasattr(self, '_timescaleChain'):
            return self._timescaleChain
        else:
            self._timescaleChain = self.Chain
        return self._timescaleChain

    @property
    def LnPrior(self):
        return np.reshape(self._LnPrior, newshape=(self._nwalkers, self._nsteps), order='F')

    @property
    def LnLikelihood(self):
        return np.reshape(self._LnLikelihood, newshape=(self._nwalkers, self._nsteps), order='F')

    @property
    def LnPosterior(self):
        return self.LnPrior + self.LnLikelihood

    @property
    def pDIC(self):
        return self._pDIC

    @property
    def dic(self):
        return self._dic

    def __repr__(self):
        return "kali.mbhb.MBHBTask(%d, %d, %d, %d, %d, %d, %f)"%(self._p, self._q, self._nthreads,
                                                                 self._nwalkers, self._nsteps,
                                                                 self._maxEvals, self._xTol)

    def __str__(self):
        line = 'p: %d; q: %d; ndims: %d\n'%(self._p, self._q, self._ndims)
        line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
        line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
        line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
        line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for \
        MCMC): %d\n'%(self._maxEvals)
        line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if ((self._nthreads == other.nthreads) and (self._nwalkers == other.nwalkers) and
                    (self._nsteps == other.nsteps) and (self._maxEvals == other.maxEvals) and
                    (self.xTol == other.xTol)):
                return True
            else:
                return False
        else:
            return False

    def __neq__(self, other):
        if self == other:
            return False
        else:
            return True

    def reset(self, nwalkers=None, nsteps=None):
        if nwalkers is None:
            nwalkers = self._nwalkers
        if nsteps is None:
            nsteps = self._nsteps
        try:
            assert nwalkers > 0, r'nwalkers must be greater than 0'
            assert isinstance(nwalkers, int), r'nwalkers must be an integer'
            assert nsteps > 0, r'nsteps must be greater than 0'
            assert isinstance(nsteps, int), r'nsteps must be an integer'
            self._nwalkers = nwalkers
            self._nsteps = nsteps
            self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps),
                                     requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnPrior = np.require(np.zeros(self._nwalkers*self._nsteps),
                                       requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnLikelihood = np.require(np.zeros(self._nwalkers*self._nsteps),
                                            requirements=['F', 'A', 'W', 'O', 'E'])
        except AssertionError as err:
            raise AttributeError(str(err))

    def check(self, Theta, tnum=None):
        if tnum is None:
            tnum = 0
        assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
        return bool(self._taskCython.check_Theta(Theta, tnum))

    def set(self, dt, Theta, tnum=None):
        if tnum is None:
            tnum = 0
        assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
        return self._taskCython.set_System(Theta, tnum)

    def Theta(self, tnum=None):
        if tnum is None:
            tnum = 0
        Theta = np.zeros(self._ndims)
        self._taskCython.get_Theta(Theta, tnum)
        return Theta

    def list(self):
        setSystems = np.zeros(self._nthreads, dtype='int32')
        self._taskCython.get_setSystemsVec(setSystems)
        return setSystems.astype(np.bool_)

    def show(self, tnum=None):
        if tnum is None:
            tnum = 0
        self._taskCython.print_System(tnum)

    def __call__(self, epochVal, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.set_Epoch(epochVal, tnum)

    def epoch(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Epoch(tnum)

    def period(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Period(tnum)

    def a1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_A1(tnum)

    def a2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_A2(tnum)

    def m1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_M1(tnum)

    def m2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_M2(tnum)

    def m12(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_M12(tnum)

    def mratio(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_M2OverM1(tnum)

    def rPeribothron1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RPeribothron1(tnum)

    def rPeribothron2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RPeribothron2(tnum)

    def rApobothron1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RApobothron1(tnum)

    def rApobothron2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RApobothron2(tnum)

    def rPeribothron(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RPeribothronTot(tnum)

    def rApobothron(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RApobothronTot(tnum)

    def rS1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RS1(tnum)

    def rS2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RS2(tnum)

    def eccentricity(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Eccentricity(tnum)

    def omega1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Omega1(tnum)

    def omega2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Omega2(tnum)

    def inclination(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Inclination(tnum)

    def tau(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Tau(tnum)

    def M(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_MeanAnomoly(tnum)

    def E(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_EccentricAnomoly(tnum)

    def nu(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_TrueAnomoly(tnum)

    def r1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_R1(tnum)

    def r2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_R2(tnum)

    def theta1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Theta1(tnum)

    def theta2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Theta2(tnum)

    def Beta1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Beta1(tnum)

    def Beta2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_Beta2(tnum)

    def radialBeta1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RadialBeta1(tnum)

    def radialBeta2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_RadialBeta2(tnum)

    def dopplerFactor1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_DopplerFactor1(tnum)

    def dopplerFactor2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_DopplerFactor2(tnum)

    def beamingFactor1(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_BeamingFactor1(tnum)

    def beamingFactor2(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_BeamingFactor2(tnum)

    def aH(self, sigmaStars=200.0, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_aH(sigmaStars, tnum)

    def aGW(self, sigmaStars=200.0, rhoStars=1000.0, H=16, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_aGW(sigmaStars, rhoStars, H, tnum)

    def durationInHardState(self, sigmaStars=200.0, rhoStars=1000.0, H=16, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_durationInHardState(sigmaStars, rhoStars, H, tnum)

    def ejectedMass(self, sigmaStars=200.0, rhoStars=1000.0, H=16, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_ejectedMass(sigmaStars, rhoStars, H, tnum)

    def simulate(self, duration=None, deltaT=None, tIn=None, fracNoiseToSignal=0.001, tnum=None):
        if tnum is None:
            tnum = 0
        if tIn is None and duration is not None:
            if deltaT is None:
                deltaT = self.period()/10.0
            numCadences = int(round(float(duration)/deltaT))
            intrinsicLC = kali.lc.mockLC(numCadences=numCadences, deltaT=deltaT,
                                         fracNoiseToSignal=fracNoiseToSignal)
        elif duration is None and tIn is not None:
            if deltaT is not None:
                raise ValueError('deltaT cannot be supplied when tIn is provided')
            numCadences = tIn.shape[0]
            t = np.require(np.array(tIn), requirements=['F', 'A', 'W', 'O', 'E'])
            intrinsicLC = kali.lc.mockLC(
                name='', band='', tIn=t, fracNoiseToSignal=fracNoiseToSignal)
            for i in xrange(intrinsicLC.numCadences):
                intrinsicLC.mask[i] = 1.0
        self._taskCython.make_IntrinsicLC(
            intrinsicLC.numCadences, intrinsicLC.dt, intrinsicLC.fracNoiseToSignal,
            intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask,
            threadNum=tnum)
        intrinsicLC._simulatedCadenceNum = numCadences - 1
        intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
        return intrinsicLC

    def observe(self, intrinsicLC, noiseSeed=None, tnum=None):
        if tnum is None:
            tnum = 0
        randSeed = np.zeros(1, dtype='uint32')
        if noiseSeed is None:
            rand.rdrand(randSeed)
            noiseSeed = randSeed[0]
        self._taskCython.add_ObservationNoise(
            intrinsicLC.numCadences, intrinsicLC.dt, intrinsicLC.fracNoiseToSignal,
            intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed,
            threadNum=tnum)
        count = int(np.sum(intrinsicLC.mask))
        y_meanSum = 0.0
        yerr_meanSum = 0.0
        for i in xrange(intrinsicLC.numCadences):
            y_meanSum += intrinsicLC.mask[i]*intrinsicLC.y[i]
            yerr_meanSum += intrinsicLC.mask[i]*intrinsicLC.yerr[i]
        if count > 0.0:
            intrinsicLC._mean = y_meanSum/count
            intrinsicLC._meanerr = yerr_meanSum/count
        else:
            intrinsicLC._mean = 0.0
            intrinsicLC._meanerr = 0.0
        y_stdSum = 0.0
        yerr_stdSum = 0.0
        for i in xrange(intrinsicLC.numCadences):
            y_stdSum += math.pow(intrinsicLC.mask[i]*intrinsicLC.y[i] - intrinsicLC._mean, 2.0)
            yerr_stdSum += math.pow(intrinsicLC.mask[i]*intrinsicLC.yerr[i] - intrinsicLC._meanerr, 2.0)
        if count > 0.0:
            intrinsicLC._std = math.sqrt(y_stdSum/count)
            intrinsicLC._stderr = math.sqrt(yerr_stdSum/count)
        else:
            intrinsicLC._std = 0.0
            intrinsicLC._stderr = 0.0

    def logPrior(self, observedLC, forced=True, widthT=0.01, widthF=0.05, tnum=None):
        if tnum is None:
            tnum = 0
        fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinIncEst = self.estimate(observedLC)
        lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)])
        highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)])
        observedLC._logPrior = self._taskCython.compute_LnPrior(
            observedLC.numCadences, observedLC.dt, observedLC.startT, lowestFlux, highestFlux,
            observedLC.t, observedLC.x,
            observedLC.y, observedLC.yerr, observedLC.mask,
            periodEst, widthT*periodEst,
            fluxEst, widthF*fluxEst,
            tnum)
        return observedLC._logPrior

    def logLikelihood(self, observedLC, forced=True, tnum=None):
        if tnum is None:
            tnum = 0
        observedLC._logPrior = self.logPrior(observedLC, forced=forced, tnum=tnum)
        if forced is True:
            observedLC._computedCadenceNum = -1
        if observedLC._computedCadenceNum == -1:
            observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(
                observedLC.numCadences, observedLC.dt, observedLC._computedCadenceNum, observedLC.t,
                observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask, tnum)
            observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
            observedLC._computedCadenceNum = observedLC.numCadences - 1
        else:
            pass
        return observedLC._logLikelihood

    def logPosterior(self, observedLC, forced=True, tnum=None):
        lnLikelihood = self.logLikelihood(observedLC, forced=forced, tnum=tnum)
        return observedLC._logPosterior

        try:
            omega1 = math.acos((1.0/eccentricity)*(
                (maxBetaParallel + minBetaParallel)/(maxBetaParallel - minBetaParallel)))
        except ValueError:
            pdb.set_trace()
        return omega1

    def estimate(self, observedLC):
        """!
        Estimate intrinsicFlux, period, eccentricity, omega, tau, & a2sini
        """
        # fluxEst
        if observedLC.numCadences > 50:
            model = gatspy.periodic.LombScargleFast()
        else:
            model = gatspy.periodic.LombScargle()
        model.optimizer.set(quiet=True, period_range=(2.0*observedLC.meandt, observedLC.T))
        model.fit(observedLC.t,
                  observedLC.y,
                  observedLC.yerr)
        periodEst = model.best_period
        if periodEst < 2.0*observedLC.meandt or periodEst > observedLC.T:
            scaled_y = (observedLC.y - observedLC.mean)/observedLC.std
            angFreqs = np.logspace(math.log10((2.0*math.pi)/(observedLC.T)),
                                   math.log10((2.0*math.pi)/(2.0*observedLC.meandt)),
                                   10000)
            periodogram = scipy.signal.spectral.lombscargle(observedLC.t, scaled_y, angFreqs)
            periodEst = 2.0*math.pi/angFreqs[np.argmax(periodogram)]

        numIntrinsicFlux = 100
        lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)])
        highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)])
        intrinsicFlux = np.linspace(np.min(observedLC.y[np.where(observedLC.mask == 1.0)]), np.max(
            observedLC.y[np.where(observedLC.mask == 1.0)]), num=numIntrinsicFlux)
        intrinsicFluxList = list()
        totalIntegralList = list()
        for f in xrange(1, numIntrinsicFlux - 1):
            beamedLC = observedLC.copy()
            beamedLC.x = np.require(np.zeros(beamedLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
            for i in xrange(beamedLC.numCadences):
                beamedLC.y[i] = observedLC.y[i]/intrinsicFlux[f]
                beamedLC.yerr[i] = observedLC.yerr[i]/intrinsicFlux[f]
            dopplerLC = beamedLC.copy()
            dopplerLC.x = np.require(np.zeros(dopplerLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
            for i in xrange(observedLC.numCadences):
                dopplerLC.y[i] = math.pow(beamedLC.y[i], 1.0/3.44)
                dopplerLC.yerr[i] = (1.0/3.44)*math.fabs(dopplerLC.y[i]*(beamedLC.yerr[i]/beamedLC.y[i]))
            dzdtLC = dopplerLC.copy()
            dzdtLC.x = np.require(np.zeros(dopplerLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
            for i in xrange(observedLC.numCadences):
                dzdtLC.y[i] = 1.0 - (1.0/dopplerLC.y[i])
                dzdtLC.yerr[i] = math.fabs((-1.0*dopplerLC.yerr[i])/math.pow(dopplerLC.y[i], 2.0))
            foldedLC = dzdtLC.fold(periodEst)
            foldedLC.x = np.require(np.zeros(foldedLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
            integralSpline = UnivariateSpline(
                foldedLC.t[np.where(foldedLC.mask == 1.0)], foldedLC.y[np.where(foldedLC.mask == 1.0)],
                1.0/foldedLC.yerr[np.where(foldedLC.mask == 1.0)], k=3, s=None, check_finite=True)
            totalIntegral = math.fabs(integralSpline.integral(foldedLC.t[0], foldedLC.t[-1]))
            intrinsicFluxList.append(intrinsicFlux[f])
            totalIntegralList.append(totalIntegral)
        fluxEst = intrinsicFluxList[
            np.where(np.array(totalIntegralList) == np.min(np.array(totalIntegralList)))[0][0]]

        # periodEst
        for i in xrange(beamedLC.numCadences):
            beamedLC.y[i] = observedLC.y[i]/fluxEst
            beamedLC.yerr[i] = observedLC.yerr[i]/fluxEst
            dopplerLC.y[i] = math.pow(beamedLC.y[i], 1.0/3.44)
            dopplerLC.yerr[i] = (1.0/3.44)*math.fabs(dopplerLC.y[i]*(beamedLC.yerr[i]/beamedLC.y[i]))
            dzdtLC.y[i] = 1.0 - (1.0/dopplerLC.y[i])
            dzdtLC.yerr[i] = math.fabs((-1.0*dopplerLC.yerr[i])/math.pow(dopplerLC.y[i], 2.0))
        if observedLC.numCadences > 50:
            model = gatspy.periodic.LombScargleFast()
        else:
            model = gatspy.periodic.LombScargle()
        model.optimizer.set(quiet=True, period_range=(2.0*observedLC.meandt, observedLC.T))
        model.fit(dzdtLC.t,
                  dzdtLC.y,
                  dzdtLC.yerr)
        periodEst = model.best_period
        if periodEst < 2.0*observedLC.meandt or periodEst > observedLC.T:
            scaled_y = (dzdtLC.y - dzdtLC.mean)/dzdtLC.std
            angFreqs = np.logspace(math.log10((2.0*math.pi)/(observedLC.T)),
                                   math.log10((2.0*math.pi)/(2.0*observedLC.meandt)),
                                   10000)
            periodogram = scipy.signal.spectral.lombscargle(observedLC.t, scaled_y, angFreqs)
            periodEst = 2.0*math.pi/angFreqs[np.argmax(periodogram)]

        # eccentricityEst & omega2Est
        # First find a full period going from rising to falling.
        risingSpline = UnivariateSpline(
            dzdtLC.t[np.where(dzdtLC.mask == 1.0)], dzdtLC.y[np.where(dzdtLC.mask == 1.0)],
            1.0/dzdtLC.yerr[np.where(dzdtLC.mask == 1.0)], k=3, s=None, check_finite=True)
        risingSplineRoots = risingSpline.roots()
        firstRoot = risingSplineRoots[0]
        if risingSpline.derivatives(risingSplineRoots[0])[1] > 0.0:
            tRising = risingSplineRoots[0]
        else:
            tRising = risingSplineRoots[1]
        # Now fold the LC starting at tRising and going for a full period.
        foldedLC = dzdtLC.fold(periodEst, tStart=tRising)
        foldedLC.x = np.require(np.zeros(foldedLC.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        # Fit the folded LC with a spline to figure out alpha and beta
        fitLC = foldedLC.copy()
        foldedSpline = UnivariateSpline(
            foldedLC.t[np.where(foldedLC.mask == 1.0)], foldedLC.y[np.where(foldedLC.mask == 1.0)],
            1.0/foldedLC.yerr[np.where(foldedLC.mask == 1.0)], k=3, s=2*foldedLC.numCadences,
            check_finite=True)
        for i in xrange(fitLC.numCadences):
            fitLC.x[i] = foldedSpline(fitLC.t[i])
        # Now get the roots and find the falling root
        tZeros = foldedSpline.roots()

        # Find tRising, tFalling, tFull, startIndex, & stopIndex via DBSCAN #######################
        # Find the number of clusters
        '''dbsObj = DBSCAN(eps = periodEst/10.0, min_samples = 1)
        db = dbsObj.fit(tZeros.reshape(-1,1))
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        unique_labels = set(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)'''

        # Find tRising, tFalling, tFull, startIndex, & stopIndex
        if tZeros.shape[0] == 1:  # We have found just tFalling
            tFalling = tZeros[0]
            tRising = fitLC.t[0]
            startIndex = 0
            tFull = fitLC.t[-1]
            stopIndex = fitLC.numCadences
        elif tZeros.shape[0] == 2:  # We have found tFalling and one of tRising or tFull
            if foldedSpline.derivatives(tZeros[0])[1] < 0.0:
                tFalling = tZeros[0]
                tFull = tZeros[1]
                stopIndex = np.where(fitLC.t < tFull)[0][-1]
                tRising = fitLC.t[0]
                startIndex = 0
            elif foldedSpline.derivatives(tZeros[0])[1] > 0.0:
                if foldedSpline.derivatives(tZeros[1])[1] < 0.0:
                    tRising = tZeros[0]
                    startIndex = np.where(fitLC.t > tRising)[0][0]
                    tFalling = tZeros[1]
                    tFull = fitLC.t[-1]
                    stopIndex = fitLC.numCadences
                else:
                    raise RuntimeError(
                        'Could not determine alpha & omega correctly because the first root is rising but \
                        the second root is not falling!')
        elif tZeros.shape[0] == 3:
            tRising = tZeros[0]
            startIndex = np.where(fitLC.t > tRising)[0][0]
            tFalling = tZeros[1]
            tFull = tZeros[2]
            stopIndex = np.where(fitLC.t < tFull)[0][-1]
        else:
            # More than 3 roots!!! Use K-Means to cluster the roots assuming we have 3 groups
            root_groups = KMeans(n_clusters=3).fit_predict(tZeros.reshape(-1, 1))
            RisingGroupNumber = root_groups[0]
            FullGroupNumber = root_groups[-1]
            RisingSet = set(root_groups[np.where(root_groups != RisingGroupNumber)[0]])
            FullSet = set(root_groups[np.where(root_groups != FullGroupNumber)[0]])
            FallingSet = RisingSet.intersection(FullSet)
            FallingGroupNumber = FallingSet.pop()
            numRisingRoots = np.where(root_groups == RisingGroupNumber)[0].shape[0]
            numFallingRoots = np.where(root_groups == FallingGroupNumber)[0].shape[0]
            numFullRoots = np.where(root_groups == FullGroupNumber)[0].shape[0]

            if numRisingRoots == 1:
                tRising = tZeros[np.where(root_groups == RisingGroupNumber)[0]][0]
            else:
                RisingRootCands = tZeros[np.where(root_groups == RisingGroupNumber)[0]]
                for i in xrange(RisingRootCands.shape[0]):
                    if foldedSpline.derivatives(RisingRootCands[i])[1] > 0.0:
                        tRising = RisingRootCands[i]
                        break

            if numFallingRoots == 1:
                tFalling = tZeros[np.where(root_groups == FallingGroupNumber)[0]][0]
            else:
                FallingRootCands = tZeros[np.where(root_groups == FallingGroupNumber)[0]]
                for i in xrange(FallingRootCands.shape[0]):
                    if foldedSpline.derivatives(FallingRootCands[i])[1] < 0.0:
                        tFalling = FallingRootCands[i]
                        break

            if numFullRoots == 1:
                tFull = tZeros[np.where(root_groups == FullGroupNumber)[0]][0]
            else:
                FullRootCands = tZeros[np.where(root_groups == FullGroupNumber)[0]]
                for i in xrange(FullRootCands.shape[0]):
                    if foldedSpline.derivatives(FullRootCands[i])[1] > 0.0:
                        tFull = FullRootCands[i]
                        break
            startIndex = np.where(fitLC.t > tRising)[0][0]
            stopIndex = np.where(fitLC.t < tFull)[0][-1]
        #

        # One full period now goes from tRising to periodEst. The maxima occurs between tRising and tFalling
        # while the minima occurs between tFalling and tRising + periodEst. Find the minima and maxima
        alpha = math.fabs(fitLC.x[np.where(np.max(fitLC.x[startIndex:stopIndex]) == fitLC.x)[0][0]])
        beta = math.fabs(fitLC.x[np.where(np.min(fitLC.x[startIndex:stopIndex]) == fitLC.x)[0][0]])
        peakLoc = fitLC.t[np.where(np.max(fitLC.x[startIndex:stopIndex]) == fitLC.x)[0][0]]
        troughLoc = fitLC.t[np.where(np.min(fitLC.x[startIndex:stopIndex]) == fitLC.x)[0][0]]
        KEst = 0.5*(alpha + beta)
        delta2 = (math.fabs(foldedSpline.integral(tRising, peakLoc)) + math.fabs(
            foldedSpline.integral(troughLoc, tFull)))/2.0
        delta1 = (math.fabs(foldedSpline.integral(peakLoc, tFalling)) + math.fabs(
            foldedSpline.integral(tFalling, troughLoc)))/2.0
        eCosOmega2 = (alpha - beta)/(alpha + beta)
        eSinOmega2 = ((2.0*math.sqrt(alpha*beta))/(alpha + beta))*((delta2 - delta1)/(delta2 + delta1))
        eccentricityEst = math.sqrt(math.pow(eCosOmega2, 2.0) + math.pow(eSinOmega2, 2.0))
        tanOmega2 = math.fabs(eSinOmega2/eCosOmega2)
        if (eCosOmega2/math.fabs(eCosOmega2) == 1.0) and (eSinOmega2/math.fabs(eSinOmega2) == 1.0):
            omega2Est = math.atan(tanOmega2)*(180.0/math.pi)
        if (eCosOmega2/math.fabs(eCosOmega2) == -1.0) and (eSinOmega2/math.fabs(eSinOmega2) == 1.0):
            omega2Est = 180.0 - math.atan(tanOmega2)*(180.0/math.pi)
        if (eCosOmega2/math.fabs(eCosOmega2) == -1.0) and (eSinOmega2/math.fabs(eSinOmega2) == -1.0):
            omega2Est = 180.0 + math.atan(tanOmega2)*(180.0/math.pi)
        if (eCosOmega2/math.fabs(eCosOmega2) == 1.0) and (eSinOmega2/math.fabs(eSinOmega2) == -1.0):
            omega2Est = 360.0 - math.atan(tanOmega2)*(180.0/math.pi)
        if omega2Est >= 180.0:
            omega1Est = omega2Est - 180.0
        if omega2Est < 180.0:
            omega1Est = omega2Est + 180.0

        # tauEst
        zDot = KEst*(1.0 + eccentricityEst)*(eCosOmega2/eccentricityEst)
        zDotLC = dzdtLC.copy()
        for i in xrange(zDotLC.numCadences):
            zDotLC.y[i] = zDotLC.y[i] - zDot
        zDotZeros = list()
        startSFactor = 2.01
        while len(zDotZeros) < 3:
            startSFactor -= 0.01
            zDotSpline = UnivariateSpline(
                zDotLC.t[np.where(zDotLC.mask == 1.0)], zDotLC.y[np.where(zDotLC.mask == 1.0)],
                1.0/zDotLC.yerr[np.where(zDotLC.mask == 1.0)], k=3, s=startSFactor*zDotLC.numCadences,
                check_finite=True)
            for i in xrange(zDotLC.numCadences):
                zDotLC.x[i] = zDotSpline(zDotLC.t[i])
            zDotZeros = zDotSpline.roots()
        zDotFoldedLC = dzdtLC.fold(periodEst)
        zDotFoldedSpline = UnivariateSpline(
            zDotFoldedLC.t[np.where(zDotFoldedLC.mask == 1.0)],
            zDotFoldedLC.y[np.where(zDotFoldedLC.mask == 1.0)],
            1.0/zDotFoldedLC.yerr[np.where(zDotFoldedLC.mask == 1.0)], k=3, s=2.0*zDotFoldedLC.numCadences,
            check_finite=True)
        for i in xrange(zDotFoldedLC.numCadences):
            zDotFoldedLC.x[i] = zDotFoldedSpline(zDotFoldedLC.t[i])
        tC = zDotFoldedLC.t[np.where(np.max(zDotFoldedLC.x) == zDotFoldedLC.x)[0][0]]
        nuC = (360.0 - omega2Est)%360.0
        tE = zDotFoldedLC.t[np.where(np.min(zDotFoldedLC.x) == zDotFoldedLC.x)[0][0]]
        nuE = (180.0 - omega2Est)%360.0
        if math.fabs(360.0 - nuC) < math.fabs(360 - nuE):
            tauEst = zDotZeros[np.where(zDotZeros > tC)[0][0]]
        else:
            tauEst = zDotZeros[np.where(zDotZeros > tE)[0][0]]
        tauEst = tauEst%periodEst + observedLC.startT

        # a2sinInclinationEst
        a2sinInclinationEst = ((KEst*periodEst*self.Day*self.c*math.sqrt(1.0 - math.pow(
            eccentricityEst, 2.0)))/self.twoPi)/self.Parsec

        return fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst

    def guess(self, periodEst, a2SinInclinationEst):
        a2Guess = 0.0
        while (a2Guess < a2SinInclinationEst):
            m1Val = math.pow(10.0, random.uniform(4.0, 10.0))
            m2Val = math.pow(10.0, random.uniform(4.0, math.log10(m1Val)))
            totalMassVal = m1Val + m2Val
            axisRatioVal = m2Val/m1Val
            aTotCube = (self.G*math.pow(periodEst*self.Day, 2.0)*totalMassVal*self.SolarMass)/self.fourPiSq
            totalAxisVal = math.pow(aTotCube, 1.0/3.0)
            a1Guess = ((axisRatioVal*totalAxisVal)/(1 + axisRatioVal))/self.Parsec
            a2Guess = (totalAxisVal/(1 + axisRatioVal))/self.Parsec
        inclinationGuess = r2d(math.asin(a2SinInclinationEst/a2Guess))
        return a1Guess, a2Guess, inclinationGuess

    def fit(self, observedLC, widthT=0.01, widthF=0.05,
            zSSeed=None, walkerSeed=None, moveSeed=None, xSeed=None):
        randSeed = np.zeros(1, dtype='uint32')
        if zSSeed is None:
            rand.rdrand(randSeed)
            zSSeed = randSeed[0]
        if walkerSeed is None:
            rand.rdrand(randSeed)
            walkerSeed = randSeed[0]
        if moveSeed is None:
            rand.rdrand(randSeed)
            moveSeed = randSeed[0]
        if xSeed is None:
            rand.rdrand(randSeed)
            xSeed = randSeed[0]
        xStart = np.require(np.zeros(self.ndims*self.nwalkers), requirements=['F', 'A', 'W', 'O', 'E'])

        fluxEst, periodEst, eccentricityEst, omega1Est, tauEst, a2sinInclinationEst = self.estimate(
            observedLC)

        for walkerNum in xrange(self.nwalkers):
            noSuccess = True
            while noSuccess:
                a1Guess, a2Guess, inclinationGuess = self.guess(periodEst, a2sinInclinationEst)
                ThetaGuess = np.array(
                    [a1Guess, a2Guess, periodEst, eccentricityEst, omega1Est, inclinationGuess, tauEst,
                        fluxEst])
                res = self.set(observedLC.dt, ThetaGuess)
                lnPrior = self.logPrior(observedLC)
                if res == 0 and not np.isinf(lnPrior):
                    noSuccess = False
            for dimNum in xrange(self.ndims):
                xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]

        lowestFlux = np.min(observedLC.y[np.where(observedLC.mask == 1.0)[0]])
        highestFlux = np.max(observedLC.y[np.where(observedLC.mask == 1.0)[0]])
        res = self._taskCython.fit_MBHBModel(
            observedLC.numCadences, observedLC.dt, observedLC.startT, lowestFlux, highestFlux,
            observedLC.t, observedLC.x,
            observedLC.y, observedLC.yerr, observedLC.mask, self.nwalkers, self.nsteps, self.maxEvals,
            self.xTol, self.mcmcA, zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain,
            self._LnPrior, self._LnLikelihood,
            periodEst, widthT*periodEst,
            fluxEst, widthF*fluxEst)

        meanTheta = list()
        for dimNum in xrange(self.ndims):
            meanTheta.append(np.mean(self.Chain[dimNum, :, self.nsteps/2:]))
        meanTheta = np.require(meanTheta, requirements=['F', 'A', 'W', 'O', 'E'])
        self.set(observedLC.dt, meanTheta)
        devianceThetaBar = -2.0*self.logLikelihood(observedLC)
        barDeviance = np.mean(-2.0*self.LnLikelihood[:, self.nsteps/2:])
        self._pDIC = barDeviance - devianceThetaBar
        self._dic = devianceThetaBar + 2.0*self.pDIC
        self.rootChain
        self.timescaleChain
        self.bestTheta
        self.bestRho
        self.bestTau
        self.auxillaryChain
        return res

    @property
    def bestTheta(self):
        if hasattr(self, '_bestTheta'):
            return self._bestTheta
        else:
            bestWalker = np.where(np.max(self.LnPosterior) == self.LnPosterior)[0][0]
            bestStep = np.where(np.max(self.LnPosterior) == self.LnPosterior)[1][0]
            self._bestTheta = copy.copy(self.Chain[:, bestWalker, bestStep])
            return self._bestTheta

    @property
    def bestRho(self):
        if hasattr(self, '_bestRho'):
            return self._bestRho
        else:
            bestWalker = np.where(np.max(self.LnPosterior) == self.LnPosterior)[0][0]
            bestStep = np.where(np.max(self.LnPosterior) == self.LnPosterior)[1][0]
            self._bestRho = copy.copy(self.rootChain[:, bestWalker, bestStep])
            return self._bestRho

    @property
    def bestTau(self):
        if hasattr(self, '_bestTau'):
            return self._bestTau
        else:
            bestWalker = np.where(np.max(self.LnPosterior) == self.LnPosterior)[0][0]
            bestStep = np.where(np.max(self.LnPosterior) == self.LnPosterior)[1][0]
            self._bestTau = copy.copy(self.timescaleChain[:, bestWalker, bestStep])
            return self._bestTau

    def clear(self):
        if hasattr(self, '_rootChain'):
            del self._rootChain
        if hasattr(self, '_timescaleChain'):
            del self._timescaleChain
        if hasattr(self, '_bestTheta'):
            del self._bestTheta
        if hasattr(self, '_bestRho'):
            del self._bestRho
        if hasattr(self, '_bestTau'):
            del self._bestTau

    def smooth(self, observedLC, startT=None, stopT=None, tnum=None):
        if tnum is None:
            tnum = 0
        if observedLC.dtSmooth is None or observedLC.dtSmooth == 0.0:
            observedLC.dtSmooth = observedLC.mindt/10.0
        observedLC.pComp = 0
        observedLC.qComp = 0

        if not startT:
            startT = observedLC.t[0]
        if not stopT:
            stopT = observedLC.t[-1]
        t = sorted(observedLC.t.tolist() + np.linspace(start=startT, stop=stopT,
                   num=int(math.ceil((stopT - startT)/observedLC.dtSmooth)), endpoint=False).tolist())
        t = _f7(t)  # remove duplicates
        observedLC.numCadencesSmooth = len(t)

        observedLC.tSmooth = np.require(np.array(t), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])
        observedLC.xSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])
        observedLC.xerrSmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=[
                                           'F', 'A', 'W', 'O', 'E'])
        observedLC.ySmooth = np.require(np.zeros(observedLC.numCadencesSmooth), requirements=[
                                        'F', 'A', 'W', 'O', 'E'])
        observedLC.yerrSmooth = np.require(
            np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E'])
        observedLC.maskSmooth = np.require(
            np.zeros(observedLC.numCadencesSmooth), requirements=['F', 'A', 'W', 'O', 'E'])
        observedLC.XSmooth = np.require(np.zeros(observedLC.numCadencesSmooth*observedLC.pComp),
                                        requirements=['F', 'A', 'W', 'O', 'E'])
        observedLC.PSmooth = np.require(np.zeros(
            observedLC.numCadencesSmooth*observedLC.pComp*observedLC.pComp),
            requirements=['F', 'A', 'W', 'O', 'E'])
        unObsErr = 0.0
        obsCtr = 0
        for i in xrange(observedLC.numCadencesSmooth):
            if observedLC.tSmooth[i] == observedLC.t[obsCtr]:
                observedLC.xSmooth[i] = 0.0
                observedLC.xerrSmooth[i] = unObsErr
                observedLC.ySmooth[i] = observedLC.y[obsCtr]
                observedLC.yerrSmooth[i] = observedLC.yerr[obsCtr]
                observedLC.maskSmooth[i] = observedLC.mask[obsCtr]
                if obsCtr < observedLC.numCadences - 1:
                    obsCtr += 1
            else:
                observedLC.xSmooth[i] = 0.0
                observedLC.xerrSmooth[i] = unObsErr
                observedLC.ySmooth[i] = 0.0
                observedLC.yerrSmooth[i] = unObsErr
                observedLC.maskSmooth[i] = 0.0

        res = self._taskCython.smooth_Lightcurve(
            observedLC.numCadencesSmooth,
            observedLC.tSmooth,
            observedLC.xSmooth, tnum)
        observedLC._isSmoothed = True
        return res

    @property
    def auxillaryChain(self):
        if hasattr(self, '_auxillaryChain'):
            return np.reshape(self._auxillaryChain, newshape=(13, self._nwalkers, self._nsteps), order='F')
        else:
            self._auxillaryChain = np.require(np.zeros(13*self.nwalkers*self.nsteps),
                                              requirements=['F', 'A', 'W', 'O', 'E'])
            MBHBTask_cython.compute_Aux(self.ndims, self.nwalkers, self.nsteps,
                                        self.sigmaStars, self.H, self.rhoStars,
                                        self._Chain, self._auxillaryChain)
            return np.reshape(self._auxillaryChain, newshape=(13, self._nwalkers, self._nsteps), order='F')

    def plotscatter(self, dimx, dimy, truthx=None, truthy=None, labelx=None, labely=None,
                    best=False, median=False,
                    fig=-6, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if dimx < self.ndims and dimy < self.ndims:
            plt.scatter(self.timescaleChain[dimx, :, self.nsteps/2:],
                        self.timescaleChain[dimy, :, self.nsteps/2:],
                        c=self.LnPosterior[:, self.nsteps/2:], edgecolors='none')
            plt.colorbar()
            if best:
                loc0 = np.where(self.LnPosterior[self.nsteps/2:] ==
                                np.max(self.LnPosterior[self.nsteps/2:]))[0][0]
                loc1 = np.where(self.LnPosterior[self.nsteps/2:] ==
                                np.max(self.LnPosterior[self.nsteps/2:]))[1][0]
                plt.axvline(x=self.timescaleChain[dimx, loc0, loc1], c=r'#ffff00', label=r'Best %s'%(labelx))
                plt.axhline(y=self.timescaleChain[dimy, loc0, loc1], c=r'#ffff00', label=r'Best %s'%(labely))
            if median:
                medx = np.median(self.timescaleChain[dimx, :, self.nsteps/2:])
                medy = np.median(self.timescaleChain[dimy, :, self.nsteps/2:])
                plt.axvline(x=medx, c=r'#ff00ff', label=r'Median %s'%(labelx))
                plt.axhline(y=medy, c=r'#ff00ff', labely=r'Median %s'%(labely))
        if truthx is not None:
            plt.axvline(x=truthx, c=r'#00ff00')
        if truthy is not None:
            plt.axhline(y=truthy, c=r'#00ff00')
        if labelx is not None:
            plt.xlabel(labelx)
        if labely is not None:
            plt.ylabel(labely)
        if doShow:
            plt.show(False)
        return newFig

    def plotwalkers(self, dim, truth=None, label=None, fig=-7, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if dim < self.ndims:
            for i in xrange(self.nwalkers):
                plt.plot(self.timescaleChain[dim, i, :], c=r'#0000ff', alpha=0.1)
            plt.plot(np.median(self.timescaleChain[dim, :, :], axis=0), c=r'#ff0000')
            plt.fill_between(range(self.nsteps),
                             np.median(self.timescaleChain[dim, :, :], axis=0) -
                             np.std(self.timescaleChain[dim, :, :], axis=0),
                             np.median(self.timescaleChain[dim, :, :], axis=0) +
                             np.std(self.timescaleChain[dim, :, :], axis=0),
                             color=r'#ff0000', alpha=0.1)
        if truth is not None:
            plt.axhline(truth, c=r'#00ff00')
        plt.xlabel(r'step \#')
        if label is not None:
            plt.ylabel(label)
        if doShow:
            plt.show(False)
        return newFig

    def plottriangle(self, doShow=False, plot_contours=True, cmap='cubehelix'):
        orbitChain = copy.copy(self.timescaleChain[0:self.r, :, self.nsteps/2:])
        flatOrbitChain = np.swapaxes(orbitChain.reshape((self.ndims, -1), order='F'), axis1=0, axis2=1)
        orbitLabels = [r'$a_{1}$ (pc)', r'$a_{2}$ (pc)', r'$T$ (d)', r'$e$', r'$\Omega$ (deg.)', r'$i$ (deg)',
                       r'$\tau$ (d)', r'$F$']
        orbitExtents = [0.9, 0.9, (0.95*np.min(self.timescaleChain[2, :, self.nsteps/2:]),
                                   1.05*np.max(self.timescaleChain[2, :, self.nsteps/2:])), 0.9, 0.9, 0.9,
                        0.9, (0.85*np.min(self.timescaleChain[7, :, self.nsteps/2:]),
                              1.15*np.max(self.timescaleChain[7, :, self.nsteps/2:]))]
        newFigOrb = orbitalTr = kali.util.triangle.corner(flatOrbitChain, labels=orbitLabels,
                                                          show_titles=True,
                                                          title_fmt='.2e',
                                                          quantiles=[0.16, 0.5, 0.84],
                                                          extents=orbitExtents,
                                                          plot_contours=plot_contours,
                                                          plot_datapoints=False,
                                                          plot_contour_lines=False,
                                                          pcolor_cmap=cm.get_cmap(cmap),
                                                          verbose=False)

        auxChain = copy.copy(self.auxillaryChain[:, :, self.nsteps/2:])
        flatAuxChain = np.swapaxes(auxChain.reshape((13, -1), order='F'), axis1=0, axis2=1)
        auxLabels = [r'$a_{1}$ (pc)', r'$a_{2}$ (pc)', r'$T$ (d)', r'$e$',
                     r'$M_{12}$ ($10^{6} \times M_{\odot}$)', r'$M_{2}/M_{1}$',
                     r'$r_{\mathrm{Peribothron}}$ (pc)', r'$r_{\mathrm{Apobothron}}$ (pc)',
                     r'$r_{\mathrm{Schwarzschild}}$ (pc)',
                     r'$a_{\mathrm{Hard}}$ (pc)', r'$a_{\mathrm{GW}}$ (pc)', r'$T_{\mathrm{Hard}}$ (yr)',
                     r'$M_{\mathrm{Eject}}$ ($10^{6} \times M_{\odot}$)']
        auxExtents = [0.9, 0.9, (0.95*np.min(self.auxillaryChain[2, :, self.nsteps/2:]),
                                 1.05*np.max(self.auxillaryChain[2, :, self.nsteps/2:])), 0.9, 0.9, 0.9,
                      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        newFigAux = kali.util.triangle.corner(flatAuxChain, labels=auxLabels,
                                              show_titles=True,
                                              title_fmt='.2e',
                                              quantiles=[0.16, 0.5, 0.84],
                                              extents=auxExtents,
                                              plot_contours=plot_contours,
                                              plot_datapoints=False,
                                              plot_contour_lines=False,
                                              pcolor_cmap=cm.get_cmap(cmap),
                                              verbose=False)
        if doShow:
            plt.show(False)
        return newFigOrb, newFigAux
