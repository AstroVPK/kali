#!/usr/bin/env python
"""	Module to perform modulated C-ARMA modelling where the presence of a MBHB causes beaming.
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
import scipy.signal.spectral
import warnings as warnings
import matplotlib.pyplot as plt
import random
import pdb as pdb

from matplotlib import cm

import multi_key_dict
import gatspy.periodic

try:
    import rand
    import MBHBCARMATask_cython as MBHBCARMATask_cython
    import kali.lc
    from kali.util.mpl_settings import set_plot_params
    import kali.util.classproperty
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

fhgt = 10
fwid = 16
set_plot_params(useTex=True)

ln10 = math.log(10)


def pogsonFlux(mag, magErr):
    flux = 3631.0*math.pow(10.0, (-1.0*mag)/2.5)
    fluxErr = (ln10/2.5)*flux*magErr
    return flux, fluxErr


def _f7(seq):
    """http://tinyurl.com/angxm5"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def MAD(self, a):
    medianVal = np.median(a)
    b = np.copy(a)
    for i in range(a.shape[0]):
        b[i] = abs(b[i] - medianVal)
    return np.median(b)


def roots(p, q, Theta):
    """!
    \brief Theta -> Roots
    """
    r = MBHBCARMATask.r
    ARPoly = np.zeros(p + 1)
    ARPoly[0] = 1.0
    for i in xrange(p):
        ARPoly[i + 1] = Theta[r + i]
    ARRoots = np.array(np.roots(ARPoly))
    MAPoly = np.zeros(q + 1)
    for i in xrange(q + 1):
        MAPoly[i] = Theta[r + p + q - i]
    MARoots = np.array(np.roots(MAPoly))
    Rho = np.zeros(r + p + q + 1, dtype='complex128')
    for i in xrange(r):
        Rho[i] = Theta[i]
    for i in xrange(p):
        Rho[r + i] = ARRoots[i]
    for i in xrange(q):
        Rho[r + p + i] = MARoots[i]
    Sigma = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
    ThetaC = np.require(np.array(Theta), requirements=['F', 'A', 'W', 'O', 'E'])
    MBHBCARMATask_cython.get_Sigma(r, p, q, ThetaC, Sigma)
    Rho[r + p + q] = math.sqrt(Sigma[0])
    return Rho


def coeffs(p, q, Rho):
    """!
    \brief Roots -> Coeffs
    """
    r = MBHBCARMATask.r
    ARRoots = np.zeros(p, dtype='complex128')
    for i in xrange(p):
        ARRoots[i] = Rho[r + i]
    ARPoly = np.array(np.poly(ARRoots))
    MARoots = np.zeros(q, dtype='complex128')
    for i in xrange(q):
        MARoots[i] = Rho[r + p + i]
    if q == 0:
        MAPoly = np.ones(1)
    else:
        MAPoly = np.array(np.poly(MARoots))
    ThetaPrime = np.require(
        np.array(Rho[0:r].tolist() + ARPoly[1:].tolist() + MAPoly.tolist()[::-1]),
        requirements=['F', 'A', 'W', 'O', 'E'])
    SigmaPrime = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
    MBHBCARMATask_cython.get_Sigma(r, p, q, ThetaPrime, SigmaPrime)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Sigma00 = math.pow(Rho[r + p + q], 2.0)
    try:
        bQ = math.sqrt(Sigma00/SigmaPrime[0])
    except ValueError:
        bQ = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in xrange(q + 1):
            MAPoly[i] = bQ*MAPoly[i]
    Theta = np.zeros(r + p + q + 1, dtype='float64')
    for i in xrange(r):
        Theta[i] = Rho[i]
    for i in xrange(p):
        Theta[r + i] = ARPoly[i + 1].real
    for i in xrange(q + 1):
        Theta[r + p + i] = MAPoly[q - i].real
    return Theta


def timescales(p, q, Rho):
    """!
    \brief Roots -> Timescales
    """
    r = MBHBCARMATask.r
    imagPairs = 0
    for i in xrange(p):
        if Rho[r + i].imag != 0.0:
            imagPairs += 1
    numImag = imagPairs/2
    numReal = numImag + (p - imagPairs)
    decayTimescales = np.zeros(numReal)
    oscTimescales = np.zeros(numImag)
    realRoots = set(Rho[r:r + p].real)
    imagRoots = set(abs(Rho[r:r + p].imag)).difference(set([0.0]))
    realAR = sorted([1.0/abs(x) for x in realRoots])
    imagAR = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
    imagPairs = 0
    for i in xrange(q):
        if Rho[r + p + i].imag != 0.0:
            imagPairs += 1
    numImag = imagPairs/2
    numReal = numImag + (q - imagPairs)
    decayTimescales = np.zeros(numReal)
    oscTimescales = np.zeros(numImag)
    realRoots = set(Rho[r + p:r + p + q].real)
    imagRoots = set(abs(Rho[r + p:r + p + q].imag)).difference(set([0.0]))
    realMA = sorted([1.0/abs(x) for x in realRoots])
    imagMA = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
    return np.array(Rho[0:r].tolist() + realAR + imagAR + realMA + imagMA + [Rho[r + p + q]])


class MBHBCARMATask(object):

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

    def __init__(self, p, q, nthreads=psutil.cpu_count(logical=True), nburn=1000000,
                 nwalkers=25*psutil.cpu_count(logical=True), nsteps=250, maxEvals=10000, xTol=0.001,
                 mcmcA=2.0):
        try:
            assert p > q, r'p must be greater than q'
            assert p >= 1, r'p must be greater than or equal to 1'
            assert isinstance(p, int), r'p must be an integer'
            assert q >= 0, r'q must be greater than or equal to 0'
            assert isinstance(q, int), r'q must be an integer'
            assert nthreads > 0, r'nthreads must be greater than 0'
            assert isinstance(nthreads, int), r'nthreads must be an integer'
            assert nburn >= 0, r'nburn must be greater than or equal to 0'
            assert isinstance(nburn, int), r'nburn must be an integer'
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
            self._ndims = self._r + self._p + self._q + 1
            self._nthreads = nthreads
            self._nburn = nburn
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
            self._taskCython = MBHBCARMATask_cython.MBHBCARMATask_cython(self._p, self._q, self._nthreads,
                                                                         self._nburn)
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
        self._taskCython = MBHBCARMATask_cython.MBHBCARMATask_cython(self._p, self._q, self._nthreads,
                                                                     self._nburn)

    @kali.util.classproperty.ClassProperty
    @classmethod
    def r(self):
        return self._r

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        try:
            assert value > self._q, r'p must be greater than q'
            assert value >= 1, r'p must be greater than or equal to 1'
            assert isinstance(value, int), r'p must be an integer'
            self._taskCython.reset_CARMATask(value, self._q, self._nburn)
            self._p = value
            self._ndims = self._p + self._q + 1
            self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps),
                                     requirements=['F', 'A', 'W', 'O', 'E'])
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        try:
            assert value > self._q, r'p must be greater than q'
            assert value >= 0, r'q must be greater than or equal to 0'
            assert isinstance(value, int), r'q must be an integer'
            self._taskCython.reset_CARMATask(self._p, value, self._nburn)
            self._q = value
            self._ndims = self._p + self._q + 1
            self._Chain = np.require(np.zeros(self._ndims*self._nwalkers*self._nsteps),
                                     requirements=['F', 'A', 'W', 'O', 'E'])
        except AssertionError as err:
            raise AttributeError(str(err))

    @property
    def nthreads(self):
        return self._nthreads

    @property
    def nburn(self):
        return self._nburn

    @nburn.setter
    def nburn(self, nburnVal):
        try:
            assert value >= 0, r'nburn must be greater than or equal to 0'
            assert isinstance(value, int), r'nburn must be an integer'
            self._taskCython.reset_CARMATask(self._p, self._q, value)
            self._nburn = value
        except AssertionError as err:
            raise AttributeError(str(err))

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
            Chain = self.Chain
            self._rootChain = np.require(
                np.zeros((self._ndims, self._nwalkers, self._nsteps), dtype='complex128'),
                requirements=['F', 'A', 'W', 'O', 'E'])
            for stepNum in xrange(self._nsteps):
                for walkerNum in xrange(self._nwalkers):
                    self._rootChain[:, walkerNum, stepNum] = roots(
                        self._p, self._q, Chain[:, walkerNum, stepNum])
        return self._rootChain

    @property
    def timescaleChain(self):
        if hasattr(self, '_timescaleChain'):
            return self._timescaleChain
        else:
            rootChain = self.rootChain
            self._timescaleChain = np.require(
                np.zeros((self._ndims, self._nwalkers, self._nsteps), dtype='float64'),
                requirements=['F', 'A', 'W', 'O', 'E'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for stepNum in xrange(self._nsteps):
                    for walkerNum in xrange(self._nwalkers):
                        self._timescaleChain[:, walkerNum, stepNum] = timescales(
                            self._p, self._q, rootChain[:, walkerNum, stepNum])
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
        return "kali.mbhbcarma.MBHBCARMATask(%d, %d, %d, %d, %d, %d, %d, %f)"%(self._p, self._q,
                                                                               self._nthreads, self._nburn,
                                                                               self._nwalkers, self._nsteps,
                                                                               self._maxEvals, self._xTol)

    def __str__(self):
        line = 'r: %d; p: %d; q: %d; ndims: %d\n'%(self._r, self._p, self._q, self._ndims)
        line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
        line += 'nburn (Number of light curve steps to burn): %d\n'%(self._nburn)
        line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
        line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
        line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for MCMC):\
        %d\n'%(self._maxEvals)
        line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)
        return line

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if ((self._r == other._r) and (self._p == other.p) and (self._q == other.q) and
                    (self._nthreads == other.nthreads) and (self._nburn == other.nburn) and
                    (self._nwalkers == other.nwalkers) and (self._nsteps == other.nsteps) and
                    (self._maxEvals == other.maxEvals) and (self.xTol == other.xTol)):
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

    def reset(self, p=None, q=None, nburn=None, nwalkers=None, nsteps=None):
        if p is None:
            p = self._p
        if q is None:
            q = self._q
        if nburn is None:
            nburn = self._nburn
        if nwalkers is None:
            nwalkers = self._nwalkers
        if nsteps is None:
            nsteps = self._nsteps
        try:
            assert p > q, r'p must be greater than q'
            assert p >= 1, r'p must be greater than or equal to 1'
            assert isinstance(p, int), r'p must be an integer'
            assert q >= 0, r'q must be greater than or equal to 0'
            assert isinstance(q, int), r'q must be an integer'
            assert nburn >= 0, r'nburn must be greater than or equal to 0'
            assert isinstance(nburn, int), r'nburn must be an integer'
            assert nwalkers > 0, r'nwalkers must be greater than 0'
            assert isinstance(nwalkers, int), r'nwalkers must be an integer'
            assert nsteps > 0, r'nsteps must be greater than 0'
            assert isinstance(nsteps, int), r'nsteps must be an integer'
            self._taskCython.reset_MBHBCARMATask(p, q, nburn)
            self._p = p
            self._q = q
            self._ndims = self._p + self._q + 1
            self._nburn = nburn
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
        assert dt > 0.0, r'dt must be greater than 0.0'
        assert isinstance(dt, float), r'dt must be a float'
        assert Theta.shape == (self._ndims,), r'Too many coefficients in Theta'
        return self._taskCython.set_System(dt, Theta, tnum)

    def dt(self, tnum=None):
        if tnum is None:
            tnum = 0
        return self._taskCython.get_dt(tnum)

    def Theta(self, tnum=None):
        if tnum is None:
            tnum = 0
        Theta = np.require(np.zeros(self._ndims), requirements=['F', 'A', 'W', 'O', 'E'])
        self._taskCython.get_Theta(Theta, tnum)
        return Theta

    def list(self):
        setSystems = np.require(
            np.zeros(self._nthreads, dtype='int32'), requirements=['F', 'A', 'W', 'O', 'E'])
        self._taskCython.get_setSystemsVec(setSystems)
        return setSystems.astype(np.bool_)

    def show(self, tnum=None):
        if tnum is None:
            tnum = 0
        self._taskCython.print_System(tnum)

    def Sigma(self, tnum=None):
        if tnum is None:
            tnum = 0
        Sigma = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E'])
        self._taskCython.get_Sigma(Sigma, tnum)
        return np.reshape(Sigma, newshape=(self._p, self._p), order='F')

    def X(self, newX=None, tnum=None):
        if tnum is None:
            tnum = 0
        if not newX:
            X = np.zeros(self._p)
            self._taskCython.get_X(X, tnum)
            return np.reshape(X, newshape=(self._p), order='F')
        else:
            self._taskCython.set_X(np.reshape(X, newshape=(self._p), order='F'), tnum)
            return newX

    def P(self, newP=None, tnum=None):
        if tnum is None:
            tnum = 0
        if not newP:
            P = np.zeros(self._p*self._p)
            self._taskCython.get_P(P, tnum)
            return np.reshape(P, newshape=(self._p, self._p), order='F')
        else:
            self._taskCython.set_P(np.reshape(P, newshape=(self._p*self._p), order='F'), tnum)
            return newP

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

    def beam(self, duration=None, tIn=None, tnum=None, tolIR=1.0e-3, startT=0.0,
             fracIntrinsicVar=0.15, fracNoiseToSignal=0.001,
             maxSigma=2.0, minTimescale=2.0, maxTimescale=0.5):
        if tnum is None:
            tnum = 0
        if tIn is None and duration is not None:
            numCadences = int(round(float(duration)/self._taskCython.get_dt(threadNum=tnum)))
            intrinsicLC = kali.lc.mockLC(name='', band='', numCadences=numCadences,
                                         deltaT=self._taskCython.get_dt(threadNum=tnum), tolIR=tolIR,
                                         startT=startT, fracIntrinsicVar=fracIntrinsicVar,
                                         fracNoiseToSignal=fracNoiseToSignal, maxSigma=maxSigma,
                                         minTimescale=minTimescale, maxTimescale=maxTimescale,
                                         pSim=self._p, qSim=self._q)
        elif duration is None and tIn is not None:
            numCadences = tIn.shape[0]
            t = np.require(np.array(tIn), requirements=['F', 'A', 'W', 'O', 'E'])
            intrinsicLC = kali.lc.mockLC(name='', band='', tIn=t, startT=startT,
                                         fracIntrinsicVar=fracIntrinsicVar,
                                         fracNoiseToSignal=fracNoiseToSignal, tolIR=tolIR, maxSigma=maxSigma,
                                         minTimescale=minTimescale, maxTimescale=maxTimescale,
                                         pSim=self._p, qSim=self._q)
        self._taskCython.make_BeamedLC(
            intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.startT, intrinsicLC.fracIntrinsicVar,
            intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr,
            intrinsicLC.mask, intrinsicLC.XSim, intrinsicLC.PSim, threadNum=tnum)
        intrinsicLC._simulatedCadenceNum = numCadences - 1
        intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
        return intrinsicLC

    def simulate(self, duration=None, tIn=None, tolIR=1.0e-3, startT=0.0,
                 fracIntrinsicVar=0.15, fracNoiseToSignal=0.001,
                 maxSigma=2.0, minTimescale=2.0, maxTimescale=0.5, burnSeed=None, distSeed=None,
                 tnum=None):
        if tnum is None:
            tnum = 0
        if tIn is None and duration is not None:
            numCadences = int(round(float(duration)/self._taskCython.get_dt(threadNum=tnum)))
            intrinsicLC = kali.lc.mockLC(name='', band='', numCadences=numCadences,
                                         deltaT=self._taskCython.get_dt(threadNum=tnum), tolIR=tolIR,
                                         startT=startT, fracIntrinsicVar=fracIntrinsicVar,
                                         fracNoiseToSignal=fracNoiseToSignal, maxSigma=maxSigma,
                                         minTimescale=minTimescale, maxTimescale=maxTimescale,
                                         pSim=self._p, qSim=self._q)
        elif duration is None and tIn is not None:
            numCadences = tIn.shape[0]
            t = np.require(np.array(tIn), requirements=['F', 'A', 'W', 'O', 'E'])
            intrinsicLC = kali.lc.mockLC(name='', band='', tIn=t, startT=startT,
                                         fracIntrinsicVar=fracIntrinsicVar,
                                         fracNoiseToSignal=fracNoiseToSignal, tolIR=tolIR, maxSigma=maxSigma,
                                         minTimescale=minTimescale, maxTimescale=maxTimescale,
                                         pSim=self._p, qSim=self._q)
            for i in xrange(intrinsicLC.numCadences):
                intrinsicLC.mask[i] = 1.0
        randSeed = np.zeros(1, dtype='uint32')
        if burnSeed is None:
            rand.rdrand(randSeed)
            burnSeed = randSeed[0]
        if distSeed is None:
            rand.rdrand(randSeed)
            distSeed = randSeed[0]
        self._taskCython.make_IntrinsicLC(
            intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.startT, intrinsicLC.fracIntrinsicVar,
            intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr,
            intrinsicLC.mask, intrinsicLC.XSim, intrinsicLC.PSim, burnSeed, distSeed, threadNum=tnum)
        intrinsicLC._simulatedCadenceNum = numCadences - 1
        intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
        return intrinsicLC

    '''def extend(
            self, intrinsicLC, duration=None, tIn=None, gap=None, distSeed=None, noiseSeed=None, tnum=None):
        if tnum is None:
            tnum = 0
        randSeed = np.zeros(1, dtype='uint32')
        if distSeed is None:
            rand.rdrand(randSeed)
            distSeed = randSeed[0]
        if noiseSeed is None:
            rand.rdrand(randSeed)
            noiseSeed = randSeed[0]
        if intrinsicLC.pSim != self.p:
            intrinsicLC.pSim = self.p
        if intrinsicLC.qSim != self.q:
            intrinsicLC.qSim = self.q
        if gap is None:
            gapSize = 0.0
        else:
            gapSize = gap
        oldNumCadences = intrinsicLC.numCadences
        gapNumCadences = int(round(float(gapSize)/self._taskCython.get_dt(threadNum=tnum)))
        if tIn is None and duration is not None:
            extraNumCadences = int(round(float(duration + gapSize)/self._taskCython.get_dt(threadNum=tnum)))
        elif duration is None and gap is None and tIn is not None:
            extraNumCadences = np.array(tIn).shape[0]
        else:
            raise ValueError('Cannot specify both tIn and gap at the same time')
        newNumCadences = intrinsicLC.numCadences + extraNumCadences
        newt = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newx = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newy = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newyerr = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        newmask = np.require(np.zeros(newNumCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(intrinsicLC.numCadences):
            newt[i] = intrinsicLC.t[i]
            newx[i] = intrinsicLC.x[i]
            newy[i] = intrinsicLC.y[i]
            newyerr[i] = intrinsicLC.yerr[i]
            newmask[i] = intrinsicLC.mask[i]
        if tIn is None and duration is not None:
            for i in xrange(intrinsicLC.numCadences, newNumCadences):
                newt[i] = newt[intrinsicLC.numCadences - 1] + gapSize + \
                    (i - intrinsicLC.numCadences + 1)*self._taskCython.get_dt(threadNum=tnum)
                newmask[i] = 1.0
        elif duration is None and gap is None and tIn is not None:
            for i in xrange(intrinsicLC.numCadences, newNumCadences):
                newt[i] = tIn[i - intrinsicLC.numCadences]
                newmask[i] = 1.0
        else:
            raise ValueError('Cannot specify both tIn and gap at the same time')
        intrinsicLC._numCadences = newNumCadences
        self._taskCython.extend_IntrinsicLC(
            intrinsicLC.numCadences, intrinsicLC._simulatedCadenceNum, intrinsicLC._tolIR,
            intrinsicLC._fracIntrinsicVar, intrinsicLC._fracNoiseToSignal, newt, newx, newy, newyerr, newmask,
            intrinsicLC.XSim, intrinsicLC.PSim, distSeed, noiseSeed, threadNum=tnum)
        if gap is not None:
            old, gap, new = np.split(newt, [oldNumCadences, oldNumCadences + gapNumCadences])
            newt = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
            old, gap, new = np.split(newx, [oldNumCadences, oldNumCadences + gapNumCadences])
            newx = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
            old, gap, new = np.split(newy, [oldNumCadences, oldNumCadences + gapNumCadences])
            newy = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
            old, gap, new = np.split(newyerr, [oldNumCadences, oldNumCadences + gapNumCadences])
            newyerr = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
            old, gap, new = np.split(newmask, [oldNumCadences, oldNumCadences + gapNumCadences])
            newmask = np.require(np.concatenate((old, new)), requirements=['F', 'A', 'W', 'O', 'E'])
        intrinsicLC._simulatedCadenceNum = newt.shape[0] - 1
        intrinsicLC._numCadences = newt.shape[0]
        intrinsicLC.t = newt
        intrinsicLC.x = newx
        intrinsicLC.y = newy
        intrinsicLC.yerr = newyerr
        intrinsicLC.mask = newmask

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
            intrinsicLC._stderr = 0.0'''

    def observe(self, intrinsicLC, noiseSeed=None, tnum=None):
        if tnum is None:
            tnum = 0
        randSeed = np.zeros(1, dtype='uint32')
        if noiseSeed is None:
            rand.rdrand(randSeed)
            noiseSeed = randSeed[0]
        if intrinsicLC._observedCadenceNum == -1:
            self._taskCython.add_ObservationNoise(
                intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar,
                intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr,
                intrinsicLC.mask, noiseSeed, threadNum=tnum)
        '''else:
            self._taskCython.extend_ObservationNoise(
                intrinsicLC.numCadences, intrinsicLC.observedCadenceNum, intrinsicLC.tolIR,
                intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x,
                intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum=tnum)
            intrinsicLC._observedCadenceNum = intrinsicLC._numCadences - 1'''
        intrinsicLC._statistics()

    def logPrior(self, observedLC, forced=True, widthT=0.01, widthF=0.05, tnum=None):
        if tnum is None:
            tnum = 0
        periodEst = self.estimate(observedLC)
        observedLC._logPrior = self._taskCython.compute_LnPrior(observedLC.numCadences, observedLC.meandt,
                                                                observedLC.tolIR,
                                                                observedLC.startT,
                                                                observedLC.maxSigma*observedLC.std,
                                                                observedLC.minTimescale*observedLC.mindt,
                                                                observedLC.maxTimescale*observedLC.T,
                                                                np.min(observedLC.y), np.max(observedLC.y),
                                                                observedLC.t, observedLC.x,
                                                                observedLC.y, observedLC.yerr,
                                                                observedLC.mask,
                                                                periodEst, widthT*periodEst,
                                                                observedLC.mean, widthF*observedLC.mean,
                                                                tnum)
        return observedLC._logPrior

    def logLikelihood(self, observedLC, widthT=0.01, widthF=0.05, forced=True, tnum=None):
        if tnum is None:
            tnum = 0
        observedLC.pComp = self.p
        observedLC.qComp = self.q
        periodEst = self.estimate(observedLC)
        observedLC._logPrior = self.logPrior(observedLC, widthT=widthT, widthF=widthF,
                                             forced=forced, tnum=tnum)
        if forced is True:
            observedLC._computedCadenceNum = -1
        if observedLC._computedCadenceNum == -1:
            for rowCtr in xrange(observedLC.pComp):
                observedLC.XComp[rowCtr] = 0.0
                for colCtr in xrange(observedLC.pComp):
                    observedLC.PComp[rowCtr + observedLC.pComp*colCtr] = 0.0
            observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(
                observedLC.numCadences, observedLC._computedCadenceNum, observedLC.tolIR, observedLC.startT,
                observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask,
                observedLC.XComp, observedLC.PComp,
                periodEst, widthT*periodEst,
                observedLC.mean, widthF*observedLC.mean,
                tnum)
            observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
            observedLC._computedCadenceNum = observedLC.numCadences - 1
        elif observedLC._computedCadenceNum == observedLC.numCadences - 1:
            pass
        '''else:
            observedLC._logLikelihood = self._taskCython.update_LnLikelihood(
                observedLC.numCadences, observedLC._computedCadenceNum, observedLC._logLikelihood,
                observedLC.tolIR, observedLC.t, observedLC.x, observedLC.y - observedLC.mean, observedLC.yerr,
                observedLC.mask, observedLC.XComp, observedLC.PComp, tnum)
            observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
            observedLC._computedCadenceNum = observedLC.numCadences - 1'''
        return observedLC._logLikelihood

    def logPosterior(self, observedLC, forced=True, tnum=None):
        lnLikelihood = self.logLikelihood(observedLC, forced=forced, tnum=tnum)
        return observedLC._logPosterior

    '''def acvf(self, start=0.0, stop=100.0, num=100, endpoint=True, base=10.0, spacing='linear'):
        if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
            lags = np.logspace(np.log10(start)/np.log10(base), np.log10(
                stop)/np.log10(base), num=num, endpoint=endpoint, base=base)
        elif spacing.lower() in ['linear', 'lin']:
            lags = np.linspace(start, stop, num=num, endpoint=endpoint)
        else:
            raise RuntimeError('Unable to parse spacing')
        acvf = np.zeros(num)
        self._taskCython.compute_ACVF(num, lags, acvf)
        return lags, acvf

    def acf(self, start=0.0, stop=100.0, num=100, endpoint=True, base=10.0, spacing='linear'):
        if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
            lags = np.logspace(np.log10(start)/np.log10(base), np.log10(
                stop)/np.log10(base), num=num, endpoint=endpoint, base=base)
        elif spacing.lower() in ['linear', 'lin']:
            lags = np.linspace(start, stop, num=num, endpoint=endpoint)
        else:
            raise RuntimeError('Unable to parse spacing')
        acvf = np.zeros(num)
        acf = np.zeros(num)
        self._taskCython.compute_ACVF(num, lags, acvf)
        acf = acvf/acvf[0]
        return lags, acf

    def sf(self, start=0.0, stop=100.0, num=100, endpoint=True, base=10.0, spacing='linear'):
        if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
            lags = np.logspace(np.log10(start)/np.log10(base), np.log10(
                stop)/np.log10(base), num=num, endpoint=endpoint, base=base)
        elif spacing.lower() in ['linear', 'lin']:
            lags = np.linspace(start, stop, num=num, endpoint=endpoint)
        else:
            raise RuntimeError('Unable to parse spacing')
        acvf = np.zeros(num)
        sf = np.zeros(num)
        self._taskCython.compute_ACVF(num, lags, acvf)
        sf = 2.0*(acvf[0] - acvf)
        return lags, sf

    def plotacvf(self, fig=-2, LC=None, newdt=None, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if LC is not None:
            lagsM, acvfM = self.acvf(start=LC.dt, stop=LC.T, num=1000, spacing='linear')
        else:
            lagsM, acvfM = self.acvf(start=0.0, stop=1000.0, num=1000, spacing='linear')
        plt.plot(lagsM, acvfM, label=r'model Autocovariance Function', color='#984ea3', zorder=5)
        if LC is not None:
            if np.sum(LC.y) != 0.0:
                lagsE, acvfE, acvferrE = LC.acvf(newdt)
                if np.sum(acvfE) != 0.0:
                    plt.errorbar(lagsE[1:], acvfE[1:], acvferrE[1:], label=r'obs. Autocovariance Function',
                                 fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=0)
                    plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\log ACVF$')
        plt.title(r'Autocovariance Function')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig

    def plotacf(self, fig=-3, LC=None, newdt=None, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if LC is not None:
            lagsM, acfM = self.acf(start=LC.dt, stop=LC.T, num=1000, spacing='linear')
        else:
            lagsM, acfM = self.acf(start=0.0, stop=1000.0, num=1000, spacing='linear')
        plt.plot(lagsM, acfM, label=r'model Autocorrelation Function', color='#984ea3', zorder=5)
        if LC is not None:
            if np.sum(LC.y) != 0.0:
                lagsE, acfE, acferrE = LC.acf(newdt)
                if np.sum(acfE) != 0.0:
                    plt.errorbar(lagsE[1:], acfE[1:], acferrE[1:], label=r'obs. Autocorrelation Function',
                                 fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=0)
                    plt.xlim(lagsE[1], lagsE[-1])
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\log ACF$')
        plt.title(r'Autocorrelation Function')
        plt.legend(loc=3)
        plt.ylim(-1.0, 1.0)
        if doShow:
            plt.show(False)
        return newFig

    def plotsf(self, fig=-4, LC=None, newdt=None, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if LC is not None and np.sum(LC.y) != 0.0:
            lagsE, sfE, sferrE = LC.sf(newdt)
            lagsM, sfM = self.sf(start=lagsE[1], stop=lagsE[-1], num=1000, spacing='log')
        else:
            lagsM, sfM = self.sf(start=0.001, stop=1000.0, num=1000, spacing='log')
        plt.plot(np.log10(lagsM[1:]), np.log10(
            sfM[1:]), label=r'model Structure Function', color='#984ea3', zorder=5)
        if LC is not None:
            if np.sum(LC.y) != 0.0:
                if np.sum(sfE) != 0.0:
                    plt.scatter(
                        np.log10(lagsE[np.where(sfE != 0.0)[0]]), np.log10(sfE[np.where(sfE != 0.0)[0]]),
                        marker='o', label=r'obs. Structure Function', color='#ff7f00', edgecolors='none',
                        zorder=0)
        plt.xlim(math.log10(lagsM[1]), math.log10(lagsM[-1]))
        plt.xlabel(r'$\delta t$')
        plt.ylabel(r'$\log SF$')
        plt.title(r'Structure Function')
        plt.legend(loc=2)
        if doShow:
            plt.show(False)
        return newFig

    def _psddenominator(self, freqs, order):
        nfreqs = freqs.shape[0]
        aList = self.Theta()[0:self.p].tolist()
        aList.insert(0, 1.0)
        psddenominator = np.zeros(nfreqs)
        if ((order % 2 == 1) or (order <= -1) or (order > 2*self.p)):
            aList.pop(0)
            return PSDVals
        else:
            for freq in xrange(nfreqs):
                val = 0.0
                for i in xrange(self.p + 1):
                    j = 2*self.p - i - order
                    if ((j >= 0) and (j < self.p + 1)):
                        val += (aList[i]*aList[j]*((2.0*math.pi*1j*freqs[freq])**(
                            2*self.p - (i + j)))*pow(-1.0, self.p - j)).real
                    psddenominator[freq] = val
            aList.pop(0)
            return psddenominator

    def _psdnumerator(self, freqs, order):
        nfreqs = freqs.shape[0]
        bList = self.Theta()[self.p:self.p+self.q+1].tolist()
        psdnumerator = np.zeros(nfreqs)
        if ((order % 2 == 1) or (order <= -1) or (order > 2*self.q)):
            return psdnumerator
        else:
            for freq in xrange(nfreqs):
                val = 0.0
                for i in xrange(self.q + 1):
                    j = 2*self.q - i - order
                    if ((j >= 0) and (j < self.q + 1)):
                        val += (bList[i]*bList[j]*((2.0*math.pi*1j*freqs[freq])**(
                            2*self.q - (i + j)))*pow(-1.0, self.q - j)).real
                    psdnumerator[freq] = val
            return psdnumerator

    def psd(self, start=0.1, stop=100.0, num=100, endpoint=True, base=10.0, spacing='log'):
        if spacing.lower() in ['log', 'logarithm', 'ln', 'log10']:
            freqs = np.logspace(np.log10(start)/np.log10(base), np.log10(
                stop)/np.log10(base), num=num, endpoint=endpoint, base=base)
        elif spacing.lower() in ['linear', 'lin']:
            freqs = np.linspace(start, stop, num=num, endpoint=endpoint)
        else:
            raise RuntimeError('Unable to parse spacing')
        maxDenomOrder = 2*self.p
        maxNumerOrder = 2*self.q

        psdnumeratorcomponent = np.zeros((num, (maxNumerOrder/2) + 1))
        psddenominatorcomponent = np.zeros((num, (maxDenomOrder/2) + 1))

        psdnumerator = np.zeros(num)
        psddenominator = np.zeros(num)
        psd = np.zeros(num)

        for orderVal in xrange(0, maxNumerOrder + 1, 2):
            psdnumeratorcomponent[:, orderVal/2] = self._psdnumerator(freqs, orderVal)

        for orderVal in xrange(0, maxDenomOrder + 1, 2):
            psddenominatorcomponent[:, orderVal/2] = self._psddenominator(freqs, orderVal)

        for freq in xrange(num):
            for orderVal in xrange(0, maxNumerOrder + 1, 2):
                psdnumerator[freq] += psdnumeratorcomponent[freq, orderVal/2]
            for orderVal in xrange(0, maxDenomOrder + 1, 2):
                psddenominator[freq] += psddenominatorcomponent[freq, orderVal/2]
            psd[freq] = psdnumerator[freq]/psddenominator[freq]
        return freqs, psd, psdnumerator, psddenominator, psdnumeratorcomponent, psddenominatorcomponent

    def plotpsd(self, fig=-5, LC=None, doShow=False, clearFig=True):
        newFig = plt.figure(fig, figsize=(fwid, fhgt))
        if clearFig:
            plt.clf()
        if LC is not None:
            start = LC.mindt
            stop = LC.T
        else:
            start = 1.0e-4
            stop = 1.0e3
        freqsM, psdM, psdNumer, psdDenom, psdNumerComp, psdDenomComp = self.psd(
            start=start, stop=stop, num=1000, spacing='log')
        plt.plot(np.log10(freqsM[1:]), np.log10(
            psdM[1:]), label=r'$\ln PSD$', color='#000000', zorder=5, linewidth=6)
        plt.plot(np.log10(freqsM[1:]), np.log10(psdNumer[1:]),
                 label=r'$\ln PSD_{\mathrm{numerator}}$', color='#1f78b4', zorder=0, linewidth=4)
        plt.plot(np.log10(freqsM[1:]), -np.log10(psdDenom[1:]),
                 label=r'$-\ln PSD_{\mathrm{denominator}}$', color='#e31a1c', zorder=0, linewidth=4)
        for i in xrange(psdNumerComp.shape[1]):
            plt.plot(np.log10(freqsM[1:]), np.log10(psdNumerComp[1:, i]),
                     color='#a6cee3', zorder=0, linewidth=2, linestyle=r'dashed')
            plt.annotate(
                r'$\nu^{%d}$'%(2*i), xy=(np.log10(freqsM[25]), np.log10(psdNumerComp[25, i])),
                xycoords='data', xytext=(np.log10(freqsM[25]) + 0.25, np.log10(psdNumerComp[25, i]) + 0.5),
                textcoords='data', arrowprops=dict(
                    arrowstyle='->', connectionstyle='angle3, angleA = 0, angleB = 90'),
                ha='center', va='center', multialignment='center', zorder=100)
        for i in xrange(psdDenomComp.shape[1]):
            plt.plot(np.log10(freqsM[1:]), -np.log10(psdDenomComp[1:, i]),
                     color='#fb9a99', zorder=0, linewidth=2, linestyle=r'dashed')
            plt.annotate(
                r'$\nu^{%d}$'%(-2*i), xy=(np.log10(freqsM[-25]), -np.log10(psdDenomComp[-25, i])),
                xycoords='data', xytext=(np.log10(freqsM[-25]) - 0.25, -np.log10(psdDenomComp[-25, i]) - 0.5),
                textcoords='data', arrowprops=dict(
                    arrowstyle='->', connectionstyle='angle3, angleA = 0, angleB = 90'),
                ha='center', va='center', multialignment='center', zorder=100)
        plt.xlim(math.log10(freqsM[1]), math.log10(freqsM[-1]))
        plt.xlabel(r'$\log \nu$')
        plt.ylabel(r'$\log PSD$')
        plt.title(r'Power Spectral Density')
        plt.legend(loc=3)
        if doShow:
            plt.show(False)
        return newFig'''

    def estimate(self, observedLC):
        """!
        Estimate period using gatspy
        """
        '''
        if observedLC.numCadences > 50:
            model = gatspy.periodic.LombScargleFast()
        else:
            model = gatspy.periodic.LombScargle()
        model.optimizer.set(quiet=True, period_range=(observedLC.mindt/10.0, observedLC.T*10.0))
        model.fit(observedLC.t,
                  observedLC.y,
                  observedLC.yerr)
        periodEst = model.best_period
        if periodEst < observedLC.mindt/10.0 or periodEst > observedLC.T*10.0:
            scaled_y = (observedLC.y - observedLC.mean)/observedLC.std
            freqs = np.logspace(10.0/math.log10(observedLC.mindt), math.log10(observedLC.T/10.0),
                                observedLC.numCadences*100)
            periodogram = scipy.signal.spectral.lombscargle(observedLC.t, scaled_y, freqs)
            periodEst = 2.0*math.pi/freqs[np.argmax(periodogram)]
        '''
        scaled_y = (observedLC.y - observedLC.mean)/observedLC.std
        freqs = np.logspace(10.0/math.log10(observedLC.mindt), math.log10(observedLC.T/10.0),
                            observedLC.numCadences*100)
        periodogram = scipy.signal.spectral.lombscargle(observedLC.t, scaled_y, freqs)
        periodEst = 2.0*math.pi/freqs[np.argmax(periodogram)]
        return periodEst

    def guess(self, periodEst):
        notGood = True
        while notGood:
            eccentricityGuess = random.uniform(0.0, 1.0)
            m1Val = math.pow(10.0, random.uniform(4.0, 10.0))
            m2Val = math.pow(10.0, random.uniform(4.0, math.log10(m1Val)))
            totalMassVal = m1Val + m2Val
            axisRatioVal = m2Val/m1Val
            aTotCube = (self.G*math.pow(periodEst*self.Day, 2.0)*totalMassVal*self.SolarMass)/self.fourPiSq
            totalAxisVal = math.pow(aTotCube, 1.0/3.0)
            a1Guess = ((axisRatioVal*totalAxisVal)/(1 + axisRatioVal))/self.Parsec
            a2Guess = (totalAxisVal/(1 + axisRatioVal))/self.Parsec
            rS1Val = (2.0*self.G*m1Val*self.SolarMass)/(pow(self.c, 2.0))
            rS2Val = (2.0*self.G*m2Val*self.SolarMass)/(pow(self.c, 2.0))
            rPeribothronTotVal = (a1Guess + a2Guess)*self.Parsec*(1.0 - eccentricityGuess)
            if rPeribothronTotVal > 10.0*(rS1Val + rS2Val):
                notGood = False
                break
        return a1Guess, a2Guess, eccentricityGuess

    def fit(self, observedLC, widthT=0.01, widthF=0.05,
            zSSeed=None, walkerSeed=None, moveSeed=None, xSeed=None):
        observedLC.pComp = self.p
        observedLC.qComp = self.q
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
        minT = observedLC.mindt*observedLC.minTimescale
        maxT = observedLC.T*observedLC.maxTimescale
        minTLog10 = math.log10(minT)
        maxTLog10 = math.log10(maxT)

        periodEst = self.estimate(observedLC)
        for walkerNum in xrange(self.nwalkers):
            noSuccess = True
            sigmaFactor = 1.0e0
            expVal = ((maxTLog10 - minTLog10)*np.random.random(self._p + self._q + 1) + minTLog10)
            a1Guess, a2Guess, eccentricityGuess = self.guess(periodEst)
            RhoGuess = np.require(np.array([a1Guess, a2Guess, periodEst, eccentricityGuess,
                                            random.uniform(0.0, 360.0), random.uniform(0.0, 90.0),
                                            random.uniform(observedLC.startT,
                                                           observedLC.startT + periodEst),
                                            observedLC.mean] + (-1.0/np.power(10.0, expVal)).tolist()),
                                  requirements=['F', 'A', 'W', 'O', 'E'])
            while noSuccess:
                RhoGuess[self._r + self._p + self._q] = sigmaFactor*observedLC.std
                ThetaGuess = coeffs(self._p, self._q, RhoGuess)
                res = self.set(observedLC.dt, ThetaGuess)
                lnPrior = self.logPrior(observedLC, widthT=widthT, widthF=widthF)
                if res == 0 and not np.isinf(lnPrior):
                    noSuccess = False
                else:
                    sigmaFactor *= 0.31622776601  # sqrt(0.1)

            for dimNum in xrange(self.ndims):
                xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]
        res = self._taskCython.fit_CARMAModel(
            observedLC.dt, observedLC.numCadences, observedLC.meandt, observedLC.tolIR,
            observedLC.maxSigma*observedLC.std, observedLC.minTimescale*observedLC.mindt,
            observedLC.maxTimescale*observedLC.T, np.min(observedLC.y), np.max(observedLC.y),
            observedLC.startT, observedLC.t, observedLC.x, observedLC.y, observedLC.yerr, observedLC.mask,
            self.nwalkers, self.nsteps, self.maxEvals, self.xTol, self.mcmcA,
            zSSeed, walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPrior, self._LnLikelihood,
            periodEst, widthT*periodEst,
            observedLC.mean, widthF*observedLC.mean)

        meanTheta = list()
        for dimNum in xrange(self.ndims):
            meanTheta.append(np.mean(self.Chain[dimNum, :, self.nsteps/2:]))
        meanTheta = np.require(meanTheta, requirements=['F', 'A', 'W', 'O', 'E'])
        self.set(observedLC.dt, meanTheta)
        devianceThetaBar = -2.0*self.logLikelihood(observedLC)
        barDeviance = np.mean(-2.0*self.LnLikelihood[:, self.nsteps/2:])
        self._pDIC = barDeviance - devianceThetaBar
        self._dic = devianceThetaBar + 2.0*self.pDIC
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

    def smooth(self, observedLC, startT=None, stopT=None, tnum=None):
        if tnum is None:
            tnum = 0
        if observedLC.dtSmooth is None or observedLC.dtSmooth == 0.0:
            observedLC.dtSmooth = observedLC.mindt/10.0
        observedLC.pComp = self.p
        observedLC.qComp = self.q

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
        unObsErr = math.sqrt(sys.float_info.max)
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

        res = self._taskCython.smooth_RTS(
            observedLC.numCadencesSmooth, -1, observedLC.tolIR, observedLC.startT,
            observedLC.tSmooth, observedLC.xSmooth,
            observedLC.ySmooth, observedLC.yerrSmooth, observedLC.maskSmooth,
            observedLC.XComp, observedLC.PComp, observedLC.XSmooth, observedLC.PSmooth,
            observedLC.xSmooth, observedLC.xerrSmooth, tnum)
        observedLC._isSmoothed = True
        return res

    @classmethod
    def _auxillary(cls, a1, a2, T, eccentricity, sigmaStars=200.0, rhoStars=1000.0, H=16.0):
        a1 = a1*cls.Parsec
        a2 = a2*cls.Parsec
        T = T*cls.Day
        MTot = ((cls.fourPiSq*math.pow(a1 + a2, 3.0))/(cls.G*math.pow(T, 2.0)))/cls.SolarMass
        MRat = a1/a2
        m1 = (MTot*(1.0/(1.0 + MRat)))*cls.SolarMass
        m2 = (MTot*(MRat/(1.0 + MRat)))*cls.SolarMass
        MRed = m1*m2/(m1 + m2)
        rPeri = ((a1 + a2)*(1.0 - eccentricity))/cls.Parsec
        rApo = ((a1 + a2)*(1.0 + eccentricity))/cls.Parsec
        rSch = ((2.0*cls.G*(MTot*cls.SolarMass))/(math.pow(cls.c, 2.0)))/cls.Parsec
        aHard = ((cls.G*MRed)/(4.0*math.pow((sigmaStars*cls.kms2ms), 2.0)))/cls.Parsec
        numer = 64.0*math.pow(cls.G, 2.0)*m1*m2*(MTot*cls.SolarMass)*(cls.kms2ms*sigmaStars)
        denom = 5.0*H*math.pow(cls.c, 5.0)*(cls.SolarMassPerCubicParsec*rhoStars)
        aGW = math.pow(numer/denom, 0.2)/cls.Parsec
        THard = ((sigmaStars*cls.kms2ms
                  )/(H*cls.G*(cls.SolarMassPerCubicParsec*rhoStars)*aGW*cls.Parsec))/cls.Year
        MEject = (MTot*math.log(aHard/aGW))/1.0e6
        MTot = MTot/1.0e6
        return MTot, MRat, rPeri, rApo, rSch, aHard, aGW, THard, MEject

    @property
    def auxillaryChain(self):
        if hasattr(self, '_auxillaryChain'):
            return self._auxillaryChain
        else:
            self._auxillaryChain = np.require(np.zeros((13, self._nwalkers, self._nsteps)),
                                              requirements=['F', 'A', 'W', 'O', 'E'])
            for stepNum in xrange(self.nsteps):
                for walkerNum in xrange(self.nwalkers):
                    a1 = self.Chain[0, walkerNum, stepNum]
                    a2 = self.Chain[1, walkerNum, stepNum]
                    T = self.Chain[2, walkerNum, stepNum]
                    eccentricity = self.Chain[3, walkerNum, stepNum]
                    mTot, q, rPeri, rApo, rSch, aHard, aGW, THard, MEject = self._auxillary(a1, a2,
                                                                                            T, eccentricity)
                    self._auxillaryChain[0, walkerNum, stepNum] = a1
                    self._auxillaryChain[1, walkerNum, stepNum] = a2
                    self._auxillaryChain[2, walkerNum, stepNum] = T
                    self._auxillaryChain[3, walkerNum, stepNum] = eccentricity
                    self._auxillaryChain[4, walkerNum, stepNum] = mTot
                    self._auxillaryChain[5, walkerNum, stepNum] = q
                    self._auxillaryChain[6, walkerNum, stepNum] = rPeri
                    self._auxillaryChain[7, walkerNum, stepNum] = rApo
                    self._auxillaryChain[8, walkerNum, stepNum] = rSch
                    self._auxillaryChain[9, walkerNum, stepNum] = aHard
                    self._auxillaryChain[10, walkerNum, stepNum] = aGW
                    self._auxillaryChain[11, walkerNum, stepNum] = THard
                    self._auxillaryChain[12, walkerNum, stepNum] = MEject
            return self._auxillaryChain

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
            plt.axvline(x=truthx, c=r'#000000')
        if truthy is not None:
            plt.axhline(y=truthy, c=r'#000000')
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
            plt.axhline(truth, c=r'#000000')
        plt.xlabel(r'step \#')
        if label is not None:
            plt.ylabel(label)
        if doShow:
            plt.show(False)
        return newFig

    def plottriangle(self, doShow=False):
        orbitChain = copy.copy(self.timescaleChain[0:self.r, :, self.nsteps/2:])
        flatOrbitChain = np.swapaxes(orbitChain.reshape((self.r, -1), order='F'), axis1=0, axis2=1)
        orbitLabels = [r'$a_{1}$ (pc)', r'$a_{2}$ (pc)', r'$T$ (d)', r'$e$', r'$\Omega$ (deg.)', r'$i$ (deg)',
                       r'$\tau$ (d)', r'$F$']
        orbitExtents = [0.9, 0.9, (0.95*np.min(self.timescaleChain[2, :, self.nsteps/2:]),
                                   1.05*np.max(self.timescaleChain[2, :, self.nsteps/2:])), 0.9, 0.9, 0.9,
                        0.9, (0.85*np.min(self.timescaleChain[7, :, self.nsteps/2:]),
                              1.15*np.max(self.timescaleChain[7, :, self.nsteps/2:]))]
        newFigOrb = kali.util.triangle.corner(flatOrbitChain, labels=orbitLabels,
                                              show_titles=True,
                                              title_fmt='.2e',
                                              quantiles=[0.16, 0.5, 0.84],
                                              extents=orbitExtents,
                                              plot_contours=False,
                                              plot_datapoints=True,
                                              plot_contour_lines=False,
                                              pcolor_cmap=cm.gist_earth,
                                              verbose=False)

        stochasticChain = copy.copy(self.timescaleChain[self.r:, :, self.nsteps/2:])
        flatStochasticChain = np.swapaxes(stochasticChain.reshape((self.ndims - self.r, -1), order='F'),
                                          axis1=0, axis2=1)
        stochasticLabels = []
        for i in xrange(self.p):
            stochasticLabels.append(r'$\tau_{\mathrm{AR,}, %d}$ (d)'%(i + 1))
        for i in xrange(self.q):
            stochasticLabels.append(r'$\tau_{\mathrm{MA,}, %d}$ (d)'%(i + 1))
        stochasticLabels.append(r'$\mathrm{Amp.}$')
        newFigSto = kali.util.triangle.corner(flatStochasticChain, labels=stochasticLabels,
                                              show_titles=True,
                                              title_fmt='.2e',
                                              quantiles=[0.16, 0.5, 0.84],
                                              plot_contours=False,
                                              plot_datapoints=True,
                                              plot_contour_lines=False,
                                              pcolor_cmap=cm.gist_earth,
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
                                              plot_contours=False,
                                              plot_datapoints=True,
                                              plot_contour_lines=False,
                                              pcolor_cmap=cm.gist_earth,
                                              verbose=False)
        if doShow:
            plt.show(False)
        return newFigSto, newFigOrb, newFigAux
