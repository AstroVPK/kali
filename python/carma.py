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
    import lc
    from util.mpl_settings import set_plot_params
except ImportError:
    print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
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
    ARPoly = np.zeros(p + 1)
    ARPoly[0] = 1.0
    for i in xrange(p):
        ARPoly[i + 1] = Theta[i]
    ARRoots = np.array(np.roots(ARPoly))
    MAPoly = np.zeros(q + 1)
    for i in xrange(q + 1):
        MAPoly[i] = Theta[p + q - i]
    MARoots = np.array(np.roots(MAPoly))
    Rho = np.zeros(p + q + 1, dtype='complex128')
    for i in xrange(p):
        Rho[i] = ARRoots[i]
    for i in xrange(q):
        Rho[p + i] = MARoots[i]
    Sigma = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
    ThetaC = np.require(np.array(Theta), requirements=['F', 'A', 'W', 'O', 'E'])
    CARMATask_cython.get_Sigma(p, q, ThetaC, Sigma)
    Rho[p + q] = math.sqrt(Sigma[0])
    return Rho


def coeffs(p, q, Rho):
    ARRoots = np.zeros(p, dtype='complex128')
    for i in xrange(p):
        ARRoots[i] = Rho[i]
    ARPoly = np.array(np.poly(ARRoots))
    MARoots = np.zeros(q, dtype='complex128')
    for i in xrange(q):
        MARoots[i] = Rho[p + i]
    if q == 0:
        MAPoly = np.ones(1)
    else:
        MAPoly = np.array(np.poly(MARoots))
    ThetaPrime = np.require(
        np.array(ARPoly[1:].tolist() + MAPoly.tolist()[::-1]), requirements=['F', 'A', 'W', 'O', 'E'])
    SigmaPrime = np.require(np.zeros(p*p), requirements=['F', 'A', 'W', 'O', 'E'])
    CARMATask_cython.get_Sigma(p, q, ThetaPrime, SigmaPrime)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Sigma00 = math.pow(Rho[p + q], 2.0)
    try:
        bQ = math.sqrt(Sigma00/SigmaPrime[0])
    except ValueError:
        bQ = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in xrange(q + 1):
            MAPoly[i] = bQ*MAPoly[i]
    Theta = np.zeros(p + q + 1, dtype='float64')
    for i in xrange(p):
        Theta[i] = ARPoly[i + 1].real
    for i in xrange(q + 1):
        Theta[p + i] = MAPoly[q - i].real
    return Theta


def timescales(p, q, Rho):
    imagPairs = 0
    for i in xrange(p):
        if Rho[i].imag != 0.0:
            imagPairs += 1
    numImag = imagPairs/2
    numReal = numImag + (p - imagPairs)
    decayTimescales = np.zeros(numReal)
    oscTimescales = np.zeros(numImag)
    realRoots = set(Rho[0:p].real)
    imagRoots = set(abs(Rho[0:p].imag)).difference(set([0.0]))
    realAR = sorted([1.0/abs(x) for x in realRoots])
    imagAR = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
    imagPairs = 0
    for i in xrange(q):
        if Rho[i].imag != 0.0:
            imagPairs += 1
    numImag = imagPairs/2
    numReal = numImag + (q - imagPairs)
    decayTimescales = np.zeros(numReal)
    oscTimescales = np.zeros(numImag)
    realRoots = set(Rho[p:p + q].real)
    imagRoots = set(abs(Rho[p:p + q].imag)).difference(set([0.0]))
    realMA = sorted([1.0/abs(x) for x in realRoots])
    imagMA = sorted([(2.0*math.pi)/abs(x) for x in imagRoots])
    return np.array(realAR + imagAR + realMA + imagMA + [Rho[p + q]])


class CARMATask(object):

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
            self._ndims = self._p + self._q + 1
            self._nthreads = nthreads
            self._nburn = nburn
            self._nwalkers = nwalkers
            self._nsteps = nsteps
            self._maxEvals = maxEvals
            self._xTol = xTol
            self._mcmcA = mcmcA
            self._Chain = np.require(
                np.zeros(self._ndims*self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
            self._LnPosterior = np.require(
                np.zeros(self._nwalkers*self._nsteps), requirements=['F', 'A', 'W', 'O', 'E'])
            self._taskCython = CARMATask_cython.CARMATask_cython(self._p, self._q, self._nthreads,
                                                                 self._nburn)
        except AssertionError as err:
            raise AttributeError(str(err))

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
            self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
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
            self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
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
            self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
            self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
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
            self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
            self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
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
            for stepNum in xrange(self._nsteps):
                for walkerNum in xrange(self._nwalkers):
                    self._timescaleChain[:, walkerNum, stepNum] = timescales(
                        self._p, self._q, rootChain[:, walkerNum, stepNum])
        return self._timescaleChain

    @property
    def LnPosterior(self):
        return np.reshape(self._LnPosterior, newshape=(self._nwalkers, self._nsteps), order='F')

    def __repr__(self):
        return "libcarma.task(%d, %d, %d, %d, %d, %d, %d, %f)"%(self._p, self._q, self._nthreads,
                                                                self._nburn, self._nwalkers, self._nsteps,
                                                                self._maxEvals, self._xTol)

    def __str__(self):
        line = 'p: %d; q: %d; ndims: %d\n'%(self._p, self._q, self._ndims)
        line += 'nthreads (Number of hardware threads to use): %d\n'%(self._nthreads)
        line += 'nburn (Number of light curve steps to burn): %d\n'%(self._nburn)
        line += 'nwalkers (Number of MCMC walkers): %d\n'%(self._nwalkers)
        line += 'nsteps (Number of MCMC steps): %d\n'%(self.nsteps)
        line += 'maxEvals (Maximum number of evaluations when attempting to find starting location for MCMC):\
        %d\n'%(self._maxEvals)
        line += 'xTol (Fractional tolerance in optimized parameter value): %f'%(self._xTol)
        return line

    def __eq__(self, other):
        if isinstance(other, task):
            if ((self._p == other.p) and (self._q == other.q) and (self._nthreads == other.nthreads) and
                (self._nburn == other.nburn) and (self._nwalkers == other.nwalkers) and
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
            self._taskCython.reset_CARMATask(p, q, nburn)
            self._p = p
            self._q = q
            self._ndims = self._p + self._q + 1
            self._nburn = nburn
            self._nwalkers = nwalkers
            self._nsteps = nsteps
            self._Chain = np.zeros(self._ndims*self._nwalkers*self._nsteps)
            self._LnPosterior = np.zeros(self._nwalkers*self._nsteps)
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

    def simulate(self, duration, tolIR=1.0e-3, fracIntrinsicVar=0.15, fracNoiseToSignal=0.001, maxSigma=2.0,
                 minTimescale=2.0, maxTimescale=0.5, burnSeed=None, distSeed=None, noiseSeed=None, tnum=None):
        if tnum is None:
            tnum = 0
        numCadences = int(round(float(duration)/self._taskCython.get_dt(threadNum=tnum)))
        intrinsicLC = lc.basicLC(
            numCadences, dt=self._taskCython.get_dt(threadNum=tnum), tolIR=tolIR,
            fracIntrinsicVar=fracIntrinsicVar, fracNoiseToSignal=fracNoiseToSignal, maxSigma=maxSigma,
            minTimescale=minTimescale, maxTimescale=maxTimescale, pSim=self._p, qSim=self._q)
        randSeed = np.zeros(1, dtype='uint32')
        if burnSeed is None:
            rand.rdrand(randSeed)
            burnSeed = randSeed[0]
        if distSeed is None:
            rand.rdrand(randSeed)
            distSeed = randSeed[0]
        self._taskCython.make_IntrinsicLC(
            intrinsicLC.numCadences, intrinsicLC.tolIR, intrinsicLC.fracIntrinsicVar,
            intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x, intrinsicLC.y, intrinsicLC.yerr,
            intrinsicLC.mask, intrinsicLC.XSim, intrinsicLC.PSim, burnSeed, distSeed, threadNum=tnum)
        intrinsicLC._simulatedCadenceNum = numCadences - 1
        intrinsicLC._T = intrinsicLC.t[-1] - intrinsicLC.t[0]
        return intrinsicLC

    def extend(self, intrinsicLC, duration, gap=None, distSeed=None, noiseSeed=None, tnum=None):
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
        extraNumCadences = int(round(float(duration + gapSize)/self._taskCython.get_dt(threadNum=tnum)))
        newNumCadences = intrinsicLC.numCadences + extraNumCadences
        newt = np.require(np.zeros(newNumCadences), requirements=[
                          'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
        newx = np.require(np.zeros(newNumCadences), requirements=[
                          'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
        newy = np.require(np.zeros(newNumCadences), requirements=[
                          'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed fluxes.
        newyerr = np.require(np.zeros(newNumCadences), requirements=[
                             'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed flux errors.
        newmask = np.require(np.zeros(newNumCadences), requirements=[
                             'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        for i in xrange(intrinsicLC.numCadences):
            newt[i] = intrinsicLC.t[i]
            newx[i] = intrinsicLC.x[i]
            newy[i] = intrinsicLC.y[i]
            newyerr[i] = intrinsicLC.yerr[i]
            newmask[i] = intrinsicLC.mask[i]
        for i in xrange(intrinsicLC.numCadences, newNumCadences):
            newt[i] = newt[intrinsicLC.numCadences - 1] + gapSize + \
                (i - intrinsicLC.numCadences + 1)*self._taskCython.get_dt(threadNum=tnum)
            newmask[i] = 1.0
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

        count = int(np.sum(self.mask))
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
        else:
            self._taskCython.extend_ObservationNoise(
                intrinsicLC.numCadences, intrinsicLC.observedCadenceNum, intrinsicLC.tolIR,
                intrinsicLC.fracIntrinsicVar, intrinsicLC.fracNoiseToSignal, intrinsicLC.t, intrinsicLC.x,
                intrinsicLC.y, intrinsicLC.yerr, intrinsicLC.mask, noiseSeed, threadNum=tnum)
            intrinsicLC._observedCadenceNum = intrinsicLC._numCadences - 1

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

    def logPrior(self, observedLC, forced=True, tnum=None):
        if tnum is None:
            tnum = 0
        observedLC._logPrior = self._taskCython.compute_LnPrior(observedLC.numCadences, observedLC.tolIR,
                                                                observedLC.maxSigma*observedLC.std,
                                                                observedLC.minTimescale*observedLC.mindt,
                                                                observedLC.maxTimescale*observedLC.T,
                                                                observedLC.t, observedLC.x, observedLC.y,
                                                                observedLC.yerr, observedLC.mask, tnum)
        return observedLC._logPrior

    def logLikelihood(self, observedLC, forced=True, tnum=None):
        if tnum is None:
            tnum = 0
        observedLC.pComp = self.p
        observedLC.qComp = self.q
        observedLC._logPrior = self.logPrior(observedLC, forced=forced, tnum=tnum)
        if forced is True:
            observedLC._computedCadenceNum = -1
        if observedLC._computedCadenceNum == -1:
            for rowCtr in xrange(observedLC.pComp):
                observedLC.XComp[rowCtr] = 0.0
                for colCtr in xrange(observedLC.pComp):
                    observedLC.PComp[rowCtr + observedLC.pComp*colCtr] = 0.0
            observedLC._logLikelihood = self._taskCython.compute_LnLikelihood(
                observedLC.numCadences, observedLC._computedCadenceNum, observedLC.tolIR, observedLC.t,
                observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]),
                observedLC.yerr, observedLC.mask, observedLC.XComp, observedLC.PComp, tnum)
            observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
            observedLC._computedCadenceNum = observedLC.numCadences - 1
        elif observedLC._computedCadenceNum == observedLC.numCadences - 1:
            pass
        else:
            observedLC._logLikelihood = self._taskCython.update_LnLikelihood(
                observedLC.numCadences, observedLC._computedCadenceNum, observedLC._logLikelihood,
                observedLC.tolIR, observedLC.t, observedLC.x, observedLC.y -
                np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr, observedLC.mask,
                observedLC.XComp, observedLC.PComp, tnum)
            observedLC._logPosterior = observedLC._logPrior + observedLC._logLikelihood
            observedLC._computedCadenceNum = observedLC.numCadences - 1
        return observedLC._logLikelihood

    def logPosterior(self, observedLC, forced=True, tnum=None):
        lnLikelihood = self.logLikelihood(observedLC, forced=forced, tnum=tnum)
        return observedLC._logPosterior

    def acvf(self, start=0.0, stop=100.0, num=100, endpoint=True, base=10.0, spacing='linear'):
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
        return newFig

    def fit(self, observedLC, zSSeed=None, walkerSeed=None, moveSeed=None, xSeed=None):
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

        for walkerNum in xrange(self.nwalkers):
            noSuccess = True
            sigmaFactor = 1.0e0
            RhoGuess = -1.0 / \
                np.power(10.0, ((maxTLog10 - minTLog10)*np.random.random(self.p + self.q + 1) + minTLog10))
            while noSuccess:
                RhoGuess[self.p + self.q] = sigmaFactor*observedLC.std
                ThetaGuess = coeffs(self.p, self.q, RhoGuess)
                res = self.set(observedLC.dt, ThetaGuess)
                lnPrior = self.logPrior(observedLC)
                if res == 0 and lnPrior == 0.0:
                    noSuccess = False
                else:
                    print 'SigmaTrial: %e'%(RhoGuess[self.p + self.q])
                    sigmaFactor *= 0.31622776601  # sqrt(0.1)

            for dimNum in xrange(self.ndims):
                xStart[dimNum + walkerNum*self.ndims] = ThetaGuess[dimNum]
        res = self._taskCython.fit_CARMAModel(
            observedLC.dt, observedLC.numCadences, observedLC.tolIR, observedLC.maxSigma*observedLC.std,
            observedLC.minTimescale*observedLC.mindt, observedLC.maxTimescale*observedLC.T, observedLC.t,
            observedLC.x, observedLC.y - np.mean(observedLC.y[np.nonzero(observedLC.mask)]), observedLC.yerr,
            observedLC.mask, self.nwalkers, self.nsteps, self.maxEvals, self.xTol, self.mcmcA, zSSeed,
            walkerSeed, moveSeed, xSeed, xStart, self._Chain, self._LnPosterior)
        return res

    def smooth(self, observedLC, tnum=None):
        if tnum is None:
            tnum = 0
        if observedLC.dtSmooth is None or observedLC.dtSmooth == 0.0:
            observedLC.dtSmooth = observedLC.mindt/10.0

        observedLC.pComp = self.p
        observedLC.qComp = self.q

        t = sorted(observedLC.t.tolist() + np.linspace(start=observedLC.t[0], stop=observedLC.t[
                   -1], num=int(math.ceil(observedLC.T/observedLC.dtSmooth)), endpoint=False).tolist())
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
                obsCtr += 1
            else:
                observedLC.xSmooth[i] = 0.0
                observedLC.xerrSmooth[i] = unObsErr
                observedLC.ySmooth[i] = 0.0
                observedLC.yerrSmooth[i] = unObsErr
                observedLC.maskSmooth[i] = 0.0

        preSmoothYMean = np.mean(observedLC.ySmooth[np.nonzero(observedLC.maskSmooth)])
        res = self._taskCython.smooth_RTS(
            observedLC.numCadencesSmooth, -1, observedLC.tolIR, observedLC.tSmooth, observedLC.xSmooth,
            observedLC.ySmooth - preSmoothYMean, observedLC.yerrSmooth, observedLC.maskSmooth,
            observedLC.XComp, observedLC.PComp, observedLC.XSmooth, observedLC.PSmooth, tnum)
        for i in xrange(observedLC.numCadencesSmooth):
            observedLC.xSmooth[i] = observedLC.XSmooth[i*observedLC.pComp] + preSmoothYMean
            try:
                observedLC.xerrSmooth[i] = math.sqrt(observedLC.PSmooth[i*observedLC.pComp*observedLC.pComp])
            except ValueError:
                observedLC.xerrSmooth[i] = 0.0
        observedLC._isSmoothed = True
        return res
