#!/usr/bin/env python
"""	Module to simulate binary SMBHs.

	For a demonstration of the module, please run the module as a command line program eg.
	bash-prompt$ python binarySMBHDemo.py --help
	and
	bash-prompt$ python binarySMBHDemo.py -rPer 0.001 -m12 6.0e8 -q 0.2 -e 0.75 -omega 90.0 -i 90.0
"""
import math as math
import cmath as cmath
import numpy as np
from numpy import vectorize
import scipy.optimize as opt
import pdb

from .python.util.mpl_settings import *
from . import lib.bSMBH as bSMBH

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXXXSmall']
set_plot_params(fontfamily='serif', fontstyle='normal', fontvariant='normal',
                fontweight='normal', fontstretch='normal', fontsize=AxisSize, useTex='True')

G = 6.67408e-11  # m^3/kg s^2
c = 299792458.0  # m/s
AU = 1.4960e11  # m
Parsec = 3.0857e16  # m
Year = 31557600.0  # s
Day = 86400.0  # s
PiSq = math.pow(math.pi, 2.0)
SolarMass = 1.98855e30  # kg
kms2ms = 1.0e3  # m/s
SolarMassPerCubicParsec = 1.98855e30/math.pow(3.0857e16, 3.0)  # kg/m^3

SigmaOoM = 200.0  # km/s
HOoM = 16.0
RhoOoM = 1000.0  # SolarMasses/pc^3


def d2r(d):
    return (math.pi/180.0)*d


def r2d(r):
    return (180.0/math.pi)*r

if __name__ == "__main__":
    import argparse as argparse
    import matplotlib.pyplot as plt
    from matplotlib import animation

    parser = argparse.ArgumentParser()
    parser.add_argument('-rPer', '--rPer', type=float, default=0.01,
                        help=r'Seperation at periapsis i.e. closest approach (parsec), default = 0.01 parsec')
    parser.add_argument('-m12', '--m12', type=float, default=1.0e7,
                        help=r'Sum of masses of black holes (M_Sun), default = 10^7 M_Sun')
    parser.add_argument('-q', '--q', type=float, default=1.0,
                        help=r'Mass ratio of black holes (dimensionless), default = 1.0')
    parser.add_argument('-e', '--e', type=float, default=0.0,
                        help=r'Orbital eccentricity (dimensionless), default = 0.0')
    parser.add_argument('-omega', '--omega', type=float, default=90.0,
                        help=r'Argument of periapsis (degree), default = 0.0 degree')
    parser.add_argument('-i', '--i', type=float, default=90.0,
                        help=r'Inclination of orbit (radian), default = 90 degree')
    parser.add_argument('-tau', '--tau', type=float, default=0.0,
                        help=r'MJD at periapsis (day), default = 0.0 day')
    parser.add_argument('-alpha1', '--alpha1', type=float, default=-0.44,
                        help=r'SED power-law spectral index of 1st black hole (dimensionless), default = -0.44')
    parser.add_argument('-alpha2', '--alpha2', type=float, default=-0.44,
                        help=r'SED power-law spectral index of 2nd black hole (dimensionless), default = -0.44')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=r'Make plots of mean anomoly, eccentric anomoloy, orbital velocity etc...?')
    args = parser.parse_args()

    Num = 1000
    numOrbits = 3
    Num = numOrbits*Num
    intervalLen = 1.0/60.0
    numFrames = int((1.0/intervalLen)*10)

    A = bSMBH.bSMBH(rPer=args.rPer, m12=args.m12, q=args.q, e=args.e,
                    omega=args.omega, i=args.i, tau=args.tau, alpha1=args.alpha1, alpha2=args.alpha2)
    T = A.getPeriod()
    dt = T/numFrames

    fig = plt.figure(1, figsize=(plot_params['fwid'], plot_params['fwid']))
    ax1 = plt.subplot(111, projection='polar')
    line1, = ax1.plot([], [], 'o', ms=20, color='#000000', label=r'$m_{1}$ \& $m_{2}$')
    rmax = (A.getA2()*(1.0 + A.getEllipticity()))
    ax1.set_rmax(1.25*rmax)
    ax1.grid(True)
    ax1.set_title('Orbit of binary SMBH', va='bottom')
    ax1.set_xlabel(r'$r$ (parsec)')
    ax1.legend()

    def init():
        line1.set_data([], [])
        return line1,

    def animate(i):
        times = i*dt
        r1, theta1, b1, rB1, dF1, bF1 = A.getCoordinates('m1', times)
        r2, theta2, b2, rB2, dF2, bF2 = A.getCoordinates('m2', times)
        line1.set_data([d2r(theta1), d2r(theta2)], [r1, r2])
        return line1,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=numFrames, interval=intervalLen, blit=True)

    times = np.linspace(0.0, (numOrbits*T), num=Num)

    r1 = np.zeros(Num)
    theta1 = np.zeros(Num)
    r2 = np.zeros(Num)
    theta2 = np.zeros(Num)
    b1 = np.zeros(Num)
    b2 = np.zeros(Num)
    rB1 = np.zeros(Num)
    rB2 = np.zeros(Num)
    dF1 = np.zeros(Num)
    dF2 = np.zeros(Num)
    bF1 = np.zeros(Num)
    bF2 = np.zeros(Num)

    for i in xrange(Num):
        r1[i], theta1[i], b1[i], rB1[i], dF1[i], bF1[i] = A.getCoordinates('m1', times[i])
        r2[i], theta2[i], b2[i], rB2[i], dF2[i], bF2[i] = A.getCoordinates('m2', times[i])

    fig2 = plt.figure(2, figsize=(plot_params['fwid'], plot_params['fhgt']))
    plt.plot(times/T, b1, color='#377eb8', label=r'$\beta_{m_{1}}(t/T)$')
    plt.plot(times/T, rB1, color='#7570b3', label=r'$\beta_{m_{1},\parallel}(t/T)$')
    plt.plot(times/T, b2, color='#e41a1c', label=r'$\beta_{m_{2}}(t/T)$')
    plt.plot(times/T, rB2, color='#d95f02', label=r'$\beta_{m_{2},\parallel}(t/T)$')
    plt.xlabel(r'$t/T$ ($T = %3.2f$ yr)'%(T))
    plt.ylabel(r'$\beta$, $\beta_{\parallel}$')
    plt.grid(True)
    plt.legend()

    fig3 = plt.figure(3, figsize=(plot_params['fwid'], plot_params['fhgt']))
    plt.plot(times/T, dF1, color='#7570b3', label=r'$D_{m_{1}}(t/T)$')
    plt.plot(times/T, bF1, color='#377eb8', label=r'$D_{m_{1}}^{3-\alpha}(t/T)$')
    plt.plot(times/T, dF2, color='#d95f02', label=r'$D_{m_{2}}(t/T)$')
    plt.plot(times/T, bF2, color='#e41a1c', label=r'$D_{m_{2}}^{3-\alpha}(t/T)$')
    plt.xlabel(r'$t/T$ ($T = %3.2f$ yr)'%(T))
    plt.ylabel(r'$D$, $D^{3-\alpha}$')
    plt.grid(True)
    plt.legend()

    plt.show()
