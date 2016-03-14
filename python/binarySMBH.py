#!/usr/bin/env python
"""	Module to simulate binary SMBHs.

	For a demonstration of the module, please run the module as a command line program using 
	bash-prompt$ python makeMockLC.py --help
	and
	bash-prompt$ python makeMockLC.py $PWD/examples/taskTest taskTest01.ini
"""
import math as math
import cmath as cmath
import numpy as np
from numpy import vectorize
import scipy.optimize as opt
import pdb

from python.util.mpl_settings import *

LabelSize = plot_params['LabelXLarge']
AxisSize = plot_params['AxisLarge']
AnnotateSize = plot_params['AnnotateXLarge']
LegendSize = plot_params['LegendXXXSmall']
set_plot_params(fontfamily = 'serif', fontstyle = 'normal', fontvariant = 'normal', fontweight = 'normal', fontstretch = 'normal', fontsize = AxisSize, useTex = 'True')

MAXITER = 1000000
TOL = 1.0e-9

SigmaOoM = 200.0 # km/s
HOoM = 16.0
RhoOoM = 1000.0 # SolarMasses/pc^3

@vectorize
def d2r(d):
	return (math.pi/180.0)*d

@vectorize
def r2d(r):
	return (180.0/math.pi)*r

class binarySMBH(object):
	"""!
	\brief Class to represent beaming due to the orbital motion of binary Supermassive Black Holes (binary SMBH).

	This class keeps track of the orbital parameters of a pair of supermassive Black Holes in orbit. It is able to compute a beaming factor for the system given the orbital phase.
	"""
	G = 6.67408e-11 # m^3/kg s^2
	c = 299792458.0 # m/s
	AU = 1.4960e11 # m
	Parsec = 3.0857e16 # m
	Year = 31557600.0 # s
	Day = 86400.0 # s
	PiSq = math.pow(math.pi, 2.0)
	SolarMass = 1.98855e30 # kg
	kms2ms = 1.0e3 # m/s
	SolarMassPerCubicParsec = 1.98855e30/math.pow(3.0857e16, 3.0) # kg/m^3

	def __init__(self, rPer = 0.01, m12 = 1.0e7, q = 1.0, e = 0.0, omega = 0.0, i = 90.0, tau = 0.0, alpha1 = -0.44, alpha2 = -0.44):
		"""!
		\brief Initialize the binarySMBH object.

		Check to see if the parameter values are sane. 

		List of keyword arguments.
		\param[in] rPer:        Separation at periapsis i.e. closest separation (default 0.01 parsec)
		\param[in] m12:         Combined mass of binary SMBHs in Solar mass (default 1.0e7 Solar mass)
		\param[in] q:           Mass ratio of secondary to primary (default 1.0)
		\param[in] e:           Ellipticity of orbit (default 0.0)
		\param[in] omega        Argument of periapsis in degree (default 0.0 degree)
		\param[in] i            Inclination of the orbit in degree (default 90 degree)
		\param[in] tau          MJD at periapsis in day (default 0.0 day)
		\param[in] alpha1:       Power-law SED spectral index of 1st black hole (default -0.44)
		\param[in] alpha2:       Power-law SED spectral index of 2nd black hole (default -0.44)
		"""
		if m12 > 0.0:
			self.m12 = m12*self.SolarMass
		else:
			raise ValueError('Total mass of binary SMBH must be > 0.0 M_sun')
		if q <= 0.0:
			raise ValueError('Mass ratio m_1/m_2 must be > 0.0')
		elif q > 1.0:
			self.q = 1.0/q
		else:
			self.q = q
		self.m1 = self.m12/(1.0 + self.q)
		self.m2 = self.m12 - self.m1
		self.mu = (self.m1*self.m2)/self.m12
		if e >= 0.0:
			self.e = e
		else: 
			raise ValueError('Orbital ellipticity must be >= 0.0') 
		self.ellipticityFactor = math.sqrt((1.0 + self.e)/(1.0 - self.e))
		if rPer > 0.0:
			self.rPer = rPer*self.Parsec
			self.a1 = (self.rPer*self.m2)/(self.m12*(1.0 - self.e))
			self.a2 = (self.rPer*self.m1)/(self.m12*(1.0 - self.e))
		else:
			raise ValueError('Separation at periapsis must be > 0.0 parsec')
		self.omega1 = d2r(omega)
		self.omega2 = self.omega1 + math.pi
		self.i = d2r(i)
		self.tau = tau*self.Day
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.Period()
		self.t = 0.0
		self.M = 2.0*math.pi*(self.t - self.tau)/self.T # compute Mean Anomoly
		self.E = opt.newton(self.KE, 0.0, fprime = self.KEPrime, fprime2 = self.KEPrimePrime, tol = TOL, args = (self.e, self.M), maxiter = MAXITER) # solve Kepler's Equation to compute the Eccentric Anomoly
		self.r1 = self.a1*(1.0 - self.e*math.cos(self.E)) # current distance of m1 from COM
		self.r2 = (self.m1*self.r1)/self.m2 # current distance of m2 from COM
		self.nu = 2.0*math.atan(self.ellipticityFactor*math.tan(self.E/2.0)) # current true anomoly of m1
		if self.nu < 0.0:
			self.nu = 2.0*math.pi + self.nu
		self.theta1 = self.nu + self.omega1
		self.theta2 = self.nu + self.omega2

	@staticmethod
	def KE(E, e, M):
		val = E - e*math.sin(E) - M
		return val

	@staticmethod
	def KEPrime(E, e, M):
		val = 1.0 - e*math.cos(E)
		return val

	@staticmethod
	def KEPrimePrime(E, e, M):
		val = e*math.sin(E)
		return val

	def Period(self):
		"""!
		\brief Orbital period (years).
		"""
		self.T = math.sqrt((4.0*self.PiSq*math.pow(self.a1,3.0)*math.pow(self.m12, 2.0))/(self.G*math.pow(self.m2, 3.0)))
		return self.T/self.Year

	def __call__(self, t):
		"""!
		\brief Orbital beta.
		"""
		if t != self.t:
			self.t = t
			self.M = 2.0*math.pi*(self.t - self.tau)/self.T # compute Mean Anomoly
			self.E = opt.newton(self.KE, 0.0, fprime = self.KEPrime, fprime2 = self.KEPrimePrime, tol = TOL, args = (self.e, self.M), maxiter = MAXITER) # solve Kepler's Equation to compute the Eccentric Anomoly
			self.r1 = self.a1*(1.0 - self.e*math.cos(self.E)) # current distance of m1 from COM
			self.r2 = (self.m1*self.r1)/self.m2 # current distance of m2 from COM
			self.nu = 2.0*math.atan(self.ellipticityFactor*math.tan(self.E/2.0)) # current true anomoly of m1
			if self.nu < 0.0:
				self.nu = 2.0*math.pi + self.nu
			self.theta1 = self.nu + self.omega1
			self.theta2 = self.nu + self.omega2

	def beta(self, t):
		"""!
		\brief Orbital beta.
		"""
		self(t)
		b1 = math.sqrt(((self.G*math.pow(self.m2, 2.0))/self.m12)*((2.0/self.r1) - (1.0/self.a1)))/self.c
		b2 = math.sqrt(((self.G*math.pow(self.m1, 2.0))/self.m12)*((2.0/self.r2) - (1.0/self.a2)))/self.c
		return b1, b2

	def radialBeta(self, t):
		"""!
		\brief Transverse beta.
		"""
		self(t)
		rB1 = ((((2.0*math.pi*self.a1)/self.T)*math.sin(self.i)/math.sqrt(1.0 - math.pow(self.e, 2.0)))*(math.cos(self.nu + self.omega1) + self.e*math.cos(self.omega1)))/self.c
		rB2 = ((((2.0*math.pi*self.a2)/self.T)*math.sin(self.i)/math.sqrt(1.0 - math.pow(self.e, 2.0)))*(math.cos(self.nu + self.omega2) + self.e*math.cos(self.omega2)))/self.c
		return rB1, rB2

	def dopplerFactor(self, t):
		b1, b2 = self.beta(t)
		rB1, rB2 = self.radialBeta(t)
		dF1 = math.sqrt(1.0 - math.pow(b1, 2.0))/(1.0 - rB1)
		dF2 = math.sqrt(1.0 - math.pow(b2, 2.0))/(1.0 - rB2)
		return dF1, dF2

	def beamingFactor(self, t):
		b1, b2 = self.beta(t)
		rB1, rB2 = self.radialBeta(t)
		bF1 = math.pow((math.sqrt(1.0 - math.pow(b1, 2.0)))/(1.0 - rB1), 3.0 - self.alpha1)
		bF2 = math.pow((math.sqrt(1.0 - math.pow(b2, 2.0)))/(1.0 - rB2), 3.0 - self.alpha2)
		return bF1, bF2

	def aH(self, sigma = SigmaOoM):
		self.aH = (self.G*self.mu)/(4.0*math.pow(sigma*self.kms2ms, 2.0))
		return self.aH/self.Parsec

	def aGW(self, sigma = SigmaOoM, rho = RhoOoM, H = HOoM):
		self.aGW = math.pow((64.0*math.pow(self.G*self.mu, 2.0)*self.m12*sigma*self.kms2ms)/(5.0*H*math.pow(self.c, 5.0)*rho*self.SolarMassPerCubicParsec), 0.2)
		return self.aGW/self.Parsec

	def durationInHardState(self, sigma = SigmaOoM, rho = RhoOoM, H = HOoM):
		self.durationInHardstate = ((sigma*self.kms2ms)/(H*self.G*rho*self.SolarMassPerCubicParsec*self.aGW))
		return self.durationInHardstate/self.Year

	def ejectedMass(self, sigma = SigmaOoM, rho = RhoOoM, H = HOoM):
		self.ejectedMass = self.m12*math.log(self.aH/self.aGW)
		return self.ejectedMass


if __name__ == "__main__":
	import argparse as argparse
	import matplotlib.pyplot as plt
	from matplotlib import animation
	#plt.ion()

	parser = argparse.ArgumentParser()
	parser.add_argument('-rPer', '--rPer', type = float, default = 0.01, help = r'Seperation at periapsis i.e. closest approach (parsec), default = 0.01 parsec')
	parser.add_argument('-m12', '--m12', type = float, default = 1.0e7, help = r'Sum of masses of black holes (M_Sun), default = 10^7 M_Sun')
	parser.add_argument('-q', '--q', type = float, default = 1.0, help = r'Mass ratio of black holes (dimensionless), default = 1.0')
	parser.add_argument('-e', '--e', type = float, default = 0.0, help = r'Orbital eccentricity (dimensionless), default = 0.0')
	parser.add_argument('-omega', '--omega', type = float, default = 0.0, help = r'Argument of periapsis (degree), default = 0.0 degree')
	parser.add_argument('-i', '--i', type = float, default = 90.0, help = r'Inclination of orbit (radian), default = 90 degree')
	parser.add_argument('-tau', '--tau', type = float, default = 0.0, help = r'MJD at periapsis (day), default = 0.0 day')
	parser.add_argument('-alpha1', '--alpha1', type = float, default = -0.44, help = r'SED power-law spectral index of 1st black hole (dimensionless), default = -0.44')
	parser.add_argument('-alpha2', '--alpha2', type = float, default = -0.44, help = r'SED power-law spectral index of 2nd black hole (dimensionless), default = -0.44')
	parser.add_argument('-v', '--verbose', action = 'store_true', help = r'Make plots of mean anomoly, eccentric anomoloy, orbital velocity etc...?')
	args = parser.parse_args()

	Num = 1000
	numOrbits = 3
	Num = numOrbits*Num
	numFrames = 100

	A = binarySMBH(rPer = args.rPer, m12 = args.m12, q = args.q, e = args.e, omega = args.omega, i = args.i, tau = args.tau, alpha1 = args.alpha1, alpha2 = args.alpha2)
	dt = A.T/numFrames

	fig = plt.figure(1, figsize = (plot_params['fwid'], plot_params['fwid']))
	ax1 = plt.subplot(111, projection='polar')
	line1, = ax1.plot([], [], 'o', ms = 20, color = '#377eb8', label = r'$m_{1}$ \& $m_{2}$')
	rmax = (A.a2*(1.0 + A.e))/A.Parsec
	ax1.set_rmax(1.1*rmax)
	ax1.grid(True)
	ax1.set_title('Orbit of binary SMBH', va='bottom')
	ax1.set_xlabel(r'$r$ (parsec)')
	ax1.legend()

	def init():
		line1.set_data([], [])
		return line1,

	def animate(i):
		times = i*dt
		A(times)
		r1 = A.r1
		theta1 = A.theta1
		r2 = A.r2
		theta2 = A.theta2
		line1.set_data([theta1, theta2], [r1/A.Parsec, r2/A.Parsec])
		return line1,

	anim = animation.FuncAnimation(fig, animate, init_func = init, frames = numFrames, interval = 20, blit = True)

	times = np.linspace(0.0, numOrbits*A.T, num = Num)

	betaFac1 = np.zeros(Num)
	betaFac2 = np.zeros(Num)
	radialBetaFac1 = np.zeros(Num)
	radialBetaFac2 = np.zeros(Num)
	dopplerFac1 = np.zeros(Num)
	dopplerFac2 = np.zeros(Num)
	beamingFac1 = np.zeros(Num)
	beamingFac2 = np.zeros(Num)

	if args.verbose:
		theta1 = np.zeros(Num)
		r1 = np.zeros(Num)
		theta2 = np.zeros(Num)
		r2 = np.zeros(Num)
		MList = np.zeros(Num)
		EList = np.zeros(Num)
		nuList = np.zeros(Num)
		for i in xrange(Num):
			r1[i] = A.r1
			theta1[i] = A.theta1
			r2[i] = A.r2
			theta2[i] = A.theta2
			MList[i] = A.M
			EList[i] = A.E
			nuList[i] = A.nu
			etaFac1[i], betaFac2[i] = A.beta(times[i])
			radialBetaFac1[i], radialBetaFac2[i] = A.radialBeta(times[i])
			dopplerFac1[i], dopplerFac2[i] = A.dopplerFactor(times[i])
			beamingFac1[i], beamingFac2[i] = A.beamingFactor(times[i])
	else:
		for i in xrange(Num):
			betaFac1[i], betaFac2[i] = A.beta(times[i])
			radialBetaFac1[i], radialBetaFac2[i] = A.radialBeta(times[i])
			dopplerFac1[i], dopplerFac2[i] = A.dopplerFactor(times[i])
			beamingFac1[i], beamingFac2[i] = A.beamingFactor(times[i])

	fig2 = plt.figure(2, figsize = (plot_params['fwid'], plot_params['fhgt']))
	plt.plot(times/A.T, betaFac1, color = '#377eb8', label = r'$\beta_{m_{1}}(t/T)$')
	plt.plot(times/A.T, radialBetaFac1, color = '#7570b3', label = r'$\beta_{m_{1},\parallel}(t/T)$')
	plt.plot(times/A.T, betaFac2, color = '#e41a1c', label = r'$\beta_{m_{2}}(t/T)$')
	plt.plot(times/A.T, radialBetaFac2, color = '#d95f02', label = r'$\beta_{m_{2},\parallel}(t/T)$')
	plt.xlabel(r'$t/T$ ($T = %3.2f$ yr)'%(A.T/A.Year))
	plt.ylabel(r'$\beta$, $\beta_{\parallel}$')
	plt.grid(True)
	plt.legend()

	fig3 = plt.figure(3, figsize = (plot_params['fwid'], plot_params['fhgt']))
	plt.plot(times/A.T, dopplerFac1, color = '#7570b3', label = r'$D_{m_{1}}(t/T)$') 
	plt.plot(times/A.T, beamingFac1, color = '#377eb8', label = r'$D_{m_{1}}^{3-\alpha}(t/T)$')
	plt.plot(times/A.T, dopplerFac2, color = '#d95f02', label = r'$D_{m_{2}}(t/T)$')
	plt.plot(times/A.T, beamingFac2, color = '#e41a1c', label = r'$D_{m_{2}}^{3-\alpha}(t/T)$')
	plt.xlabel(r'$t/T$ ($T = %3.2f$ yr)'%(A.T/A.Year))
	plt.ylabel(r'$D$, $D^{3-\alpha}$')
	plt.grid(True)
	plt.legend()

	if args.verbose:
		fig4 = plt.figure(4, figsize = (plot_params['fwid'], plot_params['fhgt']))
		plt.plot(times/A.T, r2d(MList), label = r'$M$ (degree)')
		plt.plot(times/A.T, r2d(EList), label = r'$E$ (degree)')
		plt.plot(times/A.T, r2d(nuList), label = r'$\nu$ (degree)')
		plt.plot(times/A.T, r2d(theta1), color = '#377eb8', label = r'$\theta_{1}$ (degree)')
		plt.plot(times/A.T, r2d(theta2), color = '#377eb8', label = r'$\theta_{2}$ (degree)')
		plt.xlabel(r'$t/T$ ($T = %3.2f$ yr)'%(A.T/A.Year))
		plt.ylabel(r'$M$, $E$, $\nu$, \& $\theta$ (degree)')
		plt.grid(True)
		plt.legend()
	
		cosTheta1 = np.cos(theta1)
		cosTheta2 = np.cos(theta2)
		fig5 = plt.figure(5, figsize = (plot_params['fwid'], plot_params['fhgt']))
		plt.plot(times/A.T, cosTheta1, label = r'$\cos(\theta_{1})$')
		plt.plot(times/A.T, cosTheta2, label = r'$\cos(\theta_{2})$')

	plt.show()

	pdb.set_trace()