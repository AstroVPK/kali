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
import scipy.optimize as opt
import pdb


MAXITER = 1000000
TOL = 1.0e-6

SigmaOoM = 200.0 # km/s
HOoM = 16.0
RhoOoM = 1000.0 # SolarMasses/pc^3

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

	def __init__(self, a = 0.01, m12 = 1.0e7, q = 1.0, e = 0.0, omega = 0.0, i = math.pi/2.0, tau = 0.0, alpha = -0.44):
		"""!
		\brief Initialize the binarySMBH object.

		Check to see if the parameter values are sane. 

		List of keyword arguments.
		\param[in] a:           Semimajor axis of the orbit in Parsec (default 0.01 parsec)
		\param[in] m12:         Combined mass of binary SMBHs in Solar masses (default 1.0e7 solar masses)
		\param[in] q:           Mass ratio of secondary to primary (default 1.0)
		\param[in] e:           Ellipticity of orbit (default 0.0)
		\param[in] omega        Argument of periapsis (radians)
		\param[in] i            Inclination of the orbit (radians)
		\param[in] tau          MJD at periapsis
		\param[in] alpha:       Power-law SED spectral index (default -0.44)
		"""
		if a > 0.0:
			self.a = a*self.Parsec
		else:
			raise ValueError('Semi-major axis of orbit must be > 0.0 parsec')
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
		if e >= 0.0:
			self.e = e
		else: 
			raise ValueError('Orbital ellipticity must be >= 0.0') 
		self.ellipticityFactor = math.sqrt((1.0 + self.e)/(1.0 - self.e))
		self.omega = omega
		self.i = i
		self.tau = tau
		self.alpha = alpha
		self.m1 = self.m12/(1.0 + self.q)
		self.m2 = self.m12 - self.m1
		self.mu = (self.m1*self.m2)/self.m12
		self.Period()
		self.t = 0.0
		self.M = 2.0*math.pi*(self.t - self.tau)/self.T # compute Mean Anomoly
		self.E = opt.newton(self.KE, 0.0, fprime = self.KEPrime, fprime2 = self.KEPrimePrime, tol = TOL, args = (self.e, self.M), maxiter = MAXITER) # solve Kepler's Equation to compute the Eccentric Anomoly
		self.r = self.a*(1.0 - self.e*math.cos(self.E)) # current distance from primary
		self.nu = 2.0*math.atan(self.ellipticityFactor*math.tan(self.E/2.0)) # current true anomoly'''

	def KeplersEquation(self, M):
		"""!
		\brief Less accurate computation of E from M
		"""
		E = M + (self.e - (math.pow(self.e, 3.0)/8.0))*math.sin(M) + 0.5*math.pow(self.e, 2.0)*math.sin(2.0*M) + 0.375*math.pow(self.e, 3.0)*math.sin(3.0*M)
		return E

	@staticmethod
	def KE(E, e, M):
		val = E - e*math.sin(E) - M
		return val

	@staticmethod
	def KEPrime(E, e, M):
		val = 1 - e*math.cos(E)
		return val

	@staticmethod
	def KEPrimePrime(E, e, M):
		val = e*math.sin(E)
		return val

	def Period(self):
		"""!
		\brief Orbital period (years).
		"""
		self.T = (math.sqrt((4*self.PiSq*math.pow(self.a,3.0))/(self.G*self.m12)))
		return self.T/self.Year

	def __call__(self, t):
		"""!
		\brief Orbital beta.
		"""
		if t != self.t:
			self.t = t
			self.M = 2.0*math.pi*(self.t - self.tau)/self.T # compute Mean Anomoly
			self.E = opt.newton(self.KE, 0.0, fprime = self.KEPrime, fprime2 = self.KEPrimePrime, tol = TOL, args = (self.e, self.M), maxiter = MAXITER)# solve Kepler's Equation to compute the Eccentric Anomoly
			self.r = self.a*(1.0 - self.e*math.cos(self.E)) # current distance from primary
			self.nu = 2.0*math.atan(self.ellipticityFactor*math.tan(self.E/2.0)) + math.pi # current true anomoly

	def getPosition(self, t):
		self(t)
		return self.r, self.nu

	def beta(self, t):
		"""!
		\brief Orbital beta.
		"""
		r, nu = self.getPosition(t)
		b = math.sqrt(self.G*self.m12*((2.0/r) - (1.0/self.a)))/self.c
		return b

	def transverseBeta(self, t):
		"""!
		\brief Transverse beta.
		"""
		r, nu = self.getPosition(t)
		tB = ((((2.0*math.pi*self.a)/self.T)*math.sin(i)/math.sqrt(1.0 - math.pow(self.e, 2.0)))*(math.cos(nu + self.omega) + self.e*math.cos(self.omega)))/self.c
		return tB

	def dopplerFactor(self, t):
		return (math.sqrt(1.0 - math.pow(self.beta(t), 2.0)))/(1.0 - self.transverseBeta(t))

	def beamingFactor(self, t):
		return math.pow((math.sqrt(1.0 - math.pow(self.beta(t), 2.0)))/(1.0 - self.transverseBeta(t)), 3.0 - self.alpha)

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
	import matplotlib.pyplot as plt
	Num = 1000
	e = 0.0
	A = binarySMBH(e = e)

	times = np.linspace(0.0, A.T, num = Num)
	angles = np.zeros(Num)
	dists = np.zeros(Num)
	betaFac = np.zeros(Num)
	transBetaFac = np.zeros(Num)
	dopplerFac = np.zeros(Num)
	beamingFac = np.zeros(Num)
	for i in xrange(Num):
		dists[i], angles[i] = A.getPosition(times[i])
		betaFac[i] = A.beta(times[i])
		transBetaFac[i] = A.transverseBeta(times[i])
		dopplerFac[i] = A.dopplerFactor(times[i])
		beamingFac[i] = A.beamingFactor(times[i])
	pdb.set_trace()

	ax = plt.subplot(111, projection='polar')
	ax.plot(angles, dists/A.Parsec, color='r', linewidth=3)
	ax.set_rmax(0.025)
	ax.grid(True)
	ax.set_title('Orbit of binary SMBH', va='bottom')

	plt.figure(2)
	plt.plot(angles*(180.0/(2.0*math.pi)),betaFac, label = r'$\beta$')
	plt.plot(angles*(180.0/(2.0*math.pi)),transBetaFac, label = r'$\beta_{\perp}$')
	plt.legend()

	plt.figure(3)
	plt.plot(angles*(180.0/(2.0*math.pi)), dopplerFac, label = r'$D$')
	plt.plot(angles*(180.0/(2.0*math.pi)), beamingFac, label = r'$D^{3-\alpha}$')
	plt.legend()

	plt.figure(4)
	plt.plot(times/A.T,angles*(180.0/(2.0*math.pi)))

	plt.show()