# distutils: language = c++
import math

cdef extern from 'binarySMBH.hpp':
	cpdef double d2r(double)
	cpdef double r2d(double)
	cdef cppclass binarySMBH:
		binarySMBH() except +
		binarySMBH(double rPericenterTotal, double m1, double m2, double ellipticity, double omega, double inclination, double tau, double alpha1, double alpha2) except +
		#void call 'operator()'(double epoch)
		void setEpoch(double epoch)
		double getEpoch()
		double getPeriod()
		double getA1()
		double getA2()
		double getEccentricity()
		double getR1()
		double getR2()
		double getTheta1()
		double getTheta2()
		double getBeta1()
		double getBeta2()
		double getRadialBeta1()
		double getRadialBeta2()
		double getDopplerFactor1()
		double getDopplerFactor2()
		double getBeamingFactor1()
		double getBeamingFactor2()
		double aH(double sigmaStars);
		double aGW(double sigmaStars, double rhoStars, double H);
		double durationInHardState(double sigmaStars, double rhoStars, double H);
		double ejectedMass(double sigmaStars, double rhoStars, double H);

cdef double G = 6.67408e-11 # m^3/kg s^2
cdef double c = 299792458.0 # m/s
cdef double AU = 1.4960e11 # m
cdef double Parsec = 3.0857e16 # m
cdef double Day = 86164.090530833 # s
cdef double Year = 31557600.0 # s
cdef double kms = 1.0e3 # m/s
cdef double SolarMass = 1.98855e30 # kg
cdef double SolarMassPerCubicParsec = SolarMass/math.pow(Parsec, 3.0) # kg/m^3

cdef double SigmaOoM = 200.0*kms # km/s
cdef double HOoM = 16.0
cdef double RhoOoM = 1000.0*SolarMassPerCubicParsec # SolarMasses/pc^3

cdef class bSMBH:
	cdef binarySMBH *thisptr      # hold a C++ instance which we're wrapping
	def __cinit__(self, rPer = 0.01, m12 = 1.0e1, q = 1.0, e = 0.0, omega = 90.0, i = 90.0, tau = 0.0, alpha1 = -0.44, alpha2 = -0.44):
		if m12 > 0.0:
			m12 = m12
		else:
			raise ValueError('Total mass of binary SMBH must be > 0.0 M_sun')
		if q <= 0.0:
			raise ValueError('Mass ratio m_1/m_2 must be > 0.0')
		elif q > 1.0:
			q = 1.0/q
		else:
			q = q
		cdef double m1 = m12/(1.0 + q)
		cdef double m2 = m12 - m1
		cdef double a1
		cdef double a2
		if rPer > 0.0:
			rPer = rPer
			a1 = (rPer*m2)/(m12*(1.0 - e))
			a2 = (rPer*m1)/(m12*(1.0 - e))
		else:
			raise ValueError('Separation at periapsis must be > 0.0 parsec')
		self.thisptr = new binarySMBH(rPer, m1, m2, e, d2r(omega), d2r(i), tau*Day, alpha1, alpha2)
	def __dealloc__(self):
		del self.thisptr
	'''def __call__(self, epoch):
		if (epoch != self.thisptr.getEpoch()):
			self.thisptr.call(epoch*Year)'''
	def getEpoch(self):
		return self.thisptr.getEpoch()
	def setEpoch(self, epoch):
		self.thisptr.setEpoch(epoch)
	def getPeriod(self):
		return self.thisptr.getPeriod()
	def getA1(self):
		return self.thisptr.getA1()
	def getA2(self):
		return self.thisptr.getA2()
	def getEccentricity(self):
		return self.thisptr.getEccentricity()
	def getCoordinates(self, which, epoch):
		if (epoch != self.thisptr.getEpoch()):
			self.thisptr.setEpoch(epoch)
		if which == 'm1':
			return self.thisptr.getR1()/Parsec, r2d(self.thisptr.getTheta1()), self.thisptr.getBeta1(), self.thisptr.getRadialBeta1(), self.thisptr.getDopplerFactor1(), self.thisptr.getBeamingFactor1()
		if which == 'm2':
			return self.thisptr.getR2()/Parsec, r2d(self.thisptr.getTheta2()), self.thisptr.getBeta2(), self.thisptr.getRadialBeta2(), self.thisptr.getDopplerFactor2(), self.thisptr.getBeamingFactor2()
	def aH(self, sigmaStars = SigmaOoM):
		return self.thisptr.aH(sigmaStars)
	def aGW(self, sigmaStars = SigmaOoM, rhoStars = RhoOoM, H = HOoM):
		return self.thisptr.aGW(sigmaStars, rhoStars, H);
	def durationInHardState(self, sigmaStars = SigmaOoM, rhoStars = RhoOoM, H = HOoM):
		return self.thisptr.durationInHardState(sigmaStars, rhoStars, H);
	def ejectedMass(self, sigmaStars = SigmaOoM, rhoStars = RhoOoM, H = HOoM):
		return self.thisptr.ejectedMass(sigmaStars, rhoStars, H);