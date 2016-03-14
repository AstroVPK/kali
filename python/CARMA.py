import math as math
import cmath as cmath
import numpy as np
import numpy.linalg as la

class CARMA(object):
	def __init__(self, p, q):
		self.p = p
		self.q = q

		self._dt = None

		self.A = np.matrix(np.zeros((self.p, self.p), dtype = 'complex128'))
		for i in xrange(1, p):
			A[i-1,i] = 1.0 + 0.0j
		self.vr = None
		self.vrInv = None
		self.w = None
		self.expw = np.matrix(np.zeros((self.p, self.p), dtype = 'complex128'))
		self.B = np.matrix(np.zeros((self.p, 1), dtype = 'complex128'))
		self.C = None
		self.H = np.matrix(np.zeros((1, self.p), dtype = 'float64'))
		self.F = None
		self.Q = np.matrix(np.zeros((self.p, self.p), dtype = 'float64'))

		self.X = np.matrix(np.zeros((self.p, 1), dtype = 'float64'))
		self.P = None
		self.XMinus = None
		self.PMinus = None

	@property
	def dt(self):
		return self._dt

	@dt.setter
	def dt(self, value):
		self._dt = value

	def setCARMA(self, Theta):
		for i in xrange(0, self.p):
			A[i,0] = -1.0*Theta[i] + 0.0j
		self.w, self.vr = la.eig(self.A)
		self.vr = np.matrix(self.vr)
		self.vrInv = np.matrix(la.inv(self.vr))
		self.w = np.matrix(self.w)

		for i in xrange(self.p, self.p + self.q):
			B[2.0*self.p-i,0] = Theta[i] + 0.0j

		self.C = self.vrInv*self.B*np.matrix(np.transpose(self.B))*np.matrix(np.transpose(self.vrInv))

	def solveCARMA(self):
		for i in xrange(self.p):
			expw[i,i] = cmath.exp(self.w[i,i]*self.dt)
		self.F = self.vr*self.expw*self.vrInv