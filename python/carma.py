import math as math
import cmath as cmath
import numpy as np

class CARMA:
	def __init__(self):

	@staticmethod
	def checkParams(aList = None, bList = None):
		if aList is None:
			raise ValueError('#CAR > 0')
		self.p = len(aList)

		if bList is None:
			raise ValueError('#CMA > 0')
		self.q = len(bList) - 1

		hasUniqueEigenValues=1
		isStable=1
		isInvertible=1
		isNotRedundant=1
		hasPosSigma=1

		CARPoly=list()
		CARPoly.append(1.0)
		for i in xrange(p):
			CARPoly.append(aList[i])
		CARRoots=np.roots(CARPoly)
		if (len(CARRoots)!=len(set(CARRoots))):
			hasUniqueEigenValues=0
		for CARRoot in CARRoots:
			if (CARRoot.real>=0.0):
				isStable=0

		isInvertible=1
		CMAPoly=list()
		for i in xrange(q + 1):
			CMAPoly.append(bList[i])
		CMAPoly.reverse()
		CMARoots=np.roots(CMAPoly)
		if (len(CMARoots)!=len(set(CMARoots))):
			uniqueRoots=0
		for CMARoot in CMARoots:
			if (CMARoot>0.0):
				isInvertible=0

		isNotRedundant=1
		for CARRoot in CARRoots:
			for CMARoot in CMARoots:
				if (CARRoot==CMARoot):
					isNotRedundant=0
		return isStable*isInvertible*isNotRedundant*hasUniqueEigenValues*hasPosSigma

	def getPSDDenominator(freqs, aList, order):
		pVal = len(aList)
		numFreqs = freqs.shape[0]
		aList.insert(0, 1.0)
		PSDVals = npzeros(numFreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*pVal)):
			aList.pop(0)
			return PSDVals
		else:
			for freq in xrange(freqs.shape[0]):
				val = 0.0
				for i in xrange(pVal + 1):
					j = 2*pVal - i - order
					if ((j >= 0) and (j < pVal + 1)):
						val += (aList[i]*aList[j]*((2.0*pi*1j*freqs[freq])**(2*pVal - (i + j)))*pow(-1.0, pVal - j)).real
					PSDVals[freq] = val
			aList.pop(0)
			return PSDVals
	
	def getPSDNumerator(freqs, bList, order):
		qVal = len(bList) - 1
		numFreqs = freqs.shape[0]
		PSDVals = npzeros(numFreqs)
		if ((order % 2 == 1) or (order <= -1) or (order > 2*qVal)):
			return PSDVals
		else:
			for freq in xrange(freqs.shape[0]):
				val = 0.0
				for i in xrange(qVal + 1):
					j = 2*qVal - i - order
					if ((j >= 0) and (j < qVal + 1)):
						val += (bList[i]*bList[j]*((2.0*pi*1j*freqs[freq])**(2*qVal - (i + j)))*pow(-1.0, qVal - j)).real
					PSDVals[freq] = val
			return PSDVals