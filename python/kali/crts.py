import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import subprocess
import re
import argparse
import matplotlib.pyplot as plt
import pdb

try:
	import kali.lc
	import kali.carma
except ImportError:
	print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
	sys.exit(1)


class crtsLC(kali.lc.lc):

	def read(self, name, band=r'V', path=None, **kwargs):
		if path is None:
			try:
				self.path = os.environ['CRTSDATADIR']
			except KeyError:
				raise KeyError(
					'Environment variable "CRTSDATADIR" not set! Please set "CRTSDATADIR" to point where all \
					CRTS data should live first...')
		else:
			self.path = path
		self.z = kwargs.get('z', 0.0)
		extension = kwargs.get('extension', '.txt')
		source = kwargs.get('source', 'batch')
		fullPath = os.path.join(self.path, name + extension)
		with open(fullPath, 'rb') as fileOpen:
			allLines = fileOpen.readlines()
		allLines = [line.rstrip('\n') for line in allLines]
		masterID = list()
		Mag = list()
		Magerr = list()
		Flux = list()
		Fluxerr = list()
		RA = list()
		Dec = list()
		MJD = list()
		for i in xrange(1, len(allLines)-1):
			splitLine = re.split(r'[ ,|\t]+', allLines[i])
			if int(splitLine[6]) == 0:
				masterID.append(int(float(splitLine[0])))
				Mag.append(float(splitLine[1]))
				Magerr.append(float(splitLine[2]))
				flux, fluxerr = kali.carma.pogsonFlux(float(splitLine[1]), float(splitLine[2]))
				Flux.append(flux)
				Fluxerr.append(fluxerr)
				RA.append(float(splitLine[3]))
				Dec.append(float(splitLine[4]))
				MJD.append(float(splitLine[5]))
		zipped = sorted(zip(MJD, masterID, Mag, Magerr, Flux, Fluxerr, RA, Dec))
		MJD, masterID, Mag, Magerr, Flux, Fluxerr, RA, Dec = zip(*zipped)
		self.startT = MJD[0]
		MJD = [(mjd - MJD[0])/(1.0 + self.z) for mjd in MJD]
		rMJD = list()
		newMJD = list()
		newRA = list()
		newDec = list()
		newmasterID = list()
		newMagerr = list()
		newMag = list()
		newFlux = list()
		newFluxerr = list()
		regressions = list()
		tempnewFlux = list()
		tempnewMJD = list()
		totalMag = 0
		totalMJD = 0
		sumMagerr = 0
		count = 0
		objCount = 1
		initialLength = len(MJD)
		rMJD = [MJD[i]//1 for i in range (0, len(MJD))]
		
		for i in range (0, initialLength - 1):
			if rMJD[i] == rMJD[i+1] and i != initialLength - 2:
				totalMag += Mag[i]
				totalMJD += MJD[i]
				tempnewMJD.append(MJD[i])
				tempnewflux, tempnewfluxerr = kali.carma.pogsonFlux((float(Mag[i])), float(Magerr[i]))
				tempnewFlux.append(tempnewflux)
				count += 1
			else:
				newRA.append(RA[i])
				newDec.append(Dec[i])
				newmasterID.append(masterID[i])
				totalMag += Mag[i]
				totalMJD += MJD[i]
				tempnewMJD.append(MJD[i])
				tempnewflux, tempnewfluxerr = kali.carma.pogsonFlux((float(Mag[i])), float(Magerr[i]))
				tempnewFlux.append(tempnewflux)
				count += 1
				averageMJD = totalMJD/count
				newMJD.append(averageMJD)
				averageMag = totalMag/count
				newMag.append(averageMag)
				if len(tempnewMJD) == 1:
					regressions.append(0)
				else:	
					m, b = np.polyfit(tempnewMJD, tempnewFlux, 1)
					regressions.append(m)
				count = 0
				totalMag = 0
				totalMJD = 0
				tempnewMJD = []
				tempnewFlux = []
				
		for j in range (0, initialLength - 1):
			if rMJD[j] == rMJD[j+1]:
				sumMagerr += (Mag[j]-newMag[objCount-1])**2
				count += 1
			else:
				sumMagerr += (Mag[j]-newMag[objCount-1])**2
				count += 1
				if count >= 2:
					newMagerr.append(np.sqrt(sumMagerr/(count-1)))
				else:
					newMagerr.append(np.sqrt(sumMagerr/(count)))
				objCount += 1
				count = 0
				sumMagerr = 0
		
		for k in range (0, len(newMJD)-1):
			newflux, newfluxerr = kali.carma.pogsonFlux(float(newMag[k]), float(newMagerr[k]))
			newFlux.append(newflux)
			newFluxerr.append(newfluxerr)
		
		self._numCadences = len(newMJD) - 1
		self.regressions = np.require(np.array(regressions), requirements=['F', 'A', 'W', 'O', 'E'])
		self.mask = np.require(np.array(self._numCadences*[1.0]), requirements=['F', 'A', 'W', 'O', 'E'])
		self.t = np.require(np.array(newMJD[:-1]), requirements=['F', 'A', 'W', 'O', 'E'])
		self.y = np.require(np.array(newFlux), requirements=['F', 'A', 'W', 'O', 'E'])
		self.yerr = np.require(np.array(newFluxerr), requirements=['F', 'A', 'W', 'O', 'E'])
		self.x = np.require(np.array(self._numCadences*[0.0]), requirements=['F', 'A', 'W', 'O', 'E'])
		self.mag = np.require(np.array(newMag), requirements=['F', 'A', 'W', 'O', 'E'])
		self.magerr = np.require(np.array(newMagerr), requirements=['F', 'A', 'W', 'O', 'E'])
		self.RA = np.require(np.array(newRA), requirements=['F', 'A', 'W', 'O', 'E'])
		self.Dec = np.require(np.array(newDec), requirements=['F', 'A', 'W', 'O', 'E'])
		self.masterID = np.require(np.array(newmasterID), requirements=['F', 'A', 'W', 'O', 'E'])
		self._name = str(name)  # The name of the light curve (usually the object's name).
		self._band = str(r'V')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
		self._xunit = r'$d$'  # Unit in which time is measured (eg. s, sec, seconds etc...).
		self._yunit = r'$F$ (Jy)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

	def write(self, name, path=None, **kwrags):
		pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(r'-n', r'--name', type=str, default=r'PG1302-102', help=r'Object name')
	parser.add_argument(r'-z', r'--z', type=float, default=0.2784, help=r'Object redshift')
	# parser.add_argument(r'-n', r'--name', type = str, default = r'OH287', help = r'Object name')
	# parser.add_argument(r'-z', r'--redShift', type = float, default = 0.305, help = r'Object redshift')
	args = parser.parse_args()

	LC = crtsLC(name=args.name, band='V', z=args.z)

	LC.plot()
	plt.show()
