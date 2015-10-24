import numpy as np		
import math as m
import CARMAFast as CF
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy.polynomial.polynomial as nppoly
import pdb


numFreqs = 1000
T_incr = 0.02
T = 4000

colormapping = {'0': '#a6cee3', '2': '#b2df8a', '4': '#fb9a99', '6': '#fdbf6f', '8': '#cab2d6', '10': '#ffff99'}

freqs = np.logspace(m.log10(1./T),m.log10(1./T_incr),numFreqs)

aListRoots = [-0.73642081, -0.01357919+0.75j, -0.01357919-0.75j]
aPoly = nppoly.polyfromroots(aListRoots)
aPoly = aPoly.tolist()
aPoly.pop(0)
aPoly.reverse()

print "aPoly"
print aPoly

#aList = [0.75,0.01]
aList = aPoly
bList = [7.0e-9,1.2e-9]

if (CF.checkParams(aList,bList) == 1):
	maxDenomOrder = 2*len(aList)
	maxNumerOrder = 2*(len(bList)-1)

	numerPSD = np.zeros((numFreqs,(maxNumerOrder/2) + 2))
	denomPSD = np.zeros((numFreqs,(maxDenomOrder/2) + 2))

	PSD = np.zeros((numFreqs))

	#plt.ion()
	plt.figure(1)
	
	for orderVal in xrange(0, maxNumerOrder + 1, 2):
		numerPSD[:,orderVal/2] = CF.getPSDNumerator(freqs,bList,orderVal)
		plt.loglog(freqs,numerPSD[:,orderVal/2],linestyle='--',color=colormapping[str(orderVal)],linewidth=2)

	for orderVal in xrange(0, maxDenomOrder + 1, 2):
		denomPSD[:,orderVal/2] = CF.getPSDDenominator(freqs,aList,orderVal)
		plt.loglog(freqs,1.0/denomPSD[:,orderVal/2],linestyle='--',color=colormapping[str(orderVal)],linewidth=2)
	for freq in xrange(freqs.shape[0]):
		for orderVal in xrange(0, maxNumerOrder + 1, 2):
			numerPSD[freq,(maxNumerOrder/2) + 1] += numerPSD[freq,orderVal/2]
		for orderVal in xrange(0, maxDenomOrder + 1, 2):
			denomPSD[freq,(maxDenomOrder/2) + 1] += denomPSD[freq,orderVal/2]
		PSD[freq] = numerPSD[freq,(maxNumerOrder/2) + 1]/denomPSD[freq,(maxDenomOrder/2) + 1]

	plt.loglog(freqs,1.0/denomPSD[:,(maxDenomOrder/2) + 1],linestyle=':',color='#e31a1c',linewidth=4)
	plt.loglog(freqs,numerPSD[:,(maxNumerOrder/2) + 1],linestyle=':',color='#1f78b4',linewidth=4)
	plt.loglog(freqs,PSD,linestyle='-',color='#000000',linewidth=4)

	plt.show()

else:
	print "Bad params!!"