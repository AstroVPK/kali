import numpy as np
import math as m
import KalmanFast as KF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from mpl_settings import *
import triangleVPK as VPK
import sys as s
import pdb

secPerSiderealDay = 86164.0905 
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
deltat=(numIntLC*(intTime+readTime))/secPerSiderealDay

def MAD(a):
	medianVal=np.median(a)
	b=np.copy(a)
	for i in range(a.shape[0]):
		b[i]=abs(b[i]-medianVal)
	return np.median(b)

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=600
nbins=100
set_plot_params(fontsize=12)

basePath=s.argv[1]
pMax=int(s.argv[2])
chop=int(s.argv[3])
#dictDIC=dict()
#fiftiethQ=dict()

#resultFilePath=basePath+'result.dat'
#resultFile=open(resultFilePath,'w')

for pNum in range(1,pMax+1,1):
	for qNum in range(0,pNum,1):
		TriFilePath=basePath+'mcmcOut_%d_%d.dat'%(pNum,qNum)
		TriFile=open(TriFilePath)
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		nsteps=int(values[1])
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		nwalkers=int(values[1])
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		ndim=int(values[1])
		walkers=np.zeros((nsteps,nwalkers,ndim))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=TriFile.readline()
				line.rstrip("\n")
				values=line.split()
				for k in range(ndim):
					walkers[i,j,k]=float(values[k+4])
		TriFile.close()

		medianWalker=np.zeros((nsteps,ndim))
		medianDevWalker=np.zeros((nsteps,ndim))
		for i in range(nsteps):
			for k in range(ndim):
				medianWalker[i,k]=np.median(walkers[i,:,k])
				medianDevWalker[i,k]=MAD(walkers[i,:,k])
		stepArr=np.arange(nsteps)

		plt.figure(1,figsize=(fwid,fhgt))
		for k in range(ndim):
			plt.subplot(ndim,1,k+1)
			for j in range(nwalkers):
				plt.plot(walkers[:,j,k],c='#000000',alpha=0.05,zorder=-5)
			plt.fill_between(stepArr[:],medianWalker[:,k]+medianDevWalker[:,k],medianWalker[:,k]-medianDevWalker[:,k],color='#ff0000',edgecolor='#ff0000',alpha=0.5,zorder=5)
			plt.plot(stepArr[:],medianWalker[:,k],c='#dc143c',linewidth=1,zorder=10)
			plt.xlabel('$step$')
			if (0<k<pNum+1):
				plt.ylabel("$\phi_{%d}$"%(k))
			elif ((k>=pNum+1) and (k<pNum+qNum+1)):
				plt.ylabel("$\\theta_{%d}$"%(k-pNum))
			else:
				plt.ylabel("$\sigma_{w}$")
		plt.savefig(basePath+"mcmcWalkers_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		plt.clf()

		samples=walkers[chop:,:,:].reshape((-1,ndim))
		lbls=list()
		lbls.append("$\sigma_{w}$")
		for i in range(pNum):
			lbls.append("$\phi_{%d}$"%(i+1))
		for i in range(qNum):
			lbls.append("$\\theta_{%d}$"%(i+1))
		figVPK,quantiles,qvalues=VPK.corner(samples,labels=lbls,show_titles=True,title_args={"fontsize": 12},quantiles=[0.16, 0.5, 0.84],verbose=False,plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth)
		figVPK.savefig(basePath+"mcmcVPKTriangles_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		figVPK.clf()

		del walkers
		del stepArr
		del medianWalker
		del medianDevWalker