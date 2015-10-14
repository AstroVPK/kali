import numpy as np
import math as m
import random as r
import CARMAFast as CF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from mpl_settings import *
import scipy.linalg as la
import numpy.linalg as npla 
import triangle as VPK
import sys as s
import pdb

s1=2
s2=9
fwid=8.5
fhgt=5.25
dotsPerInch=600
nbins=100
set_plot_params(fontsize=12)

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

basePath=s.argv[1]
pMax=int(s.argv[2])
chop=int(s.argv[3])
dictDIC=dict()
fiftiethQ=dict()

resultFilePath=basePath+'result.dat'
resultFile=open(resultFilePath,'w')

for pNum in range(0,pMax+1,1):
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
		deviances=np.zeros((nsteps,nwalkers))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=TriFile.readline()
				line.rstrip("\n")
				values=line.split()
				for k in range(ndim):
					walkers[i,j,k]=float(values[k+4])
				deviances[i,j]=-2.0*float(values[-1])
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
			plt.xlabel('stepNum')
			if (0 <= k < pNum):
				plt.ylabel("$a_{%d}$"%(k+1))
			elif ((k >= pNum) and (k < pNum + qNum + 1)):
				plt.ylabel("$b_{%d}$"%(k-pNum))

		#plt.tight_layout()
		plt.savefig(basePath+"mcmcWalkers_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		plt.clf()

		samples=walkers[chop:,:,:].reshape((-1,ndim))
		sampleDeviances=deviances[chop:,:].reshape((-1))
		DIC=0.5*m.pow(np.std(sampleDeviances),2.0) + np.mean(sampleDeviances)
		dictDIC["%d %d"%(pNum,qNum)]=DIC
		lbls=list()
		for i in range(pNum):
			lbls.append("$a_{%d}$"%(i+1))
		for i in range(qNum+1):
			lbls.append("$b_{%d}$"%(i))
		figVPK,quantiles,qvalues=VPK.corner(samples,labels=lbls,fig_title="DIC: %f"%(dictDIC["%d %d"%(pNum,qNum)]),show_titles=True,title_args={"fontsize": 12},quantiles=[0.16, 0.5, 0.84],verbose=False,plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth)

		figVPK.savefig(basePath+"mcmcVPKTriangles_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		figVPK.clf()

		line1="p: %d; q: %d\n"%(pNum,qNum)
		resultFile.write(line1)
		line2="DIC: %e\n"%(DIC)
		resultFile.write(line2)
		for k in range(ndim):
			if (0 <= k < pNum):
				line3="a_%d\n"%(k)
			elif ((k >= pNum) and (k < pNum + qNum + 1)):
				line3="b_%d\n"%(k-pNum)
			resultFile.write(line3)
			fiftiethQ["%d %d %s"%(pNum,qNum,line.rstrip("\n"))]=float(qvalues[k][1])
			for i in range(len(quantiles)):
				line4="Quantile: %.2f; Value: %e\n"%(quantiles[i],qvalues[k][i])
				resultFile.write(line4)
		line5="\n"
		resultFile.write(line5)

		del walkers
		del deviances
		del stepArr
		del medianWalker
		del medianDevWalker

sortedDICVals=sorted(dictDIC.items(),key=operator.itemgetter(1))
pBest=int(sortedDICVals[0][0].split()[0])
qBest=int(sortedDICVals[0][0].split()[1])
line="Model Appropriateness (in descending order of appropriateness) & Relative Likelihood (i.e. Relative Likelihood of Minimal Infrmation Loss)\n"
resultFile.write(line)
line="Model       DIC Value    Relative Likelihood\n"
resultFile.write(line)
line="-----       ---------    -------------------\n"
resultFile.write(line)
for i in range(len(sortedDICVals)):
	RelProbOfMinInfLoss=100.0*m.exp(0.5*(float(sortedDICVals[0][1])-float(sortedDICVals[i][1])))
	line='{:>4}   {:> 13.3f}    {:> 18.2f}%\n'.format(sortedDICVals[i][0],float(sortedDICVals[i][1]),RelProbOfMinInfLoss)
	resultFile.write(line);

bestFilePath=basePath+'mcmcOut_%d_%d.dat'%(pBest,qBest)
bestFile=open(bestFilePath)
line=bestFile.readline()
line.rstrip("\n")
values=line.split()
nsteps=int(values[1])
line=bestFile.readline()
line.rstrip("\n")
values=line.split()
nwalkers=int(values[1])
line=bestFile.readline()
line.rstrip("\n")
values=line.split()
ndim=int(values[1])
walkers=np.zeros((nsteps,nwalkers,ndim))
for i in range(nsteps):
	for j in range(nwalkers):
		line=bestFile.readline()
		line.rstrip("\n")
		values=line.split()
		for k in range(ndim):
			walkers[i,j,k]=float(values[k+4])
bestFile.close()

randStep=r.randint(chop,nsteps-1)
randWalker=r.randint(0,nwalkers-1)

aBest=[dim for dim in walkers[randStep,randWalker,0:pBest].tolist()]
bBest=[dim for dim in walkers[randStep,randWalker,pBest:pBest+qBest+1].tolist()]

line="\n"
resultFile.write(line);
line="Randomly chosen values from posterior distribution\n"
resultFile.write(line);
line="--------------------------------------------------\n"
resultFile.write(line);
for i in xrange(pBest):
	line="a_%d: %e\n"%(i+1,aBest[i])
	resultFile.write(line);
for i in xrange(qBest+1):
	line="b_%d: %e\n"%(i,bBest[i])
	resultFile.write(line);
resultFile.close()

yFilePath=basePath+'y.dat'
yFile=open(yFilePath)
line=yFile.readline()
line.rstrip("\n")
values=line.split()
numPts=int(values[1])
line=yFile.readline()
line.rstrip("\n")
values=line.split()
numObs=int(values[1])
line=yFile.readline()
line.rstrip("\n")
values=line.split()
meanY=float(values[1])

t=np.zeros((numPts,2))
y=np.zeros((numPts,2))
mask=np.zeros(numPts)
x=np.zeros((numPts,2))
v=np.zeros((numPts,2))

for i in range(numPts):
	line=yFile.readline()
	line.rstrip("\n")
	values=line.split()
	t[i,0]=int(values[0])
	t[i,1]=deltat*i
	mask[i]=float(values[1])
	y[i,0]=float(values[2])
	y[i,1]=float(values[3])

(p,A,B,F,I,D,Q,H,R,K)=CF.makeSystem(pBest)
(X,P,XMinus,PMinus,F,I,D,Q)=CF.setSystem(deltat,p,aBest,bBest,A,B,F,I,D,Q)
LnLike=CF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
r,x=CF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)

plt.figure(2,figsize=(fwid,2*fhgt))

plt.subplot(211)
yMax=np.max(y[np.nonzero(y[:,0]),0])
yMin=np.min(y[np.nonzero(y[:,0]),0])
plt.ylabel('$F$ (arb units)')
plt.xlabel('$t$ (d)')
for i in range(numPts):
	if (mask[i]==1.0):
		plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=-5)
plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=10)
plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=1,zorder=5)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(yMin,yMax)

plt.subplot(212)
vMax=np.std(v[np.nonzero(v[:,0]),0])
vMin=-np.std(v[np.nonzero(v[:,0]),0])

nBins=50
binVals,binEdges=np.histogram(v[~np.isnan(v[:,0]),0],bins=nBins,range=(vMin,vMax))
binMax=np.nanmax(binVals)
newBinVals=np.zeros(nBins)
newBinErrors=np.zeros(nBins)
tMax=np.nanmax(t[:,1])
for i in range(nBins):
	newBinVals[i]=(tMax/4.0)*(float(binVals[i])/binMax)
	newBinErrors[i]=(tMax/4.0)*(m.sqrt(float(binVals[i]))/binMax)
binWidth=np.asscalar(binEdges[1]-binEdges[0])

plt.ylabel('$\Delta F$ (arb units)')
plt.xlabel('$t$ (d)')
for i in range(numPts):
	if (mask[i]==1.0):
		plt.errorbar(t[i,1],v[i,0],yerr=v[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
plt.barh(binEdges[0:nBins],newBinVals[0:nBins],xerr=newBinErrors[0:nBins],height=binWidth,alpha=0.4,zorder=10)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(vMin,vMax)

plt.tight_layout()
plt.savefig(basePath+"lc_%d_%d.jpg"%(pBest,qBest),dpi=dotsPerInch)
plt.clf()