import CARMAFast as CF
import math as math
import numpy as np
import random as r
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_settings import *
import scipy.optimize as opt
import emcee
import triangle
import time
import pdb
from Utilities import mail

goldenRatio=1.61803398875
fhgt=10.0
fwid=fhgt*goldenRatio
dotsPerInch=300

AnnotateLarge = 32
AnnotateMedium = 28
AnnotateSmall = 24

LegendLarge = 24
LegendMedium = 20
LegendSmall = 16

LabelXLarge = 32
LabelLarge = 28
LabelMedium = 24
LabelSmall = 20
LabelXSmall = 16

AxisXXLarge = 32
AxisXLarge = 28
AxisLarge = 24
AxisMedium = 20
AxisSmall = 16

normalFontSize=32
smallFontSize=24
footnoteFontSize=20
scriptFontSize=16
tinyFontSize=12

set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=AxisMedium,useTex='True')

percentile = 95.0
lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
upper = 100.0 - lower

nbins=100

redShift=0.0

intTime=6.019802903
readTime=0.5189485261
numIntLC=270
Deltat=((intTime+readTime)*numIntLC)/(1.0+redShift)
secPerSiderealDay=86164.0905

dt=0.02
T=2.0
numPts=int(T/dt)

TalkPath="/home/vish/Documents/AASWinter2015/Talk/"
CARMAPath="/home/vish/Documents/Research/CARMAEstimation/"
KeplerPath="/home/vish/Documents/Research/Kepler/"
startCadence=21145

burnSeed=r.randint(1000000000,9999999999)
distSeed=r.randint(1000000000,9999999999)
noiseSeed=r.randint(1000000000,9999999999)

#burnSeed=3471320269
#distSeed=231195107
#noiseSeed=3699689922

outPath=CARMAPath

#aMaster=[0.75,0.01]

aRoots = [-0.73642081+0.0j, -0.01357919-0.0j]#, -0.578976+0.25j, -0.578976-0.25j]
aPoly=(np.polynomial.polynomial.polyfromroots(aRoots)).tolist()
aPoly.reverse()
aPoly.pop(0)
aMaster=[coeff.real for coeff in aPoly]

sigma = 7.0e-9
bRoots=[-5.83333333]#-0.53642081+0j, -0.00563952+0j]
bPoly=(np.polynomial.polynomial.polyfromroots(bRoots)).tolist()
bPoly.reverse()
divisor=bPoly[-1].real
bMaster=[sigma*(coeff.real/divisor) for coeff in bPoly]
#bMaster = [1.2e-9, 7.0e-9]

print 'aMaster: '+str(aMaster)
print 'bMaster: '+str(bMaster)
aNum=len(aMaster)
bNum=len(bMaster)
m=aNum
p=aNum
q=bNum-1

noiseMaster=math.pow(10.0,-12.0)
numBurn=1000000

ndim=aNum+bNum
nwalkers=100
nsteps=500
chop=int(nsteps/2.0)

def CARMALnLikeMissing(Theta,dt,y,mask,p,q,m,A,B,F,I,Q,H,R,K):
	pTrial=Theta[0:m].tolist()
	qTrial=Theta[m:2*m+1].tolist()
	if (CF.checkParams(aList=pTrial,bList=qTrial)==0):
		logLike=-np.inf
	else:
		(X,P,XMinus,PMinus,F,Q)=CF.setSystem(dt,p,q,m,pTrial,qTrial,A,B,F,Q)
		logLike=CF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,Q,H,R,K)
	return logLike

def CARMANegLnLikeMissing(Theta,dt,y,mask,p,q,m,A,B,F,I,Q,H,R,K):
	return -CARMALnLikeMissing(Theta,dt,y,mask,p,q,m,A,B,F,I,Q,H,R,K)


(CARRoots,CMARoots)=CF.getRoots(aList=aMaster,bList=bMaster)
counter=0
for CARRoot in CARRoots:
	print "C-AR Root %d: %f+%fi"%(counter,CARRoot.real,CARRoot.imag)
	counter+=1
counter=0
for CMARoot in CMARoots:
	print "C-MA Root %d: %f+%fi"%(counter,CMARoot.real,CMARoot.imag)
	counter+=1

if CF.checkParams(aList=aMaster,bList=bMaster):

	t=np.zeros((numPts,2))
	y=np.zeros((numPts,2))
	mask=np.zeros(numPts)
	x=np.zeros((numPts,2))
	v=np.zeros((numPts,2))

	(p,q,m,A,B,F,I,Q,H,R,K)=CF.makeSystem(p,q,m)
	(X,P,XMinus,PMinus,F,Q)=CF.setSystem(dt,p,q,m,aMaster,bMaster,A,B,F,Q)
	X=CF.burnSystem(m,X,F,Q,numBurn,burnSeed)

	KeplerObj="kplr006932990Carini"
	KeplerFilePath=KeplerPath+KeplerObj+"/"+KeplerObj+"-calibrated.dat"
	data=np.loadtxt(KeplerFilePath,skiprows=2)
	numCadences=data.shape[0]
	startIndex=np.where(data[:,0]==startCadence)[0][0]
	counter=0
	for i in range(startIndex,startIndex+numPts):
		t[counter,0]=data[i,0]
		t[counter,1]=dt*counter
		if (data[i,2]!=0.0):
			mask[counter]=1.0
		counter+=1
	(X,y)=CF.obsSystemMissing(m,X,F,Q,H,noiseMaster,y,mask,numPts,distSeed,noiseSeed)

	yMean=np.nanmean(y[:,0])
	for i in range(numPts):
		y[i,0]-=yMean

	plt.figure(1,figsize=(fwid,fhgt))
	yMax=np.max(y[np.nonzero(y[:,0]),0])
	yMin=np.min(y[np.nonzero(y[:,0]),0])
	plt.ylabel('$F$ (arb units)')
	plt.xlabel('$t$ (d)')
	for i in range(numPts):
		if (mask[i]==1.0):
			plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	plt.xlim(t[0,1],t[-1,1])
	#plt.ylim(yMin,yMax)
	plt.tight_layout()
	plt.draw()
	plt.savefig(outPath+"CARMA(%d,%d)_Master.jpg"%(p,q),dpi=dotsPerInch)

	(X,P,XMinus,PMinus,F,Q)=CF.setSystem(dt,p,q,m,aMaster,bMaster,A,B,F,Q)
	print "Master LnLike: %f"%(CF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,Q,H,R,K))

	thetaInit = list()
	for i in range(aNum):
		thetaInit.append(r.gauss(0.0,1e-10)+aMaster[i])
	for i in range(bNum):
		thetaInit.append(r.gauss(0.0,1e-10)+bMaster[i])
	result=opt.fmin_powell(CARMANegLnLikeMissing,x0=thetaInit,args=(dt,y,mask,p,q,m,A,B,F,I,Q,H,R,K),ftol=0.001,disp=1)

	pInferred=list()
	qInferred=list()
	for i in range(0,p):
		pInferred.append(result[i])
		print "phi[%d]: %e"%(i+1,result[i])
	for i in range(p,p+q+1):
		qInferred.append(result[i])
		print "theta[%d]: %e"%(i-m+1,result[i])

	(X,P,XMinus,PMinus,F,Q)=CF.setSystem(dt,p,q,m,pInferred,qInferred,A,B,F,Q)
	CF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,Q,H,R,K)
	nBins=50
	binVals,binEdges=np.histogram(v[~np.isnan(v[:,0]),0],bins=nBins,range=(0.1*np.nanmin(v[1:numPts,0]),0.1*np.nanmax(v[1:numPts,0])))
	binMax=np.nanmax(binVals)
	newBinVals=np.zeros(nBins)
	newBinErrors=np.zeros(nBins)
	tMax=np.nanmax(t[:,1])
	for i in range(nBins):
		newBinVals[i]=(tMax/4.0)*(float(binVals[i])/binMax)
		newBinErrors[i]=(tMax/4.0)*(math.sqrt(float(binVals[i]))/binMax)
	binWidth=np.asscalar(binEdges[1]-binEdges[0])

	plt.figure(2,figsize=(fwid,fhgt))
	yMax=np.max(y[np.nonzero(y[:,0]),0])
	yMin=np.min(y[np.nonzero(y[:,0]),0])
	plt.ylabel('$F$ (arb units)')
	plt.xlabel('$t$ (d)')
	for i in range(numPts):
		if (mask[i]==1.0):
			plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=-5)
	plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=0.75,zorder=-10)
	plt.xlim(t[0,1],t[-1,1])
	plt.tight_layout()
	plt.draw()
	plt.savefig(outPath+"CARMA(%d,%d)_PowellFit.jpg"%(p,q),dpi=dotsPerInch)

	YesNo=input("Happy? (1/0): ")
	#YesNo=1

	if (YesNo):
		pos=[np.array(result)+1e-10*np.random.randn(ndim) for i in range(nwalkers)]
		sampler=emcee.EnsembleSampler(nwalkers,ndim,CARMALnLikeMissing,a=2.0,args=(dt,y,mask,p,q,m,A,B,F,I,Q,H,R,K),threads=4)
		beginT=time.time()
		sampler.run_mcmc(pos,nsteps)
		endT=time.time()
		print "That took %f sec"%(endT-beginT)
		plt.figure(3,figsize=(fwid,fhgt))
		for j in range(ndim):
			plt.subplot(ndim,1,j+1)
			for i in range(nwalkers):
				plt.plot(sampler.chain[i,:,j],c='#000000',alpha=0.5)
			plt.xlabel('$step$')
			if (j<aNum):
				plt.ylabel("$\phi_{%d}$"%(j+1))
			elif ((j>=aNum) and (j<aNum+bNum)):
				plt.ylabel("$\\theta_{%d}$"%(j-aNum+1))
			plt.tight_layout()
			plt.draw()
		#plt.show()
		plt.savefig(outPath+"CARMA(%d,%d)_Walkers.jpg"%(p,q),dpi=dotsPerInch)
		WalkersFilePath=outPath+'CARMA(%d,%d)_Walkers.dat'%(p,q)
		WalkersFile=open(WalkersFilePath,'w')
		WalkersFile.write("nsteps: %d\n"%(nsteps))
		WalkersFile.write("nwalkers: %d\n"%(nwalkers))
		WalkersFile.write("ndim: %d\n"%(ndim))
		aMasterNum=len(aMaster)
		bMasterNum=len(bMaster)
		WalkersFile.write("pMasterNum: %d\n"%(aMasterNum))
		WalkersFile.write("qMasterNum: %d\n"%(bMasterNum-1))
		for i in range(aMasterNum):
			WalkersFile.write("p_%d: %e\n"%(i+1,aMaster[i]))
		for i in range(bMasterNum):
			WalkersFile.write("q_%d: %e\n"%(i+1,bMaster[i]))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=''
				for k in range(ndim):
					line+='%+17.16e '%(sampler.chain[j,i,k])
				line += '%+17.16e'%(sampler.lnprobability[j,i])
				line+='\n'
				WalkersFile.write(line)
		WalkersFile.close()
		samples=sampler.chain[:,chop:,:].reshape((-1,ndim))

		lbls=[]
		for i in range(aNum):
			lbls.append("$\phi_{%d}$"%(i+1))
		for i in range(bNum):
			lbls.append("$\\theta_{%d}$"%(i+1))
		fig,quantiles,allqvalues = triangle.corner(samples,labels=lbls,truths=aMaster+bMaster,truth_color='#000000',show_titles=True,title_args={"fontsize":12},quantiles=[lower/100.0, 0.5, upper/100.0],plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth,bins=nbins)
		fig.savefig(outPath+"CARMA(%d,%d)_MCMC.jpg"%(p,q),dpi=dotsPerInch)

		pMCMC=list()
		qMCMC=list()
		distMCMC=0.0
		for i in range(0,p):
			pMCMC.append(np.percentile(samples[:,i],50.0))
			print "phi[%d]: %e"%(i,pMCMC[i])
		for i in range(p,p+q+1):
			qMCMC.append(np.percentile(samples[:,i],50.0))
			print "theta[%d]: %e"%(i-m,qMCMC[i-m])

		(X,P,XMinus,PMinus,F,Q)=CF.setSystem(dt,p,q,m,pMCMC,qMCMC,A,B,F,Q)
		CF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,Q,H,R,K)
		nBins=50
		binVals,binEdges=np.histogram(v[~np.isnan(v[:,0]),0],bins=nBins,range=(0.1*np.nanmin(v[1:numPts,0]),0.1*np.nanmax(v[1:numPts,0])))
		binMax=np.nanmax(binVals)
		newBinVals=np.zeros(nBins)
		newBinErrors=np.zeros(nBins)
		tMax=np.nanmax(t[:,1])
		for i in range(nBins):
			newBinVals[i]=(tMax/4.0)*(float(binVals[i])/binMax)
			newBinErrors[i]=(tMax/4.0)*(math.sqrt(float(binVals[i]))/binMax)
		binWidth=np.asscalar(binEdges[1]-binEdges[0])

		plt.figure(5,figsize=(fwid,fhgt))
		plt.subplot(211)
		yMax=np.max(y[np.nonzero(y[:,0]),0])
		yMin=np.min(y[np.nonzero(y[:,0]),0])
		plt.ylabel('$F$ (arb units)')
		plt.xlabel('$t$ (d)')
		for i in range(numPts):
			if (mask[i]==1.0):
				plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=-5)
		plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=0.75,zorder=-10)
		plt.xlim(t[0,1],t[-1,1])
		#plt.ylim(yMin,yMax)
		plt.tight_layout()
		plt.draw()
		plt.subplot(212)
		vMax=np.max(v[np.nonzero(v[:,0]),0])
		vMin=np.min(v[np.nonzero(v[:,0]),0])
		plt.ylabel('$\Delta F$ (arb units)')
		plt.xlabel('$t$ (d)')
		for i in range(numPts):
			if (mask[i]==1.0):
				plt.errorbar(t[i,1],v[i,0],yerr=v[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		plt.barh(binEdges[0:nBins],newBinVals[0:nBins],xerr=newBinErrors[0:nBins],height=binWidth,alpha=0.4,zorder=10)
		plt.xlim(t[0,1],t[-1,1])
		plt.ylim(0.1*vMin,0.1*vMax)
		plt.tight_layout()
		plt.draw()
		plt.savefig(outPath+"CARMA(%d,%d)_MCMCFit.jpg"%(p,q),dpi=dotsPerInch)
		#plt.show()
		#mail('vishal.kasliwal@gmail.com','Earth to Vishal...',"All done!",outPath+"CARMA(%d,%d)_Master.jpg"%(aNum,bNum),outPath+"CARMA(%d,%d)_PowellFit.jpg"%(aNum,bNum),outPath+"CARMA(%d,%d)_Walkers.jpg"%(aNum,bNum),outPath+"CARMA(%d,%d)_MCMC.jpg"%(aNum,bNum),outPath+"CARMA(%d,%d)_MCMCFit.jpg"%(aNum,bNum))

else:
	print "Bad Initial Params!"
