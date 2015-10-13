from math import pow,log,pi,sqrt
from numpy import transpose,roots,kron,reshape,isnan,nan,sum,inf
from numpy import zeros as npzeros
from numpy.matlib import matrix,zeros,identity
from numpy.linalg import inv,solve,det
from scipy.linalg import expm
from scipy.integrate import quad
from random import gauss, seed
from sys import float_info
import numpy as np
import pdb

intTime=6.019802903
readTime=0.5189485261
numIntLC=270
Deltat=(intTime+readTime)*numIntLC
secPerSiderealDay=86164.0905
dt=Deltat/secPerSiderealDay

def makeSystem(m):

	A=zeros((m,m))
	for i in xrange(m-1):
		A[i,i+1]=1.0
	B=zeros((m,1))

	F=zeros((m,m))
	I=identity(m)
	D=zeros((m,1))
	Q=zeros((m,m))
	H=zeros((1,m))
	R=zeros((1,1))
	K=zeros((m,1))

	return (m,A,B,F,I,D,Q,H,R,K)

def checkParams(aList=None,bList=None):
	if aList is None:
		raise ValueError('#CAR > 0')
	p = len(aList)

	if bList is None:
		raise ValueError('#CMA > 0')
	if (len(bList)!=len(aList)):
		raise ValueError('#CMA == #CAR (pad with 0s in front!)')

	uniqueRoots=1
	isStable=1
	CARPoly=list()
	CARPoly.append(1.0)
	for i in xrange(p):
		CARPoly.append(aList[i])
	CARRoots=roots(CARPoly)
	if (len(CARRoots)!=len(set(CARRoots))):
		uniqueRoots=0
	for CARRoot in CARRoots:
		if (CARRoot.real>=0.0):
			isStable=0

	isInvertible=1
	CMAPoly=list()
	for i in xrange(p):
		CMAPoly.append(bList[i])
	CMARoots=roots(CMAPoly)
	if (len(CMARoots)!=len(set(CMARoots))):
		uniqueRoots=0
	for CMARoot in CMARoots:
		if (CMARoot>=0.0):
			isInvertible=0

	isNotRedundant=1
	for CARRoot in CARRoots:
		for CMARoot in CMARoots:
			if (CARRoot==CMARoot):
				isNotRedundant=0

	return isStable*isInvertible*isNotRedundant*uniqueRoots

def getRoots(aList=None,bList=None):
	if aList is None:
		raise ValueError('#CAR > 0')
	p = len(aList)

	if bList is None:
		raise ValueError('#CMA > 0')
	if (len(bList)!=len(aList)):
		raise ValueError('#CMA == #CAR (pad with 0s in front!)')

	CARPoly=list()
	CARPoly.append(1.0)
	for i in xrange(p):
		CARPoly.append(aList[i])
	CARRoots=roots(CARPoly)

	CMAPoly=list()
	for i in xrange(p):
		CMAPoly.append(bList[i])
	CMARoots=roots(CMAPoly)

	return (CARRoots,CMARoots)

def Qint(dchi,A,B,i,j):
	ans = ((expm(A*dchi))*B*(B.T)*((expm(A*dchi)).T))[i,j]
	return ans

def setSystem(dt,m,aList,bList,A,B,F,I,D,Q):

	for i in xrange(m):
		A[i,0]=-aList[i]
		B[i,0]=bList[i]

	F=expm(A*dt)

	'''for i in xrange(m):
		for j in xrange(i+1):
			Q[i,j]=quad(Qint,0,dt,args=(A,B,i,j),epsrel=0.0001)[0]
			Q[j,i]=Q[i,j]
		D[i,0]=sqrt(Q[i,i])

	DPrime = zeros((m,1))
	QPrime = zeros((m,m))'''
	for i in xrange(m):
		D[i,0]=sqrt(quad(Qint,0,dt,args=(A,B,i,i),epsrel=0.0001)[0]) # Tested
		'''D[i,0]=quad(Qint,0,dt,args=(A,B,i,i),epsrel=0.0001)[0]''' # Untested
	Q = D*(D.T)

	X=zeros((m,1))
	try:
		P=matrix(reshape(solve(kron(I,I)-kron(F,F),reshape(Q,(m*m,1))),(m,m)))
	except np.linalg.linalg.LinAlgError:
		print "Computation of P failed!"
		print aList
		print bList
		print
		P=zeros((m,m))
		for i in range(m):
			P[i,i]=10**7
	XMinus=zeros((m,1))
	PMinus=zeros((m,m))
	return (X,P,XMinus,PMinus,F,I,D,Q)

def setSystemDiffuse(dt,m,aList,bList,A,B,F,I,D,Q):
	for i in xrange(p):
		A[i,0]=-aList[i]
		B[i]=bList[i]

	F=expm(newA*dt)

	for i in xrange(m):
		for j in xrange(i+1):
			Q[i,j]=quad(Qint,0,dt,args=(A,B,i,j),epsrel=0.001)[0]
			Q[j,i]=Q[i,j]
		D[i,0]=sqrt(Q[i,i])

	X=zeros((m,1))
	P=zeros((m,m))
	for i in range(m):
		P[i,i]=10**7
	XMinus=zeros((m,1))
	PMinus=zeros((m,m))
	H[0,0]=1.0
	return (X,P,XMinus,PMinus,F,I,D,Q)

def burnSystemFixed(X,F,D,numBurn,burnRand):
	for i in range(numBurn):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(burnRand[i])
		print "X_-"
		print X'''
		X=F*X+burnRand[i]*D
		'''print "X_+"
		print X
		print'''
	return X

def burnSystem(X,F,D,numBurn,burnSeed):
	burnRand=npzeros(numBurn)
	seed(burnSeed)
	for i in range(numBurn):
		burnRand[i]=gauss(0.0,1.0)
	for i in range(numBurn):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(burnRand[i])
		print "X_-"
		print X'''
		X=F*X+burnRand[i]*D
		'''print "X_+"
		print X
		print'''
	return X

def obsSystemFixed(X,F,D,H,noise,y,numObs,distRand,noiseRand):
	H[0,0]=1.0
	for i in range(numObs):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+distRand[i]*D
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+noiseRand[i]
		y[i,1]=noise
		'''print "y"
		print y[i,0]'''
	return (X,y)

def obsSystemFixedMissing(X,F,D,H,noise,y,mask,numObs,distRand,noiseRand):
	for i in range(numObs):
		H[0,0]=mask[i]
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+distRand[i]*D
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+noiseRand[i]
		if (mask[i]==1.0):
			y[i,1]=noise
		else:
			y[i,1]=inf
		'''print "y"
		print y[i,0]'''
	return (X,y)

def obsSystem(X,F,D,H,noise,y,numObs,distSeed,noiseSeed):
	H[0,0]=1.0
	distRand=npzeros(numObs)
	noiseRand=npzeros(numObs)
	seed(distSeed)
	for i in range(numObs):
		distRand[i]=gauss(0.0,1.0)
	seed(noiseSeed)
	for i in range(numObs):
		noiseRand[i]=gauss(0.0,noise)
	for i in range(numObs):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+distRand[i]*D
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+noiseRand[i]
		y[i,1]=noise
		'''print "y"
		print y[i,0]'''
	return (X,y)

def obsSystemMissing(X,F,D,H,noise,y,mask,numObs,distSeed,noiseSeed):
	veryLarge = float_info[0]
	distRand=npzeros(numObs)
	noiseRand=npzeros(numObs)
	seed(distSeed)
	for i in range(numObs):
		distRand[i]=gauss(0.0,1.0)
	seed(noiseSeed)
	for i in range(numObs):
		noiseRand[i]=gauss(0.0,noise)
	for i in range(numObs):
		H[0,0]=mask[i]
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+distRand[i]*D
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+H[0,0]*noiseRand[i]
		if (mask[i]==1.0):
			y[i,1]=noise
		else:
			y[i,1]=sqrt(veryLarge)
		'''print "y"
		print y[i,0]'''
	return (X,y)

def gaussian(x, mu, sigma):
	numPts = array(x).shape[0]
	return exp(-power(x - mu, 2.0)/(2.0*power(sigma, 2.0)))

def plotLnLike(t,y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	veryLarge = float_info[0]
	numPts=y.shape[0]
	LnLike=0.0
	ptCounter=0.0

	plotPath = '/home/exarkun/Desktop/KalmanPlots/'
	PDF = False
	HList = list()
	RList = list()
	XList = list()
	PList = list()
	XMinusList = list()
	PMinusList = list()
	vList = list()
	SList = list()
	KList = list()
	LnLikeList = list()

	XList.append(X)
	PList.append(P)
	H[0,0] = 0.0
	HList.append(H)
	R[0,0] = float_info[0]
	RList.append(R)
	XMinusList.append(XMinus)
	PMinusList.append(PMinus)
	vList.append(0.0)
	SList.append(float_info[0])
	KList.append(K)
	LnLikeList.append(0.0)
	tList = t[:,1].tolist()
	tList.insert(0,-1.0)

	for i in range(numPts):
		H[0,0]=mask[i]
		R[0,0]=y[i,1]*y[i,1]

		HList.append(H)
		RList.append(R)

		XMinus=F*X

		PMinus=F*P*transpose(F)+Q

		XMinusList.append(XMinus)
		PMinusList.append(PMinus)

		v=y[i,0]-H*XMinus

		S=H*PMinus*transpose(H)+R

		vList.append(v)
		SList.append(S)

		inverseS=inv(S)

		K=PMinus*transpose(H)*inverseS

		KList.append(K)

		IMinusKH=I-K*H

		X=K*y[i,0]+IMinusKH*XMinus

		P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)

		XList.append(X)
		PList.append(P)

		LnLike+=(-0.5*(transpose(v)*inverseS*v)-0.5*log(det(S)))[0,0]

		LnLikeList.append((-0.5*(transpose(v)*inverseS*v)-0.5*log(det(S)))[0,0])

	LnLike += -0.5*numPts*log(2.0*pi)

	fig1 = plt.figure(1,figsize=(fwid,fhgt))
	fig2 = plt.figure(2,figsize=(fwid,fwid))
	gs = gridspec.GridSpec(1000, 1000)
	ax1 = fig1.add_subplot(gs[:,:])
	ax2 = fig2.add_subplot(gs[:,:])
	for i in xrange(y.shape[0]):
		if (y[i,1] < 1e+154):
			ax1.errorbar(t[i,1],y[i,0],y[i,1],fmt='o',capsize=0,markeredgecolor='none',color='#d95f02',zorder=10)
			ax1.annotate(r'$y_{%d}$'%(i),xy=(t[i,1],y[i,0]),fontsize=AnnotateSmall,xycoords='data',xytext=(8,-6),textcoords='offset points')
	Pt = 4
	fill_t=list()
	fill_x=list()
	fill_xl=list()
	fill_xu=list()
	for i in xrange(Pt+1):
		fill_t.append(tList[i])
		fill_x.append(XList[i][0,0])
		fill_xl.append(XList[i][0,0]-sqrt(PList[i][0,0]))
		fill_xu.append(XList[i][0,0]+sqrt(PList[i][0,0]))
	ax1.plot(fill_t,fill_x,'-',color='#666666',zorder=5)
	ax1.fill_between(fill_t,fill_xl,fill_xu,color='#b3b3b3',zorder=0)
	ax1.set_xlim(-1.0,8.5)
	ax1.set_ylim(-4.0e-8,1e-8)

	ax2.annotate(r'',xy=(XList[Pt][0,0],XList[Pt][1,0]),fontsize=AnnotateSmall,xycoords='data',xytext=(0,0),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#4daf4a'),zorder=10)
	ax2.annotate(r'$\widehat{\textbf{\textit{x}}}^{+}_{%d}$'%(Pt-1),fontsize=AnnotateSmall,xy=(XList[Pt][0,0],XList[Pt][1,0]),xycoords='data',xytext=(36,-16),textcoords='offset points')
	larger=max(XList[Pt][0,0],XList[Pt][1,0])
	#ax2.set_xlim(5.0*larger,-5.0*larger)
	#ax2.set_ylim(5.0*larger,-5.0*larger)
	ax2.set_xlim(3.0*larger,-3.0*larger)
	ax2.set_ylim(3.0*larger,-3.0*larger)
	eigVals,eigVecs=eigh(PList[Pt])
	larger=max(eigVals[0],eigVals[1])
	smaller=min(eigVals[0],eigVals[1])
	ang=180.0*atan((2.0*PList[Pt][1,0])/(pow(PList[Pt][0,0],2.0)-pow(PList[Pt][1,1],2.0)))
	if (larger == eigVals[0]):
		wid = sqrt(larger)*2.4477
		hgt = sqrt(smaller)*2.4477
	else:
		wid = sqrt(smaller)*2.4477
		hgt = sqrt(larger)*2.4477
	elle=Ellipse(xy=array([XList[Pt][0,0],XList[Pt][1,0]]),width=wid, height=hgt,angle=ang)
	elle.set_facecolor('#ccebc5')
	ax2.add_artist(elle)
	ax2.annotate(r'$\textbf{\textsf{P}}^{+}_{%d}$'%(Pt-1),fontsize=AnnotateSmall,xy=(XList[Pt][0,0],XList[Pt][1,0]),xycoords='data',xytext=(-4,10),textcoords='offset points')
	fig1.savefig(plotPath+'Fig1_Step1.jpg',dpi=300)
	fig2.savefig(plotPath+'Fig2_Step1.jpg',dpi=300)
	if PDF:
		fig1.savefig(plotPath+'Fig1_Step1.pdf',dpi=300)
		fig2.savefig(plotPath+'Fig2_Step1.pdf',dpi=300)


	ax1.plot([fill_t[-1],fill_t[-1]+1.0],[fill_x[-1],XMinusList[Pt+1][0,0]],color='#377eb8',linestyle='--',zorder=5)
	ax1.errorbar(fill_t[-1]+1.0,XMinusList[Pt+1][0,0],sqrt(PMinusList[Pt+1][0,0]),fmt='o',color='#377eb8',capsize=0,markeredgecolor='none',zorder=5)
	ax1.annotate(r'2. $\textbf{\textsf{P}}^{-}_{%d} = \textbf{\textsf{F}} \textbf{\textsf{P}}^{+}_{%d} \textbf{\textsf{F}}^{\top} + \textbf{\textsf{Q}}$'%(Pt,Pt-1),fontsize=AnnotateSmall,xy=(fill_t[-1]+1.0, XMinusList[Pt+1][0,0]), xycoords='data',xytext=(-252,56), textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="angle3"))
	ax1.annotate(r'1. $\widehat{\textbf{\textit{x}}}^{-}_{%d} = \textbf{\textsf{F}} \widehat{\textbf{\textit{x}}}^{+}_{%d}$'%(Pt,Pt-1),fontsize=AnnotateSmall,xy=(fill_t[-1]+1.0, XMinusList[Pt+1][0,0]), xycoords='data',xytext=(-252,92), textcoords='offset points')

	ax2.annotate(r'',xy=(XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0]), xycoords='data',xytext=(0,0), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#377eb8'),zorder=10)
	ax2.annotate(r'$\widehat{\textbf{\textit{x}}}^{-}_{%d}$'%(Pt),fontsize=AnnotateSmall,xy=(XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0]),xycoords='data',xytext=(48,-16),textcoords='offset points')
	larger=max(XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0])
	eigVals,eigVecs=eigh(PMinusList[Pt+1])
	larger=max(eigVals[0],eigVals[1])
	smaller=min(eigVals[0],eigVals[1])
	ang=180.0*atan((2.0*PMinusList[Pt+1][1,0])/(pow(PMinusList[Pt+1][0,0],2.0)-pow(PMinusList[Pt+1][1,1],2.0)))
	if (larger == eigVals[0]):
		wid = sqrt(larger)*2.4477
		hgt = sqrt(smaller)*2.4477
	else:
		wid = sqrt(smaller)*2.4477
		hgt = sqrt(larger)*2.4477
	elle=Ellipse(xy=array([XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0]]), width=wid, height=hgt, angle=ang)
	elle.set_facecolor('#b3cde3')
	ax2.add_artist(elle)
	ax2.annotate(r'$\textbf{\textsf{P}}^{-}_{%d}$'%(Pt),fontsize=AnnotateSmall,xy=(XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0]),xycoords='data',xytext=(-4,10),textcoords='offset points')
	fig1.savefig(plotPath+'Fig1_Step2.jpg',dpi=300)
	fig2.savefig(plotPath+'Fig2_Step2.jpg',dpi=300)
	if PDF:
		fig1.savefig(plotPath+'Fig1_Step2.pdf',dpi=300)
		fig2.savefig(plotPath+'Fig2_Step2.pdf',dpi=300)


	mu = XMinusList[Pt+1][0,0]
	sigma = sqrt(SList[Pt+1])
	min_len = mu-3.0*sigma
	max_len = mu+3.0*sigma
	eval_locs = linspace(min_len,max_len,100)
	eval_vals = gaussian(eval_locs,mu,sigma)
	ax1.plot(fill_t[-1]+1.0+eval_vals,eval_locs,color='#000000',linestyle='-.',zorder=0)
	ax1.annotate(r'4. $S_{%d} = \textbf{\textsf{H}} \textbf{\textsf{P}}^{-}_{%d} \textbf{\textsf{H}}^{\top} + \sigma^{2}_{N,%d}$'%(Pt,Pt-1,Pt),fontsize=AnnotateSmall,xy=(fill_t[-1]+1.0+eval_vals[14], eval_locs[14]), xycoords='data',xytext=(48,48), textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="angle3"))

	ax1.annotate(r'',xy=(fill_t[-1]+1.0,y[Pt,0]), xycoords='data',xytext=(fill_t[-1]+1.0,eval_locs[49]), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#000000'),zorder=15)
	ax1.annotate(r'',xy=(fill_t[-1]+1.0,eval_locs[49]), xycoords='data',xytext=(fill_t[-1]+1.0,y[Pt,0]), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#000000'),zorder=15)
	ax1.annotate(r'3. $r_{%d} = y_{%d} - \textbf{\textsf{H}} \widehat{\textbf{\textit{x}}}^{-}_{%d}$'%(Pt,Pt,Pt),fontsize=AnnotateSmall,xy=(fill_t[-1]+1.0,(y[Pt,0]+eval_locs[49])/2.0), xycoords='data',xytext=(96,-48), textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="angle3"))
	fig1.savefig(plotPath+'Fig1_Step3.jpg',dpi=300)
	fig2.savefig(plotPath+'Fig2_Step3.jpg',dpi=300)
	if PDF:
		fig1.savefig(plotPath+'Fig1_Step3.pdf',dpi=300)
		fig2.savefig(plotPath+'Fig2_Step3.pdf',dpi=300)

	ax2.annotate(r'',xy=(XMinusList[Pt+1][0,0]+vList[Pt+1][0,0]*KList[Pt+1][0,0],XMinusList[Pt+1][1,0]+vList[Pt+1][0,0]*KList[Pt+1][1,0]), xycoords='data',xytext=(XMinusList[Pt+1][0,0],XMinusList[Pt+1][1,0]),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#e41a1c'),zorder=15)
	ax2.annotate(r'$r_{%d}\textbf{\textsf{K}}_{%d}$'%(Pt,Pt),fontsize=AnnotateSmall,xy=(XMinusList[Pt+1][0,0]+vList[Pt+1][0,0]*KList[Pt+1][0,0],XMinusList[Pt+1][1,0]+vList[Pt+1][0,0]*KList[Pt+1][1,0]), xycoords='data',xytext=(-24,-24),textcoords='offset points',zorder=15)
	fig1.savefig(plotPath+'Fig1_Step4.jpg',dpi=300)
	fig2.savefig(plotPath+'Fig2_Step4.jpg',dpi=300)
	if PDF:
		fig1.savefig(plotPath+'Fig1_Step4.pdf',dpi=300)
		fig2.savefig(plotPath+'Fig2_Step4.pdf',dpi=300)

	ax2.annotate(r'',xy=(XList[Pt+1][0,0],XList[Pt+1][1,0]),fontsize=AnnotateSmall,xycoords='data',xytext=(0,0),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='#4daf4a'),zorder=10)
	ax2.annotate(r'$\widehat{\textbf{\textit{x}}}^{+}_{%d}$'%(Pt),fontsize=AnnotateSmall,xy=(XList[Pt+1][0,0],XList[Pt+1][1,0]),xycoords='data',xytext=(36,-16),textcoords='offset points')
	larger=max(XList[Pt+1][0,0],XList[Pt+1][1,0])
	eigVals,eigVecs=eigh(PList[Pt+1])
	larger=max(eigVals[0],eigVals[1])
	smaller=min(eigVals[0],eigVals[1])
	ang=180.0*atan((2.0*PList[Pt+1][1,0])/(pow(PList[Pt+1][0,0],2.0)-pow(PList[Pt+1][1,1],2.0)))
	if (larger == eigVals[0]):
		wid = sqrt(larger)*2.4477
		hgt = sqrt(smaller)*2.4477
	else:
		wid = sqrt(smaller)*2.4477
		hgt = sqrt(larger)*2.4477
	elle=Ellipse(xy=array([XList[Pt+1][0,0],XList[Pt+1][1,0]]),width=wid, height=hgt,angle=ang)
	elle.set_facecolor('#ccebc5')
	ax2.add_artist(elle)
	ax2.annotate(r'$\textbf{\textsf{P}}^{+}_{%d}$'%(Pt),fontsize=AnnotateSmall,xy=(XList[Pt+1][0,0],XList[Pt+1][1,0]),xycoords='data',xytext=(-4,10),textcoords='offset points')

	fill_t=list()
	fill_x=list()
	fill_xl=list()
	fill_xu=list()
	for i in xrange(Pt+2):
		fill_t.append(tList[i])
		fill_x.append(XList[i][0,0])
		fill_xl.append(XList[i][0,0]-sqrt(PList[i][0,0]))
		fill_xu.append(XList[i][0,0]+sqrt(PList[i][0,0]))
	ax1.plot(fill_t,fill_x,'-',color='#666666',zorder=5)
	ax1.fill_between(fill_t,fill_xl,fill_xu,color='#b3b3b3',zorder=0)

	ax1.annotate(r'5. $\textbf{\textsf{K}}_{%d} = \textbf{\textsf{P}}^{-}_{%d} \textbf{\textsf{K}}_{%d} S_{%d}^{-1}$'%(Pt,Pt,Pt,Pt),fontsize=AnnotateSmall,xy=(fill_t[-1],fill_x[-1]),xycoords='data',xytext=(-284,168), textcoords='offset points')
	ax1.annotate(r'6. $\widehat{\textbf{\textit{x}}}^{+}_{%d} = \widehat{\textbf{\textit{x}}}^{+}_{%d} + r_{%d}\textbf{\textsf{K}}_{%d}$'%(Pt,Pt,Pt,Pt),fontsize=AnnotateSmall,xy=(fill_t[-1],fill_x[-1]),xycoords='data',xytext=(-284,132), textcoords='offset points')
	ax1.annotate(r'7. $\textbf{\textsf{P}}^{+}_{%d} = (\textbf{\textsf{I}} - \textbf{\textsf{K}}_{%d} \textbf{\textsf{H}}_{%d}) \textbf{\textsf{P}}^{-}_{%d} (\textbf{\textsf{I}} - \textbf{\textsf{K}}_{%d} \textbf{\textsf{H}}_{%d})^{\top} + \textbf{\textsf{K}}_{%d} \sigma^{2}_{N,%d} \textbf{\textsf{K}}^{\top}_{%d}$'%(Pt,Pt,Pt,Pt,Pt,Pt,Pt,Pt,Pt),fontsize=AnnotateSmall,xy=(fill_t[-1],fill_x[-1]), xycoords='data',xytext=(-284,96), textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="angle3"))

	fig1.savefig(plotPath+'Fig1_Step5.jpg',dpi=300)
	fig2.savefig(plotPath+'Fig2_Step5.jpg',dpi=300)
	if PDF:
		fig1.savefig(plotPath+'Fig1_Step5.pdf',dpi=300)
		fig2.savefig(plotPath+'Fig2_Step5.pdf',dpi=300)

	pdb.set_trace()

	return LnLike

def getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	veryLarge = float_info[0]
	numPts=y.shape[0]
	LnLike=0.0
	ptCounter=0.0

	'''print "X"
	print "------"
	print X
	print "P"
	print "------"
	print P

	print "XMinus"
	print "------"
	print XMinus
	print "PMinus"
	print "------"
	print PMinus'''

	for i in range(numPts):
		'''print ""
		print "i: %d"%(i)'''

		H[0,0]=mask[i]
		R[0,0]=y[i,1]*y[i,1]

		'''print "mask[i]: %f"%(mask[i])
		print "y[i,0]: %f"%(y[i,0])
		print "y[i,1]: %f"%(y[i,1])

		print "H[0,0]: %f"%(H[0,0])
		print "R[0,0]: %f"%(R[0,0])'''
		
		XMinus=F*X
		
		'''print "XMinus"
		print "------"
		print XMinus'''
		
		PMinus=F*P*transpose(F)+Q
		
		'''print "PMinus"
		print "------"
		print PMinus'''
		
		v=y[i,0]-H*XMinus

		'''print "  v   "
		print "------"
		print v'''

		S=H*PMinus*transpose(H)+R
	
		'''print "  S   "
		print "------"
		print S'''

		inverseS=inv(S)

		'''print "inverseS"
		print "--------"
		print inverseS'''

		K=PMinus*transpose(H)*inverseS
		
		'''print "  K   "
		print "------"
		print K'''

		IMinusKH=I-K*H
		
		'''print "IMinusKH"
		print "--------"
		print IMinusKH'''

		X=K*y[i,0]+IMinusKH*XMinus

		'''print "   X    "
		print "--------"
		print X'''

		P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)

		'''print "   P    "
		print "--------"
		print P
		if (isnan(P[0,0])):
			pdb.set_trace()'''

		LnLike+=(-0.5*(transpose(v)*inverseS*v)-0.5*log(det(S)))[0,0]
		#ptCounter+=mask[i]

		'''print " LnLike "
		print "--------"
		print LnLike'''

	LnLike += -0.5*numPts*log(2.0*pi)
	return LnLike

def getLnLikeMissing(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	numPts=y.shape[0]
	numObs=sum(mask)
	LnLike=0.0
	for i in range(numPts):
		R[0,0]=y[i,1]*y[i,1]
		XMinus=F*X
		PMinus=F*P*transpose(F)+Q
		if (mask[i]==1.0):
			H[0,0]=1.0
			v=y[i,0]-H*XMinus
			S=H*PMinus*transpose(H)+R
			inverseS=inv(S)
			K=PMinus*transpose(H)*inverseS
			IMinusKH=I-K*H
			X=K*y[i,0]+IMinusKH*XMinus
			P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)
			LnLike+=(-0.5*(transpose(v)*inverseS*v)-0.5*log(det(S)))[0,0]
		else:
			X=XMinus
			P=PMinus
			LnLike+=0.0
	LnLike += -0.5*numObs*log(2.0*pi)
	return LnLike

def getResiduals(y,r,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	numPts=y.shape[0]
	for i in range(numPts):
		R[0,0]=y[i,1]*y[i,1]
		XMinus=F*X
		PMinus=F*P*transpose(F)+Q
		if (isnan(y[i,0])==False):
			H[0,0]=1.0
			v=y[i,0]-H*XMinus
			S=H*PMinus*transpose(H)+R
			inverseS=inv(S)
			K=PMinus*transpose(H)*inverseS
			IMinusKH=I-K*H
			X=K*y[i,0]+IMinusKH*XMinus
			P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)
			r[i,0]=v[0,0]
			r[i,1]=sqrt(S[0,0])
		else:
			X=XMinus
			P=PMinus
			r[i,0]=nan
			r[i,1]=nan
	return r

def fixedIntervalSmoother(y,r,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	m=F.shape[0]
	numPts=y.shape[0]
	XArr=npzeros((numPts,m,1))
	PArr=npzeros((numPts,m,m))
	XMinusArr=npzeros((numPts,m,1))
	PMinusArr=npzeros((numPts,m,m))
	smoothXArr=npzeros((numPts,m,1))
	smoothPArr=npzeros((numPts,m,m))
	for i in range(numPts):
		R[0,0]=y[i,1]*y[i,1]
		XMinus=F*X
		XMinusArr[i,:,:]=XMinus
		PMinus=F*P*transpose(F)+Q
		PMinusArr[i,:,:]=PMinus
		v=y[i,0]-H*XMinus
		S=H*PMinus*transpose(H)+R
		inverseS=inv(S)
		K=PMinus*transpose(H)*inverseS
		IMinusKH=I-K*H
		X=K*y[i,0]+IMinusKH*XMinus
		XArr[i,:,:]=X
		P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)
		PArr[i,:,:]=P
		r[i,0]=v[0,0]
		#try:
		r[i,1]=sqrt(S[0,0])
		#except ValueError:
		#	pdb.set_trace()
	smoothPArr[numPts-1,:,:]=PArr[numPts-1,:,:]
	smoothXArr[numPts-1,:,:]=XArr[numPts-1,:,:]
	for i in range(numPts-2,-1,-1):
		IMinus=inv(matrix(PMinusArr[i+1,:,:]))
		K=matrix(PArr[i,:,:])*transpose(F)*IMinus
		smoothPArr[i,:,:]=matrix(PArr[i,:,:])-K*(matrix(PMinusArr[i+1,:,:])-matrix(smoothPArr[i+1,:,:]))*transpose(K)
		smoothXArr[i,:,:]=matrix(XArr[i,:,:])+K*(matrix(smoothXArr[i+1,:,:])-matrix(XMinusArr[i+1,:,:]))
	for i in range(numPts):
		x[i,0]=smoothXArr[i,0,0]
		#try:
		x[i,1]=sqrt(smoothPArr[i,0,0])
		#except ValueError:
		#	pdb.set_trace()
	return (r,x)

'''def fixedIntervalSmootherMissing(y,r,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	m=F.shape[0]
	numPts=y.shape[0]
	XArr=npzeros((numPts,m,1))
	PArr=npzeros((numPts,m,m))
	XMinusArr=npzeros((numPts,m,1))
	PMinusArr=npzeros((numPts,m,m))
	smoothXArr=npzeros((numPts,m,1))
	smoothPArr=npzeros((numPts,m,m))
	for i in range(numPts):
		R[0,0]=y[i,1]*y[i,1]
		XMinus=F*X
		XMinusArr[i,:,:]=XMinus
		PMinus=F*P*transpose(F)+Q
		PMinusArr[i,:,:]=PMinus
		if (isnan(y[i,0])==False):
			H[0,0]=1.0
			v=y[i,0]-H*XMinus
			S=H*PMinus*transpose(H)+R
			inverseS=inv(S)
			K=PMinus*transpose(H)*inverseS
			IMinusKH=I-K*H
			X=K*y[i,0]+IMinusKH*XMinus
			XArr[i,:,:]=X
			P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)
			PArr[i,:,:]=P
			r[i,0]=v[0,0]
			r[i,1]=sqrt(S[0,0])
		else:
			X=XMinus
			XArr[i,:,:]=X
			P=PMinus
			PArr[i,:,:]=P
			r[i,0]=nan
			r[i,1]=nan
	smoothPArr[numPts-1,:,:]=PArr[numPts-1,:,:]
	smoothXArr[numPts-1,:,:]=XArr[numPts-1,:,:]
	for i in range(numPts-2,-1,-1):
		IMinus=inv(matrix(PMinusArr[i+1,:,:]))
		K=matrix(PArr[i,:,:])*transpose(F)*IMinus
		smoothPArr[i,:,:]=matrix(PArr[i,:,:])-K*(matrix(PMinusArr[i+1,:,:])-matrix(smoothPArr[i+1,:,:]))*transpose(K)
		smoothXArr[i,:,:]=matrix(XArr[i,:,:])+K*(matrix(smoothXArr[i+1,:,:])-matrix(XMinusArr[i+1,:,:]))
	for i in range(numPts):
		x[i,0]=smoothXArr[i,0,0]
		x[i,1]=sqrt(smoothPArr[i,0,0])
	return (r,x)'''
