from math import pow,log,pi,sqrt
from cmath import exp as cexp
from numpy import transpose,roots,kron,reshape,isnan,nan,sum,inf
from numpy import zeros as npzeros
from numpy import array
from numpy.matlib import matrix,zeros,identity
from numpy.linalg import inv,solve,det,eig,cholesky
from numpy.random import multivariate_normal
from scipy.linalg import expm
from scipy.integrate import quad
from random import gauss,seed
from sys import float_info
import numpy as np
import pdb

intTime=6.019802903
readTime=0.5189485261
numIntLC=270
Deltat=(intTime+readTime)*numIntLC
secPerSiderealDay=86164.0905
dt=Deltat/secPerSiderealDay

def makeSystem(p,q,m):

	A=zeros((m,m))
	for i in xrange(m):
		A[i-1,i]=1.0
	B=zeros((m,1))

	F=zeros((m,m))
	I=identity(m)
	Q=zeros((m,m))
	H=zeros((1,m))
	R=zeros((1,1))
	K=zeros((m,1))

	return (p,q,m,A,B,F,I,Q,H,R,K)

def checkParams(aList=None,bList=None):
	if aList is None:
		raise ValueError('#CAR > 0')
	p = len(aList)

	if bList is None:
		raise ValueError('#CMA > 0')
	q = len(bList)-1

	hasUniqueEigenValues=1
	isStable=1
	isInvertible=1
	isNotRedundant=1
	hasPosSigma=1

	CARPoly=list()
	CARPoly.append(1.0)
	for i in xrange(p):
		CARPoly.append(aList[i])
	CARRoots=roots(CARPoly)
	#print 'C-AR Roots: ' + str([rootVal for rootVal in CARRoots])
	if (len(CARRoots)!=len(set(CARRoots))):
		hasUniqueEigenValues=0
	for CARRoot in CARRoots:
		if (CARRoot.real>=0.0):
			isStable=0
	#print 'isStable: %d'%(isStable)

	isInvertible=1
	CMAPoly=list()
	for i in xrange(q + 1):
		CMAPoly.append(bList[i])
	CMAPoly.reverse()
	CMARoots=roots(CMAPoly)
	#print 'C-MA Roots: ' + str([rootVal for rootVal in CMARoots])
	if (len(CMARoots)!=len(set(CMARoots))):
		uniqueRoots=0
	for CMARoot in CMARoots:
		if (CMARoot>0.0):
			isInvertible=0
	#print 'isInvertible: %d'%(isInvertible)

	isNotRedundant=1
	for CARRoot in CARRoots:
		for CMARoot in CMARoots:
			if (CARRoot==CMARoot):
				isNotRedundant=0

	if (bList[0] <= 0.0):
		hasPosSigma = 0

	retVal = isStable*isInvertible*isNotRedundant*hasUniqueEigenValues*hasPosSigma
	#print 'retVal: %d'%(retVal)

	return retVal

def getRoots(aList=None,bList=None):
	if aList is None:
		raise ValueError('#CAR > 0')
	p = len(aList)

	if bList is None:
		raise ValueError('#CMA > 0')
	q = len(bList)-1

	CARPoly=list()
	CARPoly.append(1.0)
	for i in xrange(p):
		CARPoly.append(aList[i])
	CARRoots=roots(CARPoly)

	CMAPoly=list()
	for i in xrange(q+1):
		CMAPoly.append(bList[i])
	CMARoots=roots(CMAPoly)

	return (CARRoots,CMARoots)

def setSystem(dt,p,q,m,aList,bList,A,B,F,Q):
	for i in xrange(len(aList)):
		A[i,0]=-1.0*aList[i]

	for i in xrange(len(bList)):
		B[m-1-i,0] = bList[q-i];

	F=expm(A*dt)

	lam,vr = eig(A)
	vr = matrix(vr)
	vrTrans = transpose(vr)
	vrInv = inv(vr)

	C = vrInv*B*transpose(B)*transpose(vrInv)

	Q=zeros((m,m))
	for i in xrange(m):
		for j in xrange(m):
			for k in xrange(m):
				for l in xrange(m):
					Q[i,j] += vr[i,k]*C[k,l]*vrTrans[l,j]*((cexp((lam[k] + lam[l])*dt) - 1.0)/(lam[k] + lam[l]))

	T=cholesky(Q)

	X=zeros((m,1))
	Sigma=zeros((m,m))
	P=zeros((m,m))
	for i in xrange(m):
		for j in xrange(m):
			for k in xrange(m):
				for l in xrange(m):
					Sigma[i,j] += vr[i,k]*C[k,l]*vrTrans[l,j]*(-1.0/(lam[k] + lam[l]))
			P[i,j] = Sigma[i,j]

	XMinus=zeros((m,1))
	PMinus=zeros((m,m))

	'''print 'A'
	print A
	print
	print 'B'
	print B
	print
	print 'F'
	print F
	print
	print 'C'
	print C
	print
	print 'Q'
	print Q
	print
	print 'T'
	print T
	print
	print 'Sigma'
	print Sigma
	print
	print 'X'
	print X
	print
	print 'P'
	print P'''

	return (X,P,XMinus,PMinus,F,Q)

def setSystemDiffuse(dt,p,q,m,aList,bList,A,B,F,Q):
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
	return (X,P,XMinus,PMinus,F,Q)

def burnSystem(m,X,F,Q,numBurn,burnSeed):
	seed(burnSeed)
	burnRand = multivariate_normal(array(m*[0.0]),Q,numBurn)
	for i in range(numBurn):
		X = F*X + transpose(matrix(burnRand[i,:]))
	return X

def obsSystemMissing(m,X,F,Q,H,noise,y,mask,numObs,distSeed,noiseSeed):
	veryLarge = sqrt(float_info[0])
	#veryLarge = sqrt(1.0e300)
	noiseRand=npzeros(numObs)
	seed(distSeed)
	distRand = multivariate_normal(array(m*[0.0]),Q,numObs)
	seed(noiseSeed)
	for i in range(numObs):
		noiseRand[i]=gauss(0.0,noise)
	for i in range(numObs):
		H[0,0]=mask[i]
		X = F*X + transpose(matrix(distRand[i,:]))
		y[i,0]=H*X+H[0,0]*noiseRand[i]
		if (mask[i]==1.0):
			y[i,1]=noise
		else:
			y[i,1]=veryLarge
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

def getLnLike(y,mask,X,P,XMinus,PMinus,F,I,Q,H,R,K):
	veryLarge = float_info[0]
	numPts=y.shape[0]
	numObs=sum(mask)
	LnLike=0.0
	ptCounter=0.0
	for i in range(numPts):
		H[0,0]=mask[i]
		R[0,0]=y[i,1]*y[i,1]
		XMinus=F*X
		PMinus=F*P*transpose(F)+Q
		v=mask[i]*y[i,0] - H*XMinus
		S=H*PMinus*transpose(H)+R
		inverseS=1.0/S[0,0]
		K=PMinus*transpose(H)*inverseS
		IMinusKH=I-K*H
		X=K*y[i,0]+IMinusKH*XMinus
		P=IMinusKH*PMinus*transpose(IMinusKH)+K*R*transpose(K)
		LnLike+=-0.5*(v[0,0]*v[0,0]*inverseS)-0.5*log(S[0,0])
	#LnLike += -0.5*numObs*log(2.0*pi)
	return LnLike

def getLnLikeMissing(y,mask,X,P,XMinus,PMinus,F,I,Q,H,R,K):
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

def getResiduals(y,r,X,P,XMinus,PMinus,F,I,Q,H,R,K):
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

def fixedIntervalSmoother(y,r,x,X,P,XMinus,PMinus,F,I,Q,H,R,K):
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
		try:
			x[i,1]=sqrt(smoothPArr[i,0,0])
		except ValueError:
			pdb.set_trace()
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

def getACF(times, A, Sigma, H):
	ACF = npzeros(numtimes)
	for time in xrange(times.shape[0]):
		ACF[time] = (transpose(H)*expm(A*times[time])*Sigma*H)[0,0]
	return ACF