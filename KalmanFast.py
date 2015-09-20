from math import log,pi,sqrt 
from numpy import transpose,roots,kron,reshape,isnan,nan,sum,inf
from numpy import zeros as npzeros
from numpy.matlib import matrix,zeros,identity
from numpy.linalg import inv,solve,det
from random import gauss, seed
from sys import float_info
import pdb

def makeSystem(p,q):
	m=max(p,q+1)

	F=zeros((m,m))
	for i in range(0,m-1):
		F[i,i+1]=1.0
	I=identity(m)
	D=zeros((m,1))
	D[0,0]=1.0
	Q=zeros((m,m))
	H=zeros((1,m))
	R=zeros((1,1))
	K=zeros((m,1))

	return (m,p,q,F,I,D,Q,H,R,K)

def checkParams(pList=None,qList=None,dist=None):
	if pList is None:
		pList=list()
	p=len(pList)

	if qList is None:
		qList=list()
	q=len(qList)

	isStable=1
	ARPoly=pList[::-1]
	for i in range(0,p):
		ARPoly[i]*=-1.0
	ARPoly.append(1.0)
	ARRoots=roots(ARPoly)
	for ARRoot in ARRoots:
		if (ARRoot.conjugate()*ARRoot<1.0):
			isStable=0

	isInvertible=1
	MAPoly=qList[::-1]
	MAPoly.append(1.0)
	MARoots=roots(MAPoly)
	for MARoot in MARoots:
		if (MARoot.conjugate()*MARoot<1.0):
			isInvertible=0

	isNotRedundant=1
	for ARRoot in ARRoots:
		for MARoot in MARoots:
			if (ARRoot==MARoot):
				isNotRedundant=0

	isReasonable=1
	if dist is None:
		dist=0.0
	if (dist<0.0):
		isReasonable=0

	return isStable*isInvertible*isNotRedundant*isReasonable

def setSystem(p,q,m,pList,qList,dist,F,I,D,Q):
	for i in range(0,min(q,m)):
		D[i+1,0]=qList[i]
	Q=D*dist*dist*transpose(D)
	for i in range(0,min(p,m)):
		F[i,0]=pList[i]
	vecQ=reshape(Q,(m*m,1))
	IKron=kron(I,I)
	FKron=kron(F,F)
	X=zeros((m,1))
	P=matrix(reshape(solve(IKron-FKron,vecQ),(m,m)))
	XMinus=zeros((m,1))
	PMinus=zeros((m,m))
	return (X,P,XMinus,PMinus,F,I,D,Q)

def setSystemDiffuse(p,q,m,pList,qList,dist,F,I,D,Q):
	for i in range(0,min(q,m)):
		D[i+1,0]=qList[i]
	Q=D*dist*dist*transpose(D)
	for i in range(0,min(p,m)):
		F[i,0]=pList[i]
	X=zeros((m,1))
	P=zeros((m,m))
	for i in range(m):
		P[i,i]=10**7
	XMinus=zeros((m,1))
	PMinus=zeros((m,m))
	H[0,0]=1.0
	return (X,P,XMinus,PMinus,F,I,D,Q)

def burnSystemFixed(X,F,D,dist,numBurn,burnRand):
	for i in range(numBurn):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(burnRand[i])
		print "X_-"
		print X'''
		X=F*X+D*burnRand[i]
		'''print "X_+"
		print X
		print'''
	return X

def burnSystem(X,F,D,dist,numBurn,burnSeed):
	burnRand=npzeros(numBurn)
	seed(burnSeed)
	for i in range(numBurn):
		burnRand[i]=gauss(0.0,dist)
	for i in range(numBurn):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(burnRand[i])
		print "X_-"
		print X'''
		X=F*X+D*burnRand[i]
		'''print "X_+"
		print X
		print'''
	return X

def obsSystemFixed(X,F,D,H,dist,noise,y,numObs,distRand,noiseRand):
	H[0,0]=1.0
	for i in range(numObs):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+D*distRand[i]
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+noiseRand[i]
		y[i,1]=noise
		'''print "y"
		print y[i,0]'''
	return (X,y)

def obsSystemFixedMissing(X,F,D,H,dist,noise,y,mask,numObs,distRand,noiseRand):
	for i in range(numObs):
		H[0,0]=mask[i]
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+D*distRand[i]
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

def obsSystem(X,F,D,H,dist,noise,y,numObs,distSeed,noiseSeed):
	H[0,0]=1.0
	distRand=npzeros(numObs)
	noiseRand=npzeros(numObs)
	seed(distSeed)
	for i in range(numObs):
		distRand=gauss(0.0,dist)
	seed(noiseSeed)
	for i in range(numObs):
		noiseRand=noiseRand[i]
	for i in range(numObs):
		'''print
		print "i: %d"%(i)
		print "Disturbance: %f"%(distRand[i])
		print "X_-"
		print X'''
		X=F*X+D*distRand[i]
		'''print "X_+"
		print X
		print "Noise: %f"%(noiseRand[i])'''
		y[i,0]=H*X+noiseRand[i]
		y[i,1]=noise
		'''print "y"
		print y[i,0]'''
	return (X,y)

def obsSystemMissing(X,F,D,H,dist,noise,y,mask,numObs,distSeed,noiseSeed):
	veryLarge = float_info[0]
	distRand=npzeros(numObs)
	noiseRand=npzeros(numObs)
	seed(distSeed)
	for i in range(numObs):
		distRand[i]=gauss(0.0,dist)
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
		X=F*X+D*distRand[i]
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

def getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K):
	veryLarge = float_info[0]
	numPts=y.shape[0]
	LnLike=0.0

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

		#try:
		LnLike+=(-0.5*(transpose(v)*inverseS*v)-0.5*log(det(S)))[0,0]
		#except ValueError:
		#	pdb.set_trace()

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
