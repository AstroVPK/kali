import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.integrate as sint
import math as m
import matplotlib.pyplot as plt
import pdb


def vec(A):
   reshape(transpose(A), (-1,1))

def Qint(dchi,A,B,i,j):
	#ans = ((sla.expm(A*dchi))*B*(B.T)*(sla.expm((A.T)*dchi)))[i,j]
	ans = ((sla.expm(A*dchi))*B*(B.T)*((sla.expm(A*dchi)).T))[i,j]
	return ans

dt = 0.02
#dt = 0.020489543562895264
A = np.matrix(np.zeros((2,2)))
B = np.matrix(np.zeros((2,1)))
D = np.matrix(np.zeros((2,1)))

A[0,0] = -0.75
A[1,0] = -0.01
A[0,1] = 1.0

print "A"
print A
print

w,vr = la.eig(A)
vrInv = vr.I

print "w"
print w
print

print "vr"
print vr
print

expw = np.matrix(np.diag(np.exp(w*dt)))

print "expw"
print expw
print

F = sla.expm(A*dt)

print "F"
print F
print

I = np.matrix(np.identity(2))

B[0,0] = 1.2*m.pow(10.0,-9.0)
B[1,0] = 7.0*m.pow(10.0,-9.0)

print "B"
print B
print

for i in xrange(2):
	D[i,0]=m.sqrt(sint.quad(Qint,0,dt,args=(A,B,i,i),epsrel=0.0001)[0]) # Tested
	'''D[i,0]=sint.quad(Qint,0,dt,args=(A,B,i,i),epsrel=0.0001)[0]''' # Untested
Q = D*(D.T)

print "D"
print D
print

print "Q"
print Q
print

P = np.matrix(np.reshape(la.solve(np.kron(I,I)-np.kron(F,F),np.reshape(Q,(2*2,1))),(2,2)))

print "P"
print P
print

X = np.matrix(np.zeros((2,1)))
H = np.matrix(np.zeros((1,2)))

H[0,0] = 1.0

numBurn = 1

burnDist = np.random.normal(0.0, 1.0, numBurn)

for i in xrange(numBurn):
	'''print "X"
	print X
	print
	print "F"
	print F
	print
	print "F*X"
	print F*X
	print
	print "D"
	print D
	print
	print "w[%d]"%(i)
	print burnDist[i]
	print
	print "w[%d]*D"%(i)
	print burnDist[i]*D
	print
	print "F*X + w[%d]*D"%(i)
	print F*X + burnDist[i]*D
	print
	print'''
	X = F*X + burnDist[i]*D

print X
print

numObs = 4000

t = np.zeros((numObs))
y = np.zeros((numObs))

obsDist = np.random.normal(0.0, 1.0, numObs)
for i in xrange(numObs):
	X = F*X + obsDist[i]*D
	t[i] = dt*i
	y[i] = (H*X)[0,0]

plt.ion()
plt.figure(1)
plt.plot(t,y)

pdb.set_trace()