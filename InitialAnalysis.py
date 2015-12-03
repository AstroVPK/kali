import numpy as np
import math as m
import socket
HOST = socket.gethostname()
if HOST in ['dirac.physics.drexel.edu','sun.physics.drexel.edu']:
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_settings import *
import sys as s
import pdb

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=600
nbins=100
set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=20,useTex='True')

secPerSiderealDay = 86164.0905 
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
deltat=(numIntLC*(intTime+readTime))/secPerSiderealDay

acfFile = True
pacfFile = True
sfFile = True

basePath=s.argv[1]
step=int(s.argv[2])

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

acfFilePath=basePath+'acf.dat'
try:
	acfFile=open(acfFilePath)
except IOError:
	acfFile = False
if acfFile:
	line=acfFile.readline()
	line=acfFile.readline()
	line=acfFile.readline()
	acf=np.zeros((numPts,2))
	for i in xrange(numPts):
		line=acfFile.readline()
		line.rstrip("\n")
		values=line.split()
		acf[i,0]=deltat*i
		acf[i,1]=float(values[0])


pacfFilePath=basePath+'pacf.dat'
try:
	pacfFile=open(pacfFilePath)
except IOError:
	pacfFile = True
if pacfFile:
	line=pacfFile.readline()
	line=pacfFile.readline()
	line=pacfFile.readline()
	line.rstrip("\n")
	values=line.split()
	maxLag=int(values[1])
	pacf=np.zeros((maxLag,2))
	for i in xrange(maxLag):
		line=pacfFile.readline()
		line.rstrip("\n")
		values=line.split()
		pacf[i,0]=deltat*i
		pacf[i,1]=float(values[0])

plt.figure(1,figsize=(fwid,fhgt))

if acfFile and pacfFile:
	plt.subplot(311)
elif (acfFile and not pacfFile) or (pacfFile and not acfFile):
	plt.subplot(211)
yMax=np.max(y[np.nonzero(y[:,0]),0])
yMin=np.min(y[np.nonzero(y[:,0]),0])
plt.ylabel(r'$F$ (arb units)')
plt.xlabel(r'$t$ (d)')
for i in xrange(0,numPts,step):
	if (y[i,0]!=0.0):
		plt.errorbar(t[i,1],y[i,0],y[i,1],fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',zorder=10)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(yMin,yMax)

if acfFile and pacfFile:
	plt.subplot(312)
if acfFile and not pacfFile:
	plt.subplot(212)
if acfFile:
	plt.ylabel('r$\rho(\Delta t)$')
	plt.xlabel('r$\Delta t$ (d)')
	plt.plot(acf[:,0],acf[:,1],color='#1a1a1a')
	#plt.fill_between(acf[:,0],0,acf[:,1],color='#999999',zorder=-20)
	plt.hlines(y=[1.96/m.sqrt(numPts),-1.96/m.sqrt(numPts)],xmin=0,xmax=numPts-1,linewidth=1, color='#ef8a62',linestyle='dashed')
	plt.hlines(y=0,xmin=0,xmax=numPts-1,linewidth=2, color='#000000')
	plt.ylim(-1.0,1.0)
	plt.xlim(0,acf[-1,0])

if acfFile and pacfFile:
	plt.subplot(313)
if pacfFile and not acfFile:
	plt.subplot(212)
if pacfFile:
	plt.ylabel(r'$\alpha(\Delta t)$')
	plt.xlabel(r'$\Delta t$ (d)')
	plt.plot(pacf[:,0],pacf[:,1],color='#1a1a1a')
	#plt.fill_between(acf[:,0],0,acf[:,1],color='#999999',zorder=-20)
	plt.hlines(y=[1.96/m.sqrt(maxLag),-1.96/m.sqrt(maxLag)],xmin=0,xmax=numPts-1,linewidth=1, color='#ef8a62',linestyle='dashed')
	plt.hlines(y=0,xmin=0,xmax=maxLag-1,linewidth=2, color='#000000')
	plt.ylim(-1.0,1.0)
	plt.xlim(0,acf[-1,0])

plt.tight_layout()
plt.savefig(basePath+"cfs.jpg",dpi=dotsPerInch)
plt.clf()