from numpy import *
from matplotlib import *
from matplotlib.pyplot import *
import os
import math
import scipy
import numpy
import random
from scipy.integrate import quad
from scipy.linalg import svd
from scipy.optimize import fmin
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.special import gammaincinv
import pdb
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders

def vec(A):
	reshape(transpose(A), (-1,1))

class myColors:
	def __init__(self,colors=0):
		if (isinstance(colors,list)):
			if (len(colors)>0):
				self.colorList=colors
			else:
				self.colorList.append('#000000')
		elif (isinstance(colors,int)):
			if (colors == 0):
				self.colorList=[]
			elif (colors == 1):
				self.colorList=['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
			elif (colors == 2):
				self.colorList=['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef','#e6f5d0','#b8e186','#7fbc41','#4d9221','#276419']
			elif (colors == 3):
				self.colorList=['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b']
			elif (colors == 4):
				self.colorList=['#7f3b08','#b35806','#e08214','#fdb863','#fee0b6','#d8daeb','#b2abd2','#8073ac','#542788','#2d004b']
			elif (colors == 5):
				self.colorList=['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
			elif (colors == 6):
				self.colorList=['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#e0e0e0','#bababa','#878787','#4d4d4d','#1a1a1a']
			elif (colors == 7):
				self.colorList=['#a50026','#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
			elif (colors == 8):
				self.colorList=['#a50026','#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850','#006837']
			elif (colors == 9):
				self.colorList=['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']
			elif (colors == 10):
				self.colorList=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
			elif (colors == 11):
				self.colorList=['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
		self.numColors=len(self.colorList)
		self.colorCounter=0
	def getColor(self):
		if (len(self.colorList)==0):
			return '#000000'
		returnColor=self.colorList[self.colorCounter]
		self.colorCounter=self.colorCounter+1
		if (self.colorCounter==self.numColors):
			self.colorCounter=0
		return returnColor
	def addColor(self,newColor):
		self.colorList.append(newColor)
		self.numColors=self.numColors+1	
		
gmail_user='sun.physics.drexel.edu@gmail.com'
gmail_pwd='palantir'
   
def mail(to,subject,text,*attach):
	msg = MIMEMultipart()
	msg['From']=gmail_user
	msg['To']=to
	msg['Subject']=subject
	msg.attach(MIMEText(text))
	n=len(attach)
	for i in range(n):
		part = MIMEBase('application','octet-stream')
		part.set_payload(open(attach[i],'rb').read())
		Encoders.encode_base64(part)
		part.add_header('Content-Disposition','attachment; filename="%s"' % os.path.basename(attach[i]))
		msg.attach(part)
	mailServer=smtplib.SMTP('smtp.gmail.com',587)
	mailServer.ehlo()
	mailServer.starttls()
	mailServer.ehlo()
	mailServer.login(gmail_user, gmail_pwd)
	mailServer.sendmail(gmail_user, to, msg.as_string())
   # Should be mailServer.quit(), but that crashes...
	mailServer.close()

def NDimHist(dataArray, numBins):
    numcol = dataArray.shape[1]
    numrow = dataArray.shape[0]
    numBinsTuple = tuple(numBins)
    cells = zeros(shape=numBinsTuple)
    step=[]
    minvals=[]
    maxvals=[]
    for i in range(numcol):
        high=(max(dataArray[:,i]))
        low=(min(dataArray[:,i]))
        newmin = floor(low)
        newmax = ceil(high)
        maxvals.append(newmax)
        minvals.append(newmin)
        step.append((newmax - newmin)/numBins[i])
        dataArray[:,i] = dataArray[:,i]-newmin
    for i in range(numrow):
        position=[]
        for j in range(numcol):
            position.append(floor(dataArray[i,j]/step[j]))
        positionTuple=tuple(position)
        try:
            cells[positionTuple]=cells[positionTuple]+1
        except IndexError:
            pdb.set_trace()
    return (cells,minvals,maxvals,step)

def median(dataArray):
	sortedDataArray=numpy.sort(dataArray)
	dataLength=sortedDataArray.shape[0]
	if (math.fmod(dataLength,2)==1.0):
		medianValue=sortedDataArray[(dataLength/2)]
	else:
		medianValue=(sortedDataArray[dataLength/2]+sortedDataArray[(dataLength/2)-1])/2.0
	return medianValue

def medianDeviation(dataArray):
	medianValue=median(dataArray)
	arrayLength=dataArray.shape[0]
	arrayResiduals=numpy.zeros(arrayLength)
	for counter in range(arrayLength):
		arrayResiduals[counter]=numpy.absolute(dataArray[counter]-medianValue)
	medianDeviationValue=median(arrayResiduals)
	return medianDeviationValue

class RandomDistribution:
	def __init__(self,cdf):
		self.cdf=cdf
		self.randomObj=random.SystemRandom()
	def f(self,xyc):
		x,y,c=xyc
		z=numpy.array([y-self.cdf(x),y-c,c-c])
		return z
	def random(self):
		c=self.randomObj.random()
		result=fsolve(self.f,[0.5,0.5,c],xtol=1.490128*math.pow(10,-24),maxfev=int(math.pow(10,6)))[0]
		return result
		
class incgammavariate:
	def __init__(self,a=1.0):
		self.randomObj=random.SystemRandom()
		self.a=float(a)
	def random(self):
		c=self.randomObj.random()
		return gammaincinv(self.a,c)

def setTickLabels(axis,axisMin,axisMax,dataMin,dataMax,XOrY='x',nLabels=3,rotation=0,fontsize=12):
	commonIncr = [0.01,0.1,0.25,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0,25.0,35.0,45.0,75.0,100.0]
	dataMinOrder = math.floor(math.log10(dataMin))
	dataMaxOrder = math.floor(math.log10(dataMax))
	dataRangeOrder = math.floor(math.log10(dataMax - dataMin))
	exponentSize = int(math.floor(math.log10(dataMin)))
	baseStepSize = math.pow(10.0,dataRangeOrder-1)
	for i in xrange(len(commonIncr)):
		if ((dataMax - dataMin)/(commonIncr[i]*baseStepSize) <= nLabels):
			stepSize = commonIncr[i-1]*baseStepSize
			break
	tickList = list()
	labelList = list()
	startVal = round(axisMin,-int(math.floor(math.log10(math.fabs(axisMin)))))
	if (startVal > axisMin):
		startVal = round(axisMin,-int(math.floor(math.log10(math.fabs(axisMin))))+1)
	stopVal = round(axisMax,-int(math.floor(math.log10(axisMax))))
	i = 0
	if (exponentSize != 0):
		while((startVal + i*stepSize) < axisMax):
			tickList.append(startVal + i*stepSize)
			labelList.append(r'$%.1f \times 10^{%d}$'%((startVal + i*stepSize)/math.pow(10.0,exponentSize),exponentSize))
			i += 1
	else:
		while((startVal + i*stepSize) < axisMax):
			tickList.append(startVal + i*stepSize)
			labelList.append(r'$%.1f$'%(startVal + i*stepSize))
			i += 1
	if (tickList[-1] < axisMax):
		if (exponentSize != 0):
			tickList.append(startVal + i*stepSize)
			labelList.append(r'$%.1f \times 10^{%d}$'%((startVal + i*stepSize)/math.pow(10.0,exponentSize),exponentSize))
		else:
			tickList.append(startVal + i*stepSize)
			labelList.append(r'$%.1f$'%(startVal + i*stepSize))
	if (tickList[0] < axisMin):
		tickList.pop(0)
		labelList.pop(0)
	if ((tickList[0] - axisMin) < stepSize/2.0):
		tickList.pop(0)
		labelList.pop(0)
	tickList = numpy.array(tickList)
	if (XOrY == 'x'):
		axis.set_xticks(tickList)
		axis.set_xticklabels(labelList, rotation = rotation, fontsize = fontsize)
	else:
		axis.set_yticks(tickList)
		axis.set_yticklabels(labelList, rotation = rotation, fontsize = fontsize)
	axis.set_ylim(axisMin,axisMax)
	#pdb.set_trace()
	return 1