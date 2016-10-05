import urllib2
import re
import os
import sys
import pdb

lightcurvesPath = 'http://archive.stsci.edu/pub/kepler/lightcurves/'
target_pixel_filesPath = 'http://archive.stsci.edu/pub/kepler/target_pixel_files/'
cbvPath = 'http://archive.stsci.edu/pub/kepler/cbv/'


dataFolder = str(sys.argv[1])
if (dataFolder[-1] != '/'):
    dataFolder += '/'
targetListFile = dataFolder+'kplr_targetList.dat'

# Test to see if path is correct
if (os.access(dataFolder, os.F_OK) != True):
    print "Base directory of data tree %s incorrect or does not exist."%(dataFolder)
    exit(1)
if (os.access(targetListFile, os.F_OK) != True):
    print "Target list %s not found in base directory of data tree."%(targetListFile)
    exit(1)

cbvFolder = dataFolder + 'cbv/'
if (os.access(cbvFolder, os.F_OK) == False):
    os.mkdir(cbvFolder)
cbvpath = urllib2.urlopen(cbvPath)
string = cbvpath.read().decode('utf-8')
cbvPattern = re.compile('kplr' + '[0-9]*-q[0-9]*-d[0-9]*_lcbv.fits')
cbvList = cbvPattern.findall(string)
for i in range(len(cbvList)/2):
    remoteCBVFile = urllib2.urlopen(cbvPath + cbvList[2*i])
    localCBVFile = open(cbvFolder + cbvList[2*i], 'wb')
    localCBVFile.write(remoteCBVFile.read())
    localCBVFile.close()
    remoteCBVFile.close()

tLFile = open(targetListFile)
for target in tLFile:
    if (target[0] != '#'):
        target = target.rstrip('\n')
        objFolder = dataFolder + target
        if (os.access(objFolder, os.F_OK) == False):
            os.mkdir(objFolder)
        FITSFolder = objFolder+'/llc/'
        simFolder = objFolder+'/sim/'
        lpdtargFolder = objFolder+'/lpd-targ/'
        if (os.access(FITSFolder, os.F_OK) == False):
            os.mkdir(FITSFolder)
        if (os.access(simFolder, os.F_OK) == False):
            os.mkdir(simFolder)
        if (os.access(lpdtargFolder, os.F_OK) == False):
            os.mkdir(lpdtargFolder)
        XXXX = target[4:8]
        KKKKKKKKK = target[4:13]
        urlpath = urllib2.urlopen(lightcurvesPath + XXXX + '/' + KKKKKKKKK)
        targpath = urllib2.urlopen(target_pixel_filesPath + XXXX + '/' + KKKKKKKKK)
        string = urlpath.read().decode('utf-8')
        stringTarg = targpath.read().decode('utf-8')
        quarterPattern = re.compile(target + '-[0-9]*_llc.fits')
        targPattern = re.compile(target + '-[0-9]*_lpd-targ.fits.gz')
        quarterList = quarterPattern.findall(string)
        targList = targPattern.findall(stringTarg)
        epochListFile = objFolder + '/' + target + '-epochList.dat'
        eLFile = open(epochListFile, 'w')
        numQuarters = len(quarterList)/2
        for i in range(numQuarters-1):
            eLFileLine = '%s.dat\n'%(quarterList[2*i][0:31])
            eLFile.write(eLFileLine)
            remoteQuarterFile = urllib2.urlopen(
                lightcurvesPath + XXXX + '/' + KKKKKKKKK + '/' + quarterList[2*i])
            remoteTargFile = urllib2.urlopen(
                target_pixel_filesPath + XXXX + '/' + KKKKKKKKK + '/' + targList[2*i])
            localQuarterFile = open(FITSFolder + quarterList[2*i], 'wb')
            localTargFile = open(lpdtargFolder + targList[2*i], 'wb')
            localQuarterFile.write(remoteQuarterFile.read())
            localTargFile.write(remoteTargFile.read())
            localQuarterFile.close()
            remoteQuarterFile.close()
            localTargFile.close()
            remoteTargFile.close()
        eLFileLine = '%s.dat'%(quarterList[2*(numQuarters-1)][0:31])
        eLFile.write(eLFileLine)
        eLFile.close()
        remoteQuarterFile = urllib2.urlopen(
            lightcurvesPath + XXXX + '/' + KKKKKKKKK + '/' + quarterList[2*(numQuarters-1)])
        remoteTargFile = urllib2.urlopen(
            target_pixel_filesPath + XXXX + '/' + KKKKKKKKK + '/' + targList[2*(numQuarters-1)])
        localQuarterFile = open(FITSFolder + quarterList[2*(numQuarters-1)], 'wb')
        localTargFile = open(lpdtargFolder + targList[2*(numQuarters-1)], 'wb')
        localQuarterFile.write(remoteQuarterFile.read())
        localTargFile.write(remoteTargFile.read())
        localQuarterFile.close()
        remoteQuarterFile.close()
        localTargFile.close()
        remoteTargFile.close()
tLFile.close()
