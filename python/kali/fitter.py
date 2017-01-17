import math
import cmath
import numpy as np
import sys
import os
import psutil
import time
import re
import pdb

try:
    import kali.lc
except ImportError:
    print 'Could not import kali.lc! kali may not be setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

try:
    import kali.util.triangle
except ImportError:
    print 'Could not import kali.util.triangle! kali may not be setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


def fitModel(lc, model, pVal, qVal,
             nthreads, nsteps,
             maxEvals, xTol, mcmcA):
    modulename = 'kali.' + model
    if modulename not in sys.modules:
        try:
            exec('import ' + modulename)
        except ImportError:
            print 'Could not import kali.%s! kali may not be setup. \
            Setup kali by sourcing bin/setup.sh'%(model)
            sys.exit(1)
    exec('fitTask = kali.%s.%sTask(p=%d, q=%d, \
    nthreads=%d, nwalkers=%d, nsteps=%d, \
    maxEvals=%d, xTol=%f, mcmcA=%f)'%(model, model.upper(), pVal, qVal,
                                      nthreads, nwalkers, nsteps,
                                      maxEvals, xTol, mcmcA))
    print 'Starting kali.%s fitting for p = %d and q = %d...'%(model, pVal, qVal)
    startTask = time.time()
    fitTask.fit(lc)
    stopTask = time.time()
    timeTask = stopTask - startTask
    print 'kali.%s took %4.3f s = %4.3f min = %4.3f hrs'%(model,
                                                          timeTask,
                                                          timeTask/60.0,
                                                          timeTask/3600.0)
    return fitTask


def fit(lc, models, path=None,
        nthreads=psutil.cpu_count(logical=True), nwalkers=25*psutil.cpu_count(logical=False), nsteps=10000,
        maxEvals=10000, xTol=0.001, mcmcA=2.0):
    if not isinstance(lc, kali.lc.lc):
        raise ValueError('No light curve supplied!')
    if not os.path.isfile(os.path.join(path, 'kali.lc_%s_%s_%s.pkl'%(lc.name, lc.band, lc.z))):
        pickle.dump(lc, open(os.path.join(path, 'kali.lc_%s_%s_%s.pkl'%(lc.name, lc.band, lc.z)), 'wb'))
    if not models:
        raise ValueError('No models specified!')
    if path is None:
        raise ValueError('Must supply output path!')
    for model in models:
        original = model
        model = model.lower()
        kaliRegex = re.compile('kali\.')
        res = re.findall(kaliRegex, model)
        if res:
            model = model.split('.')[1]
        orderRegEx = re.compile('[0-9]+')
        orderList = re.findall(orderRegEx, model)
        if len(orderList) == 2:
            model = model.split('(')[0]
            orderP = int(orderList[0])
            orderQList = [int(orderList[1])]
        elif len(orderList) == 1:
            model = model.split('(')[0]
            orderP = int(orderList[0])
            orderQList = [i for i in xrange(orderP)]
        elif len(orderList) == 0:
            if model != 'mbhb':
                raise ValueError('Model %s could not be interpreted!'%(original))
            else:
                orderP = 0
                orderQList = [0]
        elif len(orderList) > 2:
            raise ValueError('Model %s could not be interpreted!'%(original))
        for orderQ in orderQList:
            fitTask = fitModel(lc, model, orderP, orderQ,
                               nthreads=nthreads, nwalkers=nwalkers, nsteps=nsteps,
                               maxEvals=maxEvals, xTol=xTol, mcmcA=mcmcA)
            filename = 'kali.lc_%s_%s_%s_kali.%s.%d.%d.pkl'%(lc.name, lc.band, lc.z, model, pVal, qVal)
            pickle.dump(fitTask, open(os.path.join(path, filename), 'wb'))
