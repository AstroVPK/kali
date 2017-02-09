import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import subprocess
import argparse
import matplotlib.pyplot as plt
import pdb

try:
    import libcarma as libcarma
except ImportError:
    print 'libcarma is not setup. Setup libcarma by sourcing bin/setup.sh'
    sys.exit(1)


class surveyLC(libcarma.basicLC):

    def read(self, name, band='?', path=os.environ['?DATADIR'], **kwargs):
        # CODE here to open the data file ####

        # CODE HERE to construct t, x, y, yerr, & mask + numCadences, startT + other properties you want to
        # track.

        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'V')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$d$'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        # self._yunit = r'who the f*** knows?' ## Unit in which the flux is measured (eg Wm^{-2} etc...).
        self._yunit = r'$F$ (Jy)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--n', type=str, default='???????', help=r'Object name')
    parser.add_argument('-band', '--b', type=str, default='???????', help=r'System bandpass')
    parser.add_argument('-pwd', '--pwd', type=str, default='???????', help=r'Directory to work with data')
    args = parser.parse_args()

    LC = surveyLC(name=args.name, band=args.band, path=args.pwd)

    LC.plot()
    LC.plotacf()
    LC.plotsf()
    plt.show(False)
