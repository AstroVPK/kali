import math
import cmath
import numpy as np
import argparse as argparse
import pdb

import libcarma
import sdss
from util.mpl_settings import set_plot_params

fhgt = 10
fwid = 16
set_plot_params(useTex = True)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type = str, default = '/home/vpk24/Documents', help = r'Path to working directory')
parser.add_argument('-n', '--name', type = str, default = 'LightCurveSDSS_1.csv', help = r'SDSS Filename')
args = parser.parse_args()

sdssLC = sdss.sdss_gLC(supplied = args.name, pwd = args.pwd)

nt = libcarma.basicTask(3, 1)

Theta = np.array([6.79360723e-01, 5.68739766e-01, 9.55019388e-03, 1.76440554e-07, 9.28333039e-07])

nt.set(sdssLC.dt, Theta)

print "logLikelihood: %+8.7e"%(nt.logLikelihood(sdssLC))