import numpy as np
import argparse
import os
import sys
import pdb

try:
    import kali.carma
    import kali.s82
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type=str, default=os.path.join(
    os.environ['KALI'], 'examples/data'), help=r'Path to working directory')
parser.add_argument('-n', '--name', type=str, default='rand', help=r'SDSS ID')
args = parser.parse_args()

nl = kali.s82.sdssLC(name=args.name, band='g', pwd=args.pwd)

nt = kali.carma.CARMATask(3, 1)

Rho = np.array([-1.0/100.0, -1.0/55.0, -1.0/10.0, -1.0/25.0, 2.0e-08])
Theta = kali.carma.coeffs(3, 1, Rho)

nt.set(nl.dt, Theta)

print "logPrior: %+8.7e"%(nt.logPrior(nl))

print "logLikelihood: %+8.7e"%(nt.logLikelihood(nl))

print "logPosterior: %+8.7e"%(nt.logPosterior(nl))
