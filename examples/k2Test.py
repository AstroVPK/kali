import numpy as np
import os
import argparse
import pdb

from astropy import units
from astropy.coordinates import SkyCoord

try:
    import kali.carma
    import kali.k2
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type=str, default=os.path.join(
    os.environ['KALI'], 'examples/data'), help=r'Path to working directory')
parser.add_argument('-n', '--name', type=str, default='rand', help=r'K2 EPIC ID')
args = parser.parse_args()

nl = kali.k2.k2LC(
    name='212141173', campaign='c05', band='Kepler', pwd=os.path.join(os.environ['KALI'], 'examples/data'),
    processing='vj',
    coordinates=SkyCoord('08 31 35.483 +22 50 42.64', unit=(units.hourangle, units.deg), frame='icrs'))

nt = kali.carma.CARMATask(3, 1)

Rho = np.array([-1.0/35.0, -1.0/23.0, -1.0/2.0, -1.0/0.5, 2.0e-08])
Theta = kali.carma.coeffs(3, 1, Rho)

nt.set(nl.dt, Theta)

print "logPrior: %+8.7e"%(nt.logPrior(nl))

print "logLikelihood: %+8.7e"%(nt.logLikelihood(nl))

print "logPosterior: %+8.7e"%(nt.logPosterior(nl))
