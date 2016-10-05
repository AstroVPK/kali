#!/usr/bin/env python
"""	Module to draw hardware random numbers.

    For a demonstration of the module, please run the module as a command line program eg.
    bash-prompt$ python randDemo.py --help
    and
    bash-prompt$ python randDemo.py -n 100
"""

import numpy as np
from . import lib.rand as rand

if __name__ == '__main__':
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numRand', type=int, default=10,
                        help=r'Number of hardware random numbers to generate...')
    args = parser.parse_args()

    A = np.zeros(args.numRand, dtype='uint32')
    success = rand.rdrand(A)
    for i in xrange(args.numRand):
        print '%s'%(str(A[i]))
