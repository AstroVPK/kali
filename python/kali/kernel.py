import math
import cmath
import numpy as np
import pdb


class Kernel(object):

    def __init__(self):
        pass

    def __call__():
        pass


class Lanczos(Kernel):

    def __init__(self, a=3):
        self.a = a

    def _lanczos(self, z):
        if z == 0.0:
            return 1.0
        elif math.fabs(z) >= self.a:
            return 0.0
        else:
            return (self.a*math.sin(math.pi*z)*math.sin((math.pi*z)/self.a))/(math.pow(math.pi,
                                                                              2.0)*math.pow(z, 2.0))

    def _lanczos1d(self, x, y):
        return self._lanczos(x - y)

    def __call__(self, srcLC, destLC):
        for i in range(destLC.numCadences):
            for j in range(srcLC.numCadences):
                destLC.x[i] += self._lanczos1d(srcLC.t[j], destLC.t[i])*srcLC.mask[j]*srcLC.x[j]
                destLC.y[i] += self._lanczos1d(srcLC.t[j], destLC.t[i])*srcLC.mask[j]*srcLC.y[j]
                destLC.yerr[i] += math.pow(self._lanczos1d(srcLC.t[j],
                                           destLC.t[i])*srcLC.mask[j]*srcLC.yerr[j], 2.0)
            destLC.yerr[i] = math.sqrt(destLC.yerr[i])
