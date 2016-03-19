import numpy as np

import lc
import CARMATask

numCadences = 1000
dt = 0.1

r = lc.basicLC(numCadences)
newTask = CARMATask.CARMATask(1,0)

Theta = np.zeros(2)
Theta[0] = 0.1
Theta[1] = 1.0

res = newTask.printSystem(0.1, Theta)

res = newTask.makeIntrinsicLC(Theta, numCadences, dt, False, 1.0e-3, 0.15, 0.001, 2.0, 1.0, 100.0, r.t, r.x, r.y, r.yerr, r.mask, 78676897, 78687678)

newTask.guard()

print r