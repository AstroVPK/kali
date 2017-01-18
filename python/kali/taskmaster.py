import math
import cmath
import numpy as np
import pdb


class taskMaster(object):
    """!
\anchor taskMaster_

\brief Efficiently run a set of tasks on a set of light curves.

\section kali_taskMaster_Contents Contents
  - \ref kali_taskMaster__Purpose
  - \ref kali_taskMaster_Initialize
  - \ref kali_taskMaster_Run
  - \ref kali_taskMaster_Config
  - \ref kali_taskMaster_Debug
  - \ref kali_taskMaster_Example

\section kali_taskMaster_Purpose	Description

\copybrief taskMaster_

The taskMaster provides us with a high level tool for pipelining the fitting of multiple light curves to
multiple models (ie. multiple kali Task objects). Since individual Tasks may allocate extra memory
(tranparently in the background) to fit light curves, part of the design goal of the taskMaster is to re-use
each Task as many times as possible

\section kali_taskMaster_Initialize       Task initialization
\copydoc \_\_init\_\_

\section kali_taskMaster_Run       Invoking the Task
\copydoc run

\section kali_taskMaster_Config       Configuration parameters
See \ref AssembleCoaddConfig_

\section kali_taskMaster_Debug		Debug

\section kali_taskMaster_Example	A complete example of using the taskMaster

    """


def __init__(self, *args, **kwargs):
    pass
