import pdb

from python.plotPSD import plotPSDTask
from python.writeMockLC import writeMockLCTask

plotPSDTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/','Config.ini').run()
writeMockLCTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/','Config.ini').run()
#writeMockLCTask('/home/vish/code/trunk/cpp/libcarma/examples/writeMockLCTest/','Config.ini', DateTime = '12252015131853').run()