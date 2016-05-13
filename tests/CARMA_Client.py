import zmq, sys, os, time, cPickle, StringIO, numpy as np, re, random
import carmcmc as cm
import multiprocessing as mp
from JacksTools import jio



def getRandLC():

	context = zmq.Context()
	socket = context.socket(zmq.REQ)
	socket.connect('tcp://76.124.106.126:5001')

	socket.send(b"randLC\n0")
	fname, z, data = socket.recv_pyobj()
	socket.close()
	return fname, z, data

if __name__ == '__main__':

	print getRandLC()	

