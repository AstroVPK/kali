import zmq, sys, os, time, cPickle, StringIO, numpy as np, re, random
import multiprocessing as mp
from JacksTools import jio


class Server:

	__version__ = "0.0.1"

	def __init__(self):

		self.context = zmq.Context(io_threads = 4)

		self.socket = self.context.socket(zmq.REP)
		self.socket.bind('tcp://*:5001')
		self.running = True
		self.randBytes = None

		self.commands = {
		'getLC':self.getLC, 
		'randLC':self.randLC,
		'IDList':self.getIDList,
		'isValid':self.isValidCommand,
		'argCount':self.getArgCount,
		'ping': self.respond,
		'getRawLC':self.getRawLC,
		'getAllRawLC':self.getAllRawLC,
		'getNearestLC':self.getNearestLC,
		'getFileInfo':self.getFileInfo,
		'getAllFileInfo':self.getAllFileInfo}
	
		print "Server Started on %s" % FileManager.root

	#Below here are functions that can be called by the user

	def respond(self): #respond to arbitrary request

		self.socket.send_pyobj(1)
		print "Server has been pinged"

	def getArgCount(self, command):

		result = self.commands[command].func_code.co_argcount
		self.socket.send_pyobj(result)
		print "Sent arg count for %s" % command

	def isValidCommand(self, server_func):

		isValid = False
		if server_func in self.commands:
			isValid = True
		self.socket.send_pyobj(isValid)
		print "Sent %s" % str(isValid)

	def getLC(self, ID):

		fname = os.path.join(FileManager.LCDir, FileManager.getFileName(ID))
		data = jio.load(fname, delimiter = ',', headed = True)
		z = FileManager.getRedshift(ID)
		package = (fname, z, data)	
		self.socket.send_pyobj(package)
		print "Sent %s" % FileManager.getFileName(ID)

	def getNearestLC(self, ID, tol):

		tol = float(tol)
		ra = float(ScienceManager.hmsToDeg(ID[:9]))
		dec = float(ScienceManager.dmsToDeg(ID[9:]))
		coords = FileManager.getCoordList()		
		coords = zip(range(len(coords)), coords)
		nearest = None
		sep = 360
		for i, (Ra, Dec) in coords:
			dist = np.arccos(np.sin(dec)*np.sin(Dec)+np.cos(dec)*np.cos(Dec)*np.cos(ra - Ra))
			if dist < tol:
				if (dist < sep):
					sep = dist
					nearest = i
		if nearest is not None:
			ID = FileManager.IDList(FileManager.getLCList())[nearest]
			self.getLC(ID)		
		else:
			raise IndexError("No objects in list within tolerance")

	def getIDList(self):

		idlist = FileManager.IDList(FileManager.getLCList())
		self.socket.send_pyobj(idlist)
		print "Sent %s" % "id list"

	def randLC(self):

		ID = random.choice(FileManager.IDList(FileManager.getLCList()))
		self.getLC(ID)

	def getRawLC(self, ID):

		fname = os.path.join(FileManager.LCDir, FileManager.getFileName(ID))
		with open(fname,'rb') as f:
			raw = f.read()
		self.socket.send_pyobj(raw)
		print "Sent raw data for %s" % ID

	def getAllRawLC(self, *IDList): #get all the data for all IDs in the list

		DataList = []
		for ID in IDList:
			fname = os.path.join(FileManager.LCDir, FileManager.getFileName(ID))
			with open(fname,'rb') as f:
				DataList.append(f.read())
		self.socket.send_pyobj(DataList)
		print "Sent all the data on disk"

	def getFileInfo(self, filename):

		package = FileManager.getFileInfo(filename)
		self.socket.send_pyobj(package)
		print "Sent file information for %s" % ID

	def getAllFileInfo(self):

		
		package = map(FileManager.getFileInfo, FileManager.getLCList())
		self.socket.send_pyobj(package)
		print "Sent information on all files"

	#below here are server specific (sort of private) functions

	def sync(self, server): #sync the LCDir of the server to another server
		'''steps for the sync:
		1. Get List of Files on remote server
		2. Compare file to local files
		2.a. Get files that do not exist on this server
		2.b. Get files that have been updated less recently
		3. sync files
		4. update last updated time
		'''
		try:
			context = zmq.Context()
			socket = context.socket(zmq.REQ)
			socket.LINGER = False
			socket.connect(server)
			
		except zmq.ZMQError as e:
			print e
		else:
			try:
				socket.send(b"IDList\n")
				if socket.poll(timeout = 2000, flags = zmq.POLLIN):
					idList = socket.recv_pyobj()
					print "Got List of ID numbers"
				else:
					idList = []
					print "Sync Failed, could not get ID List"
				socket.send(b"getAllFileInfo\n")
				if socket.poll(timeout = 10000, flags = zmq.POLLIN):
					stats = socket.recv_pyobj()
					print "Get File Information"
				else:
					print "Sync Failed, could not get Stats List"
			except zmq.ZMQError as e:
				print e
			else:
				localIDs = FileManager.IDList(FileManager.getLCList())
				localStats = dict(zip(localIDs,map(FileManager.getFileInfo, FileManager.getLCList())))
				stats = dict(zip(idList, stats))
				matchIDs = list((set(localIDs) & set(idList)))
				new = list(set(idList) - set(localIDs))
				updates = [ID for ID in matchIDs if stats[ID].st_ctime > localStats[ID].st_ctime]			
				idList = list(set(new)|set(updates))
				print "About to sync %i files" % len(idList)
				print "Please wait, this may take up to 20 minutes" 
				IDList = " ".join(idList)
				try:
					for i, ID in enumerate(idList):
						socket.send(b"getRawLC\n%s" % ID)
						if socket.poll(timeout = 5000, flags = zmq.POLLIN):
							print "Got %s" % ID, "%i / %i" % (i, len(idList))
							result = socket.recv_pyobj()
							FileManager.updateLC([ID], [result])
						else:
							print "Timeout, Failed to get %s" & ID
					print "Sync Complete"
				except zmq.ZMQError as e:
					print e
					print "Sync Failed, Server Error"

	def ErrorMsg(self, *args):
		
		error = Exception(*args)
		self.socket.send_pyobj(error)
		print "Sent Exception %s" % args[0]

	def isRunning(self):

		return self.running

	def stop(self):

		self.running = False

	def quit(self):

		self.running = False
		self.socket.close()
		sys.exit(0)

	def processCMD(self, cmd): # process a command input from keyboard

		if cmd.lower() in ['exit','quit']:
			print "Quitting"
			self.quit()
		elif cmd.lower() in ['stop']:
			if self.isRunning():
				print "Stopping Server"
				self.stop()
				return True
			else:
				print "The server is already stopped"
				return True
		elif cmd.lower() in ['start']:
			if self.isRunning():
				print "The server is already running"
				return True
			else:
				print "Starting Server"
				self.start()
				return False
		elif cmd.lower() in ['sync']:
			server = raw_input("Server: ")
			self.sync(server)
	

	def start(self):

		self.running = True
		while self.isRunning():
			try:
				message = self.socket.recv().split('\n')
				command = message[0]
				args = message[1].split()
				print "GOT:",command
				print "   ARGS:", args
				self.commands[command](*args)

			except KeyboardInterrupt as k:
				print "KeyboadInterrupt detected, would you like to do something?"
				cmd = raw_input("=> ")
				self.processCMD(cmd)
				
			except Exception as e:
				print "Failed"
				print e
				self.ErrorMsg(*e.args)

		cmd = raw_input("=> ")
		while self.processCMD(cmd):
			cmd = raw_input("=> ")

class FileManager:

	if "S82DATADIR" in os.environ:
		root = os.environ['S82DATADIR']
	else:
		print "S82DATADIR not found"
		root = ''
		while not os.path.isdir(root):
			root = raw_input("Root Dir: ")

	Pattern = r"LC_(.*)_Calibrated\.csv"	
	LCDir = os.path.join(root,"Data/LC/Calibrated")
	PickleDir = os.path.join(root,"/Pickles")
	RedshiftFile = os.path.join(root,"Data/Stripe82ObjectList.dat")
	zList = []
	regex = re.compile(Pattern)
	with open(RedshiftFile,'rb') as f:
		zList = [line.split() for line in f]

	def __init__(self):
		pass
	
	@classmethod
	def getzList(self):

		return self.zList

	@classmethod
	def getLCList(self):

		return os.listdir(self.LCDir)

	@classmethod
	def getPickleList(self):

		return os.listdir(self.PickleDir)

	@classmethod
	def IDList(self, List):

		return self.regex.findall('\n'.join(List))

	@classmethod
	def getProcessed(self):

		return self.IDList(self.getPickleList())

	@classmethod
	def getUnprocessed(self):

		return list(set(self.IDList(self.getLCList())) - set(self.getProcessed()))

	@classmethod
	def getFileName(self, ID):

		return self.getLCList()[self.IDList(self.getLCList()).index(ID)]

	@classmethod
	def getFileInfo(self, filename):
		
		return os.stat(os.path.join(self.LCDir,filename))

	@classmethod
	def getRedshift(self, ID):

		index = [i[0] for i in self.zList].index(ID) #1 is the index for the ID number
		return self.zList[index][3] #3 is the index for redshift

	@classmethod
	def getCoordList(self):

		return map(lambda x: (float(ScienceManager.hmsToDeg(x[:9])), float(ScienceManager.dmsToDeg(x[9:]))), self.IDList(self.getLCList()))

	@classmethod
	def updateLC(self, IDList, dataList):
	
		for ID, data in zip(IDList, dataList):
			filename = os.path.join(self.LCDir, ID.join(("LC_","_Calibrated.csv")))	
			with open(filename,'w') as f:
				f.write(data)

	#below here manage the server files

class ScienceManager:

	@classmethod
	def degTohms(self, deg):
	
		deg = float(deg)	
		h = deg/360.0*24
		m = (h - int(h))*60
		s = (m - int(m))*60
		h = str(int(h)).zfill(2)
		m = str(int(m)).zfill(2)
		s = '.'.join((str(int(s)).zfill(2), str(int((s - int(s))*100)).zfill(2)))
		return ''.join((h,m,s))

	@classmethod
	def degTodms(self, deg):
		
		deg = float(deg)
		d = dec*(-1 if deg < 1 else 1)
		m = (d - int(d))*60
		s = (m - int(m))*60
		d = str(int(d)).zfill(2)
		m = str(int(m)).zfill(2)
		s = '.'.join((str(int(s)).zfill(2), str(int((s - int(s))*100))[0]))
		return ''.join((d,m,s))

	@classmethod			
	def dmsToDeg(self, dms):

		d, m, s = map(float,(dms[:3], dms[3:5], dms[5:]))
		if d > 0:
			deg = d + m/60.0 + s/60.0/60.0
		else:
			deg = d - m/60.0 - s/60.0/60.0
		return str(deg)

	@classmethod
	def hmsToDeg(self, hms):
		h,m,s = map(float, (hms[:2], hms[2:4], hms[4:]))
		deg = h/24.0*360.0 + m/24.0*360.0/60.0 + s/24.0*360.0/60.0/60.0
		return str(deg)


if __name__ == '__main__':

	S = Server()
	S.start()
	
