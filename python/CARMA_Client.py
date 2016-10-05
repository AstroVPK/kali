import zmq
import warnings


TIMEOUT = 1000  # milliseconds
VERBOSE = False
RETRY = True  # Should we try to get another server if we can't connect?

if __name__ == '__main__':
    VERBOSE = True


def printV(*args):

    if VERBOSE:
        for arg in args:
            print arg,
        print ''


class SDSSError(Exception):  # custom SDSSError that relates to serverside issues

    def __init__(self, message, errors=None):

        super(SDSSError, self).__init__(message)
        self.errors = errors


class ServerList(dict):  # dictionary like class that manages the possible servers

    def __init__(self, *args, **kwargs):

        super(ServerList, self).__init__(*args, **kwargs)
        self.best = None

    def addServer(self, name, address, priority):  # add a server to our list of servers
        server = {"address": address, "priority": priority}
        self[name] = server

    def testServer(self, server):

        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.LINGER = False
            socket.connect(server['address'])
            socket.send(b"ping\n", flags=zmq.NOBLOCK)
            if socket.poll(timeout=1000, flags=zmq.POLLIN):
                return True
            else:
                return False
        except zmq.ZMQError as e:
            raise SDSSError(e.message, e.errno)

    def getBestServer(self):  # determine the best server
        for key, server in sorted(self.items(), key=lambda x: x[1]['priority']):
            isGood = self.testServer(server)
            if isGood:
                printV("Best Server is %s" % key)
                self.best = server['address']
                break
        else:
            self.best = None
            raise SDSSError("No good servers available at the moment", self.best)

    def setBestServer(self, server):  # manually override the best server

        printV("Testing Server %s" % server)
        isGood = self.testServer(self[server])
        if isGood:
            self.best = self[server]['address']
            printV("%s is now connected" % server)
        else:
            raise SDSSError("Bad Server: %s" % server, server)


servers = ServerList()
try:
    servers.addServer("newton", "tcp://newton.physics.drexel.edu:5001", 2)
except CARMA_Client.SDSSError as err:
    warnings.warn(str(err))
try:
    servers.addServer("echidna", "tcp://173.75.227.192:5001", 1)
except CARMA_Client.SDSSError as err:
    warnings.warn(str(err))
try:
    servers.addServer("vish15", "tcp://vish15.physics.upenn.edu:5001", 0)
except CARMA_Client.SDSSError as err:
    warnings.warn(str(err))

servers.getBestServer()


def getSocket():

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.LINGER = False
    socket.connect(servers.best)
    return socket


def zmqSocketDecorator(func):  # a decorator that handles the zmq sockets and raises SDSS exceptions

    def wrapper(*args, **kwargs):
        try:
            socket = getSocket()
            return func(socket, *args, **kwargs)
        except zmq.ZMQError as e:
            raise SDSSError(e.message, e.errno)

    return wrapper


@zmqSocketDecorator
def getCommandResult(socket, cmd):  # send a command to the server and return the result
    global RETRY

    socket.send(cmd)
    if socket.poll(timeout=TIMEOUT, flags=zmq.POLLIN):
        result = socket.recv_pyobj(flags=zmq.NOBLOCK)
    else:
        if RETRY:
            printV("Server Disconnected.  Attempting to Connect to Another Server")
            servers.getBestServer()
            RETRY = False
            result = getCommandResult(cmd)
            RETRY = True
            return result
        else:
            raise SDSSError("Socket timed out", TIMEOUT)
    if isinstance(result, Exception):
        raise SDSSError(*result.args)

    return result


def createCommand(server_func, *args):  # get the command string for a function and it's arguments

    if len(args):
        args = " ".join(map(str, args))
    else:
        args = ''
    cmd = b"%s\n%s" % (server_func, args)
    return cmd


def isValid(server_func):  # checks if a server_func is valid

    cmd = createCommand('isValid', server_func)
    result = getCommandResult(cmd)
    return result


def commandArgCount(server_func):  # gets information about the server func

    cmd = createCommand('argCount', server_func)
    result = getCommandResult(cmd)
    return result


def _createFunction(server_func, docstr=None):
    # Create a function object that acts on a server side func with name 'server_func'
    if isValid(server_func):
        nargs = commandArgCount(server_func) - 1
    else:
        raise SDSSError("Invalid Function: %s" % server_func, server_func)

    def Func(*args):
        if len(args) != nargs:
            message = "%s takes exactly %i arguments (%i given)" % (server_func, nargs, len(args))
            raise TypeError(message)
        if docstr is not None:
            Func.__doc__ = docstr
        cmd = createCommand(server_func, *args)
        result = getCommandResult(cmd)
        return result
    return Func


def createFunction(server_func, docstr=None):

    def initialFunc(*args):
        initalFunc = _createFunction(server_func, docstr)
        return initalFunc(*args)
    return initialFunc


# define our client-side functions below
getRandLC = createFunction("randLC",
                           """
args: None
returns:
    filename, redshift, data (tuple):
        filename (str): name of the file on disk
        redshift (float): redshift of the object
        data (numpy structure array): structured array of the data from the LC file
""")

getLC = createFunction("getLC",
                       """
args:
    ID (str): SDSS J2000 name
returns:
    filename, redshift, data (tuple):
        filename (str): name of the file on disk
        redshift (float): redshift of the object
        data (numpy structure array): structured array of the data from the LC file
""")

getIDList = createFunction("IDList",
                           """
args: None
returns:
    IDList (list): List of strings of SDSS Objects names on disk
""")

getNearestLC = createFunction('getNearestLC',
                              """
args:
    ID (str): SDSS J200 name
    tol (float): matching tolerance in degrees
returns:
    filename, reshift, data (tuple):
        see above
""")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print "Test"
    if sys.argv[1] == '--check':
        for name in sys.argv[2:]:
            try:
                getNearestLC(name, 2/60.0/60.0)
            except SDSSError as e:
                if 'No objects in list' in e.message:
                    print "LC does not exist in data base", 0, name
            except IndexError as e:
                print "No File Specified"
            else:
                print "LC does exist in database     ", 1, name
    elif sys.argv[1] == '--rand':
        print getRandLC()
    elif sys.argv[1] == '--list':
        print getIDList()
    elif sys.argv[1] == '--all':
        good = 0
        print "Trying to get a random LC"
        try:
            getRandLC()
            print "Good"
            good += 1
        except Exception as e:
            print e
            print "Bad"
        print "Trying to get the list of ID numbers"
        try:
            getIDList()
            print "Good"
            good += 1
        except Exception as e:
            print e
            print "Bad"
        print "Trying to get a specific LC"
        try:
            getLC("013013.62+000253.5")
            print "Good"
            good += 1
        except Exception as e:
            print e
            print "Bad"
        print "Trying to catch an invalid LC"
        try:
            getLC("013013.62+000253.6")
        except SDSSError as e:
            print "Good"
            good += 1
        except Exception as e:
            print e
            print "Bad"
        if good == 4:
            print "All tests successful"
        else:
            print "Some tests failed, Errors were found"
