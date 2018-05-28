import sys
import struct
import io
import socket
import time

import platform
# RaspberryPi B+ : excluded imports
if platform.machine() != 'armv6l':
	# Required so that openCV can display with imshow in multiple threads
	from matplotlib import pyplot as plt


# Callback Function that does nothing
def nothing(*arg):
	pass


# SocketDescriptor extends class socket.socket
#   It introduces new attributes and a .copy() static method
#   for creating new class instances from original socket objects.
#   Since .accept() method returns an original socket obj, we can use
#   the .copy() method to cast it into a SocketDescriptor
#   Another way would be to override .accept()
class SocketDescriptor(socket.socket):
	def __init__(self, *args, **kwargs):
		super(SocketDescriptor, self).__init__(*args, **kwargs)
		self.c_name = "undefined"
		self.c_state = 'idle'
		self.c_created = time.time()
		self.c_idle_time = 0
		self.c_fstream = None
	
	@classmethod
	def copy(cls, sock):
		fd = socket.dup(sock.fileno())
		copy = cls(sock.family, sock.type, sock.proto, fileno=fd)
		copy.settimeout(sock.gettimeout())
		return copy


# Open a socket listening for connections on address = (interface-ip port)
#    (0.0.0.0 port) means all interfaces in that port
def startListener(address):
	print("> Starting listener socket {0}.\n  Listening at {1} {2}".format(socket.gethostbyname(socket.gethostname()),
	                                                                       address[0], address[1]))
	sock = socket.socket()
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind(address)
	sock.listen(1)
	return sock


# Send message through a socket
#   using a <stream-length> header of 4 bytes
def netThrow(sock, message):
	try:
		sock.send("{:04}".format(len(message)).encode())
		sock.send(message.encode())
	except socket.error as SE:
		print("Connection issue. Sending message failed:\n", SE)


# Retrieve a message from a socket
#   using a <stream-length> header of 4 bytes
def netCatch(sock):
	try:
		ln = sock.recv(4)
		data = sock.recv(int(ln.decode()))
		return data
	except socket.error as SE:
		print("[error]: connection failed!")
		return False
	# raise
	except:
		data = ln + sock.recv(100)
		print("[error]: protocol failed: ", sys.exc_info()[0], "Invalid data from client")
		return data
# raise


# Send binary data through a file like object connector
#   using a <stream-length> header of type 32-bit unsigned int
def binThrow(conn, stream):
	try:
		stream_len = stream.seek(0, 2)
		conn.write(struct.pack('<L', stream_len))
		# conn.flush()
		stream.seek(0)
		conn.write(stream.read())
		conn.flush()
	except:
		print("Connection issue. Sending message failed:\n", sys.exc_info()[0])


# Retrieve a message from a socket
#   using a <stream-length> header of 4 bytes
def binCatch(conn):
	try:
		stream = io.BytesIO()
		stream_len = struct.unpack('<L', conn.read(struct.calcsize('<L')))[0]
		stream.write(conn.read(stream_len))
		stream.seek(0)
		return stream_len, stream
	except:
		print("[error] (binCatch) : stream failed: ", sys.exc_info()[0])
		return 0, None


# Sends a string message and gets back an answer
def netAsk(sock, msgNo, str):
	# Send message
	netThrow(sock, str)
	print("[{0}]({1}): {2}".format(socket.gethostname(), msgNo, str))
	# Receive answer
	data = netCatch(sock)
	print("[{0}]({1}): {2}".format(sock.getpeername()[0], msgNo, data.decode('UTF-8')))
	# Return answer received
	return data.decode('UTF-8')


# Listens for message and sends back
def netReply(sock, msgNo, str):
	# Receive message
	data = netCatch(sock)
	print("[{0}]({1}): {2}".format(sock.getpeername()[0], msgNo, data.decode('UTF-8')))
	# Sends reply
	netThrow(sock, str)
	print("[{0}]({1}): {2}".format(socket.gethostname(), msgNo, str))
	# Return message received
	return data.decode('UTF-8')


