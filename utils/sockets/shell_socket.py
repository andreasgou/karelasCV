import socket
import sys
import time


class ShellSocket:
	
	def __init__(self, host="", port=0):
		self.host = host
		self.port = port
		self.socket = None
		self.inputs = []
		self.outputs = []
		self.queues = {}
		self.streams = []
		self.commands = {}
		self.welcome = None
		self.active = False
		self.status = None
	
	def init(self, name):
		self.socket = SocketDescriptor(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.c_name = name
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.status = "init"
	
	# Retrieve a binary message from a socket.
	# The message begins with a 4-byte header containing the length of the actual message
	def get_msg(self, sock):
		try:
			ln = sock.recv(4)
			data = sock.recv(int(ln.decode()))
			return data
		except socket.error as SE:
			print("[error]: connection failed!")
			return False
		except:
			if sock.c_state != "closed":
				data = ln + sock.recv(100)
				print("[error]: message protocol failed: ", sys.exc_info()[0], "Invalid data from client")
				return data
			else:
				return None
	
	# Send message using a queue
	#   using a <stream-length> header of 4 bytes
	def put_msg(self, sock, message):
		try:
			self.queues[sock].put(message)
			if sock not in self.outputs:
				self.outputs.append(sock)
		except socket.error as SE:
			print("Connection issue. Sending message failed:\n", SE)
	
	def close(self, sock, remote=False):
		self.inputs.remove(sock)
		if sock in self.outputs:
			self.outputs.remove(sock)
		sock.close()
		sock.c_state = "closed"
		del self.queues[sock]
	
	def set_commands(self, cmd_list):
		self.commands = cmd_list
	
	def append_command(self, cmd):
		self.commands.update(cmd)


# SocketDescriptor extends class socket.socket
#   It introduces new attributes and a .copy() static method
#   for creating new class instances from original socket objects.
#   Since .accept() method returns an original socket obj, we can use
#   the .copy() method to cast it into a SocketDescriptor
#   Another way would be to override .accept()
# Credits to augurar: https://stackoverflow.com/a/45209878/7521854
class SocketDescriptor(socket.socket):
	def __init__(self, *args, **kwargs):
		super(SocketDescriptor, self).__init__(*args, **kwargs)
		self.c_name = "undefined"
		self.c_state = 'idle'
		self.c_created = time.time()
		self.c_idle_time = 0
		self.c_fstream = None
	
	@classmethod
	def copy(self, sock):
		fd = socket.dup(sock.fileno())
		nsoc = self(sock.family, sock.type, sock.proto, fileno=fd)
		nsoc.settimeout(sock.gettimeout())
		return nsoc
	
	def log(self, level, msg):
		# print("[{}] {}:".format(level, self.c_name), msg)
		print("[{}]".format(self.c_name), msg)
