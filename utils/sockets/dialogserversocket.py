import sys
import select
import socket
import io

from .helper_sockets import SocketDescriptor
from .helper_sockets import netCatch
from .helper_sockets import netThrow
from .helper_sockets import binThrow


# Open server socket for chat with incoming connections
# This socket is able to handle multiple client connections
# ------------------------------------------------------------------
class DialogServer:
	
	def __init__(self, port, private=True):
		self.port = port
		
		self.srvsock = SocketDescriptor(socket.AF_INET, socket.SOCK_STREAM)
		self.srvsock.c_name = "Server socket"
		self.srvsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.srvsock.bind(("", port))
		self.srvsock.listen(2)
		
		self.descriptors = [self.srvsock]
		self.outputs = []
		self.streams = []
		self.commands = {}
		self.private = private
		self.welcome = None
		
		print("DialogServer started on port {}\nPrivate mode is {}".format(port, self.private))
	
	def run(self):
		evt = 0
		while 1:
			# Await an event on a readable socket descriptor
			(sread, swrite, sexc) = select.select(self.descriptors, [], [])
			
			# Iterate through the tagged read descriptors
			for sock in sread:
				
				# Received a connect to the server (listening) socket
				if sock == self.srvsock:
					self.accept_new_connection()
				else:
					
					# Received something on a client socket
					# stin = sock.recv(100).decode()
					stin = netCatch(sock)
					
					# Check to see if the peer socket closed
					if stin == '':
						self.close(sock)
					else:
						try:
							host, port = sock.getpeername()
						except:
							print("[error]", sys.exc_info()[0], "Closing socket")
							self.descriptors.remove(sock)
							sock.close()
							continue
						
						stin = stin.decode()
						stout = '[%s:%s] %s' % (host, port, stin)
						if stin.rstrip() == 'quit':
							netThrow(sock, "quit")
							self.close(sock)
						elif stin.rstrip() == 'stream-listener':
							# Move socket to streams list
							self.descriptors.remove(sock)
							self.streams.append(sock)
							# Make a file-like object out of the connection
							sock.c_fstream = sock.makefile('wb')
							binThrow(sock.c_fstream, io.BytesIO(b"--EOS--"))
							print("Stream socket request completed for {}:{}".format(host, port))
						elif stin.rstrip() == 'close-stream-listener':
							so = self.streams[0]
							host, port = so.getpeername()
							self.streams.remove(so)
							# so.c_fstream.close()
							binThrow(so.c_fstream, io.BytesIO(b"quit"))
							# time.sleep(5)
							netThrow(sock, "--EOS--")
							# time.sleep(5)
							stout = 'Stream socket closed %s:%s\n' % (host, port)
							self.broadcast_string(stout, sock)
						# so.close()
						else:
							self.broadcast_string(stout, sock)
							stcmd = stin.rstrip().split(' ')
							cmd = self.commands.get(stcmd[0])
							if cmd:
								print("Executing command '{} {}' for {}:{}".format(cmd.__name__, stcmd[0], host, port))
								cmd(self, sock, stcmd)
							
							netThrow(sock, "--EOS--")
	
	def close(self, sock):
		host, port = sock.getpeername()
		stout = 'Client left %s:%s\n' % (host, port)
		self.broadcast_string(stout, sock)
		self.descriptors.remove(sock)
		sock.close()
	
	def broadcast_string(self, string, omit_sock):
		if not self.private:
			for sock in self.descriptors:
				if sock != self.srvsock and sock != omit_sock:
					netThrow(sock, string)
		print("{}".format(string.rstrip()))
	
	def accept_new_connection(self):
		so, (remhost, remport) = self.srvsock.accept()
		# The .accept() method returns a socket.socket object
		# thus we need to cast it into a SocketDescriptor class
		# by invoking it's .copy() static method we defined.
		newso = SocketDescriptor.copy(so)
		newso.c_name = "Client {}:{}".format(remhost, remport)
		self.descriptors.append(newso)
		# Close the original socket for safety.
		so.close()
		
		netThrow(newso, self.welcome())
		stout = 'Client joined %s:%s\n' % (remhost, remport)
		self.broadcast_string(stout, newso)
	
	def set_commands(self, cmd_list):
		self.commands = cmd_list
	
	def append_command(self, cmd):
		self.commands.update(cmd)

