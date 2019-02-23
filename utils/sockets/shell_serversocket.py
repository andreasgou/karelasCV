import sys
import select
import socket
import queue
import time

from utils.sockets.shell_socket import ShellSocket
from utils.sockets.shell_socket import SocketDescriptor


# Open server socket for chat with incoming connections
# This socket is able to handle multiple client connections
# ------------------------------------------------------------------
class ShellServerSocket(ShellSocket):
	
	def __init__(self, host="", port=0, private=True):
		ShellSocket.__init__(self, host, port)
		self.private = private
		
	def init(self, name):
		ShellSocket.init(self, name)
		try:
			self.socket.bind((self.host, self.port))
		except OSError as OSE:
			print("[error]:", OSE, "(port {})".format(self.port))
			self.active = False
			self.status = 'failure'
		else:
			self.socket.listen(2)
			self.inputs.append(self.socket)
			self.active = True
			self.status = 'ready'
			print("ServerSocket started on port {}\nPrivate mode is {}".format(self.port, self.private))
	
	def run(self):
		while 1:
			# Await an event on a readable socket descriptor
			(sread, swrite, sexc) = select.select(self.inputs, self.outputs, self.inputs)
			
			# Iterate through input sockets
			for sock in sread:
				# Received a connect to the server (listening) socket
				if sock == self.socket:
					self.accept_new_connection()
				else:
					stin = self.get_msg(sock)
					# Check to see if the peer socket closed
					if stin:
						stin = stin.decode()
						stin = stin.rstrip()
						stout = "[{}] {}".format(sock.c_name, stin)

						stcmd = stin.split(' ')
						cmd = self.commands.get(stcmd[0])
						send_eos = True
						if cmd:
							# sock.log("command", cmd.__name__)
							sock.log("command", stin)
							send_eos = cmd(self, sock, stcmd)
						else:
							sock.log("info", stin)
							self.broadcast_string(stout, sock)
		
						if send_eos:
							self.put_msg(sock, "--EOS--")
					else:
						self.close(sock)

			# Iterate through output sockets
			for sock in swrite:
				try:
					message = self.queues[sock].get_nowait()
				except queue.Empty:
					self.outputs.remove(sock)
				except socket.error as SE:
					print("Connection issue. Sending message failed:\n", SE)
				else:
					sock.send("{:04}".format(len(message)).encode())
					sock.send(message.encode())
			
			# Iterate through exceptions
			for sock in sexc:
				sock.close()

	def close(self, sock, remote=False):
		ShellSocket.close(self, sock)
		sock.log("info", "Client left")
		stout = "[{}] {}".format(sock.c_name, "Client left")
		self.broadcast_string(stout, sock)
	
	def broadcast_string(self, string, omit_sock):
		if not self.private:
			for sock in self.inputs:
				if sock != self.socket and sock != omit_sock:
					self.put_msg(sock, string)
	
	def accept_new_connection(self):
		so, (remhost, remport) = self.socket.accept()
		# The .accept() method returns a socket.socket object
		# thus we need to cast it into a SocketDescriptor class
		# by invoking it's .copy() static method we defined.
		newso = SocketDescriptor.copy(so)
		newso.c_name = "{}:{}".format(remhost, remport)
		self.inputs.append(newso)
		# Close the original socket for safety.
		so.close()
		
		self.queues[newso] = queue.Queue()
		self.put_msg(newso, self.welcome(self, newso))
		stout = "Client joined {}".format(newso.c_name)
		self.socket.log("info", stout)
		self.broadcast_string(stout, newso)

