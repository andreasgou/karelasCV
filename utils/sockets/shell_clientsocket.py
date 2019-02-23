import select
from threading import *
import queue
from .shell_socket import *


# Client socket for chatting with DialogServer
# --------------------------------------------
class ShellClientSocket(ShellSocket, Thread):
	
	def __init__(self, host, port, hdl_terminate):
		ShellSocket.__init__(self, host, port)
		Thread.__init__(self)
		self.terminate = hdl_terminate
	
	def init(self, name, confirm):
		ShellSocket.init(self, name)
		try:
			self.socket.connect((self.host, self.port))
			if confirm:
				data = self.get_msg(self.socket)
				if data:
					print(data.decode())
			self.inputs.append(self.socket)
			self.queues[self.socket] = queue.Queue()
			self.active = True
			self.status = 'ready'
			
			self.start()
		except ConnectionRefusedError as SRE:
			self.status = 'failure'
			print("[error]:", SRE)
		
		return self.status
	
	def run(self):
		self.status = 'running'
		while self.active:
			# Await an event on a readable socket descriptor
			(sread, swrite, sexc) = select.select(self.inputs, self.outputs, self.inputs, 1)

			# Iterate through input sockets
			for sock in sread:
				# Received a connect to the server (listening) socket
				stin = self.get_msg(sock)

				# Check to see if the peer socket closed
				if stin:
					stin = stin.decode()
					stin = stin.rstrip()
					
					if stin == '--EOS--':
						self.status = "running"
					else:
						sock.log("info", stin)
					
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
				print("Exception from socket")
				sock.close()
		
		self.status = 'idle'
	
	# Send a message to the remote host.
	# This method mimics sync communication with the remote host if wait=True.
	# IMPORTANT NOTE: This method should be called from another thread (not the main thread loop).
	def send(self, msg, wait=False):
		self.put_msg(self.socket, msg)
		if wait:
			self.status = "waiting"
			while self.status == 'waiting':
				time.sleep(1)
	
	# Close current socket.
	# If remote=True, the socket is closed from the main thread loop method run() above.
	# This is the case when we need to notify the remote host before closing the connection.
	# The server is notified that we want to close the connection through a previous command, it executes the
	# necessary code and then it closes the socket. When the main thread loop gets an exception caused from the closed
	# connection, it calls close() with remote=False and gracefully terminates the client socket.
	# IMPORTANT NOTE: Calling this method with remote=True should be called from another thread, outside
	#                 the main thread loop.
	def close(self, sock, remote=False):
		sock.c_state = "closed"
		if remote:
			while self.status != "idle":
				time.sleep(1)
			ShellSocket.close(self, sock)
		else:
			self.active = False

	

