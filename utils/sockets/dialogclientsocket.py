import time
from threading import *

from .helper_sockets import *


# Client socket for chatting with DialogServer
# --------------------------------------------
class DialogClient(Thread):
	
	def __init__(self, host, port, hdl_terminate):
		Thread.__init__(self)
		self.status = "init"
		self.host = host
		self.port = port
		self.socket = socket.socket()
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.terminate = hdl_terminate
	
	def init_socket(self, confirm):
		try:
			self.socket.connect((self.host, self.port))
			if confirm:
				data = netCatch(self.socket)
				if data:
					print(data.decode())
			self.start()
		except ConnectionRefusedError as SRE:
			self.status = 'failed'
			print("[error]:", SRE)
		
		return self.status
	
	def run(self):
		self.status = 'running'
		while True:
			data = netCatch(self.socket)
			if data:
				data = data.decode()
				if data == '--EOS--':
					self.status = "running"
				elif data == 'quit':
					self.close()
					break
				elif data == 'pause':
					self.status = 'paused'
					break
				else:
					print(" ", data)
			else:
				self.close()
				self.status = 'init'
				# pass control to the custom terminate handler
				self.terminate()
				break
	
	# This method causes sync communication to the caller
	#   if the wait param is set to true.
	def send(self, msg, wait=False):
		netThrow(self.socket, msg)
		if wait:
			self.status = "working"
			while self.status == 'working':
				time.sleep(1)
	
	def close(self):
		self.socket.close()
		self.status = "init"

