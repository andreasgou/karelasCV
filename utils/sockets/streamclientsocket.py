import sys
from threading import *

from .helper_sockets import *


# Client socket for streaming from DialogServer
# ---------------------------------------------
class StreamClient(Thread):
	
	def __init__(self, host, port, hdl_terminate):
		Thread.__init__(self)
		self.status = "init"
		self.pipe = None
		self.consumer = None
		self.host = host
		self.port = port
		self.socket = socket.socket()
		# self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.settimeout(60)
		self.terminate = hdl_terminate
	
	def init_socket(self, confirm):
		try:
			self.socket.connect((self.host, self.port))
			if confirm:
				self.status = 'negotiate'
				data = netCatch(self.socket)
				if data:
					netThrow(self.socket, "stream-listener")
			# Make a file-like object
			self.pipe = self.socket.makefile('rb')
			self.start()
		except (TimeoutError, socket.timeout) as STO:
			self.status = 'failed'
			print("[error]:", STO)
		except ConnectionRefusedError as SRE:
			self.status = 'failed'
			print("[error]:", SRE)
		except:
			self.status = 'failed'
			print("[error]: protocol failed: ", sys.exc_info()[0])
		
		return self.status
	
	def run(self):
		while True:
			(ln, data) = binCatch(self.pipe)
			if self.status == 'purge':
				# consume all data from the pipe
				self.pipe.flush()
				print("[info] (StreamClient.run) : Video stream purged")
				continue
			
			# print("  ", self.status, ln, data)
			if data:
				if self.status == 'negotiate':
					data = data.getvalue().decode()
					if data == '--EOS--':
						self.status = "running"
					elif data == 'quit':
						break
					else:
						print(" ", data)
				else:
					self.consumer(self, ln, data)
			else:
				self.close()
				# pass control to the custom terminate handler
				# self.terminate()
				break
		
		self.status = "init"
		print("Stream socket stopped")
	
	def set_consumer(self, consumer):
		self.consumer = consumer
	
	def close(self):
		if self.status != 'failed':
			self.status == 'purge'
			self.pipe.flush()
			self.pipe.close()
			self.socket.close()
			self.status = "init"
		print("Stream socket closed")
	
	def purge_negotiate(self):
		self.status = 'purge'

