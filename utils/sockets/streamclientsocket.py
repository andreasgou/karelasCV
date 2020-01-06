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
		self.pause = False
		self.pause_time = 0.0
	
	def init_socket(self):
		try:
			self.socket.connect((self.host, self.port))
			self.status = 'negotiate'
			data = netCatch(self.socket)
			if data:
				netThrow(self.socket, "stream-listener")
			# Make a file-like object
			self.pipe = self.socket.makefile('rb')
			self.start()
		except (TimeoutError, socket.timeout) as STO:
			self.status = 'failed'
			self.printout("[error]: {}".format(STO))
		except ConnectionRefusedError as SRE:
			self.status = 'failed'
			self.printout("[error]: {}".format(SRE))
		except:
			self.status = 'failed'
			self.printout("[error]: protocol failed: {}".format(sys.exc_info()[0]))
		
		return self.status
	
	def run(self):
		while True:
			if self.pause:
				if self.consumer is not None:
					self.consumer(self, None, None)
				continue
			
			(ln, data) = binCatch(self.pipe)
			if self.status == 'purge' and data:
				# consume all data from the pipe
				self.pipe.flush()
				self.printout("[info]: Video stream purged")
				continue
			
			# self.printout("  ", self.status, ln, data)
			if data:
				if self.status == 'negotiate':
					data = data.getvalue().decode()
					if data == '--EOS--':
						self.status = "running"
					elif data == 'quit':
						break
					else:
						self.printout("  {}".format(data))
				else:
					self.consumer(self, ln, data)
			else:
				self.close()
				# pass control to the custom terminate handler
				# self.terminate()
				break
		
		self.status = "init"
		self.printout("Stream socket stopped")
	
	def set_consumer(self, consumer):
		self.consumer = consumer
	
	def close(self):
		self.printout("Closing data stream.. (status='{}')".format(self.status))
		if self.status != 'failed':
			self.status == 'purge'
			self.pipe.flush()
			self.pipe.close()
			self.socket.close()
			self.status = "init"
		self.printout("Stream socket closed")
	
	def purge_negotiate(self):
		self.status = 'purge'
	
	def printout(self, msg):
		print(type(self).__name__, msg)

