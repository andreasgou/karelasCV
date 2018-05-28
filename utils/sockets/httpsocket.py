import sys
from threading import *

from .helper_sockets import *


# Client socket for data streaming from an HTTP server
# ----------------------------------------------------
class StreamHttpClient(Thread):
	
	def __init__(self, host, port, path, hdl_terminate):
		Thread.__init__(self)
		self.status = "init"
		self.pipe = None
		self.consumer = None
		self.host = host
		self.port = port
		self.path = path
		self.url = "http://" + self.host + ":" + str(self.port) + self.path
		self.timeout = 60
		self.cache = bytearray()
		self.terminate = hdl_terminate
	
	def init_socket(self, confirm):
		try:
			# if confirm:
			#     print("[info] Connected. Getting video stream..")
			#     self.status = 'negotiate'
			#     self.status = 'running'
			#     self.socket.sendall("GET {} HTTP/1.1\r\n\r\n".format(self.path).encode('utf-8'))
			# Make a file-like object
			# self.pipe = self.socket.makefile('rb')
			(ln, data, self.cache) = jpgStream(self.url, self.cache)
			if ln > 0:
				print("[info] Opening {}".format(self.url))
				self.status = 'running'
				self.start()
			else:
				print("[warning] No data from server or server not accessible.")
				self.status = 'failed'
		except (TimeoutError, socket.timeout) as STO:
			self.status = 'failed'
			print("[error]:", STO)
		except ConnectionRefusedError as SRE:
			self.status = 'failed'
			print("[error]:", SRE)
		except:
			self.status = 'failed'
			print("[error]: protocol failed: ", sys.exc_info())
		
		return self.status
	
	def run(self):
		# class variable to be used for video stream only
		self.cache = bytearray()
		while True:
			try:
				(ln, data, self.cache) = self.read_jpg_stream(self.url, self.cache)
				if self.status == 'purge':
					# consume all data from the pipe
					# self.pipe.flush()
					print("[info] (StreamClient.run) : Video stream purged")
					continue
				
				# print("  ", self.status, ln, data)
				if ln > 0:
					if self.status == 'negotiate':
						data = data.getvalue().decode()
						if data == '--EOS--':
							self.status = "running"
						elif data == 'quit':
							break
						else:
							print(" ", data)
					else:
						if self.consumer is not None:
							self.consumer(self, ln, data)
						else:
							self.close()
							break
				else:
					self.close()
					# pass control to the custom terminate handler
					# self.terminate()
					break
			except TimeoutError:
				print("[error] stream failed: ", sys.exc_info()[0])
				self.status = 'failed'
				self.close()
				break
		
		self.status = "init"
		print("Stream socket stopped")
	
	def set_consumer(self, consumer):
		self.consumer = consumer
	
	def close(self):
		self.status = "init"
		print("Stream socket closed")
	
	def purge_negotiate(self):
		self.status = 'purge'
	
	def read_jpg_stream(self, url, cache):
		try:
			bt = cache
			
			img_resp = requests.get(url)
			img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
			stream = cv2.imdecode(img_array, -1)
			# stream = img_array
			stream_len = len(stream)
			
			return stream_len, stream, bt
		except requests.exceptions.ConnectionError:
			print("[error] (jpgStream) : connection error: ", sys.exc_info()[0])
			return 0, None, None
		except:
			print("[error] (jpgStream) : stream failed: ", sys.exc_info())
			return 0, None, None

		

