import sys
import time
import cv2
from threading import *
from imutils import paths
from utils.datasets import SimpleDatasetLoader
from utils.preprocessing import SimplePreprocessor

from .helper_sockets import *


# Server socket for streaming images from a local folder
# ------------------------------------------------------------------
class StreamFileListClient(Thread):
	
	def __init__(self, dim, path, hdl_terminate):
		Thread.__init__(self)
		self.status = "init"
		self.pipe = None
		self.consumer = None
		self.path = path
		self.data = None
		self.idletime = 5
		self.terminate = hdl_terminate
		self.dim = dim
	
	def init_socket(self):
		try:
			imagePaths = list(paths.list_images(self.path))
			print(imagePaths, self.path)
			# aqcuire first image dimensions and use them for all
			imagePath = imagePaths[0].replace("\\ ", " ")
			img = cv2.imread(imagePath)
			if img is None:
				raise Exception('Image not found?', imagePath)
			if self.dim is None:
				self.dim = img.shape[:2]
			else:
				self.dim = (self.dim[1], self.dim[0])
			# initialize the image preprocessors
			sp = SimplePreprocessor(self.dim[1], self.dim[0])
			# iap = ImageToArrayPreprocessor()
			
			sdl = SimpleDatasetLoader(preprocessors=[sp])
			# sdl = SimpleDatasetLoader()
			(data, labels, _) = sdl.load(imagePaths, verbose=100)
			# data = data.astype("float") / 255.0

			if len(data) > 0:
				print("[info] Image matrix shape: {}".format(data.shape))
				self.data = data
				# self.status = 'running'
				# self.start()
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
		self.status = 'running'
		print("[debug] Thread is running..")
		while True:
			try:
				for i in self.data:
					if self.status == 'purge':
						# consume all data from the pipe
						# self.pipe.flush()
						self.consumer = None
						print("[info] (StreamClient.run) : Video stream purged")
						break
					
					else:
						data = i.copy()
						ln = len(data)
						if ln > 0 and self.consumer is not None:
							self.consumer(self, ln, data)
						else:
							self.status = 'purge'
							# pass control to the custom terminate handler
							# self.terminate()
							break

					time.sleep(self.idletime)
					
				if self.status == 'purge':
					self.consumer = None
					print("[info] (StreamClient.run) : Video stream purged")
					break

			except UserWarning:
				print("[error] stream failed: ", sys.exc_info()[0])
				self.status = 'failed'
				self.close()
				break
		
		self.status = "init"
		print("Stream socket stopped")
	
	def set_consumer(self, consumer):
		self.consumer = consumer
	
	def close(self):
		print("Closing data stream..")
		while self.status != "init":
			time.sleep(1)
		print("Stream socket closed")
	
	def purge_negotiate(self):
		self.status = 'purge'

