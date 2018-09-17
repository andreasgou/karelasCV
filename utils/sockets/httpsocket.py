import sys
from threading import *
import requests
import numpy as np
import cv2
import time
import json

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
		self.pause = False
		self.pause_time = 0.0
	
	def init_socket(self):
		try:
			(ln, data, self.cache) = self.read_jpg_stream(self.url, self.cache)
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
			if self.pause:
				continue
			
			try:
				(ln, data, self.cache) = self.read_jpg_stream(self.url, self.cache)
				if self.status == 'purge':
					# consume all data from the pipe
					# self.pipe.flush()
					print("[info] (StreamClient.run) : Video stream purged")
					break
				
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
		print("Closing data stream..")
		while self.status != "init":
			time.sleep(1)
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
			print("[error] (read_jpg_stream) : connection error: ", sys.exc_info()[0])
			return 0, None, None
		except:
			print("[error] (read_jpg_stream) : stream failed: ", sys.exc_info())
			return 0, None, None
	
	def set_command(self, cmd, args):
		if cmd == "get":
			url = "http://" + self.host + ":" + str(self.port) + "/status.json?show_avail=0"
			res = requests.get(url)
			obj = json.loads(res.content)
			if len(args) > 1:
				field = args[1]
				obj = obj["curvals"][field]
			else:
				field = "all"
				obj = obj["curvals"]
			print("Current setting for {}: {}".format(field, obj))

		elif cmd == "set":
			if len(args) < 3:
				url = "http://" + self.host + ":" + str(self.port) + "/status.json?show_avail=1"
				res = requests.get(url)
				obj = json.loads(res.content)
				if len(args) == 1:
					print("Available fields:\n{}".format([s for s in obj["avail"].keys()]))
				else:
					field = args[1]
					print("Available values for {}: {}".format(field, obj["avail"][field]))
				return

			field = args[1]
			value = args[2]
			if field == "focus":
				value = "nofocus" if value == "off" else "focus"
				url = "http://{}:{}/{}".format(self.host, str(self.port), value)
			else:
				url = "http://{}:{}/settings/{}?set={}".format(self.host, str(self.port), field, value)
			res = requests.get(url)
			print(res.content)
	
		

