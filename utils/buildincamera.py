import cv2
import time
from threading import *


class BuildinCamera(Thread):
	
	def __init__(self):
		Thread.__init__(self)
		self.camera = None
		self.conf = None
		self.stream = None
		self.consumer = None
		self.log = None
		self.plugins = []
		self.socket = None
		self.setfname = None
		self.state = 'init'
		self.video_port = False
		self.video_path = None
		self.frame_no = 0
		self.streamLength = 0
		self.stm = time.time()
		
		self.conf = {
			'resolution'   : (1024, 768),
			'framerate'    : 5,
			'video_port'   : False,
			'exposure_mode': 'auto',
			'awb_mode'     : 'auto',
			'iso'          : 0,
			'shutter_speed': 0,
			'zoom'         : (0.0, 0.0, 1.0, 1.0)
		}
	
	def init_socket(self, video_path=None):
		self.video_path = video_path
		ret = self.init_camera(self)
		self.start()
		return ret
	
	def init_camera(self, conf=None, stream=None, fname="undefined.json"):
		if self.state == 'init':
			if conf:
				self.conf = conf
			
			self.setfname = self.conf['file_name'] if 'file_name' in conf else fname
			if self.video_path is None:
				self.camera = cv2.VideoCapture(0)
			else:
				self.camera = cv2.VideoCapture(self.video_path)
			# set default camera properties
			# if 'video_port' in conf:
			#     self.video_port = self.conf['video_port']
			self.stream = stream
			self.state = "idle"
		else:
			print("Buildin camera already initialized. Current state is '{}'".format(self.state))
		
		# self.log = get_camera_settings(self)
		# print(self.log)
		
		return self.state
	
	def close(self):
		self.close_camera()
	
	def close_camera(self):
		if self.state == 'init':
			return
		if self.state == 'streaming':
			self.state = 'closing'
			i=0
			while self.state != 'closed':
				i += 1
				print("Finishing current capture{}".format('.' * i), end='\r')
				time.sleep(1)
		self.stream.flush()
		self.camera.release()
	# self.stream = None
	
	def set_consumer(self, handler):
		self.consumer = handler
	
	def set_stream(self, stream):
		self.stream = stream
	
	def run(self):
		self.state = "streaming"
		self.stm = time.time()
		self.frame_no = 0
		
		while True:
			if self.state == 'streaming':
				# for foo in self.camera.capture_continuous(stream, 'jpeg', use_video_port=self.video_port):
				ret, frame = self.camera.read()
				self.frame_no += 1
				self.streamLength = len(frame)
				
				self.stream.write(frame)
				self.stream.flush()
				
				# persistent processors
				for plugin in self.plugins:
					plugin(self)
				
				if self.streamLength > 0 and self.consumer is not None:
					self.consumer(self, self.streamLength, frame)
				
				self.stm = time.time()
				
				if self.state in ['pause', 'closing']:
					break
		
		self.state = "closed"
		print("PiCamera thread stopped!")
