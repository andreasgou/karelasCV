# ipwebcam.py

# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3

import urllib
import requests
import cv2
import numpy as np
import time
from PIL import Image

from utils import helper_sockets as socs

# Socket plugin functions
# ---------------
class VideoProcessor:
	
	def __init__(self, winname, winsize):
		self.winname = winname
		self.winsize = winsize
		self.plugins = []
		self.action = None
		self.action_args = None
		self.double_monitor = False
		self.last_frame = None
		self.histogram = None
	
	
	def init_video_monitor(self):
		# construct empty image
		blank_image = np.zeros((self.winsize[1], self.winsize[0], 3), np.uint8)
		self.putText(blank_image, "Waiting for video stream.. Press 'q' to quit")
		
		# Initialize openCV image window
		cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
		cv2.moveWindow(self.winname, 0, 0)
		self.imgshow(blank_image)
		
		cv2.waitKey(1)
	
	
	def use(self, dialog, length, stream):
		# image = Image.open(stream)
		# convert to numpy array and flip channels R-B or B-R
		# self.last_frame = iv.Pil2Numpy(image, 3)
		self.last_frame = stream
		# print("stream length: {}".format(length))
		img = self.last_frame
		
		# persistent processors
		for plugin in self.plugins:
			img = plugin[0](self, img, plugin[1])
		
		if self.action:
			self.action(self, img, self.action_args)
			self.action = None
			self.action_args = None
		
		# resize image
		img = cv2.resize(img, self.winsize, interpolation=cv2.INTER_CUBIC)
		
		self.imgshow(img)
	
	def show_hist(self, type='curve'):
		wname = "Histogram"
		if self.histogram is None and type != "off":
			cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
			cv2.moveWindow(wname, 0, 0)
			self.histogram = type
			print("[info] histogram enabled: {}".format(type))
		elif self.histogram is not None and type == "off":
			self.histogram = None
			cv2.destroyWindow(wname)
			print("[info] histogram disabled")
		elif type != "off":
			self.histogram = type
			print("[info] histogram updated: {}".format(type))
	
	def append_plugin(self, handler, args):
		plugin = self.get_plugin(handler)
		if plugin is None:
			self.plugins.append([handler, args])
			print("[info] plugin created: {}".format(args))
		else:
			plugin[1] = args
			print("[info] plugin updated: {}".format(args))
		
		# open plugin monitor window
		if "monitor" in args:
			wname = handler.__name__
			cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
			cv2.moveWindow(wname, self.winsize[0], 0)
	
	
	def set_action(self, handler, args):
		self.action = handler
		self.action_args = args
	
	def imgshow(self, img):
		# OpenCV gui solution
		cv2.imshow(self.winname, img)
		if self.histogram is not None:
			wname = "Histogram"
			hist = hf.get_histogram_image(img, self.histogram)
			cv2.imshow(wname, hist)
	
	def set_plugin_param(self, level):
		# Not yet implemented
		plugin = self.get_plugin(hdr_contours)
		pname = plugin[0].__name__
		if plugin[0] is not None and pname == "plugin_contours":
			# plugin[1] = level
			print("Plugin is running: {}".format(plugin[1]))
	
	def get_plugin(self, handler):
		retval = None
		for plugin in self.plugins:
			if plugin[0] == handler:
				retval = plugin
				break
		return retval
	
	def putText(self, img, text, cord=(10, 100), size=0.4, color=(200, 255, 0)):
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, text, cord, font, size, color, 1, cv2.LINE_AA)

# Init socket and open video stream
def video_start(pihost, piport, pipath, hw_quit):
	global video_monitor, video_so, winname, winsize
	
	# Create a video monitor instance
	video_monitor = VideoProcessor(winname, winsize)
	
	# initialize video stream and player
	video_so = socs.StreamHttpClient(pihost, piport, pipath, hw_quit)
	video_so.set_consumer(video_monitor.use)
	
	# print("[info] Initializing video stream. Socket timeout: {}".format(video_so.socket.gettimeout()))
	print("[info] Initializing video stream. Socket timeout: {}".format(video_so.timeout))
	if video_so.init_socket(True) == 'failed':
		print("[error]: Unable to create video binary stream!")
		video_so.close()
		return False
	else:
		print("[info]: Video stream created. Fetching data..")
		video_monitor.init_video_monitor()
		
		# send and wait for the reply
		# hostso.send("start", wait=True)
		return True

def hw_quit():
	global force_quit
	force_quit = True
	print("Press any key to quit!")

# Replace the URL with your own IPwebcam shot.jpg IP:port
# url='http://192.168.2.22:8080/'
rmhost = "192.168.2.105"
rmport = 5503
rmpath = "/shot.jpg"
timeout = 10

winname = "Camera-feed"
winsize = (720, 405)
force_quit = False
video_so = None
video_monitor = None

connected = video_start(rmhost, rmport, rmpath, hw_quit)

while connected:
	# Use urllib to get the image from the IP camera
	# imgResp = urllib.urlopen(url)
	
	# print("Trying to open {}\nTimeout is set to {}".format(url, timeout))
	# imgResp = requests.get(url, timeout=timeout)
	
	# Numpy to convert into a array
	# imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	# imgNp = np.array(bytearray(imgResp.content),dtype=np.uint8)
	
	# Finally decode the array to OpenCV usable format ;)
	# img = cv2.imdecode(imgNp,-1)
	
	
	# put the image on screen
	# cv2.imshow('IPWebcam',img)
	
	#To give the processor some less stress
	#time.sleep(0.1)
	
	# Quit if q is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break