# Video Processor interface
# -------------------------

import sys
import numpy as np
import cv2
from PIL import Image

from utils import helper_visuals as iv
from utils import cvhist as hf


class VideoProcessor:
	
	def __init__(self, winname, winsize):
		self.winname = winname
		self.winsize = (winsize[1], winsize[0])
		self.plugins = []
		self.action = None
		self.action_args = None
		self.double_monitor = False
		self.last_frame = None
		self.save_frame = None
		self.histogram = None
		self.camtype = 0
		self.key = None
		
		self.refPt = []
		self.cropping = False
		self.selection = False
	
	def init_video_monitor(self):
		# construct empty image
		blank_image = np.zeros((self.winsize[0], self.winsize[1], 3), np.uint8)
		self.putText(blank_image, "waiting for video stream..")
		
		# Initialize openCV image window
		cv2.namedWindow(self.winname, cv2.WINDOW_AUTOSIZE)
		cv2.moveWindow(self.winname, 0, 0)
		cv2.setMouseCallback(self.winname, self.click_and_crop)
		self.imgshow(blank_image)
		
		cv2.waitKey(1)
	
	def use(self, dialog, length, stream):
		if self.cropping:
			return
		
		if self.camtype == 1:
			self.last_frame = stream
		elif self.camtype == 2:
			self.last_frame = stream
		else:
			image = Image.open(stream)
			# convert to numpy array and flip channels R-B or B-R
			self.last_frame = iv.Pil2Numpy(image, 3)
		
		img = self.last_frame.copy()
		
		# persistent processors
		for plugin in self.plugins:
			img = plugin[0](self, img, plugin[1])
		
		if self.action:
			itr = 0
			try:
				self.action(self, img, self.action_args)
				if "repeat" in self.action_args:
					idx = self.action_args.index("repeat")
					itr = int(self.action_args[idx+1]) - 1
				if itr <= 0:
					self.action = None
					self.action_args = None
				else:
					self.action_args[idx+1] = itr
			except:
				print("[error]: ", sys.exc_info()[0], sys.exc_info()[1])
		
		# resize image if required
		if self.winsize != img.shape[:2]:
			img = cv2.resize(img, (self.winsize[1], self.winsize[0]), interpolation=cv2.INTER_CUBIC)
		
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
	
	def append_plugin(self, handler, param):
		plugin = self.get_plugin(handler)
		if plugin is None:
			self.plugins.append([handler, param])
			print("[info] plugin created: {}".format(param))
		else:
			plugin[1] = param
			print("[info] plugin updated: {}".format(param))
		
		# open plugin monitor window
		if "monitor" in param:
			wname = handler.__name__
			cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
			cv2.moveWindow(wname, self.winsize[1], 0)
	
	def set_action(self, handler, args):
		self.action = handler
		self.action_args = args
	
	def imgshow(self, img):
		# OpenCV gui solution
		if self.selection:
			cv2.rectangle(img, self.refPt[0], self.refPt[1], (255, 0, 0), 2)
		
		cv2.imshow(self.winname, img)
		if self.histogram is not None:
			wname = "Histogram"
			hist = hf.get_histogram_image(img, self.histogram)
			cv2.imshow(wname, hist)
			
	def click_and_crop(self, event, x, y, flags, param):
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			if self.winsize != self.last_frame.shape[:2]:
				print("\r[warning] cropping not allowed: preview image is resized!\n> ", end='')
				print("\r           source: ({}, {})\n> ".format(self.last_frame.shape[1], self.last_frame.shape[0]), end='')
				print("\r          preview: ({}, {})\n> ".format(self.winsize[1], self.winsize[0]), end='')
				return
			
			self.selection = False
			self.refPt = [(x, y)]
			self.cropping = True
		
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			if self.winsize != self.last_frame.shape[:2]:
				return
			# Reset selection when clicked into
			if self.refPt[0] == (x, y):
				self.cropping = False
				self.selection = False
				self.refPt = []
				print("\r[info] mask removed\n> ", end='')
				self.imgshow(self.last_frame)
				return
			
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			self.refPt.append((x, y))
			self.cropping = False
			self.selection = True
			
			# draw a rectangle around the region of interest
			print("\r[info] mask created: {} x {}          \n> ".format(x-self.refPt[0][0], y-self.refPt[0][1]), end='')
			img = self.last_frame.copy()
			self.imgshow(img)
		
			# key = cv2.waitKey(1) & 0xFF
		
		elif event == cv2.EVENT_MOUSEMOVE:
			if not self.cropping:
				return
			print("\r[debug] mask dimensions: {} x {}    ".format(x-self.refPt[0][0], y-self.refPt[0][1]), end='')
			img = self.last_frame.copy()
			cv2.rectangle(img, self.refPt[0], (x, y), (255, 0, 0), 1)
			self.imgshow(img)
	
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

