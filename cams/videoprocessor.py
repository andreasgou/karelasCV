# Video Processor interface
# -------------------------

import sys, traceback
import numpy as np
import cv2
import time
import datetime
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
		self.action_time = None
		self.double_monitor = False
		self.last_frame = None
		self.save_frame = None
		self.pause_frame = False
		self.histogram = None
		self.camtype = 0
		self.key = None
		self.socket = None
		self.pause_start = None
		
		self.refPt = []
		self.cropping = False
		self.selection = False
		self.tracking = False
		self.tracker = None
		self.movePt = []
		self.move = False
	
	def init_video_monitor(self):
		# construct empty image
		blank_image = np.zeros((self.winsize[0], self.winsize[1], 3), np.uint8)
		self.putText(blank_image, "waiting for video stream..")
		
		# Initialize openCV image window
		cv2.namedWindow(self.winname, cv2.WINDOW_AUTOSIZE)
		cv2.moveWindow(self.winname, 0, 0)
		cv2.setMouseCallback(self.winname, self.mouse_control)
		self.imgshow(blank_image)
		
		cv2.waitKey(1)

	def pause(self):
		self.pause_frame = True
		self.socket.pause = True
		self.pause_start = time.time()
		
	def unpause(self):
		self.pause_frame = False
		self.socket.pause = False
		self.socket.pause_time = self.socket.pause_time + time.time() - self.pause_start
	
	def use(self, dialog, length, stream):
		if self.pause_frame:
			return
		
		if self.camtype == 1:
			self.last_frame = stream
		elif self.camtype >= 2:
			self.last_frame = stream
		else:
			image = Image.open(stream)
			# convert to numpy array and flip channels R-B or B-R
			self.last_frame = iv.Pil2Numpy(image, 3)
		
		img = self.last_frame.copy()
		self.save_frame = img
		
		# persistent processors
		for plugin in self.plugins:
			try:
				img = plugin[0](self, img, plugin)
				self.save_frame = img
			except:
				exc_type, exc_value, exc_traceback = sys.exc_info()
				print("[error] ", exc_type, exc_value)
				traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
				print("\n[warning] Plugin {} is not functional, trying to remove..".format(plugin[1]))
				self.remove_plugin(plugin[0])
		
		if self.action:
			itr = 0
			try:
				dt = datetime.datetime.now()
				if (dt - self.action_time).total_seconds() > 0:
					print("\r[info] running action: {}\n> ".format(self.action_args), end='')
					self.action(self, img, self.action_args)
					if "repeat" in self.action_args:
						if "idle" in self.action_args:
							idx = self.action_args.index("idle") + 1
							idle = int(self.action_args[idx])
							self.action_time += datetime.timedelta(seconds=idle)
						idx = self.action_args.index("repeat") + 1
						itr = int(self.action_args[idx]) - 1
					if itr <= 0:
						self.action = None
						self.action_args = None
					else:
						self.action_args[idx] = itr
			except:
				print("[error]: ", sys.exc_info()[0], sys.exc_info()[1])
		
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
	
	def get_plugin(self, handler):
		retval = None
		for plugin in self.plugins:
			if plugin[0] == handler:
				retval = plugin
				break
		return retval
	
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
			
	def remove_plugin(self, handler):
		plugin = self.get_plugin(handler)
		if plugin is not None:
			if 'monitor' in plugin[1]:
				cv2.destroyWindow(plugin[0].__name__)
			self.plugins.remove(plugin)
			if plugin[0].__name__ in ["hdr_tracker_cmt", "hdr_tracker"]:
				self.tracking = False
				self.tracker = None
				self.selection = len(self.refPt) > 0
			print("[info] Plugin {} removed".format(plugin[1]))
	
	def remove_all_plugins(self):
		for plugin in self.plugins:
			if 'monitor' in plugin[1]:
				cv2.destroyWindow(plugin[0].__name__)
			if plugin[0].__name__ in ["hdr_tracker_cmt", "hdr_tracker"]:
				self.tracking = False
				self.tracker = None
				self.selection = len(self.refPt) > 0
		self.plugins = []
		print("[info] All plugins removed")
		return
	
	def set_action(self, handler, args):
		if self.action is None:
			self.action = handler
			self.action_args = args
			self.action_time = datetime.datetime.now()
			print("[info] action {} activated.".format(self.action_args))
		elif handler is None:
			self.action = None
			self.action_args = None
		else:
			print("[warning] running action in progress: {}. New action was not activated!".format(self.action_args))
	
	def imgshow(self, img):
		# resize image if required
		imshape = img.shape[:2]
		if self.winsize != imshape:
			img = cv2.resize(img, (self.winsize[1], self.winsize[0]), interpolation=cv2.INTER_CUBIC)
		
		if self.selection:
			c = self.adjust_coords(imshape, self.refPt)
			# cv2.rectangle(img, self.refPt[0], self.refPt[1], (255, 0, 0), 2)
			cv2.rectangle(img, c[0], c[1], (255, 0, 0), 2)
		elif self.tracking and self.tracker.has_result:
			c = self.adjust_coords(imshape, [self.tracker.tl, self.tracker.br])
			# cv2.rectangle(img, self.tracker.tl, self.tracker.br, (0, 255, 0), 1)
			cv2.rectangle(img, c[0], c[1], (0, 255, 0), 1)
		
		cv2.imshow(self.winname, img)
		if self.histogram is not None:
			hist = hf.get_histogram_image(img, self.histogram)
			cv2.imshow("Histogram", hist)
			
	def mouse_control(self, event, x, y, flags, param):
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			# if self.winsize != self.last_frame.shape[:2]:
			if self.winsize != self.save_frame.shape[:2]:
				print("\r[warning] cropping not allowed: preview image is resized!\n> ", end='')
				print("\r           source: ({}, {})\n> ".format(self.save_frame.shape[1], self.save_frame.shape[0]), end='')
				print("\r          preview: ({}, {})\n> ".format(self.winsize[1], self.winsize[0]), end='')
				return
		
			# Move selection
			if self.selection \
					and self.refPt[0][0] < x < self.refPt[1][0] \
					and self.refPt[0][1] < y < self.refPt[1][1]:

				# self.save_frame = self.last_frame.copy()
				self.movePt = [x, y]
				self.move = True
				# self.pause_frame = True
				self.pause()
				print("\r[debug] move selection: {} x {}    ".format(x, y), end='')
				return
			
			# Initiate selection
			# self.save_frame = self.last_frame.copy()
			self.selection = False
			self.refPt = [(x, y)]
			self.cropping = True
			# self.pause_frame = True
			self.pause()
		
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# Selection not permitted if preview image is resized
			# if self.winsize != self.last_frame.shape[:2]:
			if self.winsize != self.save_frame.shape[:2]:
				return
			
			# Reset selection when clicked (mouse down-up at the same location)
			if self.refPt[0] == (x, y) \
					or (self.move and tuple(self.movePt) == (x, y)):
				self.selection = False
				self.cropping = False
				self.refPt = []
				self.move = False
				self.movePt = []
				# self.pause_frame = False
				self.unpause()
				print("\n[info] mask removed\n> ", end='')
				self.imgshow(self.last_frame)
				return
				
			if self.move:
				posX = x - self.movePt[0]
				posY = y - self.movePt[1]
				self.refPt[0] = (self.refPt[0][0] + posX, self.refPt[0][1] + posY)
				self.refPt[1] = (self.refPt[1][0] + posX, self.refPt[1][1] + posY)
				self.movePt = []
				self.move = False
				# self.pause_frame = False
				self.unpause()
				print("\r[info] mask selection moved to: {}x{}    \n> ".format(self.refPt[0], self.refPt[1]), end='')
				return
				
			# record the ending (x, y) coordinates
			tl = (min(self.refPt[0][0], x),
			      min(self.refPt[0][1], y))
			br = (max(self.refPt[0][0], x),
			      max(self.refPt[0][1], y))
			self.refPt = [tl, br]
			self.cropping = False
			self.selection = True
			# self.pause_frame = False
			self.unpause()
			
			# draw a rectangle around the region of interest
			print("\r[info] mask created: {}x{}          \n> ".format(br[0]-tl[0], br[1]-tl[1]), end='')
			img = self.save_frame.copy()
			self.imgshow(img)
		
			# key = cv2.waitKey(1) & 0xFF
		
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.cropping:
				tl = self.refPt[0]
				br = (x, y)
				print("\r[debug] mask dimensions: {}x{}    ".format(br[0]-tl[0], br[1]-tl[1]), end='')
				img = self.save_frame.copy()
				cv2.rectangle(img, tl, br, (255, 0, 0), 1)
				self.imgshow(img)
				return
			
			if self.move:
				posX = x - self.movePt[0]
				posY = y - self.movePt[1]
				tl = (self.refPt[0][0] + posX, self.refPt[0][1] + posY)
				br = (self.refPt[1][0] + posX, self.refPt[1][1] + posY)
				print("\r[debug] move selection to: {}{}".format(tl, br), end='')
				img = self.save_frame.copy()
				cv2.rectangle(img, tl, br, (255, 0, 0), 1)
				self.imgshow(img)
				return
		
		elif event == cv2.EVENT_RBUTTONDOWN:
			# Not in use
			if self.selection:
				tl = self.refPt[0]
				br = self.refPt[1]
				self.resize_monitor((br[0]-tl[0], br[1]-tl[1]))

				self.cropping = False
				self.selection = False
				# self.pause_frame = False
				self.unpause()
				self.move = False
				self.refPt = []
				print("\r[info] mask removed\n> ", end='')

	def putText(self, img, text, cord=(10, 100), size=0.4, color=(200, 255, 0)):
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, text, cord, font, size, color, 1, cv2.LINE_AA)

	def resize_monitor(self, dim):
		self.winsize = dim
		cv2.resizeWindow(self.winname, self.winsize[1], self.winsize[0])
		cv2.moveWindow(self.winname, 0, 0)
	
	def adjust_coords(self, imgsize, cords):
		tx = int((self.winsize[1] / imgsize[1]) * cords[0][0])
		bx = int((self.winsize[1] / imgsize[1]) * cords[1][0])
		ty = int((self.winsize[0] / imgsize[0]) * cords[0][1])
		by = int((self.winsize[0] / imgsize[0]) * cords[1][1])
		return [(tx, ty), (bx, by)]
