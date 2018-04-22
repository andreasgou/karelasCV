import os
import cv2
import datetime
import numpy as np
import argparse
import imutils as iu
from imutils import contours
from PIL import Image
# from PIL import ImageTk

from utils import helper_sockets as socs
from utils import helper_visuals as iv
from utils import cvhist as hf

# Thread free for UI
# import matplotlib
# matplotlib.use('TkAgg')

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
		self.putText(blank_image, "waiting for video stream..")
	
		# Initialize openCV image window
		cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
		cv2.moveWindow(self.winname, 0, 0)
		self.imgshow(blank_image)
		
		cv2.waitKey(1)
	

	def use(self, dialog, length, stream):
		image = Image.open(stream)
		# convert to numpy array and flip channels R-B or B-R
		self.last_frame = iv.Pil2Numpy(image, 3)
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
		
def hw_quit():
	global force_quit
	force_quit = True
	print("Press any key to quit!")

# Actions and plugins for VideoProcessor
# --------------------------------------
def action_grab(proc, img, args):
	pic_name = "grab {}.png".format(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))
	cv2.imwrite('pics/' + pic_name, img)
	print("\npicture saved as 'pics/{}'\n> ".format(pic_name), end='')
	return True

def action_blocks(proc, img, args):
	tm = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
	patches = iv.blocks(img, (80, 80, 3))
	cnt = 0
	for p in patches[:]:
		cnt += 1
		pic_name = "patch {} {:04}.png".format(tm, cnt)
		cv2.imwrite('pics/' + pic_name, p)
	print("\n{} image patches saved in 'pics/{}'\n> ".format(cnt, pic_name), end='')
	return True

def action_windows(proc, img, args):
	if len(args) < 4:
		print("\nWrong number of parameters\n> ", end='')
		return False

	dm = int(args[1])
	ds = int(args[2])
	pref = args[3]

	tm = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
	patches = iv.windows(img, (dm, dm, 3), ds)
	cnt = 0
	os.makedirs("pics/%s" % tm)
	for p in patches[:]:
		cnt += 1
		pic_name = "{}/{}_{:04}.png".format(tm, pref, cnt)
		cv2.imwrite('pics/' + pic_name, p)
		print('.', end='')
	print("\n{} image patches saved in 'pics/{}/{}_*.png'\n> ".format(cnt, tm, pref), end='')
	return True

def action_gui():
	pass

def plugin_grid(proc, img, args):
	return iv.gridlines(img, 16, 1, np.array([0, 0, 0], dtype="uint8"))

def plugin_checker(proc, img, args):
	return iv.checkerboard(img, 64, np.array([0, 0, 0], dtype="uint8"), np.array([100, 100, 100], dtype="uint8"))

# Apply a filter on the image using convolution with a filter K
def plugin_filter(proc, args):
	global kernelBank

	hdr = kernelBank[args[1]]
	hdr_type = type(hdr).__name__

	if hdr_type == 'function':
		if len(args) > 2 and args[2] == 'remove':
			plugin = proc.get_plugin(hdr)
			if plugin is not None:
				if 'monitor' in plugin[1]:
					cv2.destroyWindow(plugin[0].__name__)
				proc.plugins.remove(plugin)
				print("[info] Plugin {} removed".format(plugin[1]))
		else:
			proc.append_plugin(hdr, args)
			print("[info] Plugin {} enabled".format(hdr_type))
	elif hdr_type == 'str':
		if hdr == 'remove':
			for plugin in proc.plugins:
				if 'monitor' in plugin[1]:
					cv2.destroyWindow(plugin[0].__name__)
			proc.plugins = []
			print("[info] All plugins removed")
		elif hdr == 'list':
			for plugin in proc.plugins:
				print(plugin[1])

# construct average blurring kernels used to smooth an image
def hdr_smallBlur(proc, img, args):
	smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
	# custom convolution
	# return = iv.convolve(img, smallBlur)
	return cv2.filter2D(img, -1, smallBlur)

def hdr_largeBlur(proc, img, args):
	largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
	return cv2.filter2D(img, -1, largeBlur)

# construct a sharpening filter
def hdr_sharpen(proc, img, args):
	sharpen = np.array((
		[0, -1, 0],
		[-1, 5, -1],
		[0, -1, 0]), dtype="int")
	return cv2.filter2D(img, -1, sharpen)

# construct the Laplacian kernel used to detect edge-like regions of an image
def hdr_laplacian(proc, img, args):
	laplacian = np.array((
		[0, 1, 0],
		[1, -4, 1],
		[0, 1, 0]), dtype="int")
	return cv2.filter2D(img, -1, laplacian)

# construct the Sobel x-axis kernel
def hdr_sobelX(proc, img, args):
	sobelX = np.array((
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]), dtype="int")
	return cv2.filter2D(img, -1, sobelX)

# construct the Sobel y-axis kernel
def hdr_sobelY(proc, img, args):
	sobelY = np.array((
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]), dtype="int")
	return cv2.filter2D(img, -1, sobelY)

# construct an emboss kernel
def hdr_emboss(proc, img, args):
	emboss = np.array((
		[-2, -1, 0],
		[-1, 1, 1],
		[0, 1, 2]), dtype="int")
	return cv2.filter2D(img, -1, emboss)

def hdr_sobel(proc, img, args):
	ks = 1 if len(args) < 3 else int(args[2])
	# Calculate gradient
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ks)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ks)
	# Calculate gradient magnitude and direction ( in degrees )
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	gx = np.uint8(np.absolute(gx))
	gy = np.uint8(np.absolute(gy))
	out = cv2.bitwise_or(gx, gy)
	return out

def hdr_threshold(proc, img, args):
	# check if input is thresholded
	if len(img.shape) == 2:
		print("[warning] Adaptive threshold cannot be applied to B&W images. Reset filters and reapply.")
		return img

	# check if input is in grayscale
	if (img.shape[2] == 3):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
		
	threshold_type = 'normal'
	if len(args) > 2:
		threshold_type = args[2]

	if threshold_type == 'normal':
		thresh = 0 if len(args) < 4 else int(args[3])
		out = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
	elif threshold_type == 'adapt-mean':
		neib = 11 if len(args) < 4 else int(args[3])
		out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, neib, 2)
	elif threshold_type == 'adapt-gauss':
		neib = 11 if len(args) < 4 else int(args[3])
		out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neib, 2)
	elif threshold_type == 'otsu':
		thresh = 0 if len(args) < 4 else int(args[3])
		out = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	else:
		plugin_filter(proc, ['filter', 'threshold'])
		print("[warning] Valid thresholds: (normal | otsu) [thresh] | (adapt-mean | adapt-gauss) [neighbors]")
		print("     thresh: 0-255 default 0")
		print("  neighbors: 3, 5, 7, 9,.. (odd number)")
		out = img

	return out

def hdr_canny(proc, img, args):
	# get min-max thresholds
	min = 100 if len(args) < 3 else int(args[2])
	max = 200 if len(args) < 4 else int(args[3])
	return cv2.Canny(img, min, max)

def hdr_equalizer(proc, img, args):
	# check if input is in grayscale
	if (img.shape[2] == 3):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	return cv2.equalizeHist(gray)

def hdr_contours(proc, img, args):
	# gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

	# check if input is not binarized
	if len(img.shape) > 2:
		# threshold the image to reveal the digits
		# thresh = hdr_threshold(proc, img, args)
		thresh = hdr_canny(proc, img, [args[0], args[1]])
	else:
		thresh = img

	# find contours in the image, keeping only the four largest ones
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if iu.is_cv2() else cnts[1]
	# keep those with specific ratio w:h
	if len(args) == 5:
		ratio_low = float(args[3])
		ratio_high = float(args[4])
		cn = []
		for c in cnts:
			# compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(c)
			ratio = round(abs(w**2 - h**2)/(w*h), 2)
			if ratio_low <= ratio <= ratio_high:
				cn.append(c)
		cnts = cn
	
	# sort by size
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

	# short by position from left to right (throws an error sometimes)
	# cnts = contours.sort_contours(cnts)[0]
	# output = cv2.merge([thresh] * 3)

	# set the image to be overlayed to the original frame, bypassing filters
	output = proc.last_frame

	# loop over the contours
	for c in cnts:
		# compute the bounding box for the contour
		(x, y, w, h) = cv2.boundingRect(c)
		roi = output[y - 5:y + h + 5, x - 5:x + w + 5]
	
		# pre-process the ROI and classify it
		# roi = preprocess(roi, 28, 28)
		# roi = np.expand_dims(img_to_array(roi), axis=0)
		# pred = model.predict(roi).argmax(axis=1)[0] + 1
		# predictions.append(str(pred))
	
		# draw the prediction on the output image
		cv2.rectangle(output, (x - 2, y - 2),
					  (x + w + 2, y + h + 2), (0, 255, 0), 1)

		ratio = round(abs((w**2 - h**2)/(w*h)), 2)
		proc.putText(output, str(ratio), cord=(x-2, y-2), color=(0, 100, 255))


	if "monitor" in args:
		wn = "plugin_contours"
		# img_out = cv2.resize(output, proc.winsize, interpolation=cv2.INTER_CUBIC)
		# img_out = output
		cv2.createTrackbar("slider", wn, 3, 7, proc.set_plugin_param )
		# cv2.imshow(wn, img)

	return output

# App console commands
# --------------------
def video_start(pihost, piport, hw_quit, args):
	global video_monitor, video_so, winname, winsize

	# Create a video monitor instance
	video_monitor = VideoProcessor(winname, winsize)

	# initialize video stream and player
	video_so = socs.StreamClient(pihost, piport, hw_quit)
	video_so.set_consumer(video_monitor.use)

	if video_so.init_socket(True) == 'failed':
		print("[error]: Unable to create video binary stream!")
		video_so.close()
	else:
		video_monitor.init_video_monitor()
		
		# send and wait for the reply
		hostso.send("start", wait=True)


def video_stop():
	global video_so, video_monitor
	
	if video_so:
		video_so.status = 'purge'
		video_so.set_consumer(None)
		
		# send and wait for the reply
		hostso.send("stop", wait=True)
		video_so.status = 'negotiate'
		hostso.send("close-stream-listener", wait=True)

		video_so.close()
		cv2.destroyAllWindows()
		print("Video thread is {}alive".format("" if video_so.is_alive() else "not "))
		video_so = None
		# Tkinter gui solution (not applicable: creates video lag)
		# video_monitor.root.quit()

def video_histogram(vm, args):
	ca = len(args)
	if ca > 1:
		vm.show_hist(type=args[1])
	else:
		vm.show_hist()
		

# construct the kernel bank, a list of functions applying kernels or filters
kernelBank = {
	"blur"      : hdr_smallBlur,
	"blur-more" : hdr_largeBlur,
	"sharpen"   : hdr_sharpen,
	"laplacian" : hdr_laplacian,
	"sobel-x"   : hdr_sobelX,
	"sobel-y"   : hdr_sobelY,
	"emboss"    : hdr_emboss,
	"sobel"     : hdr_sobel,
	"threshold" : hdr_threshold,
	"canny"     : hdr_canny,
	"equalizer" : hdr_equalizer,
	"contours"  : hdr_contours,
	"remove"    : "remove",
	"list"      : "list"
}


# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-host", "--host", type=str, required=True,
				help = "host address")
ap.add_argument("-p", "--port", type=int, required=True,
				help = "port number")
ap.add_argument("-s", "--shutter", type=int, required=False,
				help = "shutter speed")
args = ap.parse_args()

# Create global variables and classes
# -----------------------------------
pihost = args.host
piport = args.port

winname = "Camera-feed"
winsize = (640, 480)
force_quit = False
video_so = None
video_monitor = None

# Create and configure camera module instance

# ----------------
# ---   MAIN   ---
# ----------------

# Connect to Pi host asynchronously
hostso = socs.DialogClient(pihost, piport, hw_quit)
if hostso.init_socket(True) == 'failed':
	print("Quiting program!")
	quit()

# Enter interactive console mode.
# Type 'quit' to exit
q = ""
while q.lower() not in {'quit'}:
	q = input("> ")
	if force_quit:
		quit()
	if q.rstrip() != '':
		qr = q.rstrip().split(' ')
		q = ' '.join(str(x) for x in qr)
		
		# video states
		if qr[0] == 'quit':
			video_stop()
			# send and wait for the reply
			hostso.send(q, wait=True)

		elif qr[0] == 'start':
			video_start(pihost, piport, hw_quit, qr)
		elif qr[0] == 'stop':
			video_stop()
		elif qr[0] == 'histogram':
			video_histogram(video_monitor, qr)

		# Plugins (one condition)
		elif qr[0] == 'grid':
			video_monitor.append_plugin(plugin_grid, qr)
			print("Plugin {} enabled".format(q))

		elif qr[0] == 'checker':
			video_monitor.append_plugin(plugin_checker, qr)
			print("Plugin {} enabled".format(q))

		elif qr[0] == 'filter':
			if len(qr) < 2:
				print("[error]: Filter not defined\nUsage: filter {}"
					  .format("|".join(key for key in kernelBank)))
			elif qr[1] not in kernelBank:
				print("[error]: Invalid filter {}\nUsage: filter {}> "
					  .format(qr[1], "|".join(key for key in kernelBank)))
			else:
				plugin_filter(video_monitor, qr)

		# Actions (two words)
		elif qr[0] == 'grab':
			video_monitor.set_action(action_grab)
			print("Action {} activated".format(q))
		elif qr[0] == 'blocks':
			video_monitor.set_action(action_blocks)
			print("Action {} activated".format(q))
		elif qr[0] == 'windows':
			video_monitor.set_action(action_windows, qr)
			print("Action {} activated".format(q))
		elif qr[0] == 'gui':
			action_gui()
			print("Action {} activated".format(q))

		# Anything else just send it over
		else:
			# send and wait for the reply
			hostso.send(q, wait=True)


# Close connection
hostso.close()
print("Bye..")
