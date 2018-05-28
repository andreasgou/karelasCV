import os
import cv2
import datetime
import numpy as np
import argparse
import imutils as iu
import json

from cams import VideoProcessor
from utils.sockets.dialogclientsocket import DialogClient
from utils.sockets.streamclientsocket import StreamClient
from utils.sockets.httpsocket import StreamHttpClient
from utils.sockets.filelistsocket import StreamFileListClient

from utils import helper_visuals as iv

# Client socket for streaming from DialogServer
# ---------------------------------------------
# class StreamClient(Thread):
#
# 	def __init__(self, host, port, hdl_terminate):
# 		Thread.__init__(self)
# 		self.status = "init"
# 		self.pipe = None
# 		self.consumer = None
# 		self.host = host
# 		self.port = port
# 		self.socket = socket.socket()
# 		# self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# 		self.socket.settimeout(60)
# 		self.terminate = hdl_terminate
#
# 	def init_socket(self, confirm):
# 		try:
# 			self.socket.connect((self.host, self.port))
# 			if confirm:
# 				self.status = 'negotiate'
# 				data = netCatch(self.socket)
# 				if data:
# 					netThrow(self.socket, "stream-listener")
# 			# Make a file-like object
# 			self.pipe = self.socket.makefile('rb')
# 			self.start()
# 		except (TimeoutError, socket.timeout) as STO:
# 			self.status = 'failed'
# 			print("[error]:", STO)
# 		except ConnectionRefusedError as SRE:
# 			self.status = 'failed'
# 			print("[error]:", SRE)
# 		except:
# 			self.status = 'failed'
# 			print("[error]: protocol failed: ", sys.exc_info()[0])
#
# 		return self.status
#
# 	def run(self):
# 		while True:
# 			(ln, data) = binCatch(self.pipe)
# 			if self.status == 'purge':
# 				# consume all data from the pipe
# 				self.pipe.flush()
# 				print("[info] (StreamClient.run) : Video stream purged")
# 				continue
#
# 			# print("  ", self.status, ln, data)
# 			if data:
# 				if self.status == 'negotiate':
# 					data = data.getvalue().decode()
# 					if data == '--EOS--':
# 						self.status = "running"
# 					elif data == 'quit':
# 						break
# 					else:
# 						print(" ", data)
# 				else:
# 					self.consumer(self, ln, data)
# 			else:
# 				self.close()
# 				# pass control to the custom terminate handler
# 				# self.terminate()
# 				break
#
# 		self.status = "init"
# 		print("Stream socket stopped")
#
# 	def set_consumer(self, consumer):
# 		self.consumer = consumer
#
# 	def close(self):
# 		if self.status != 'failed':
# 			self.status == 'purge'
# 			self.pipe.flush()
# 			self.pipe.close()
# 			self.socket.close()
# 			self.status = "init"
# 		print("Stream socket closed")
#
# 	def purge_negotiate(self):
# 		self.status = 'purge'


# Thread free for UI
# import matplotlib
# matplotlib.use('TkAgg')

# Socket plugin functions
# ---------------
def hw_quit():
	global force_quit
	force_quit = True
	print("Press any key to quit!")

# Actions and plugins for VideoProcessor
# --------------------------------------
# Gab image and save locally to disk
# If a selection mask is active, save the selection area only
def action_grab(proc, img, args):
	dt = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
	pic_name = "raw/grab_{}.jpg".format(dt)
	if len(args) > 1:
		pic_name = "{}_{}.jpg".format(args[1], dt)
	if proc.selection:
		# compute the bounding box for the contour
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		imout = img[ty:by, tx:bx].copy()
	else:
		imout = img.copy()

	cv2.imwrite(pic_name, imout)
	print("\npicture saved as '{}'\n> ".format(pic_name), end='')
	return True

# Slice image into blocks and save
def action_blocks(proc, img, args):
	if len(args) < 4 or \
			img.shape[1] % int(args[1]) != 0 or \
			img.shape[0] % int(args[2]) != 0:

		rows = []
		cols = []
		for i in range(int(img.shape[0]/2), 1, -1):
			if img.shape[0] % i == 0:
				rows.append(i)
		for i in range(int(img.shape[1]/2), 1, -1):
			if img.shape[1] % i == 0:
				cols.append(i)

		print("Usage: blocks <width> <height> <path>")
		print("Valid block dimensions are:\n  ROWS = {}\n  Block Height = {}\n\n  COLS = {}\n  Block Width = {}"
			  .format(rows, img.shape[0]//np.array(rows), cols, img.shape[1]//np.array(cols)))

		return False

	w = int(args[1])
	h = int(args[2])
	path_prefix = args[3]
	# pic_name = args[1]
	# tm = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
	print("[info] Slicing image {}x{} into blocks of size {}x{}".format(img.shape[1], img.shape[0], w, h))
	patches = iv.blocks(img, (w, h, img.shape[2]))
	cnt = 0
	for p in patches[:]:
		cnt += 1
		pic_name = "{}{:04}.jpg".format(path_prefix, cnt)
		cv2.imwrite(pic_name, p)

	print("\n{} image patches saved in '{}*'\n> ".format(cnt, path_prefix), end='')
	
	return True

# Sliding window and save
def action_windows(proc, img, args):
	if len(args) < 5:
		print("\nWrong number of parameters\n> ", end='')
		return False

	dx = int(args[1])   # width
	dy = int(args[2])   # height
	ds = int(args[3])   # sliding step
	pref = args[4]      # output file name prefix

	tm = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
	patches = iv.windows(img, (dy, dx, 3), ds)
	cnt = 0
	os.makedirs("raw/%s" % tm)
	for p in patches[:]:
		cnt += 1
		pic_name = "raw/{}/{}_{:04}.jpg".format(tm, pref, cnt)
		cv2.imwrite(pic_name, p)
		print('.', end='')
	print("\n{} image patches saved in 'raw/{}/{}_*.jpg'\n> ".format(cnt, tm, pref), end='')
	return True

def action_gui():
	pass

def plugin_grid(proc, img, args):
	return iv.gridlines(img, 16, 1, np.array([0, 0, 0], dtype="uint8"))

def plugin_checker(proc, img, args):
	return iv.checkerboard(img, 64, np.array([0, 0, 0], dtype="uint8"), np.array([100, 100, 100], dtype="uint8"))

# Apply a filter on the image using convolution with a filter K
def plugin_filter(proc, args):
	global filter_bank

	hdr = filter_bank[args[1]]
	hdr_type = type(hdr).__name__

	if hdr_type == 'function':
		if len(args) > 2 and args[2] == 'none':
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
		elif hdr == 'load':
			if len(args) > 2:
				fn = "{}.{}".format(args[2], "json")
				try:
					f = open(fn, 'r')
					obj = json.load(f)
					for plugin in obj:
						if plugin[0] == "filter":
							plugin_filter(proc, plugin)
						elif plugin[0] == "crop":
							proc.refPt = [tuple(plugin[1][0]), tuple(plugin[1][1])]
							proc.selection = True
					f.close()
					print("[info] {} plugins loaded".format(len(obj)))
				except FileNotFoundError:
					return "[error] File not found: '{}'".format(fn)
			else:
				print("[warning] Wrong number of arguments")
		elif hdr == 'save':
			if len(args) > 2:
				obj = []
				for plugin in proc.plugins:
					obj.append(plugin[1])
				if proc.selection:
					obj.append(["crop", proc.refPt])
				if len(obj) > 0:
					fn = "{}.{}".format(args[2], "json")
					f = open(fn, 'w')
					json.dump(obj, f, indent=4)
					# print(json.dumps(obj, indent=4))
					print("[info] {} active plugins saved in '{}'".format(len(obj), fn))
				else:
					print("[warning] No active plugin. Save cancelled!")
			else:
				print("[warning] Wrong number of arguments")
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
	# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	gx = np.uint8(np.absolute(gx))
	gy = np.uint8(np.absolute(gy))
	out = cv2.bitwise_or(gx, gy)
	return out

def hdr_threshold(proc, img, args):
	warn = False
	# convert to grayscale
	if len(img.shape) > 2:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
		
	threshold_type = 'normal'
	if len(args) > 2:
		threshold_type = args[2]

	if threshold_type == 'normal':
		thresh = 0 if len(args) < 4 else int(args[3])
		out = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
	elif threshold_type == 'otsu':
		thresh = 0 if len(args) < 4 else int(args[3])
		out = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	elif threshold_type == 'adapt-mean':
		neib = 11 if len(args) < 4 else int(args[3])
		if neib % 2 == 1:
			out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, neib, 2)
		else:
			warn = True
	elif threshold_type == 'adapt-gauss':
		neib = 11 if len(args) < 4 else int(args[3])
		if neib % 2 == 1:
			out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neib, 2)
		else:
			warn = True
	if warn:
		print("[warning] Valid thresholds: (normal | otsu) [thresh] | (adapt-mean | adapt-gauss) [neighbors]")
		print("     thresh: 0-255 default 0")
		print("  neighbors: 3, 5, 7, 9,.. (odd number)")
		out = img
		plugin_filter(proc, ['filter', 'threshold', 'none'])

	return out

def hdr_canny(proc, img, args):
	# get min-max thresholds
	min = 100 if len(args) < 3 else int(args[2])
	max = 200 if len(args) < 4 else int(args[3])
	return cv2.Canny(img, min, max)

def hdr_resize(proc, img, args):
	w = 320 if len(args) < 3 else int(args[2])
	h = 180 if len(args) < 4 else int(args[3])
	
	return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

def hdr_equalizer(proc, img, args):
	# check if input is in grayscale
	if (img.shape[2] == 3):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	return cv2.equalizeHist(gray)

def hdr_contours(proc, img, args):
	if len(img.shape) > 2:
		# binarize image with canny (default)
		thresh = hdr_canny(proc, img, [])
	else:
		# already binarized
		thresh = img

	# find contours in the image, keeping only the four largest
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if iu.is_cv2() else cnts[1]
	# keep those with specific ratio w:h
	if len(args) >= 4:
		ratio_low = float(args[2])
		ratio_high = float(args[3])
		cn = []
		for c in cnts:
			# compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# ratio = round(abs(w**2 - h**2)/(w*h), 2)
			ratio = (w**2 - h**2)/(w*h)
			if ratio_low <= ratio <= ratio_high:
				cn.append(c)
		cnts = cn
	
	# sort by size
	items = 10 if len(args) < 5 else int(args[4])
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:items]

	# short by position from left to right (throws an error sometimes)
	# cnts = contours.sort_contours(cnts)[0]
	# output = cv2.merge([thresh] * 3)

	# set the image to be overlayed to the original frame, bypassing filters
	output = proc.last_frame.copy()

	# loop over the contours
	for c in cnts:
		# compute the bounding box for the contour
		(x, y, w, h) = cv2.boundingRect(c)
		roi = output[y - 5:y + h + 5, x - 5:x + w + 5]
		# proc.save_frame = roi.copy()
	
		# pre-process the ROI and classify it
		# roi = preprocess(roi, 28, 28)
		# roi = np.expand_dims(img_to_array(roi), axis=0)
		# pred = model.predict(roi).argmax(axis=1)[0] + 1
		# predictions.append(str(pred))
	
		# draw the prediction on the output image
		cv2.rectangle(output, (x - 2, y - 2),
					  (x + w + 2, y + h + 2), (0, 255, 0), 1)

		# ratio = round(abs((w**2 - h**2)/(w*h)), 2)
		ratio = round((w**2 - h**2)/(w*h), 2)
		ovinfo = "{} ({})".format(ratio, round(w*h, 2))
		proc.putText(output, ovinfo, cord=(x-2, y-2), color=(0, 100, 255))

	# if items == 1 and len(cnts) == 1:
	# 	output = proc.save_frame

	if "monitor" in args:
		wn = "plugin_contours"
		# img_out = cv2.resize(output, proc.winsize, interpolation=cv2.INTER_CUBIC)
		# img_out = output
		# cv2.createTrackbar("slider", wn, 3, 7, proc.set_plugin_param )
		# cv2.imshow(wn, img)

	return output

def hdr_face_detection(proc, img, args):
	global face_cascade, eye_cascade
	output = img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	if face_cascade is None or eye_cascade is None:
		face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = output[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return output

def hdr_thermo_detection(proc, img, args):
	global thermo_cascade
	output = img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	if thermo_cascade is None:
		thermo_cascade = cv2.CascadeClassifier('cascades/haarcascade_thermometer.xml')
	
	thermo = thermo_cascade.detectMultiScale(gray, 1.3, 5)
	# sort by size
	items = 10 if len(args) < 3 else int(args[2])
	thermo = sorted(thermo, key=lambda item: (item[2]*item[3]), reverse=True)[:items]
	for (x,y,w,h) in thermo:
		cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)
	return output


# App console commands
# --------------------
def video_start(pihost, piport, pipath, hw_quit, args):
	global video_monitor, video_so, winname, winsize, cam_type

	# Create a video monitor instance
	video_monitor = VideoProcessor(winname, winsize)
	video_monitor.camtype = cam_type

	# initialize video stream and player
	if cam_type == 0:
		video_so = StreamClient(pihost, piport, hw_quit)
	elif cam_type == 1:
		video_so = StreamHttpClient(pihost, piport, pipath, hw_quit)
	elif cam_type == 2:
		video_so = StreamFileListClient(pihost, piport, pipath, hw_quit)
		video_so.idletime = 1
	else:
		print("[warning] -ipcam {}: invalid value. No video stream initialized".format(cam_type))
		video_monitor = None
		return

	video_so.set_consumer(video_monitor.use)

	if video_so.init_socket(True) == 'failed':
		print("[error]: Unable to create video binary stream!")
		video_so.close()
	else:
		video_monitor.init_video_monitor()
		if cam_type == 2:
			video_so.start()
	
		if cam_type == 0:
			# send and wait for the reply
			hostso.send("start", wait=True)


def video_stop():
	global video_so, video_monitor
	
	if video_so:
		video_so.status = 'purge'
		video_so.set_consumer(None)
	
		if cam_type == 0:
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


def video_resize(vm, args):
	if len(args) != 3:
		print("[warning] invalid dimensions given")
		return
	vm.winsize = (int(args[2]), int(args[1]))
	cv2.resizeWindow(vm.winname, vm.winsize[1], vm.winsize[0])
	cv2.moveWindow(vm.winname, 0, 0)


def video_histogram(vm, args):
	ca = len(args)
	if ca > 1:
		vm.show_hist(type=args[1])
	else:
		vm.show_hist()


def show_help():
	print("""Usage: python3 picam.py [--host <host>] [--port <port>] [--path <path>] [--ipcam <ipcam>]
	<host> : target host or IP.
	<port> : tcp port number, default=5501.
	<path> : path name to folder containing images.
	<ipcam>: 0|1|2 (default=0)
			 0: connect to raspberry pi running picam-server.py  (requires --host and --port)
			 1: connect to android device running IP webcam app (requires --host and --port)
			 2: specify folder path to display images in rotation (requires --path)
	
Default commands:
	start       : starts video feed from server
	stop        : stops video feed
	quit        : quits program
	histogram   : curve|lines|equalize|curve-gray|normalize|off

	Plugin commands, executed sequentially by order of definition:
	grid        : makes a 16x16 grid with stroke 1 with gridline color of (0,0,0).
	checker     : checkerboard overlay of box size 64x64 (black-transparent boxes)
	filter      : <name> [<args>]
				  Applies a named filter on the image.
				"blur"      : Soft blur 3x3 kernel
				"blur-more" : Hard blur  3x3 kernel
				"sharpen"   : Sharpen
				"laplacian" : Laplacian
				"sobel-x"   : SobelX
				"sobel-y"   : SobelY
				"emboss"    : Emboss
				"sobel"     : <kernel size>: reveal outlines. Kernel size: [1|3|5|..]
				"threshold" : (normal|otsu) [thresh] | (adapt-mean|adapt-gauss) [neighbors]
							  thresh: 0-255 default 0
							  neighbors: 3, 5, 7, 9,.. (default 11)
				"canny"     : Canny
				"equalizer" : Equalizer
				"contours"  : Contours
				"resize"    : Resize
				"faces"     : Face detection
				"none"      : Remove filter
				"list"      : List active filters
""")

# construct the kernel bank, a list of functions applying kernels or filters
filter_bank = {
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
	"resize"    : hdr_resize,
	"faces"     : hdr_face_detection,
	"thermo"    : hdr_thermo_detection,
	"none"      : "remove",
	"load"      : "load",
	"save"      : "save",
	"list"      : "list"
}

# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-host", "--host", type=str, required=False,
				help = "host address")
ap.add_argument("-p", "--port", type=int, required=False,
				help = "port number")
ap.add_argument("-path", "--path", type=str, required=False,
                help = "path to image folder")
ap.add_argument("-ipc", "--ipcam", type=int, default=0, required=False,
				help = "shutter speed")
args = ap.parse_args()

# Create global variables and classes
# -----------------------------------
pihost = args.host
piport = args.port
pipath = args.path
cam_type = args.ipcam
is_ipcam = (cam_type == 1)

winname = "Camera-feed"
winsize = (640, 480)
force_quit = False
video_so = None
video_monitor = None

if cam_type == 1:
	pipath = "/shot.jpg"
	# winsize = (720, 405)
# elif cam_type == 2:
# 	pipath = "data/thermometer/img"

# Face detector global variables
face_cascade = None
eye_cascade = None
thermo_cascade = None

# ----------------
# ---   MAIN   ---
# ----------------

# Connect to Pi host asynchronously
if cam_type == 0:
	hostso = DialogClient(pihost, piport, hw_quit)
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
			if cam_type == 0:
				# send and wait for the reply
				hostso.send(q, wait=True)

		elif qr[0] == 'start':
			video_start(pihost, piport, pipath, hw_quit, qr)
		elif qr[0] == 'stop':
			video_stop()
		elif qr[0] == 'resize':
			video_resize(video_monitor, qr)
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
				      .format("|".join(key for key in filter_bank)))
			elif qr[1] not in filter_bank:
				print("[error]: Invalid filter {}\nUsage: filter {}> "
				      .format(qr[1], "|".join(key for key in filter_bank)))
			else:
				plugin_filter(video_monitor, qr)

		# Actions (two words)
		elif qr[0] == 'grab':
			video_monitor.set_action(action_grab, qr)
			print("Action {} activated".format(q))
		elif qr[0] == 'blocks':
			video_monitor.set_action(action_blocks, qr)
			print("Action {} activated".format(q))
		elif qr[0] == 'windows':
			video_monitor.set_action(action_windows, qr)
			print("Action {} activated".format(q))
		elif qr[0] == 'gui':
			action_gui()
			print("Action {} activated".format(q))
		elif qr[0] == 'help':
			show_help()

		# Anything else just send it over
		else:
			if cam_type == 0:
				# send and wait for the reply
				hostso.send(q, wait=True)
			elif cam_type == 2:
				if q.startswith('set interval'):
					if len(qr) > 2:
						video_so.idletime = float(qr[2])


# Close connection
if cam_type == 0:
	hostso.close()
print("Bye..")
