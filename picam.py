import os
import datetime
import argparse

from cams import VideoProcessor
from utils.plugins.imfilters import *
from utils.sockets.dialogclientsocket import DialogClient
from utils.sockets.streamclientsocket import StreamClient
from utils.sockets.httpsocket import StreamHttpClient
from utils.sockets.filelistsocket import StreamFileListClient
from utils.sockets.videosocket import StreamVideoClient

from utils import helper_visuals as iv

# Thread free for UI
# import matplotlib
# matplotlib.use('TkAgg')


# Formatter subclass for argparse. It is used to combine functionality of both classes defined.
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
	pass

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
	if len(args) < 2:
		print("\rWrong number of parameters")
		print("Usage: grab <prefix> [repeat <times>] [idle <seconds>]\n> ", end='')
		return False
	
	dt = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
	pic_name = "{}_{}.jpg".format(args[1], dt)
	if proc.selection:
		# grab inner selection mask area
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		imout = img[ty:by, tx:bx].copy()
	elif proc.tracking:
		# grab object tracker mask area
		tx, ty, bx, by = proc.tracker.tl[0], proc.tracker.tl[1], proc.tracker.br[0], proc.tracker.br[1]
		imout = img[ty:by, tx:bx].copy()
	else:
		imout = img.copy()

	cv2.imwrite(pic_name, imout)
	print("\r[info] picture saved as '{}'\n> ".format(pic_name), end='')
	return True


# Stitching images side-by-side
gbl_stitching_img = None
def action_stitching(proc, img, args):
	global gbl_stitching_img
	
	if len(args) < 2:
		print("\rWrong number of parameters")
		print("Usage: stitch <prefix> [repeat <times>] [idle <seconds>]\n> ", end='')
		return False
	
	dt = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
	itr = 0
	if "repeat" in args:
		idx = args.index("repeat") + 1
		itr = int(args[idx]) - 1

	pic_name = "{}.jpg".format(args[1])
	if gbl_stitching_img is None:
		gbl_stitching_img = img.copy()
	else:
		# TODO: parameterize horizontal/vertical stiching
		gbl_stitching_img = np.vstack((gbl_stitching_img, img.copy()))

	cv2.imwrite(pic_name, gbl_stitching_img)
	print("\r[info] picture saved as '{}'\n> ".format(pic_name), end='')
	if itr == 0:
		gbl_stitching_img = None
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
		print("\rWrong number of parameters")
		print("Usage: windows <width> <height> <step> <prefix> [in|out] [repeat <times>] [idle <seconds>]\n> ", end='')
		return False

	dx = int(args[1])   # width
	dy = int(args[2])   # height
	ds = int(args[3])   # sliding step
	pref = args[4]      # output file name prefix
	if len(args) > 5:
		margin = int(args[5])
	else:
		margin = 0
	if len(args) > 6:
		invert = args[6] == "out"
	else:
		invert = False
	
	tm = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
	if len(img.shape) == 2:
		patches = iv.windows(img, (dy, dx), ds)
	else:
		patches = iv.windows(img, (dy, dx, img.shape[2]), ds)
	
	iy, ix = img.shape[:2]
	rows = iy//ds - dy//ds + 1
	cols = ix//ds - dx//ds + 1

	if proc.selection:
		# compute matrix dimensions of window patches
		# grab inner selection mask area
		tx, ty = proc.refPt[0][0] + margin, proc.refPt[0][1] + margin
		bx, by = proc.refPt[1][0] - margin, proc.refPt[1][1] - margin
		win_tl = (max(0, (ty//ds) - (dy//ds) + 1), max(0, (tx//ds) - (dx//ds) + 1))
		win_br = (max(0, (by//ds)), max(0, (bx//ds)))
		print("Selection coordinates over sliding window patches:")
		print("    rect: ({}, {}) - ({}, {})".format(ty, tx, by, bx))
		print(" windows: {}, {}".format(win_tl, win_br))

	cnt = 0
	skip_cnt = 0
	skip = invert
	os.makedirs("raw/%s" % tm)
	for p in patches[:]:
		if proc.selection:
			posY = cnt // cols
			posX = cnt % cols
			if win_tl[0] <= posY <= win_br[0] and win_tl[1] <= posX <= win_br[1]:
				skip = invert
			else:
				skip = not invert
		cnt += 1
		if not skip:
			pic_name = "raw/{}/{}_{}_{:04}.jpg".format(tm, pref, tm, cnt)
			cv2.imwrite(pic_name, p)
		else:
			skip_cnt += 1
			print("\rSkipping patch {}> ".format(cnt), end='')
	print("\n{} image patches saved in 'raw/{}/{}_*.jpg'\n> ".format(cnt-skip_cnt, tm, pref), end='')
	return True

def action_gui():
	pass

# App console commands
# --------------------
def video_start(pihost, piport, pipath, hw_quit, args):
	global video_monitor, video_so, winname, winsize, cam_type

	# initialize video stream and player
	if cam_type == 0:
		video_so = StreamClient(pihost, piport, hw_quit)
		winname = "Raspberry Pi ({}) - Live".format(piport)
	elif cam_type == 1:
		video_so = StreamHttpClient(pihost, piport, pipath, hw_quit)
		winname = "Android IPCam ({}) - Live".format(piport)
	elif cam_type == 2:
		if args.fit:
			video_so = StreamFileListClient(None, pipath, hw_quit)
		else:
			video_so = StreamFileListClient(winsize, pipath, hw_quit)
		winname = "Photo Stream - Live"
		# video_so.idletime = piidle
		# video_so.idletime = 1
	elif cam_type == 3:
		video_so = StreamVideoClient(winsize, hw_quit, video_path=pipath)
		winname = "Webcam - Live"
	else:
		print("[warning] -ipcam {}: invalid value. No video stream initialized".format(cam_type))
		video_monitor = None
		return
	
	# Create a video monitor instance
	# video_monitor = VideoProcessor(winname, winsize)
	# video_monitor.camtype = cam_type
	video_so.set_consumer(video_monitor.use)
	video_monitor.socket = video_so

	if video_so.init_socket() == 'failed':
		print("[error]: Unable to create video binary stream!")
		video_so.close()
	else:
		video_monitor.init_video_monitor()
		if cam_type == 2:
			video_monitor.winsize = video_so.dim
		if cam_type >= 2:
			video_so.start()
	
		if cam_type == 0:
			# send and wait for the reply
			hostso.send("start", wait=True)


def video_stop():
	global video_so, video_monitor
	
	if video_so:
		if video_so.status != 'init':
			video_so.status = 'purge'
		# video_so.set_consumer(None)
	
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


def video_pause():
	global video_monitor
	if video_monitor.pause_frame:
		video_monitor.unpause()
	else:
		video_monitor.pause()
	print("Video preview is {}".format("paused" if video_monitor.pause_frame else "running"))
	

def video_resize(vm, args):
	global winsize
	if len(args) != 3:
		print("[warning] invalid dimensions given")
		return
	winsize = (int(args[1]), int(args[2]))
	if vm is not None:
		vm.resize_monitor((winsize[1], winsize[0]))


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


# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser(
	description="PiCam computer vision remote camera",
	formatter_class=MyFormatter)

ap.add_argument("--host", type=str, required=False, help="host address")
ap.add_argument("--port", type=int, required=False, help="port number")
ap.add_argument("--path", type=str, required=False, help="path to image folder")
ap.add_argument("--ipcam", type=int, default=3, help="camera type")
ap.add_argument("--wsize", type=int, nargs=2, default=[320, 240], help="preview monitor dimensions")
ap.add_argument("--fit", type=bool, nargs='?', const=True, default=False,
                help="a resize filter is added to match preview monitor dimensions")
args = ap.parse_args()

# Create global variables and classes
# -----------------------------------
pihost = args.host
piport = args.port
pipath = args.path
cam_type = args.ipcam
piidle = 5
is_ipcam = (cam_type == 1)

winname = "Camera-feed"
winsize = tuple(args.wsize)
force_quit = False
video_so = None
video_monitor = None

if cam_type == 1:
	pipath = "/shot.jpg"

# Face detector global variables
# face_cascade = None
# eye_cascade = None
# thermo_cascade = None

# ----------------
# ---   MAIN   ---
# ----------------

# Connect to Pi host asynchronously
if cam_type == 0:
	hostso = DialogClient(pihost, piport, hw_quit)
	if hostso.init_socket(True) == 'failed':
		print("Quiting program!")
		quit()
	
# Create a video monitor instance
video_monitor = VideoProcessor(winname, winsize)
video_monitor.camtype = cam_type
# Create a default filter resize to match preview dimensions
if args.fit and cam_type != 2:
	plugin_filter(video_monitor, "filter resize {} {}".format(winsize[0], winsize[1]).split())


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
			video_start(pihost, piport, pipath, hw_quit, args)
		elif qr[0] == 'stop':
			video_stop()
		elif qr[0] == 'pause':
			video_pause()
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

		elif qr[0] == 'roi':
			hdr_roi_select(video_monitor, video_monitor.save_frame, None)

		# Actions (two words)
		elif qr[0] == 'grab':
			video_monitor.set_action(action_grab, qr)
		elif qr[0] == 'blocks':
			video_monitor.set_action(action_blocks, qr)
		elif qr[0] == 'windows':
			video_monitor.set_action(action_windows, qr)
		elif qr[0] == 'stitch':
			video_monitor.set_action(action_stitching, qr)
		elif qr[0] == 'gui':
			# TODO: action GUI under development
			action_gui()
		elif qr[0] == 'help':
			show_help()
		elif q == 'action stop':
			# Cancel running action
			video_monitor.set_action(None, qr)
			print("[info] active action canceled")

		# Anything else just send it over
		else:
			if cam_type == 0:
				# send and wait for the reply
				hostso.send(q, wait=True)
			elif cam_type == 1:
				if qr[0] in ['get', 'set']:
					video_so.set_command(qr[0], qr)
			elif cam_type == 2:
				if q.startswith('set idle'):
					if len(qr) > 2:
						video_so.idletime = float(qr[2])
			elif cam_type == 3:
				if q.startswith('set res'):
					video_so.set_command("res", [qr[2], qr[3]])
				elif q.startswith('set idle'):
					if len(qr) > 2:
						video_so.idletime = float(qr[2])
	else:
		# if object tracker (CMT filter) is active, switch debug on/off by hitting return
		if video_monitor is not None:
			plugin = video_monitor.get_plugin(hdr_tracker_cmt)
			if plugin is not None:
				plugin[2][1] = not plugin[2][1]
		

# Close connection
if cam_type == 0:
	hostso.close()
print("Bye..")
