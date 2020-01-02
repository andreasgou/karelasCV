import cv2
import numpy as np
import json
import imutils as iu
from imutils.perspective import four_point_transform
import time

from utils import helper_visuals as iv
import utils.plugins.CMT.CMT as CMT
from utils import modelio as mio
from keras.preprocessing.image import img_to_array

# import utils.plugins.CMT.util as cmt_utils


# Parse filter arguments
def parse_args(args):
	obj = {args[0]: args[1]}
	for i in range(2, len(args)):
		arg = args[i].split("=")
		obj[arg[0]] = None if len(arg) == 1 else arg[1]
	return obj


# Apply a filter on the image using convolution with a filter K
def plugin_filter(proc, args):
	global filter_bank
	
	filtr = parse_args(args)

	hdr = filter_bank[filtr.get("filter")]["handler"]
	hdr_type = type(hdr).__name__
	
	if hdr_type == 'function':
		if 'none' in filtr.keys():
			proc.remove_plugin(hdr)
		else:
			# TODO: replace args with filtr
			proc.append_plugin(hdr, args)

	elif hdr_type == 'str':
		if hdr == 'remove':
			proc.remove_all_plugins()
		elif hdr == 'load':
			if len(args) > 2:
				fn = "filters/{}.{}".format(args[2], "json")
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
					fn = "filters/{}.{}".format(args[2], "json")
					f = open(fn, 'w')
					json.dump(obj, f, indent=4)
					print(json.dumps(obj, indent=4))
					print("[info] {} active plugins saved in '{}'".format(len(obj), fn))
				else:
					print("[warning] No active plugin. Save cancelled!")
			else:
				print("[warning] Wrong number of arguments")
		elif hdr == 'list':
			for plugin in proc.plugins:
				print(plugin[1])
		elif hdr == 'help':
			print("\nDisplaying available filters:\n------------------------------")
			for key in filter_bank:
				name = "{:<15}".format(key)
				print("%s: %s" % (name, filter_bank[key]["desc"]))


# construct average blurring kernels used to smooth an image
def hdr_smallBlur(proc, img, plugin):
	smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
	# custom convolution
	# return = iv.convolve(img, smallBlur)
	return cv2.filter2D(img, -1, smallBlur)


def hdr_largeBlur(proc, img, plugin):
	largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
	return cv2.filter2D(img, -1, largeBlur)


def hdr_gaussian(proc, img, plugin):
	args = plugin[1]
	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	return blurred


# construct a sharpening filter
def hdr_sharpen(proc, img, plugin):
	sharpen = np.array((
		[0, -1, 0],
		[-1, 5, -1],
		[0, -1, 0]), dtype="int")
	return cv2.filter2D(img, -1, sharpen)


# construct the Laplacian kernel used to detect edge-like regions of an image
def hdr_laplacian(proc, img, plugin):
	laplacian = np.array((
		[0, 1, 0],
		[1, -4, 1],
		[0, 1, 0]), dtype="int")
	return cv2.filter2D(img, -1, laplacian)


# construct the Sobel x-axis kernel
def hdr_sobelX(proc, img, plugin):
	sobelX = np.array((
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]), dtype="int")
	return cv2.filter2D(img, -1, sobelX)


# construct the Sobel y-axis kernel
def hdr_sobelY(proc, img, plugin):
	sobelY = np.array((
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]), dtype="int")
	return cv2.filter2D(img, -1, sobelY)


# construct an emboss kernel
def hdr_emboss(proc, img, plugin):
	emboss = np.array((
		[-2, -1, 0],
		[-1, 1, 1],
		[0, 1, 2]), dtype="int")
	return cv2.filter2D(img, -1, emboss)


def hdr_sobel(proc, img, plugin):
	args = plugin[1]
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


def hdr_threshold(proc, img, plugin):
	args = plugin[1]
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
	elif threshold_type == 'help':
		warn = True

	if warn:
		print("[warning] Valid thresholds: (normal | otsu) [thresh] | (adapt-mean | adapt-gauss) [neighbors]")
		print("     thresh: 0-255 default 0")
		print("  neighbors: 3, 5, 7, 9,.. (odd number)")
		out = img
		plugin_filter(proc, ['filter', 'threshold', 'none'])
	
	return out


def hdr_canny(proc, img, plugin):
	args = plugin[1]
	# get min-max thresholds
	min = 100 if len(args) < 3 else int(args[2])
	max = 200 if len(args) < 4 else int(args[3])
	return cv2.Canny(img, min, max)


def hdr_inverse_colors(proc, img, plugin):
	return cv2.bitwise_not(img)


def hdr_resize(proc, img, plugin):
	args = plugin[1]
	imout = img
	if len(args) == 4:
		w = int(args[2])
		h = int(args[3])
		# resize image if required
		if [h, w] != img.shape[:2]:
			imout =  cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

	return imout


def hdr_rotate(proc, img, plugin):
	rows,cols = img.shape[:2]
	args = plugin[1]
	angle = int(args[2])
	r = rows//2 if len(args) < 4 else int(args[3])
	c = cols//2 if len(args) < 5 else int(args[4])

	M = cv2.getRotationMatrix2D((c, r), angle, 1)
	return cv2.warpAffine(img, M, (cols,rows))


def hdr_equalizer(proc, img, plugin):
	# check if input is in grayscale
	if (img.shape[2] == 3):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	
	return cv2.equalizeHist(gray)


def hdr_add_noise(proc, img, plugin):
	args = plugin[1]
	mean = (50, 50, 50) if len(args) <= 2 else (int(args[2]), int(args[2]), int(args[2]))
	sigma = (50, 50, 50) if len(args) <= 3 else (int(args[3]), int(args[3]), int(args[3]))
	noise = cv2.randn(img.copy(), mean, sigma)
	alpha = 0.5 if len(args) <= 4 else float(args[4])
	beta = 1 - alpha
	return cv2.addWeighted(img, alpha, noise, beta, 0.0)


def hdr_contours(proc, img, plugin):
	args = plugin[1]
	if len(img.shape) > 2:
		# binarize image with canny defaults
		thresh = hdr_canny(proc, img, [None, []])
	else:
		# already binarized
		thresh = img
		
	# get args
	cmd = dict(it.split("=") if it.find("=") != -1 else [it, ""] for it in args)
	# read args
	ratio_low = float(cmd.get("rl", 0))
	ratio_high = float(cmd.get("rh", 0))
	area_low = float(cmd.get("al", 0))
	area_high = float(cmd.get("ah", 0))
	margin = int(cmd.get("margin", 2))
	post_preview = cmd.get("pp", "off") == "on"
	extract_preview = cmd.get("extract", "off") == "on"
	
	# find contours in the image, keeping only the four largest
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if iu.is_cv2() else cnts[1]

	# apply area restrictions
	# if area_low + area_high + ratio_low + ratio_high != 0:
	cn = []

	for c in cnts:
		# keep only contours with 4 vertices
		epsilon = 0.1 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
		if len(approx) == 4:
			# compute the bounding box for the contour
			# (x, y, w, h) = cv2.boundingRect(c)
			(x, y, w, h) = cv2.boundingRect(approx)
			ratio = (w**2 - h**2)/(w*h)
			if (area_low + area_high == 0 or area_low <= w*h <= area_high) and \
					(ratio_low + ratio_high == 0 or ratio_low <= ratio <= ratio_high):
				# cn.append(c)
				cn.append(approx)

	cnts = cn
		
	# sort by size
	items = int(cmd.get("cnt", 10))
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:items]
	
	# short by position from left to right (throws an error sometimes)
	# cnts = contours.sort_contours(cnts)[0]
	# output = cv2.merge([thresh] * 3)
	
	# set the image to be overlayed to the original frame, bypassing filters
	if post_preview:
		output = img.copy()
	else:
		output = proc.last_frame.copy()
		
	if len(plugin) < 3:
		plugin.append(["contours", cnts])
	else:
		plugin[2][1] = cnts
		
	# loop over the contours
	for c in cnts:
		# compute the bounding box for the contour
		(x, y, w, h) = cv2.boundingRect(c)
	
		if extract_preview:
			birds_eye = four_point_transform(output, c.reshape(4, 2))
			return birds_eye

		# birds_eye = four_point_transform(output, c.reshape(4, 2))
		# dims = birds_eye.shape[:2]
		# output = iv.ovelray_patch(output, birds_eye, [(y, x), (y+dims[0], x+dims[1])])

		# draw the prediction on the output image
		cv2.rectangle(output, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 255, 0), 1)

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


def hdr_erosion(proc, img, plugin):
	args = plugin[1]
	ks = 5 if len(args) <= 2 else int(args[2])
	kernel = np.ones((ks, ks), np.uint8)
	return cv2.erode(img, kernel, iterations=1)


def hdr_dilation(proc, img, plugin):
	args = plugin[1]
	ks = 5 if len(args) <= 2 else int(args[2])
	kernel = np.ones((ks, ks), np.uint8)
	return cv2.dilate(img, kernel, iterations=1)


def plugin_grid(proc, img, plugin):
	# get args
	args = plugin[1]
	cmd = dict(it.split("=") if it.find("=") != -1 else [it, ""] for it in args)
	# read args
	dy = int(cmd.get("dy", 16))
	dx = int(cmd.get("dx", 16))
	return iv.gridlines(img, (dy, dx), 1, np.array([0, 0, 0], dtype="uint8"))


def plugin_checker(proc, img, plugin):
	return iv.checkerboard(img, 64, np.array([0, 0, 0], dtype="uint8"), np.array([100, 100, 100], dtype="uint8"))


# Face detector global variables
face_cascade = None
eye_cascade = None
def hdr_face_detection(proc, img, plugin):
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


# Face detector global variables
thermo_cascade = None
def hdr_thermo_detection(proc, img, plugin):
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

# Face detector global variables
nums_cascade = None
def hdr_num_detection(proc, img, plugin):
	global nums_cascade

	# get args
	args = plugin[1]
	cmd = dict(it.split("=") if it.find("=") != -1 else [it, ""] for it in args)

	# set the image to be overlayed to the original frame, bypassing filters
	post_preview = cmd.get("pp", "off") == "on"
	if post_preview:
		output = img.copy()
	else:
		output = proc.last_frame.copy()

	# convert to grayscale
	if len(img.shape) > 2:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	# load the pre-trained network
	if nums_cascade is None:
		model = cmd.get("model", "cascades/lenet_lcdigits_14x28.hdf5")
		if model is None:
			print("[warning] Model not found.")
			plugin_filter(proc, ['filter', 'numbers', 'none'])
			return output
		
		nums_cascade = mio.get_model(model)
	
	# get contours filter if active
	contours = proc.get_plugin(hdr_contours)
	
	if proc.selection:
		# grab inner selection mask area
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		gray = gray[ty:by, tx:bx].copy()
		gray = cv2.resize(gray, (14, 28), interpolation=cv2.INTER_CUBIC)
		# gray = gray[np.newaxis, :, :, np.newaxis]
		gray = img_to_array(gray)
		gray = np.expand_dims(gray, axis=0)
	elif proc.tracking:
		# grab object tracker mask area
		tx, ty, bx, by = proc.tracker.tl[0], proc.tracker.tl[1], proc.tracker.br[0], proc.tracker.br[1]
		gray = gray[ty:by, tx:bx].copy()
		gray = gray[np.newaxis, :, :, np.newaxis]
	elif not (proc.tracking or proc.selection):
		gray = cv2.resize(gray, (14, 28), interpolation=cv2.INTER_CUBIC)
		gray = img_to_array(gray)
		gray = np.expand_dims(gray, axis=0)
	elif contours is not None:
		batchROIs = None
		batchLocs = []
		for c in contours[2][1]:
			# compute the bounding box for the contour
			(tx, ty, w, h) = cv2.boundingRect(c)

			# expand contours a bit
			margin = 3
			tx -= margin
			ty -= margin
			bx = tx + w + margin
			by = ty + h + margin

			roi = gray[ty:by, tx:bx]
			roi = cv2.resize(roi, (14, 28), interpolation=cv2.INTER_CUBIC)
			# roi = roi[:, :, np.newaxis]
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			if batchROIs is None:
				batchROIs = roi
			else:
				batchROIs = np.vstack([batchROIs, roi])

			batchLocs.append((tx, ty, bx, by))
			
		if batchROIs is not None:
			# gray = batchROIs[np.newaxis, :, :]
			gray = batchROIs
		else:
			gray = None
		# print("Contours array:", gray)
	
	if gray is not None:
		predictions = nums_cascade.predict(gray, batch_size=8)
		preds = predictions.argmax(axis=1)
		# print(classification_report(preds, preds, target_names=[str(x) for x in classes]))
		# print(preds)
		
		if contours is not None:
			for i in range(len(preds)):
				ovinfo = "{}".format(preds[i])
				tx, ty, bx, by = batchLocs[i]
				# draw the prediction on the output image
				cv2.rectangle(output, (tx, ty), (bx, by), (0, 255, 0), 1)
				proc.putText(output, ovinfo, cord=(tx, by+15), size=0.6, color=(0, 100, 255))
		else:
			ovinfo = "{}".format(preds[0])
			if not (proc.tracking or proc.selection):
				proc.putText(output, ovinfo, cord=(20, 20), size=0.6, color=(0, 100, 255))
			else:
				proc.putText(output, ovinfo, cord=(tx, by+15), size=0.6, color=(0, 100, 255))
	
	return output


# CMT object tracking
def hdr_tracker_cmt(proc, img, plugin):
	output = img.copy()
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	
	if proc.selection:
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]

		cmt_tracker = CMT.CMT()
		# cmt_tracker.estimate_scale = args.estimate_scale
		cmt_tracker.estimate_rotation = False
		tl = (tx, ty)
		br = (bx, by)
		cmt_tracker.initialise(gray, tl, br)
		# plugin.append(cmt_tracker)
		plugin.append(["debug", True])
		proc.selection = False
		proc.tracking = True
		proc.tracker = cmt_tracker
		frame = 1
		plugin.append(frame)
		
	elif proc.tracking:
		# plugin = proc.get_plugin(hdr_cmt)
		# cmt_tracker = plugin[2]
		cmt_tracker = proc.tracker
		tic = time.time()
		cmt_tracker.process_frame(gray)
		toc = time.time()
		
		# Display results
		
		# Draw updated estimate
		# if cmt_tracker.has_result:
			# cv2.line(output, cmt_tracker.tl, cmt_tracker.tr, (0, 255, 0), 2)
			# cv2.line(output, cmt_tracker.tr, cmt_tracker.br, (0, 255, 0), 2)
			# cv2.line(output, cmt_tracker.br, cmt_tracker.bl, (0, 255, 0), 2)
			# cv2.line(output, cmt_tracker.bl, cmt_tracker.tl, (0, 255, 0), 2)
			# cv2.rectangle(output, cmt_tracker.tl, cmt_tracker.br, (0, 255, 0), 1)
		
		# cmt_utils.draw_keypoints(cmt_tracker.tracked_keypoints, output, (255, 255, 255))
		# this is from simplescale
		# cmt_utils.draw_keypoints(cmt_tracker.votes[:, :2], output)  # blue
		# cmt_utils.draw_keypoints(cmt_tracker.outliers[:, :2], output, (0, 0, 255))

		# Advance frame number
		frame = plugin[3] + 1
		plugin[3] = frame
		if plugin[2][1]:
			print('\r{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(cmt_tracker.center[0], cmt_tracker.center[1], cmt_tracker.scale_estimate, cmt_tracker.active_keypoints.shape[0], 1000 * (toc - tic), frame), end='')

	return output

# OpenCV KCF object tracking
def hdr_tracker(proc, img, plugin):
	output = img.copy()
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	
	# get args
	args = plugin[1]
	cmd = dict(it.split("=") if it.find("=") != -1 else [it, ""] for it in args)
	
	if cmd.get("name", "") == "cmt":
		proc.remove_plugin(hdr_tracker)
		plugin_filter(proc, ["filter", "cmt"])
		return output
	
	if proc.selection:
		# initialize a dictionary that maps strings to their corresponding
		# OpenCV object tracker implementations
		# OPENCV_OBJECT_TRACKERS = {
		# 	"csrt": cv2.TrackerCSRT_create,
		# 	"kcf": cv2.TrackerKCF_create,
		# 	"boosting": cv2.TrackerBoosting_create,
		# 	"mil": cv2.TrackerMIL_create,
		# 	"tld": cv2.TrackerTLD_create,
		# 	"medianflow": cv2.TrackerMedianFlow_create,
		# 	"mosse": cv2.TrackerMOSSE_create
		# }
		OPENCV_OBJECT_TRACKERS = {
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create
		}
		# grab the appropriate object tracker using our dictionary of
		# OpenCV object tracker objects
		intracker = OPENCV_OBJECT_TRACKERS[cmd.get("name", "kcf")]()
		
		# Get the roi from selection
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		
		intracker.init(img, (tx, ty, bx-tx, by-ty))
		# Wrapper object for tracker
		tracker = type('mytracker', (object,), {
			"tracker": intracker,
		     "has_result": False,
		     "tl": (0, 0),
		     "br": (0, 0)
		 })

		plugin.append(["debug", True])
		proc.selection = False
		proc.tracking = True
		proc.tracker = tracker
		frame = 1
		plugin.append(frame)
	
	elif proc.tracking:
		# plugin = proc.get_plugin(hdr_cmt)
		# cmt_tracker = plugin[2]
		tracker = proc.tracker
		intracker = tracker.tracker

		tic = time.time()

		# grab the new bounding box coordinates of the object
		(success, box) = intracker.update(img)
		tracker.has_result = success
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			tracker.tl = (x, y)
			tracker.br = (x + w, y + h)

		toc = time.time()
		
		frame = plugin[3] + 1
		plugin[3] = frame
		# if plugin[2][1]:
		# 	print('\rframe: {:04d}, {:04.0f}ms'.format(frame, 1000 * (toc - tic)), end='')

	return output

# ROI selection with OpenCV
def hdr_roi_select(proc, img, plugin):
	# proc.pause_frame = True
	proc.pause()
	initBB = cv2.selectROI(proc.winname, img, fromCenter=False,
	                       showCrosshair=True)
	print(type(initBB), initBB)
	# proc.pause_frame = False
	proc.unpause()
	return img

# Harris corner feature extractor
def hdr_harris_corner(proc, img, plugin):
	output = img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	
	# result is dilated for marking the corners, not important
	dst = cv2.dilate(dst, None)
	
	# Threshold for an optimal value, it may vary depending on the image.
	output[dst > 0.01*dst.max()] = [0, 0, 255]
	
	return output


# Harris corner with subPixel accuracy feature extractor
def hdr_harris_corner_subpixel(proc, img, plugin):
	output = img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	
	# result is dilated for marking the corners, not important
	dst = cv2.dilate(dst, None)
	
	# Threshold for an optimal value, it may vary depending on the image.
	# output[dst > 0.01*dst.max()] = [0, 0, 255]
	ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
	dst = np.uint8(dst)
	
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	
	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
	
	# Now draw them
	res = np.hstack((centroids, corners))
	res = np.int0(res)
	output[res[:, 1], res[:, 0]] = [0, 0, 255]
	output[res[:, 3], res[:, 2]] = [0, 255 ,0]
	
	return output


# Harris corner feature extractor
def hdr_good_features_to_track(proc, img, plugin):
	output = img
	# convert to grayscale
	if len(img.shape) > 2:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
	corners = np.int0(corners)
	
	for i in corners:
		x, y = i.ravel()
		cv2.circle(output, (x, y), 3, 255, -1)
	
	return output


# Deskew image using openCV moments
# Experimental - use a binary image with skewed digits to preview results
def hdr_deskew(proc, img, plugin):
	SZ = img.shape[:2]
	# convert to grayscale
	if len(img.shape) > 2:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	imout = gray
	
	if proc.selection:
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		m = imout[ty:by, tx:bx].copy()
	else:
		m = cv2.moments(imout)

	if abs(m['mu02']) < 1e-2:
		# no deskewing needed.
		return imout

	# Calculate skew based on central momemts.
	skew = m['mu11']/m['mu02']
	# Calculate affine transform to correct skewness.
	M = np.float32([[1, skew, -0.5*SZ[0]*skew], [0, 1, 0]])
	# Apply affine transform
	imout = cv2.warpAffine(imout, M, SZ, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return imout


# Image patch overlay into selected area.
# If an image is not given from the console, it copies the selected area patch
# By moving or re-selecting a new area, the patch is pasted into.
def hdr_patch(proc, img, plugin):
	# get args
	args = plugin[1]
	cmd = dict(it.split("=") if it.find("=") != -1 else [it, ""] for it in args)
	fname = cmd.get("file", "")

	imout = img
	if proc.selection:
		tx, ty, bx, by = proc.refPt[0][0], proc.refPt[0][1], proc.refPt[1][0], proc.refPt[1][1]
		plugin = proc.get_plugin(hdr_patch)
		
		if len(plugin) == 3:
			patch = plugin[2].copy()
		elif len(fname) > 0:
			patch = cv2.imread(fname)
			plugin.append(patch.copy())
		else:
			args.append("source")
			patch = imout[ty:by, tx:bx].copy()
			plugin.append(patch.copy())

		if patch.shape[:2] != (by-ty, bx-tx):
			patch = cv2.resize(patch, (bx-tx, by-ty), interpolation=cv2.INTER_AREA)

		imout = iv.ovelray_patch(img, patch, [(ty, tx), (by, bx)])

	return imout


# construct the kernel bank, a list of functions applying kernels or filters
filter_bank = {
	"blur"      : { "handler": hdr_smallBlur, "desc": "Soft blur 3x3 kernel" },
	"blur-more" : { "handler": hdr_largeBlur, "desc": "Hard blur  3x3 kernel" },
	"gaussian"  : { "handler": hdr_gaussian, "desc": "Classic gaussian blur" },
	"sharpen"   : { "handler": hdr_sharpen, "desc": "Sharpen" },
	"laplacian" : { "handler": hdr_laplacian, "desc": "Laplacian filter" },
	"sobel"     : { "handler": hdr_sobel, "desc": "{}{:17}{}{:<17}{}".format(
		"<kernel size>: reveal outlines\n",
		" ", "Apply sobel in both directions.\n",
		" ", "<kernel-size> (odd numbers only): [1|3|5|..]")
					},
	"sobel-x"   : { "handler": hdr_sobelX, "desc": "Apply sobel in x direction" },
	"sobel-y"   : { "handler": hdr_sobelY, "desc": "Apply sobel in y direction" },
	"emboss"    : { "handler": hdr_emboss, "desc": "Emboss" },
	"threshold" : { "handler": hdr_threshold, "desc": "{}{:<17}{}{:<17}{}".format(
		"(normal|otsu) [thresh] | (adapt-mean|adapt-gauss) [neighbors]\n",
		" ", "thresh: 0-255, default 0\n",
		" ", "neighbors (odd numbers only): 3, 5, 7, 9,.., default 11")
					},
	"canny"     : { "handler": hdr_canny, "desc": "Canny" },
	"inverse"   : { "handler": hdr_inverse_colors, "desc": "Description" },
	"equalizer" : { "handler": hdr_equalizer, "desc": "Equalizer" },
	"noise"     : { "handler": hdr_add_noise, "desc": "Description" },
	"contours"  : { "handler": hdr_contours, "desc": "Contours" },
	"erode"     : { "handler": hdr_erosion, "desc": "Description" },
	"dilate"    : { "handler": hdr_dilation, "desc": "Description" },
	"grid"      : { "handler": plugin_grid, "desc": "Description" },
	"resize"    : { "handler": hdr_resize, "desc": "Post resize frames" },
	"rotate"    : { "handler": hdr_rotate, "desc": "Description" },
	"faces"     : { "handler": hdr_face_detection, "desc": "Description" },
	"thermo"    : { "handler": hdr_thermo_detection, "desc": "Description" },
	"numbers"   : { "handler": hdr_num_detection, "desc": "Description" },
	"cmt"       : { "handler": hdr_tracker_cmt, "desc": "Description" },
	"tracker"   : { "handler": hdr_tracker, "desc": "Description" },
	"roi"       : { "handler": hdr_roi_select, "desc": "Description" },
	"harris"    : { "handler": hdr_harris_corner, "desc": "Description" },
	"harris-sp" : { "handler": hdr_harris_corner_subpixel, "desc": "Description" },
	"good-feat" : { "handler": hdr_good_features_to_track, "desc": "Description" },
	"deskew"    : { "handler": hdr_deskew, "desc": "Description" },
	"patch"     : { "handler": hdr_patch, "desc": "Description" },
	"none"      : { "handler": "remove", "desc": "{}{:<17}{}".format(
		"Remove all filters.\n",
		" ", "If none is given after a filter name, it removes this filter only")
					},
	"load"      : { "handler": "load", "desc": "Description" },
	"save"      : { "handler": "save", "desc": "Description" },
	"list"      : { "handler": "list", "desc": "List active filters" },
	"help"      : { "handler": "help", "desc": "Description"}
}

