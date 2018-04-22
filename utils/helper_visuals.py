import cv2
import numpy as np
import skimage.util as sku
from skimage.exposure import rescale_intensity

# takes a PIL image and returns a numpy array. Channels can be 1 grayscale, 3 RGB, 4 RGBA
def Pil2Numpy(img, channels=3):
	timg = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], channels)
	# PIL and openCV color formats are not the same: RGB/BGR
	# We need to swap channels: RGB to BGR
	return cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)


# Reads a PIL image into a new openCV named window and displays it
def showImg(img, wname, bgnd=None):
	lay = img
	# Image display with OpenCV
	cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
	# convert Image to numpy array
	t_img = Pil2Numpy(lay, 3)
	# resize image
	# t_img = cv2.resize(t_img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	
	# display image
	cv2.imshow(wname, t_img)

	
# Overlay gridlines of 'value' with 'stroke' dividing width and height by 'dim'
# 'value' can be any tuplet, based on 'arr' dimensions
def gridlines(arr, dim, stroke, value):
	mat = arr.copy()
	tilerow = arr.shape[0]//dim
	tilecol = arr.shape[1]//dim
	for i in range(tilerow, arr.shape[0], tilerow):
		mat[i:i+stroke,:] = value
	for i in range(tilecol, arr.shape[1], tilecol):
		mat[:,i:i+stroke] = value
	return mat


# Creates a checkerboard mask over the image array
# by extracting indexes of only the dark blocks.
# The caller can use this mask to apply custom values
# in these blocks over the original image
def checkers(arr, dim):
	rows = arr.shape[0]
	cols = arr.shape[1]
	# create a 2D index for the original array
	b = np.arange(rows*cols).reshape(rows, cols)
	# extract odd rows, odd columns
	odd_odd = b[(np.mod(np.arange(rows), dim*2) <  dim)][:, (np.mod(np.arange(cols), dim*2) <  dim)]
	# extract odd rows, even columns
	odd_evn = b[(np.mod(np.arange(rows), dim*2) <  dim)][:, (np.mod(np.arange(cols), dim*2) >=  dim)]
	# extract even rows, odd columns
	evn_odd = b[(np.mod(np.arange(rows), dim*2) >= dim)][:, (np.mod(np.arange(cols), dim*2) < dim)]
	# extract even rows, odd columns
	evn_evn = b[(np.mod(np.arange(rows), dim*2) >= dim)][:, (np.mod(np.arange(cols), dim*2) >= dim)]
	# merge indexes
	b = np.r_[odd_odd.reshape(odd_odd.size), evn_evn.reshape(evn_evn.size)]
	w = np.r_[odd_evn.reshape(odd_evn.size), evn_odd.reshape(evn_odd.size)]
	return b, w


# Overlay a checkerboard of rectangles of size dim x dim
# with values for black and white blocks
def checkerboard(arr, dim, b_val, w_val):
	mat = arr.copy()
	b, w = checkers(arr, dim)
	mat.reshape(mat.shape[0] * mat.shape[1],-1)[b] = b_val
	mat.reshape(mat.shape[0] * mat.shape[1],-1)[w] = w_val
	return mat
	

# Return an array of rectangles of dimension dim covering the image array
def blocks(arr, dim):
	view = sku.view_as_blocks(arr, dim)
	# collapse the first 3 dimensions in one
	return view.reshape(-1, view.shape[3], view.shape[4], view.shape[5])

# Return an array of images as a dim sized rolling window view
# rolling over the image by step
def windows(arr, dim, step=1):
	view = sku.view_as_windows(arr, dim, step)
	# collapse the first 3 dimensions in one
	return view.reshape(-1, view.shape[3], view.shape[4], view.shape[5])

def convolve(image, K):
	# grab the spatial dimensions of the image and kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]
	# print("Filter shape: {}".format(K.shape))
	
	# allocate memory for the output image, taking care to "pad"
	# the orders of the input image so the spatial size (i.e.,
	# width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
	                           cv2.BORDER_REPLICATE)

	# output = np.zeros((iH, iW), dtype="float")
	output = np.zeros((iH, iW, 3), dtype="float")
	
	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top-to-bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# print("ROI shape: {}".format(roi.shape))
			
			# perform the actual convolution by taking the
			# element-wise multiplication between the ROI and
			# the kernel, then summing the matrix

			# bw images
			# k = (roi * K).sum()

			# RGB color images
			k = (np.transpose(roi, (2, 0, 1)) * K).sum(1).sum(1)
			# print("k shape: {}".format(k.shape))

			# store the convolved value in the output (x, y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
	
	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output

def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array should look like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	return (arr.reshape(h // nrows, nrows, -1, ncols)
	        .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def cubify(arr, newshape):
	oldshape = np.array(arr.shape)
	repeats = (oldshape / newshape).astype(int)
	tmpshape = np.column_stack([repeats, newshape]).ravel()
	order = np.arange(len(tmpshape))
	order = np.concatenate([order[::2], order[1::2]])
	# newshape must divide oldshape evenly or else ValueError will be raised
	return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

