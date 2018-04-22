#!/usr/bin/env python

''' Histogram plotting helper for RGB images and grayscale images. Taken from opencv samples.

Functions : 1) hist_curve : returns histogram of an image drawn as curves
			2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Original version: Abid Rahman 3/14/12 debug Gary Bradski
 Updated version: by Andreas Gounaris 20/4/2018
'''

import cv2
import numpy as np

bins = np.arange(256).reshape(256,1)

def hist_curve(im):
	h = np.zeros((300,256,3))
	if len(im.shape) == 2:
		color = [(255,255,255)]
	elif im.shape[2] == 3:
		color = [ (255,0,0),(0,255,0),(0,0,255) ]
	for ch, col in enumerate(color):
		hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
		cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
		hist=np.int32(np.around(hist_item))
		pts = np.int32(np.column_stack((bins,hist)))
		cv2.polylines(h,[pts],False,col)
	y=np.flipud(h)
	return y

def hist_lines(im):
	h = np.zeros((300,256,3))
	if len(im.shape)!=2:
		print("hist_lines applicable only for grayscale images")
		#print("so converting image to grayscale for representation"
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
	cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
	hist=np.int32(np.around(hist_item))
	for x,y in enumerate(hist):
		cv2.line(h,(x,0),(x,y),(255,255,255))
	y = np.flipud(h)
	return y

def get_histogram_image(im, type):
	if (len(im.shape) == 3):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	else:
		gray = im
	hist = im

	# curve - show histogram for color image in curve mode
	if type == "curve":
		hist = hist_curve(im)

	# lines - show histogram in bin mode
	elif type == "lines":
		hist = hist_lines(gray)

	# equalize - show equalized histogram (always in bin mode)
	elif type == "equalize":
		equ = cv2.equalizeHist(gray)
		hist = hist_lines(equ)

	# curve-gray - show histogram for color image in curve mode
	elif type == "curve-gray":
		hist = hist_curve(gray)

	# normalize - show histogram for a normalized image in curve mode
	elif type == "normalize":
		norm = cv2.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
		hist = hist_lines(norm)

	else:
		print(''' Histogram plotting
		Keymap :
			 curve - show histogram for color image in curve mode
			 lines - show histogram in bin mode
		  equalize - show equalized histogram (always in bin mode)
		curve-gray - show histogram for color image in curve mode
		 normalize - show histogram for a normalized image in curve mode
		''')
	
	return hist
