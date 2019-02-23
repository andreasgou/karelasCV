# import the necessary packages
import cv2
from numpy import newaxis

class InvertBinarize:
	def __init__(self, width=64, height=64):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
	
	def preprocess(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# image = cv2.bitwise_not(image)      # invert colors
		image = image[:, :, newaxis]
		# image = cv2.GaussianBlur(image, (self.width, self.height), 0)
		# ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		return image
