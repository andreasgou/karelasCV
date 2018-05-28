# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# image = cv2.bitwise_not(image)      # inverts colors from image stored in pc (don't know why)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# image = cv2.GaussianBlur(image,(5, 5), 0)
		# ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)