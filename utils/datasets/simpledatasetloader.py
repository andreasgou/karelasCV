# import the necessary packages
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, image_paths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []
		classes = []

		# loop over the input images
		for (i, imagePath) in enumerate(image_paths):
			# load the image as grayscale and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			imagePath = imagePath.replace("\\ ", " ")
			image = cv2.imread(imagePath)
			# required to bypass system failure when local images are used ???
			cv2.imshow("dataset", image)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)
			if label not in classes:
				classes.append(label)

			# show an update every `verbose` images
			if verbose > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

		cv2.destroyWindow("dataset")
		# return a tuple of the data and labels
		return np.array(data), np.array(labels), classes
