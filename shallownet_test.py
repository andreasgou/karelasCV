# USAGE
# python shallownet_load.py --dataset ../datasets/animals --model shallownet_weights.hdf5

# import the necessary packages
from sklearn.exceptions import UndefinedMetricWarning

from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import SimplePreprocessor
from utils.datasets import SimpleDatasetLoader
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# Open a single image to acquire dimensions
image = cv2.imread(imagePaths[0])
print("[INFO] Image dimensions: {}".format(image.shape))
(iH, iW) = image.shape[:2]

# initialize the image preprocessors
sp = SimplePreprocessor(iW, iH)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels, classes) = sdl.load(imagePaths, verbose=100)
data = data.astype("float") / 255.0

classes = ['bg', 'hygrad', 'plug', 'thermo']

labels = LabelBinarizer().fit_transform(labels)

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=8).argmax(axis=1)
try:
	class_report = classification_report(labels.argmax(axis=1), preds, target_names=classes)
except UndefinedMetricWarning:
	pass

print(class_report)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
	# load the example image, draw the prediction, and display it
	# to our screen
	image = cv2.imread(imagePath)
	cv2.putText(image, "{}".format(classes[preds[i]]),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)