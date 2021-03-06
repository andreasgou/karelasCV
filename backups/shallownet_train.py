# USAGE
# python shallownet_train.py --dataset data/basement_3 --model cascades/shallownet_basement_3.hdf5

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import SimplePreprocessor
from utils.datasets import SimpleDatasetLoader
from utils.nn.conv import ShallowNet
from utils.nn.conv import ShallowNetBinary
from keras.optimizers import SGD
from keras.utils import plot_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

dataset = args["dataset"]
model_name = args["model"]

epochs = 10
batch_size = 32

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))

# Open a single image to acquire dimensions
image = cv2.imread(imagePaths[0])
print("[INFO] Image dimensions (rows, cols, depth): {}".format(image.shape))
(iH, iW) = image.shape[:2]

# initialize the image preprocessors
sp = SimplePreprocessor(iW, iH)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels, classes) = sdl.load(imagePaths, verbose=500)
print("[INFO] Dataset dimensions (rows, cols, depth): {}".format(data.shape))
print("[INFO] {} classes found: {}".format(len(classes), classes))
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=iW, height=iH, depth=3, classes=len(classes))

# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=batch_size, epochs=epochs, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(model_name)
# Save model graph to disk
model_graph = "{}.png".format(model_name)
plot_model(model, to_file=model_graph, show_shapes=True)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)

threshold = 0.5
hamm_loss = hamming_loss(testY.argmax(axis=1), predictions.argmax(axis=1) > threshold)
print("Hamming Loss: {}".format(hamm_loss))

# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1) > threshold,
                            target_names=classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("{}.train.png".format(model_name))
plt.show()
