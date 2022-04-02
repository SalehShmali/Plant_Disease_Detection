# USAGE
# python train.py --dataset dataset --model model.h5 --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
from lib2to3.pgen2.literals import test
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from model.model import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 200
BS = 16
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("___")
	labels.append(l)


data = np.array(data, dtype="float")
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0,rotation_range=25, width_shift_range=0.1,
	                               height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow(trainX, trainY, batch_size=BS)
test_data  =test_datagen.flow(testX, testY,batch_size=BS)

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = Model.build(IMAGE_DIMS, classes=len(mlb.classes_))

def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = tf.clip_by_value(y_pred, 0, 1)
	# calculate elements
	tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axis=1)
	fp = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred - y_true, 0, 1)), axis=1)
	fn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + tf.keras.backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + tf.keras.backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = tf.math.reduce_mean((1 + bb) * (p * r) / (bb * p + r + tf.keras.backend.epsilon()))
	return fbeta_score

# initialize the optimizer (SGD is sufficient)
# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
opt = Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])

# train the network
print("[INFO] training network...")
H = model.fit(train_data,
	          steps_per_epoch=len(train_data) // BS,
	          epochs=EPOCHS,
			  validation_data=test_data,
              validation_steps=len(test_data),
			  verbose=1)

print("[INFO] evaluating the model...")
loss, fbeta = model.evaluate(test_data, steps=len(test_data), verbose=1)
print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["fbeta"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_fbeta"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])