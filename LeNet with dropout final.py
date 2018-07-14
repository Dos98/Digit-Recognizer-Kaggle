# import the packages
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1671)  # for reproducibility

#define the convnet class
class LeNet:
	@staticmethod
	def build(input_shape, classes):
		model = Sequential()
		# CONV => RELU => POOL => DROPOUT
		model.add(Conv2D(32, kernel_size=5, padding="same",
			input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.75))
		# CONV => RELU => POOL => DROPOUT
		model.add(Conv2D(64, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.75))
		# CONV => RELU 
		model.add(Conv2D(128, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(2048))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		model.add(Dense(625))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		# a softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model

# network and training
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2

IMG_ROWS, IMG_COLS = 28, 28 
NB_CLASSES = 10  
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

train = pd.read_csv('../input/train.csv')
X_train = train.ix[:,1:].values.astype('float32') #consider as float
y_train = train.ix[:,0].values.astype('int32')
X_test = pd.read_csv('../input/test.csv').values.astype('float32')


K.set_image_dim_ordering("th")


X_train /= 255 
X_test /= 255  

# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
# X_train = X_train[:, np.newaxis, :, :]
# X_test = X_test[:, np.newaxis, :, :]
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
print(y_train.shape[0], 'train outputs')

print(X_train.shape)


model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])

history = model.fit(X_train, y_train, 
		batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

print("fitting done")

# score = model.evaluate(X_test, y_test, verbose=VERBOSE)
# print("\nTest score:", score[0])
# print('Test accuracy:', score[1])

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# Generate prediction for test set
predictions = model.predict_classes(X_test, verbose=0)
print(predictions)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("lenet_with_dropout_75_percent_mnist_predictions.csv", index=False, header=True)