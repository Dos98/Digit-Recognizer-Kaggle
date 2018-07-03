import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

train = pd.read_csv('/home/ayush/Desktop/all/train.csv')
train_images = train.ix[:,1:].values.astype('float32')
train_labels = train.ix[:,0].values.astype('int32')
# y = data['label']
# # y = np.eye(10)[y,:]
# test = data.drop(['label'], axis=1)

test_images = pd.read_csv('/home/ayush/Desktop/all/test.csv').values.astype('float32')
print(train_images.shape, train_labels.shape, test_images.shape)

# NB_CLASSES = 10
# X_train = X.astype('float32')
# X_test = test.astype('float32')
# y_train = np_utils.to_categorical(y, NB_CLASSES)
train_images = train_images / 255
test_images = test_images / 255
train_labels = to_categorical(train_labels)
print(train_labels[0])
print(train_images.shape, train_labels.shape)
# exit()

model = Sequential()

model.add(Dense(784, input_dim=(28*28), activation='relu')) 
model.add(Dropout(0.20))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(10, activation='softmax'))
model.summary()

opt = Adam()
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
n_epochs = 10
batch_size = 100
history=model.fit(train_images, train_labels, validation_split = 0.1, 
            epochs=15, batch_size=64)
print("fit done")


# Graphing Loss on the left and Accuracy on the right
history_dict = history.history

epochs = range(1, 16)

plt.rcParams["figure.figsize"] = [10,5]
plt.subplot(121)

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'ro')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(122)

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'ro')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

# Generate prediction for test set
predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("mnist_predictions.csv", index=False, header=True)