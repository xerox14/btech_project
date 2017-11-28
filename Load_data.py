import argparse
import cPickle
import glob
import cv2
import numpy as np
import random
import os 
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pickle
import h5py

K.set_image_dim_ordering('th')



def get_image_path(path):

    images_path = [os.path.join(root, name)
                   for root, dirs, files in os.walk(path)
                   for name in files
                   if name.endswith((".jpeg"))]
    return images_path

def get_image_data(image_path):
    im = cv2.imread(imagePath)
    
    im = cv2.resize(im, (32, 32)).astype(np.float32)
    
    
    im = im.transpose((2,0,1))
    #im = np.expand_dims(im, axis=0)
    
    return im

def get_image_labels(file_path):
	data = pd.read_csv(file_path)
	map_path_label = {}

	for i,j in zip(data['images'], data['labels']):
		map_path_label[i]=j

	return map_path_label

dataset_arg = "/Users/Apple/Documents/MATLAB/train/"
labels_arg = "/Users/Apple/Desktop/a/label_data.csv"
images_path = get_image_path(dataset_arg)

print "Total images: ", len(images_path)

map_path_label = get_image_labels(labels_arg)
print map_path_label

#prepare data to fine tune model
images_data = []
images_path_data = []
labels=[]
for imagePath in images_path:
    im = get_image_data(imagePath)
    images_data.append(im)
    images_path_data.append((imagePath, im))
    imagePath = imagePath.split('/')[-1]
    labels.append(map_path_label[imagePath])

X = np.array(images_data)
y = np.array(labels)
y = y.reshape(len(labels),1)

print X.shape, y.shape



# normalize inputs from 0-255 to 0.0-1.0
X = X.astype('float32')
X = X / 255.0


# one hot encode outputs
print y
print y.shape
y=np_utils.to_categorical(y)
print y.shape
print y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print y_test.shape
num_classes = y_test.shape[1]
print num_classes
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

def save_model(model_name):
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(model_name + ".h5")


save_model('model')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print model.predict(X_test)	
print("Accuracy: %.2f%%" % (scores[1]*100))





