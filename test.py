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

dataset_arg = "/Users/Apple/Documents/MATLAB/train"

images_path = get_image_path(dataset_arg)

print "Total images: ", len(images_path)

map_label_name={
    0:'FELDSPAR',
    1:'QUARTZ',
}
# fix 
#prepare data to fine tune model
images_data = []
images_path_data = []
labels=[]
for imagePath in images_path:
    im = get_image_data(imagePath)
    images_data.append(im)
    images_path_data.append((imagePath, im))
    imagePath = imagePath.split('/')[-1]
    

X_test = np.array(images_data)

# normalize inputs from 0-255 to 0.0-1.0
X_test = X_test.astype('float32')
X_test = X_test/ 255.0

from keras.models import model_from_json
def load_model(model_name):
    # load json and create model
    json_file = open( model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_name + ".h5")
    return model

model = load_model('model')

print X_test.shape

output  = model.predict(X_test)
output_label_name=[]
for row in output:
    row = list(row)
    idx = row.index(max(row))
    
    output_label_name.append(map_label_name[idx])
print output_label_name





