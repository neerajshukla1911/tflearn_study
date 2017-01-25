
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import os
from tflearn import data_utils
from models.vgg16 import VGG16
import h5py
import numpy as np


h5f = h5py.File('/home/neeraj/AI/aml_data/fynd/dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

X = np.reshape(X, (400, 128, 128, 3))
print("creating model...")
model = VGG16().create([None, 128, 128, 3], 4)

# training model
print("training model...")
model.fit(X, Y, n_epoch=1, shuffle=True,
          show_metric=True, batch_size=4, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_oxflowers17')

# Saving trained model
print("saving model...")
if not os.path.exists('trained_models/vgg16/fynd/'):
	os.makedirs('trained_models/vgg16/fynd/')
model.save('trained_models/vgg16/fynd/my_model.tflearn')

# print(model.evaluate (X, Y, batch_size=20))