
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import os
from tflearn import data_utils
from models.alexnet import AlexNet
import h5py
import numpy as np


h5f = h5py.File('{}/{}'.format( os.getenv("HOME"),'AI/aml_data/fynd/dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

X = np.reshape(X, (400, 128, 128, 3))
print("creating model...")
model = AlexNet().create([None, 128, 128, 3], 4)

# training model
print("training model...")
model.fit(X, Y, n_epoch=1, shuffle=True,
          show_metric=True, batch_size=4, snapshot_step=500,
          snapshot_epoch=False, run_id='alexnet')

# Saving trained model
print("saving model...")
model_name = 'my_model.tflearn'
model_path = "{}/{}".format('trained_models/alexnet', model_name)
if not os.path.exists(model_path):
	os.makedirs(model_path)
model.save(model_path)

# print(model.evaluate (X, Y, batch_size=20))