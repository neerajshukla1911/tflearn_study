from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


class RNN(object):
	"""RNN model"""
	def __init__(self):
		super(RNN, self).__init__()

	def create(self, input_shape, n_units):
		net = tflearn.input_data(shape=input_shape)
		net = tflearn.lstm(net, 128, return_seq=True)
		net = tflearn.lstm(net, 128)
		net = tflearn.fully_connected(net, n_units, activation='softmax')
		net = tflearn.regression(net, optimizer='adam',
		                         loss='categorical_crossentropy', name="output1")
		model = tflearn.DNN(net, tensorboard_verbose=2)		
		return model