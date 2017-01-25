from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


class AlexNet(object):
	"""AlexNet model"""
	def __init__(self):
		super(AlexNet, self).__init__()

	def create(self, input_shape, n_units):
		# Building 'AlexNet'
		network = input_data(shape=input_shape)
		network = conv_2d(network, 96, 11, strides=4, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = conv_2d(network, 256, 5, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = conv_2d(network, 384, 3, activation='relu')
		network = conv_2d(network, 384, 3, activation='relu')
		network = conv_2d(network, 256, 3, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = fully_connected(network, 4096, activation='tanh')
		network = dropout(network, 0.5)
		network = fully_connected(network, 4096, activation='tanh')
		network = dropout(network, 0.5)
		network = fully_connected(network, n_units, activation='softmax')
		network = regression(network, optimizer='momentum',
		                     loss='categorical_crossentropy',
		                     learning_rate=0.001)

		# Training
		model = tflearn.DNN(network, checkpoint_path='model_alexnet',
		                    max_checkpoints=1, tensorboard_verbose=2)
		return model