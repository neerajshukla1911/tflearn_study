from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


class VGG16(object):
	"""VGG16 model"""
	def __init__(self):
		super(VGG16, self).__init__()

	def create(self, input_shape, num_class):
		x = tflearn.input_data(shape=input_shape)
		x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
		x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
		x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

		x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
		x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
		x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

		x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
		x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
		x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
		x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
		x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
		x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
		x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

		x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
		x = tflearn.dropout(x, 0.5, name='dropout1')

		x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
		x = tflearn.dropout(x, 0.5, name='dropout2')
		x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
			restore=False)

		regression = tflearn.regression(x, optimizer='adam',
										loss='categorical_crossentropy',
										learning_rate=0.001, restore=False)

		model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
							max_checkpoints=3, tensorboard_verbose=2,
							tensorboard_dir="./logs")
		return model