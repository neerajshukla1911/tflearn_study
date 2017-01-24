import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


class VGG(object):
	"""VGG model"""
	def __init__(self):
		super(VGG, self).__init__()

	def create(self, input_shape, n_units):
		# Building 'VGG Network'
		# network = input_data(shape=[None, 224, 224, 3])
		# network = input_data(shape=[None, 128, 128, 3])
		network = input_data(shape = input_shape)	
		network = conv_2d(network, 64, 3, activation='relu')
		network = conv_2d(network, 64, 3, activation='relu')
		network = max_pool_2d(network, 2, strides=2)

		network = conv_2d(network, 128, 3, activation='relu')
		network = conv_2d(network, 128, 3, activation='relu')
		network = max_pool_2d(network, 2, strides=2)

		network = conv_2d(network, 256, 3, activation='relu')
		network = conv_2d(network, 256, 3, activation='relu')
		network = conv_2d(network, 256, 3, activation='relu')
		network = max_pool_2d(network, 2, strides=2)

		network = conv_2d(network, 512, 3, activation='relu')
		network = conv_2d(network, 512, 3, activation='relu')
		network = conv_2d(network, 512, 3, activation='relu')
		network = max_pool_2d(network, 2, strides=2)

		network = conv_2d(network, 512, 3, activation='relu')
		network = conv_2d(network, 512, 3, activation='relu')
		network = conv_2d(network, 512, 3, activation='relu')
		network = max_pool_2d(network, 2, strides=2)

		network = fully_connected(network, 4096, activation='relu')
		network = dropout(network, 0.5)
		network = fully_connected(network, 4096, activation='relu')
		network = dropout(network, 0.5)
		network = fully_connected(network, n_units, activation='softmax')

		network = regression(network, optimizer='rmsprop',
		                     loss='categorical_crossentropy',
		                     learning_rate=0.001)

		model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
		return model