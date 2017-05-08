import tensorflow as tf 
import numpy as np 
from config import sliceSize

class Model():
	def __init__(self, nb_genres):
		self.x = tf.placeholder(tf.float32, [None, sliceSize, sliceSize, 1])
		self.y = tf.placeholder(tf.int32, [None, nb_genres])

		# with tf.variable_scope("conv_max_1"):
		# 	w = tf.get_variable('w',
		# 						shape = [2,2,1,64],
		# 						initializer = tf.contrib.layers.xavier_initializer())
		# 	b = tf.Variable(tf.constant(0.1, shape=[64]), name = 'b')

		# 	conv_1 = tf.nn.conv2d(self.x, w,
		# 							strides = [1,1,1,1],
		# 							padding = 'SAME',
		# 							name = 'conv_1')

		# 	h = tf.nn.relu(tf.nn.bias_add(conv_1,b), name='relu')

		# 	pool_1 = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		conv1 = self.conv_max_layer(self.x, 'conv_max_1', 4,2,2)
		conv2 = self.conv_max_layer(conv1, 'conv_max_2', 16,2,2)
		conv3 = self.conv_max_layer(conv2, 'conv_max_3', 32,2,2)
		conv4 = self.conv_max_layer(conv3, 'conv_max_4', 64,2,2)



		flat = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
		# print flat.get_shape()
		full_1 = self.full_connect_layer(flat, 'full_connect_1', 1024)
		full_2 = tf.nn.dropout(tf.nn.relu(full_1), 0.5)

		full_3 = self.full_connect_layer(full_2, 'full_connect_2', nb_genres)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = full_3))
		self.predictions = tf.argmax(full_3, 1)
		self.corrections = tf.equal(self.predictions, tf.argmax(self.y, 1))
		self.acc = tf.reduce_mean(tf.cast(self.corrections, tf.float32))

		# for v in tf.trainable_variables():
			# print v,v.name
		self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.loss)

	def full_connect_layer(self, input_, name, unites):
		input_shape = input_.get_shape().as_list()
		input_dims = input_shape[-1]
		with tf.variable_scope(name):
			w = tf.get_variable('w', shape=[input_dims, unites], initializer = tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[unites]), name='b')
			return tf.matmul(input_,w)+b


	def conv_max_layer(self,input_, name, nb_filters, filter_size, kernel_size):
		input_shape = input_.get_shape().as_list()
		with tf.variable_scope(name):
			w = tf.get_variable('w',
								shape=[2,2, input_shape[-1], nb_filters],
								initializer = tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[nb_filters]), name='b')

			conv = tf.nn.conv2d(input_, w, strides=[1,1,1,1], padding='SAME')

			h = tf.nn.relu(tf.nn.bias_add(conv,b))

			pool = tf.nn.max_pool(h, ksize = [1,kernel_size, kernel_size,1], strides = [1,2,2,1], padding='SAME')

			return pool


if __name__ == '__main__':
	model = Model(10)
	# print model.accuracy.get_shape()