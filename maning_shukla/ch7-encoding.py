import tensorflow as tf
import numpy as np


class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch = 250, learning_rate = 0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate

        x = tf.placeholder(dtype = tf.float32, shape = [None, input_dim])

        with tf.name_scope("encode"):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim],
                            dtype = tf.float32), name = "weights")
            biases = tf.Variable(tf.zeros([hidden_dim]), name = "biases")
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)

        with tf.name_scope("decode"):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim],
                            dtype = tf.float32), name = "weights")
            biases = tf.Variable(tf.zeros([input_dim]), name = "biases")
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self, data):
        num_samples = len(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(s)
    def test(self, data):
