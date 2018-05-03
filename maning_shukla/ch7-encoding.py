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
                            dtype = tf.float32))

    def train(self, data):

    def test(self, data):
