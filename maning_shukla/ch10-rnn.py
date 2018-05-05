import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim = 10):
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name = "W_out")
        self.b_out = tf.Variable(tf.random_normal([1]), name = "b_out")

        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix if fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype = tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, mse = sess.run([self.train_op, self.cost],
                            feed_dict = {self.x : train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, "model.ckpt")
            print("Model saved to {}".format(save_path))
    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, "./model")
