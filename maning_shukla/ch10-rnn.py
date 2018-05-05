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

        self.x = tf.placeholder(tf.float32, [None, seq_size, i])
