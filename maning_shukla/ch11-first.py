import tensorflow as tf


input_dim = 1
seq_size = 6

input_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, seq_size, input_dim])


def make_cell(state_dim):
    return tf.contrib.rnn.LSTMCell(state_dim)

with tf.variable_scope("first_cell") as scope:
    cell = make_cell(state_dime = 10)
