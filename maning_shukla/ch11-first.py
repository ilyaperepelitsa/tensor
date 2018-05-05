import tensorflow as tf


input_dim = 1
seq_size = 6

input_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, seq_size, input_dim])


def make_cell(state_dim):
    return tf.contrib.rnn.LSTMCell(state_dim)

def make_multi_cell(state_dim, num_layers):
    

with tf.variable_scope("first_cell") as scope:
    cell = make_cell(state_dim = 10)
    outputs, states = tf.nn.dynamic_rnn(cell, input_placeholder, dtype = tf.float32)

with tf.variable_scope("second_cell") as scope:
    cell2 = make_cell(state_dim = 10)
    outputs2, states2 = tf.nn.dynamic_rnn(cell2, outputs, dtype = tf.float32)
