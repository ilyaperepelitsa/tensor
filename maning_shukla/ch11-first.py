import tensorflow as tf


input_dim = 1
seq_size = 6

input_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, ])
