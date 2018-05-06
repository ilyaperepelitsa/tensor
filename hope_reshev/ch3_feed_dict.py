import numpy as np
import tensorflow as tf

x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32)
