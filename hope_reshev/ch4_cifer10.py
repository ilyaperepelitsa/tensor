import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape,)
