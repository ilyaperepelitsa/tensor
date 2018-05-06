import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [None, 3])
y_true = tf.placeholder(tf.float32, shape = None)
w = tf.Variable([[0, 0, 0]], dtype = )
