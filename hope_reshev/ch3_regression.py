import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [None, 3])
y_true = tf.placeholder(tf.float32, shape = None)
w = tf.Variable([[0, 0, 0]], dtype = tf.float32, name = "weights")
b = tf.Variable(0, dtype = tf.float32, name = "bias")

y_pred = tf.matmul(w, tf.transpose(x)) + b

loss = tf.reduce_mean(tf.square(y_true - y_pred))
