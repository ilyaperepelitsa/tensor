import tensorflow as tf

init_val = tf.random_normal((1, 5), 0, 1)
var = tf.Variable(init_val, name = "var")
print(var)
