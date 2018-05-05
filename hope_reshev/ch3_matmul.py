import tensorflow as tf


sess = tf.InteractiveSession()
a = tf.constant([[1, 2, 3],
                [4, 5, 6]])

print(a.get_shape())

x = tf.constant([1, 0, 1])
print(x.get_shape())
