import tensorflow as tf


sess = tf.InteractiveSession()
a = tf.constant([[1, 2, 3],
                [4, 5, 6]])

print(a.get_shape())
a.eval()

x = tf.constant([1, 0, 1])
print(x.get_shape())
x.eval()
x = tf.expand_dims(x, 1)
x.eval()
print(x.get_shape())



b = tf.matmul(a, x)
b.eval()
