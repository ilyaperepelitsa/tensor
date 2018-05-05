import numpy as np
import tensorflow as tf
c = tf.constant([[1, 2, 3],
                  [4, 5, 6]])

print("Python list input: {}".format(c.get_shape()))


c = tf.constant(np.array([[[1, 2, 3],
                          [4, 5, 6]],

                          [[1, 1, 1],
                           [2, 2, 2]]]))
print("3D NumPy ")
