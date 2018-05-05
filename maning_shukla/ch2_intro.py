import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "/tmp/data"
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot = True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 19]))
