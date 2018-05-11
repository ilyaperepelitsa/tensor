import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

elemenet_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128


LOG_DIR = "logs/RNN_with_summaries"

_inputs = tf.placeholder(tf.float32, shape = [None, time_steps, elemenet_size],
                                        name = "inputs")

y = tf.
