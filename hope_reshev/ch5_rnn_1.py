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

y = tf.placeholder(tf.float32, shape = [None, num_classes], name = "labels")


batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = batch_x.reshape((batch_size, time_steps, elemenet_size))


def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([elemenet_size, hidden_layer_size]))
        variable_summaries(Wx)

    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)

    with tf.name_scope("Bias"):
        b_rnn
