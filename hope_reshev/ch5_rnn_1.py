import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

elemenet_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128


LOG_DIR = "logs/RNN_with_summaries"

_inputs = tf.placeholder(tf.float32, shape = [None, time_steps, elemenet_size], name = "inputs")

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
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


def rnn_step(previous_hidden_state, x):
    current_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
    return current_hidden_state

# Current input shape = (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm = [1, 0, 2])
# Current input shape = (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
all_hidden_states = tf.scan(rnn_step, processed_input,
                    initializer = initial_hidden, name = "states")


with tf.name_scope("linear_layer_weights") as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                                mean = 0, stddev = 0.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                                mean = 0, stddev = 0.01))
        variable_summaries(bl)

def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state)

with tf.name_scope("linear_layer_weights") as scope:
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    output = all_outputs[-1]
    tf.summary.histogram("outputs", output)



with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, lebels = y))
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf
