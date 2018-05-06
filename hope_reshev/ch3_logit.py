import numpy as np
import tensorflow as tf

N = 20000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_data = np.random.randn(N, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data_pre_noise
y_data = np.random.binomial(1, y_data_pre_noise)
y_data

import tensorflow as tf
NUM_STEPS = 10000

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32, shape = [None, 3])
    y_true = tf.placeholder(tf.float32, shape = None)

    with tf.name_scope("inference") as scope:
        w = tf.Variable([[0, 0, 0]], dtype = tf.float32, name = "weights")
        b = tf.Variable(0, dtype = tf.float32, name = "bias")
        y_pred = tf.matmul(w, tf.transpose(x)) + b
        y_pred = tf.sigmoid(y_pred)

    with tf.name_scope("loss") as scope:
        # loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # THIS
        loss = y_true * tf.log(y_pred) - (1 - y_true) * tf.log(1 - y_pred)
        # IS THE SAME AS THIS
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope("train") as scope:
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if (step % 1000 == 0):
                print(step, sess.run([w, b]))
                wb_.append(sess.run([w, b]))

        print(100000, sess.run([w, b]))
