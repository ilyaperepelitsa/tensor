import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from hope_reshev.generic import weight_variable, bias_variable, conv2d, max_pool_2x2, conv_layer, full_layer

DATA_PATH = "/Users/ilyaperepelitsa/Downloads/cifar-10-batches-py/"

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i: self._i + batch_size], self.labels[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i)
                        for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                                        for i in range(size)])
    plt.imshow(im)
    plt.show()


def one_hot(vec, vals = 10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

d = CifarDataManager()
print("Number of train images: {}".format(len(d.train.images)))
print("Number of train labels: {}".format(len(d.train.labels)))
print("Number of test images: {}".format(len(d.test.images)))
print("Number of test labels: {}".format(len(d.test.labels)))
images = d.train.images
display_cifar(images, 10)



cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)


conv1 = conv_layer(x, shape = [5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)


conv2 = conv_layer(conv1_pool, shape = [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob = keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))

train_step = tr.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc= np.mean([sess.run(accuracy, feed_dict = {x: X[i], y_: Y[i], keep_prob : 1.0})
                        for i in range(10)])
    print("Accuracy: {:.4}%".format(acc * 100))
