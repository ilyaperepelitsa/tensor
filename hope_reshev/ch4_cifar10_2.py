import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from hope_reshev.generic import weight_variable, bias_variable, conv2d, max_pool_2x2, conv_layer, full_layer


# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
#
# tf.test.gpu_device_name()

DATA_PATH = "/Users/ilyaperepelitsa/Downloads/cifar-10-batches-py/"
STEPS = 3000
BATCH_SIZE = 200

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



C1, C2, C3 = 30, 50, 80
F1 = 500


conv1_1 = conv_layer(x, shape = [3, 3, 3, C1])
conv1_2 = conv_layer(conv1_1, shape = [3, 3, C1, C1])
conv1_3 = conv_layer(conv1_2, shape = [3, 3, C1, C1])
conv
