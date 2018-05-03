import tensorflow as tf
import numpy as np

def get_batch(X, size):
    a = np.random.choice(len(X), size, replace = False)
    return X[a]

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch = 250, learning_rate = 0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate

        x = tf.placeholder(dtype = tf.float32, shape = [None, input_dim])

        with tf.name_scope("encode"):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim],
                            dtype = tf.float32), name = "weights")
            biases = tf.Variable(tf.zeros([hidden_dim]), name = "biases")
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)

        with tf.name_scope("decode"):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim],
                            dtype = tf.float32), name = "weights")
            biases = tf.Variable(tf.zeros([input_dim]), name = "biases")
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded
        self.batch_size = 10

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self, data, batch_size = 10):
        num_samples = len(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                # for j in range(num_samples):
                for j in range(500):
                    batch_data = get_batch(data, self.batch_size)
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                    # l, _ = sess.run([self.loss, self.train_op],
                    # feed_dict = {self.x: batch_data})
                if i % 10 == 0:
                    print("epoch {0}: loss = {1}".format(i, l))
                    self.saver.save(sess, "./model.ckpt")
            self.saver.save(sess, "./model.ckpt")

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, "./model.ckpt")
            hidden, reconstructed = sess.run([self.encoded, self.decoded],
            feed_dict = {self.x: data})
        print("input", data)
        print("compressed", hidden)
        print("reconstructed", reconstructed)
        return reconstructed

from sklearn import datasets

hidden_dim = 1
data = datasets.load_iris().data
input_dim = len(data[0])
ae = Autoencoder(input_dim, hidden_dim)

ae.train(data)
ae.test([[8, 4, 6, 2]])


# loading own images
from scipy.misc import imread, imresize

gray_image = imread(filepath, True)
small_gray_image = imresize(gray_image, 1./8.)
x = small_gray_image.flatten()


import pickle

def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding = "latin1")
    fo.close()
    return dict



import numpy as np
names = unpickle("/Users/ilyaperepelitsa/Downloads/cifar-10-batches-py/batches.meta")['label_names']
data, labels = [], []
for i in range(1, 6):
    filename = "/Users/ilyaperepelitsa/Downloads/cifar-10-batches-py/data_batch_" + str(i)
    batch_data = unpickle(filename)
    if len(data) > 0:
        data = np.vstack((data, batch_data["data"]))
        labels = np.hstack()
