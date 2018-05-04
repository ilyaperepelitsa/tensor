import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding = "latin1")
    fo.close()
    return dict

def read_data(directory):
    names = unpickle("{}/batches.meta".format(directory))["label_names"]
    print("names", names)

    data, labels = [], []
    for i in range(1, 6):
        filename = "{}/data_batch_{}".format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data["data"]))
            labels = np.hstack((labels, batch_data["labels"]))
        else:
            data = batch_data["data"]
            labels = batch_data["labels"]

    print(np.shape(data), np.shape(labels))

    data = clean(data)
    data = data.astype(np.float32)
    return names, data, labels

names, data, labels = read_data("/Users/ilyaperepelitsa/Downloads/cifar-10-batches-py")

x = tf.placeholder(tf.float32, [None, 24 * 24])
y = tf.placeholder(tf.float32, [None, len(names)])

W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))


W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))
