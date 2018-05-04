import pickle
import tensorflow as tf


def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding = "latin1")
    fo.close()
    return dict

import numpy as np

def clean(data):
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.mean(1)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis = 1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis = 1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds
    return normalized


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


import matplotlib.pyplot as plt
import random


def show_some_images(names, data, labels):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap = "Greys_r")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("cifar_examples.png")
    plt.show()

show_some_images(names, data, labels)

W = tf.Variable(tf.random_normal([5, 5, 1, 32]))

def show_weights(W, filename = None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap = "Greys_r", interpolation = "none")
        plt.axis("off")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_val = sess.run(W)
    show_weights(W_val, "step0_weights.png")

def show_conv_results(data, filename = None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap = "Greys_r", interpolation = "none")
        plt.axis("off")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

raw_data = data[4, :]
raw_img = np.reshape(raw_data, (24, 24))
plt.fig
