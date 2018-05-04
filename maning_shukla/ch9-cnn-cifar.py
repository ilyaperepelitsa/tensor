import pickle

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
    stdsT = stds.res
