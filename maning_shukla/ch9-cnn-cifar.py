import pickle

def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding = "latin1")
    fo.close()
    return dict

import numpy as np

def clean(data):
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.mean()
