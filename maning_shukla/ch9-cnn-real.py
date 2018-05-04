import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding = "latin1")
    fo.close()
    return dict

names, data, labels =
