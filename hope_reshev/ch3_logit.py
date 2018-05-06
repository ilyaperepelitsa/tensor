import numpy as np
import tensorflow as tf

N = 20000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_data = np.random.rand
