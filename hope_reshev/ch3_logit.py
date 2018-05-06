import numpy as np
import tensorflow as tf

N = 20000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_data = np.random.randn(N, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
wxb = np.matmul(w_real, x_data.T) + b_real
