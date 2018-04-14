import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.01
training_epochs = 40

trX = np.linspace(-1, 1, 51)

num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0

for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5

plt.scatter(trX, trY)
plt.show()


def model(X, w):
    
