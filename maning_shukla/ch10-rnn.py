import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_do):
