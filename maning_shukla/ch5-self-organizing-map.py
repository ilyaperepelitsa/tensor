import tensorflow as tf
import numpy as np

class SOM:
    def __init__(self, width, height, dim):
        self.num_iters = 100
        self.width = width
        self.height = height
        self.dim = dim
        self.node_locs = self.get_locs()

        nodes = tf.Variable(tf.random_normal([width * height, dim]))
        self.nodes = nodes

        x = tf.placeholder(tf.float32, [dim])
        iter = tf.placeholder(tf.float32)

        self.x = x
        self.iter = iter

        bmu_loc = self.get_bmu_loc(x)
        self.propagate_nodes = self.get_propagation(bmu_loc, x, iter)
        
