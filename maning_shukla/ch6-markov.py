import numpy as np
import tensorflow as tf


class HMM(object):
    def __init__(self, initial_prob, trans_prob, obs_prob):
        self.N = np.size(initial_prob)
        self.initial_prob = initial_prob
        self.trans_prob = trans_prob
        self.emission = tf.constant(obs_prob)
        self.viterbi = tf.placeholder(tf.float64)

        assert self.initial_prob.shape == (self.N, 1)
        assert self.trans_prob.shape == (self.N, self.N)
        assert obs_prob.shape[0] == self.N

        self.obs_idx = tf.placeholder(tf.int32)
        self.fwd = tf.placeholder(tf.float64)

    def get_emission(self, obs_idx):
        slice_location = [0, obs_idx]
        num_rows = tf.shape(self.emission)[0]
        slice_shape = [num_rows, 1]
        return tf.slice(self.emission, slice_location, slice_shape)

    def forward_init_op(self):
        obs_prob = self.get_emission(self.obs_idx)
        fwd = tf.multiply(self.initial_prob, obs_prob)
        return fwd

    def forward_op(self):
        transitions = tf.matmul(self.fwd, tf.transpose(self.get_emission(self.obs_idx)))
        weighted_transitions = transitions * self.trans_prob
        fwd = tf.reduce_sum(weighted_transitions, 0)
        return tf.reshape(fwd, tf.shape(self.fwd))

    def decode_op(self):
        transitions = tf.matmul(self.viterbi, tf.transpose(self.get_emissions * self.trans_prob))
        viterbi = tf.reduce_max(weighted_transitions)

def forward_algoritm(sess, hmm, observations):
    fwd = sess.run(hmm.forward_init_op(), feed_dict = {hmm.obs_idx: observations[0]})
    for t in range(1, len(observations)):
        fwd = sess.run(hmm.forward_op(), feed_dict = {hmm.obs_idx: observations[t],
                                                        hmm.fwd: fwd})
    prob = sess.run(tf.reduce_sum(fwd))
    return prob



# states = ("Rainy", "Sunny")
# observations = ("walk", "shop", "clean")
# start_probability = {"Rainy": 0.6, "Sunny": 0.4}
# transition_probability = {
#     "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
#     "Sunny": {"Rainy": 0.4, "Sunny": 0.6}
# }
# emission_probability = {
#     "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
#     "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1}
# }
# if __name__ == "__main__":
initial_prob = np.array([[0.6], [0.4]])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
obs_prob = np.array([[0.1, 0.4, 0.5],
                     [0.6, 0.3, 0.1]])
hmm = HMM(initial_prob = initial_prob, trans_prob = trans_prob, obs_prob = obs_prob)
observations = [0, 1, 1, 2, 1]
with tf.Session() as sess:
    prob = forward_algoritm(sess, hmm, observations)
    print("Probability of observing {} is {}".format(observations, prob))
