import tensorflow as tf

sess = tf.InteractiveSession()

input_dim = 1
seq_size = 6

input_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, seq_size, input_dim])


def make_cell(state_dim):
    return tf.contrib.rnn.LSTMCell(state_dim)

def make_multi_cell(state_dim, num_layers):
    cells = [make_cell(state_dim) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)

with tf.variable_scope("first_cell") as scope:
    cell = make_cell(state_dim = 10)
    outputs, states = tf.nn.dynamic_rnn(cell, input_placeholder, dtype = tf.float32)

with tf.variable_scope("second_cell") as scope:
    cell2 = make_cell(state_dim = 10)
    outputs2, states2 = tf.nn.dynamic_rnn(cell2, outputs, dtype = tf.float32)

multi_cell = make_multi_cell(state_dim = 10, num_layers = 4)
outputs4, states4 = tf.nn.dynamic_rnn(multi_cell, input_placeholder, dtype = tf.float32)


embeddings_0d = tf.constant([17, 22, 35, 51])
embeddings_4d = tf.constant([[1, 0, 0 , 0],
                             [0, 1, 0 , 0],
                             [0, 0, 1 , 0],
                             [0, 0, 0 , 1],])

embeddings_2x2d = tf.constant([[[1, 0], [0 , 0]],
                                [[0, 1], [0 , 0]],
                                [[0, 0], [1 , 0]],
                                [[0, 0], [0 , 1]],])


ids = tf.constant([1, 0, 2])
lookup_0d = sess.run(tf.nn.embedding_lookup(embeddings_0d, ids))
print(lookup_0d)

lookup_4d = sess.run(tf.nn.embedding_lookup(embeddings_4d, ids))
print(lookup_0d)

lookup_2x2d = sess.run(tf.nn.embedding_lookup(embeddings_2x2d, ids))
print(lookup_0d)


def extract_character_vocab(data):
    special_symbols = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
    set_symbols = set([character for line in data for character in line])
    all_symbols = special_symbols + list(set_symbols)
    int_to_symbols = {word_i : word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_, word in int_to_symbol.items()}

    return int_to_symbol, symbol_to_int

input_setences = ["hello stranger", "bye bye"]
output_sentences = ["hiya", "later"]
