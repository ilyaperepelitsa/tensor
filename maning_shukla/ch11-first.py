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
    int_to_symbol = {word_i : word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_i, word in int_to_symbol.items()}

    return int_to_symbol, symbol_to_int

input_sentences = ["hello stranger", "bye bye"]
output_sentences = ["hiya", "later alligator"]

input_int_to_symbol, input_symbol_to_int = extract_character_vocab(input_sentences)
output_int_to_symbol, output_symbol_to_int = extract_character_vocab(output_sentences)

NUM_EPOCS = 300
RNN_STATE_DIM = 512
RNN_NUM_LAYERS = 2
ENCODER_EMBEDDING_DIM = DECODER_EMBEDDING_DIM = 64

BATCH_SIZE = int(32)
LEARNING_RATE = 0.0003

INPUT_NUM_VOCAB = len(input_symbol_to_int)
OUTPUT_NUM_VOCAB = len(output_symbol_to_int)


# Encoder placeholders
encoder_input_seq = tf.placeholder(
    tf.int32,
    [None, None],
    name = "encoder_input_seq")

encoder_seq_len = tf.placeholder(
    tf.int32,
    (None,),
    name = "encoder_seq_len")

# Decoder placeholders
decoder_output_seq = tf.placeholder(
    tf.int32,
    [None, None],
    name = "decoder_output_seq")

decoder_seq_len = tf.placeholder(
    tf.int32,
    (None,),
    name = "decoder_seq_len")


def make_cell(state_dim):
    lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
    return tf.contrib.rnn.LSTMCell(state_dim, initializer = lstm_initializer)

def make_multi_cell(state_dim, num_layers):
    cells = [make_cell(state_dim) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)


#Encoder embedding
encoder_input_embedded = tf.contrib.layers.embed_sequence(
    encoder_input_seq,
    INPUT_NUM_VOCAB,
    ENCODER_EMBEDDING_DIM)

#Encoder output
encoder_multi_cell = make_multi_cell(RNN_STATE_DIM, RNN_NUM_LAYERS)

encoder_output, encoder_state = tf.nn.dynamic_rnn(
    encoder_multi_cell,
    encoder_input_embedded,
    sequence_length = encoder_seq_len,
    dtype = tf.float32)

del(encoder_output)

decoder_raw_seq = decoder_output_seq[:, :-1]
go_prefixes = tf.fill([BATCH_SIZE, 1], output_symbol_to_int["<GO>"])
decoder_input_seq = tf.concat([go_prefixes, decoder_raw_seq], 1)


decoder_embedding = tf.Variable(tf.random_uniform([OUTPUT_NUM_VOCAB,
                                                    DECODER_EMBEDDING_DIM]))
decoder_input_
