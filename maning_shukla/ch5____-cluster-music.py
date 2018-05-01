import tensorflow as tf
from os import listdir
from os.path import isfile, join


# filenames = tf.train.match_filenames_once('/Users/ilyaperepelitsa/Downloads/*.wav')
filenames = ["/Users/ilyaperepelitsa/Downloads/" + f for f in
                    listdir("/Users/ilyaperepelitsa/Downloads/") if
                    isfile(join("/Users/ilyaperepelitsa/Downloads/", f)) and
                    f.endswith('.wav')]
filenames

count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_files = sess.run(count_num_files)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    for i in range(num_files):
        audio_file = sess.run(filename)
        print(audio_file)

from bregman import *
