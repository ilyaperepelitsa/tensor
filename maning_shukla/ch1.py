import tensorflow as tf
x = tf.constant([[1, 2]])
x
negMatrix = tf.negative(x)
negMatrix
# print(negMatrix)



####### EXECUTING NOW
x = tf.constant([[1, 2]])
neg_op = tf.negative(x)

with tf.Session() as sess:
    result = sess.run(neg_op)
result
