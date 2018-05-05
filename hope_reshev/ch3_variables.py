import tensorflow as tf

init_val = tf.random_normal((1, 5), 0, 1)
var = tf.Variable(init_val, name = "var")
print(var)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # sess.run(init_val.eval)
    post_var = sess.run(var)
    print(var.eval())
post_var
