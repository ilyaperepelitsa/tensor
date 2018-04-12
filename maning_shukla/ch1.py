import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

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

####### EXECUTING INTERACTIVE MODE
sess = tf.InteractiveSession()
x = tf.constant([[1, 2]])
neg_op = tf.negative(x)

result = neg_op.eval()
result
sess.close()


###### EXECUTION WITH LOGGED DEVICES

x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)
with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
    result = sess.run(neg_op)
print(result)



######### TESTING THE VARIABLES
sess = tf.InteractiveSession()
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13.]
spike = tf.Variable(False)
spike.initializer.run()


for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        updater = tf.assign(spike, True)
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())

sess.close()

##### VECTORS AND WRITING VARIABLES
 ### NOT WORKING
sess = tf.InteractiveSession()
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13.]
spikes = tf.Variable([False] * len(raw_data), name = "spikes")
spikes.initializer.run()

saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

save_path = saver.save(sess, "spikes.ckpt")
print("spikes rata saved in file: %s" % save_path)

sess.close()



#### DEALING WITH ACTUAL VARIABLES

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg

avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incoming_values", curr_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.add_graph(sess.graph)
    for i in range(len(raw_data)):
        curr_avg = sess.run(update_avg, feed_dict = {curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
