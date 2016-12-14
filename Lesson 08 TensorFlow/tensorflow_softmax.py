import tensorflow as tf

output = None
logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)

# TODO: Calculate the softmax of the logits
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    output = sess.run(softmax, feed_dict={logits: logit_data})

print(output)
