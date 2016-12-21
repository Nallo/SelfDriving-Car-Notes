# Solution is available in the other "solution.py" tab
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# Create 2-layers Model with Dropout
keep_prop = tf.placeholder(tf.float32)

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prop)

# Logits are values without range
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# Software will make the logits values to be a distribution [0.0 - 1.0]
softmax = tf.nn.softmax(logits)

# Declare the initializer
init = tf.initialize_all_variables()

# Run the model
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(softmax, feed_dict={keep_prop: 0.5}))
