import tensorflow as tf

n_features = 120
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    print(weights.eval())
    print("weights shape:", weights.get_shape())

    print(bias.eval())
    print("bias shape", bias.get_shape())
