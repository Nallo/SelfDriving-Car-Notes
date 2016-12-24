import numpy as np
import tensorflow as tf

# Output depth, the number of features in the convolution output
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image with 3 color channels
input = tf.placeholder(
    tf.float32,
    shape=[None, image_width, image_height, color_channels])

# Weight: size = filter_w * filter_h * colors * depth
weight = tf.Variable(tf.truncated_normal(
    [filter_size_width, filter_size_height, color_channels, k_output]))

# And bias: size = depth
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
# The strides parameter tells Tensorflow the stride to adopt in each direction
# that are batch, input_w, input_h, input_channels.
# We usually set batch and input_channels strides to be 1 and play around
# with the strinde on width and height directions.
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)

# Appliy Max Pooling to the conv_layer with the same sizes used in the strides
conv_layer = tf.nn.max_pool(conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
