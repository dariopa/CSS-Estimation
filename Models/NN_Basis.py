import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf

# IMPLEMENTATION OF NN

def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        #   get n_input_channels:
        #   input tensor shape:
        #   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = (list(kernel_size) + [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights', shape=weights_shape)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        conv = tf.nn.relu(conv, name='activation')
        print(conv)
        return conv

def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights', shape=weights_shape)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        layer = tf.matmul(input_tensor, weights)
        layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer


def CNN(tf_x, classes):
    print()
    print('Building Neuronal Network...') 
    # 1st layer: Conv_1
    h1 = conv_layer(tf_x, name='conv_1',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=32)
    # MaxPooling
    h1_pool = tf.nn.max_pool(h1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
    # 2n layer: Conv_2
    h2 = conv_layer(h1_pool, name='conv_2',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=64)
    # MaxPooling
    h2_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    # 3rd layer: Fully Connected
    h3 = fc_layer(h2_pool, name='fc_3',
                  n_output_units=1024,
                  activation_fn=tf.nn.relu)

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')

    # 4th layer: Fully Connected (linear activation)
    h4 = fc_layer(h3_drop, name='fc_4',
                  n_output_units=classes,
                  activation_fn=None)

    return h4
