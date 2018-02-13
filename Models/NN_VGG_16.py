import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
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
##################################################
    # 1st layer: Conv_1
    h1 = conv_layer(tf_x, name='conv_1',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=64)
    # 2nd layer: Conv_2
    h2 = conv_layer(h1, name='conv_2',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=64)
    # MaxPooling
    h1_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
##################################################
    # 3rd layer: Conv_3
    h3 = conv_layer(h1_pool, name='conv_3',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=128)
    # 4th layer: Conv_4
    h4 = conv_layer(h3, name='conv_4',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=128)
    # MaxPooling
    h2_pool = tf.nn.max_pool(h4,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
##################################################
    # 5th layer: Conv_5
    h5 = conv_layer(h2_pool, name='conv_5',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=256)
    # 6th layer: Conv_6
    h6 = conv_layer(h5, name='conv_6',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=256)
    # 7th layer: Conv_7
    h7 = conv_layer(h6, name='conv_7',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=256)
    # MaxPooling
    h3_pool = tf.nn.max_pool(h7,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
##################################################
    # 8th layer: Conv_8
    h8 = conv_layer(h3_pool, name='conv_8',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=512)
    # 10th layer: Conv_10
    h9 = conv_layer(h8, name='conv_9',
                    kernel_size=(3, 3),
                    padding_mode='SAME',
                    n_output_channels=512)
    # 11th layer: Conv_11
    h10 = conv_layer(h9, name='conv_10',
                     kernel_size=(3, 3),
                     padding_mode='SAME',
                     n_output_channels=512)
    # MaxPooling
    h4_pool = tf.nn.max_pool(h10,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
##################################################
    # 11th layer: Conv_11
    h11 = conv_layer(h4_pool, name='conv_11',
                     kernel_size=(3, 3),
                     padding_mode='SAME',
                     n_output_channels=512)
    # 12th layer: Conv_12
    h12 = conv_layer(h11, name='conv_12',
                     kernel_size=(3, 3),
                     padding_mode='SAME',
                     n_output_channels=512)
    # 13th layer: Conv_13
    h13 = conv_layer(h12, name='conv_13',
                     kernel_size=(3, 3),
                     padding_mode='SAME',
                     n_output_channels=512)
    # MaxPooling
    h5_pool = tf.nn.max_pool(h13,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
##################################################
    # 14th layer: FulCon_1
    h14 = fc_layer(h5_pool, name='fc_14',
                   n_output_units=4096,
                   activation_fn=tf.nn.relu)
    # 15th layer: FulCon_2
    h15 = fc_layer(h14, name='fc_15',
                   n_output_units=4096,
                   activation_fn=tf.nn.relu)
    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h_drop = tf.nn.dropout(h15, keep_prob=keep_prob, name='dropout_layer')

    # 16th layer: FulCon_3 (linear activation)
    h16 = fc_layer(h_drop, name='fc_16',
                   n_output_units=classes,
                   activation_fn=None)

    return h16
