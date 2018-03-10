import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# IMPLEMENTATION OF NN

def conv_layer(input_tensor, name,
               kernel_size, n_output_channels, 
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape: 
        ##   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1] 

        weights_shape = (list(kernel_size) + 
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        conv = tf.nn.conv2d(input=input_tensor, 
                            filter=weights,
                            strides=strides, 
                            padding=padding_mode)
        conv = tf.nn.bias_add(conv, biases, 
                              name='net_pre-activation')
        conv = tf.nn.relu(conv, name='activation')
        print(conv)
        
        return conv

def fc_layer(input_tensor, name, 
             n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, 
                                      shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_units]))
        layer = tf.matmul(input_tensor, weights)
        layer = tf.nn.bias_add(layer, biases,
                              name='net_pre-activation')
        if activation_fn is None:
            return layer
        
        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer

######################################################################################

class NeuralNetworks():
    def classification(classes, learning_rate):   
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, 255], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes,
                                dtype=tf.float32,
                                name='tf_y_onehot')

        print('Building Neuronal Network...') 

        # 3th layer: FulCon_1
        h1 = fc_layer(tf_x, name='fc_1',
                    n_output_units=120,
                    activation_fn=tf.nn.relu)

        # 4th layer: FulCon_1
        h2 = fc_layer(h1, name='fc_2',
                    n_output_units=84,
                    activation_fn=tf.nn.relu)

        ## Dropout
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h_drop = tf.nn.dropout(h2, keep_prob=keep_prob, name='dropout_layer')

        # 4th layer: FulCon_1
        logits = fc_layer(h_drop, name='fc_3',
                    n_output_units=classes,
                    activation_fn=tf.nn.relu)

        ## Prediction
        predictions = {
            'probabilities' : tf.nn.softmax(logits, name='probabilities'),
            'labels' : tf.cast(tf.argmax(logits, axis=1), tf.int32, name='labels')
        }
        
        ## Loss Function and Optimization
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y_onehot), name='cross_entropy_loss')

        ## Optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

        ## Computing the prediction accuracy
        correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

