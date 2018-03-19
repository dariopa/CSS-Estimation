import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
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
    def build_LeNet(classes, x_row, y_col, learning_rate):   
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes,
                                dtype=tf.float32,
                                name='tf_y_onehot')

        print('\nBuilding Neuronal Network...') 
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.01
        
        # SOLUTION: Layer 1: Convolutional. Input = 28x28x1. Output = 24x24x32.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(32))
        conv1   = tf.nn.conv2d(tf_x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 24x24x32. Output = 14x14x32.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Layer 2: Convolutional. Output = 10x10x64.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(64))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        
        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x64. Output = 5x5x64.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Flatten. Input = 5x5x64. Output = 1024.
        fc0 = flatten(conv2)

        # SOLUTION: Layer 3: Fully Connected. Input = 1600 // 179776 // 73984. Output = 120.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(73984, 120), mean = mu, stddev = sigma))
        fc3_b = tf.Variable(tf.zeros(120))
        fc3 = tf.matmul(fc0, fc3_W) + fc3_b

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc4_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
        fc4_b = tf.Variable(tf.zeros(84))
        fc4 = tf.matmul(fc3, fc4_W) + fc4_b

        ## Dropout
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h_drop = tf.nn.dropout(fc4, keep_prob=keep_prob, name='dropout_layer')

        # SOLUTION: Layer 5: Fully Connected. Input = 1024. Output = 10.
        fc5_W = tf.Variable(tf.truncated_normal(shape=(84, classes), mean = mu, stddev = sigma))
        fc5_b = tf.Variable(tf.zeros(classes))
        logits = tf.matmul(h_drop, fc5_W) + fc5_b

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

###############################################################################################

    def build_LeNet_own(classes, x_row, y_col, learning_rate):   
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes,
                                dtype=tf.float32,
                                name='tf_y_onehot')

        print('\nBuilding Neuronal Network...') 
        # 1st layer: Conv_1
        h1 = conv_layer(tf_x, name='conv_1',
                        kernel_size=(5, 5),
                        padding_mode='VALID',
                        n_output_channels=32)
        # MaxPooling
        h1_pool = tf.nn.max_pool(h1,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='VALID')
        
        # 2nd layer: Conv_2
        h2 = conv_layer(h1_pool, name='conv_2',
                        kernel_size=(5, 5),
                        padding_mode='VALID',
                        n_output_channels=64)
        # MaxPooling
        h2_pool = tf.nn.max_pool(h2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='VALID')

        # 3th layer: FulCon_1
        h3 = fc_layer(h2_pool, name='fc_14',
                    n_output_units=120,
                    activation_fn=tf.nn.relu)

        # 4th layer: FulCon_1
        h4 = fc_layer(h3, name='fc',
                    n_output_units=84,
                    activation_fn=tf.nn.relu)

        ## Dropout
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h_drop = tf.nn.dropout(h4, keep_prob=keep_prob, name='dropout_layer')

        # 4th layer: FulCon_1
        logits = fc_layer(h_drop, name='fc5',
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

###############################################################################################
     
    def build_VGG16(classes, x_row, y_col, learning_rate):
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes,
                                dtype=tf.float32,
                                name='tf_y_onehot')


        print('\nBuilding Neuronal Network...')
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
        logits = fc_layer(h_drop, name='fc_16',
                    n_output_units=classes,
                    activation_fn=None)

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

###############################################################################################
     
    def build_VGG19(classes, x_row, y_col, learning_rate):
        # Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes,
                                dtype=tf.float32,
                                name='tf_y_onehot')


        print('\nBuilding Neuronal Network...')
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
        # 14th layer: Conv_14
        h14 = conv_layer(h5_pool, name='conv_14',
                        kernel_size=(3, 3),
                        padding_mode='SAME',
                        n_output_channels=512)
        # 15th layer: Conv_15
        h15 = conv_layer(h14, name='conv_15',
                        kernel_size=(3, 3),
                        padding_mode='SAME',
                        n_output_channels=512)
        # 16th layer: Conv_16
        h16 = conv_layer(h15, name='conv_16',
                        kernel_size=(3, 3),
                        padding_mode='SAME',
                        n_output_channels=512)
        # MaxPooling
        h6_pool = tf.nn.max_pool(h16,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        # 17th layer: FulCon_1
        h17 = fc_layer(h6_pool, name='fc_17',
                       n_output_units=4096,
                       activation_fn=tf.nn.relu)
        # 18th layer: FulCon_2
        h18 = fc_layer(h17, name='fc_18',
                       n_output_units=4096,
                       activation_fn=tf.nn.relu)
        # Dropout
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h_drop = tf.nn.dropout(h18, keep_prob=keep_prob, name='dropout_layer')

        # 19th layer: FulCon_3 (linear activation)
        logits = fc_layer(h_drop, name='fc_19',
                       n_output_units=classes,
                       activation_fn=None)

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

