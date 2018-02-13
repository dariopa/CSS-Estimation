import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# IMPLEMENTATION OF NN

def CNN(x, classes):   
    print()
    print('Building Neuronal Network...') 
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.01
    
    # SOLUTION: Layer 1: Convolutional. Input = 28x28x1. Output = 24x24x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

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

    # SOLUTION: Layer 3: Fully Connected. Input = 1600. Output = 120.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(1600, 120), mean = mu, stddev = sigma))
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
    
    return logits
