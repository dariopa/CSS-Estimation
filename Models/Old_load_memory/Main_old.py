import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from utils_old import train, predict, save, load


##############################################################################
# Folder Path
# call_folder = '/scratch_net/biwidl102/dariopa/Models/Preprocessed_Data_224_224/'
# call_folder = '/scratch_net/biwidl102/dariopa/Models/Preprocessed_Data_32_32/'
call_folder = 'Preprocessed_32_32/'

store_folder = './model_r_alpha/' 
Name = 'r_alpha'

# Select Net:
# from NN_VGG_16 import CNN
# from NN_VGG_19 import CNN
from NN_LeNet import CNN
# from NN_Basis import CNN

# Define hyperparameters
rate = 0.001
batch_size = 32
random_seed = 123
epochs = 2

# Classes
classes = 10

##############################################################################
# IMPORT DATA

X_train = np.load(call_folder + 'X_train.npy')[:, :, :, 0:1]
X_valid = np.load(call_folder + 'X_validation.npy')[:, :, :, 0:1]
X_test = np.load(call_folder + 'X_test.npy')[:, :, :, 0:1]
y_train = np.load(call_folder + 'y_binned_train.npy')[:, 0, 0]
y_valid = np.load(call_folder + 'y_binned_validation.npy')[:, 0, 0]
y_test = np.load(call_folder + 'y_binned_test.npy')[:, 0, 0]

arr = X_train[0]
x_row, y_col,_ = arr.shape

# X_train = np.reshape(X_train, newshape=[-1, x_row * y_col])
# X_valid = np.reshape(X_valid, newshape=[-1, x_row * y_col])
# X_test = np.reshape(X_test, newshape=[-1, x_row * y_col])

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

##############################################################################
# GRAPH TRAINING

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    # Placeholders for X and y:
    # tf_x = tf.placeholder(tf.float32, shape=[None, x_row * y_col], name='tf_x')
    # reshape x to a 4D tensor: [batchsize, width, height, 1]
    # tf_x = tf.reshape(tf_x, shape=[-1, x_row, y_col, 1], name='tf_x_reshaped')
    tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
    # build the graph
    logits = CNN(tf_x, classes)

    # Prediction
    predictions = {
        'probabilities' : tf.nn.softmax(logits, name='probabilities'),
        'labels' : tf.cast(tf.argmax(logits, axis=1), tf.int32, name='labels')
    }

    # One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes, dtype=tf.float32, name='tf_y_onehot')
    # Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y_onehot), name='cross_entropy_loss')

    # Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cross_entropy_loss, name='train_op')

    # Computing the prediction accuracy
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    # saver:
    saver = tf.train.Saver()

##############################################################################
# TRAINING
print()
print('Training... ')
with tf.Session(graph=g) as sess:
    [avg_loss_plot, val_accuracy_plot, test_accuracy_plot] = train(sess, epochs=epochs,
                                                                   training_set=(X_train, y_train),
                                                                   validation_set=(X_valid, y_valid),
                                                                   test_set=(X_test, y_test),
                                                                   batch_size=batch_size,
                                                                   initialize=True)
    
    save(saver, sess, epoch=epochs, path=store_folder)

##############################################################################
# GRAPH PREDICTION
# Calculate prediction accuracy on test set restoring the saved model

del g

# create a new graph and build the model
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    # Placeholders for X and y:
    # tf_x = tf.placeholder(tf.float32, shape=[None,  x_row * y_col], name='tf_x')
    # reshape x to a 4D tensor: [batchsize, width, height, 1]
    # tf_x = tf.reshape(tf_x, shape=[-1, x_row , y_col, 1], name='tf_x_reshaped')
    tf_x = tf.placeholder(tf.float32, shape=[None, x_row, y_col, 1], name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
    # build the graph
    logits = CNN(tf_x, classes)
    # Prediction
    predictions = {
        'probabilities' : tf.nn.softmax(logits, name='probabilities'),
        'labels' : tf.cast(tf.argmax(logits, axis=1), tf.int32, name='labels')
    }
    # One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=classes, dtype=tf.float32, name='tf_y_onehot')
    # Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y_onehot), name='cross_entropy_loss')
    # Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cross_entropy_loss, name='train_op')
    # Computing the prediction accuracy
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
    # saver:
    saver = tf.train.Saver()

##############################################################################
# PREDICTION
# create a new session and restore the model
with tf.Session(graph=g2) as sess:
    load(saver, sess, epoch=epochs, path=store_folder)

    y_pred = predict(sess, X_test, return_proba=False)
    # print(predict(sess, X_test, return_proba=False))
    test_acc = 100*np.sum((y_pred == y_test)/len(y_test))
    print('Test Accuracy: %.3f%%' % (test_acc))

    with open(os.path.join(store_folder, Name + '_AccuracyTest.txt'), 'w') as f:
        f.write('%.3f%%' % (test_acc))

    np.set_printoptions(precision=3, suppress=True)
    y_pred_proba = predict(sess, X_test, return_proba=True)
    print(y_pred_proba)
    np.save(os.path.join(store_folder, Name + '_pred_proba.npy'), y_pred_proba)


    np.save(os.path.join(store_folder, Name + '_avg_loss_plot.npy'), avg_loss_plot)
    np.save(os.path.join(store_folder, Name + '_val_accuracy_plot.npy'), val_accuracy_plot)
    np.save(os.path.join(store_folder, Name + '_test_accuracy_plot.npy'), test_accuracy_plot)
