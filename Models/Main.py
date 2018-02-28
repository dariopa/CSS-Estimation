import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from utils import train, predict, save, load
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If a operation is not define it the default device, let it execute in another.


##############################################################################
# Folder Path
# call_folder = '/scratch_net/biwidl102/dariopa/Data_32_32/'
call_folder = '/scratch_net/biwidl102/dariopa/Data_224_224/'

# call_folder = '/home/dario/Documents/SemThes_Local/Data_32_32/'
# call_folder = '/home/dario/Documents/SemThes_Local/Data_224_224/'

store_folder = './model_r_alpha/' 
Name = 'r_alpha'

# Select Net:
from NN_VGG_16 import CNN
# from NN_VGG_19 import CNN
# from NN_LeNet import CNN
# from NN_Basis import CNN

# Define hyperparameters
rate = 0.001
batch_size = 32
epochs = 15

# Classes
classes = 10

np.random.seed(123)
##############################################################################
# IMPORT DATA

X_train = np.load(call_folder + 'X_train.npy')
X_valid = np.load(call_folder + 'X_validation.npy')
X_test = np.load(call_folder + 'X_test.npy')
y_train = np.load(call_folder + 'y_binned_train.npy')[:, 0, 0]
y_valid = np.load(call_folder + 'y_binned_validation.npy')[:, 0, 0]
y_test = np.load(call_folder + 'y_binned_test.npy')[:, 0, 0]

img = np.asarray(Image.open(X_train[0]), dtype=np.uint8)
print(img.shape)
x_row, y_col,_ = img.shape
del img

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

##############################################################################
# GRAPH TRAINING

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    # Placeholders for X and y:
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
    # saver:
    saver = tf.train.Saver()

##############################################################################
# TRAINING
print()
print('Training... ')
with tf.Session(graph=g, config=config) as sess:
    [avg_loss_plot, val_accuracy_plot] = train(sess, epochs=epochs,
                                               training_set=(X_train, y_train),
                                               validation_set=(X_valid, y_valid),
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
    tf.set_random_seed(123)
    # Placeholders for X and y:
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cross_entropy_loss, name='train_op')
    # saver:
    saver = tf.train.Saver()

##############################################################################
# PREDICTION
# create a new session and restore the model

with tf.Session(graph=g2, config=config) as sess:
    load(saver, sess, epoch=epochs, path=store_folder)

    # NO PROBABILITIES
    y_pred = np.full((len(X_test)),0)
    X = np.full((1, x_row, y_col, 1), 0)                              # NUMERISCHER WERT - ÄNDERN!

    for i in range(len(X_test)):
        X[0, :, :, :] = np.array(Image.open(str(X_test[i])))[:,:,0:1] # NUMERISCHER WERT - ÄNDERN!
        y_pred[i] = predict(sess, X, return_proba=False)              # NUMERISCHER WERT - ÄNDERN!
    test_acc = 100*np.sum((y_pred == y_test)/len(y_test))
    print(' Test Acc: %7.3f%%' % test_acc)
    with open(os.path.join(store_folder, Name + '_AccuracyTest.txt'), 'w') as f:
        f.write('%.3f%%' % (test_acc))

    # PROBABILITIES
    np.set_printoptions(precision=3, suppress=True)

    y_pred_proba = np.full((len(X_test), 10),0, dtype=np.float16)     # NUMERISCHER WERT - ÄNDERN!
    X = np.full((1, x_row, y_col, 1), 0, dtype=np.uint8)

    for i in range(len(X_test)):
        X[0, :, :, :] = np.array(Image.open(str(X_test[i])))[:,:,0:1] # NUMERISCHER WERT - ÄNDERN!
        y_pred_proba[i] = predict(sess, X, return_proba=True)
    print(y_pred_proba)
    np.save(os.path.join(store_folder, Name + '_pred_proba.npy'), y_pred_proba)


    np.save(os.path.join(store_folder, Name + '_avg_loss_plot.npy'), avg_loss_plot)
    np.save(os.path.join(store_folder, Name + '_val_accuracy_plot.npy'), val_accuracy_plot)
    np.save(os.path.join(store_folder, Name + '_test_accuracy_plot.npy'), test_accuracy_plot)
