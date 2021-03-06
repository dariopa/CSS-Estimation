import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from utils import train, predict, save, load
from PIL import Image
from Neural_Networks import NeuralNetworks

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not define it the default device, let it execute in another.


##############################################################################
# Folder Path
call_folder = '/scratch_net/biwidl102/dariopa/Data_MNIST/'

# call_folder = '/home/dario/Documents/SemThes_Local/Data_MNIST/'

store_folder = './model_MNIST_test/' 
Name = 'MNIST_test'

## Define hyperparameters
learning_rate = 1e-4
random_seed = 123
np.random.seed(random_seed)
batch_size = 64
epochs = 100

# Select Net
CNN = NeuralNetworks.build_LeNet_own
# CNN = NeuralNetworks.build_VGG16

# Classes
classes = 10

##############################################################################
# IMPORT DATA

X_train = np.load(call_folder + 'X_train.npy')
X_valid = np.load(call_folder + 'X_validation.npy')
X_test = np.load(call_folder + 'X_test.npy')
y_train = np.load(call_folder + 'y_binned_train.npy')
y_valid = np.load(call_folder + 'y_binned_validation.npy')
y_test = np.load(call_folder + 'y_binned_test.npy')

img = np.asarray(Image.open(X_train[0]), dtype=np.uint8)
print(img.shape)
x_row, y_col = img.shape
del img

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

##############################################################################
# GRAPH TRAINING

## create a graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    CNN(classes, x_row, y_col, learning_rate)
    ## saver:
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

np.save(os.path.join(store_folder, Name + '_avg_loss_plot.npy'), avg_loss_plot)
np.save(os.path.join(store_folder, Name + '_val_accuracy_plot.npy'), val_accuracy_plot)
##############################################################################
# GRAPH PREDICTION
# Calculate prediction accuracy on test set restoring the saved model

del g

# create a new graph and build the model
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    CNN(classes, x_row, y_col, learning_rate)
    ## saver:
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

    y_pred_proba = np.full((len(X_test), 10), 0, dtype=np.float16)     # NUMERISCHER WERT - ÄNDERN!
    X = np.full((1, x_row, y_col, 1), 0, dtype=np.uint8)

    for i in range(len(X_test)):
        X[0, :, :, :] = np.array(Image.open(str(X_test[i])))[:, :, 0:1] # NUMERISCHER WERT - ÄNDERN!
        y_pred_proba[i] = predict(sess, X, return_proba=True)
    print(y_pred_proba)
    np.save(os.path.join(store_folder, Name + '_pred_proba.npy'), y_pred_proba)
