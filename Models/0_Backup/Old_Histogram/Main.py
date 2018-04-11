import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from utils import train, predict, save, load
from PIL import Image
from Neural_Networks import NeuralNetworks

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not define it the default device, let it execute in another.


##############################################################################
# Folder Path
# call_folder = '/scratch_net/biwidl102/dariopa/Data_32_32/'
# call_folder = '/scratch_net/biwidl102/dariopa/Data_150_150/'
# call_folder = '/scratch_net/biwidl102/dariopa/Data_150_150_5_classes/'
# call_folder = '/scratch_net/biwidl102/dariopa/Data_224_224/'
# call_folder = '/scratch_net/biwidl102/dariopa/Data_224_224_5_classes/'

# call_folder = '/home/dario/Documents/SemThes_Local/Data_32_32/'
call_folder = '/home/dario/Documents/SemThes_Local/Data_150_150/'
# call_folder = '/home/dario/Documents/SemThes_Local/Data_224_224/'

store_folder = './model_histogram/' 
if not os.path.isdir(store_folder):
    os.makedirs(store_folder)
Name = 'r_alpha'

## Define hyperparameters
learning_rate = 1e-4
random_seed = 123
np.random.seed(random_seed)
batch_size = 64
epochs = 150

# Select Net
# CNN = NeuralNetworks.build_LeNet_own
CNN = NeuralNetworks.classification

# Classes
classes = 10

##############################################################################
# IMPORT DATA

X_train = np.load(call_folder + 'X_train.npy')[:, :, 0]
X_valid = np.load(call_folder + 'X_validation.npy')[:, :, 0]
X_test = np.load(call_folder + 'X_test.npy')[:, :, 0]
y_train = np.load(call_folder + 'y_binned_train.npy')[:, 0]
y_valid = np.load(call_folder + 'y_binned_validation.npy')[:, 0]
y_test = np.load(call_folder + 'y_binned_test.npy')[:, 0]

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
    CNN(classes, learning_rate)

##############################################################################
# TRAINING
print()
print('Training... ')
with tf.Session(graph=g, config=config) as sess:
    [avg_loss_plot, val_accuracy_plot, test_accuracy_plot] = train(sess=sess, epochs=epochs,
                                                             training_set=(X_train, y_train),
                                                             validation_set=(X_valid, y_valid),
                                                             test_set=(X_test, y_test),
                                                             batch_size=batch_size,
                                                             initialize=True)
                                                             
    np.save(os.path.join(store_folder, Name + '_avg_loss_plot.npy'), avg_loss_plot)
    np.save(os.path.join(store_folder, Name + '_val_accuracy_plot.npy'), val_accuracy_plot)
    np.save(os.path.join(store_folder, Name + '_test_accuracy_plot.npy'), test_accuracy_plot)
##############################################################################
# PREDICTION

    # LABELS
    y_pred = predict(sess, X_test, return_proba=False)
    test_acc = 100*np.sum((y_pred == y_test)/len(y_test))
    print('Test Accuracy: %.3f%%' % (test_acc))
    with open(os.path.join(store_folder, Name + '_AccuracyTest.txt'), 'w') as f:
        f.write('%.3f%%' % (test_acc))

    # PROBABILITIES
    np.set_printoptions(precision=3, suppress=True)
    y_pred_proba = predict(sess, X_test, return_proba=True)
    print(y_pred_proba)
    np.save(os.path.join(store_folder, Name + '_pred_proba.npy'), y_pred_proba)
