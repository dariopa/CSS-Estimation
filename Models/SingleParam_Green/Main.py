import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from PIL import Image
from utils_training import train, predict, save, load
from utils_NN import NeuralNetworks

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not defined in the default device, let it execute in another.

random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

##############################################################################
# Which dataset to use?
call_folder = '../../Data_150_150_5_classes_sigma/'

# Name of analysed channel
channel = 'green'

# Name of analysed parameter:
parameter = 'sigma'

# Select Net
CNN = NeuralNetworks.build_VGG16

## Define hyperparameters
learning_rate = 1e-4
batch_size = 64
epochs = 40
classes = 5

# In which folder to store images?
store_folder = './model_' + str(channel) + '_' + str(parameter) + '_classes_' + str(classes) + '_' + 'VGG16_150_no_preprocessing/'
##############################################################################

if channel == 'red':
    k = 0
elif channel == 'green':
    k = 1
elif channel == 'blue':
    k = 2
else:
    k = None

if parameter == 'alpha':
    j = 0
elif parameter == 'sigma':
    j = 1
else:
    j = None

if not os.path.isdir(store_folder):
    os.makedirs(store_folder)

with open(os.path.join(store_folder, 'Hyperparameters.csv'), 'w+') as fp:
    line = "Learning_rate:" + "," + str(learning_rate) + "\n"
    fp.write(line)
    line = "Batch_size:" + "," + str(batch_size) + "\n"
    fp.write(line)
    line = "Epochs:" + "," + str(epochs) + "\n"
    fp.write(line)
##############################################################################
# IMPORT DATA

X_train = np.load(call_folder + 'X_train.npy')
X_valid = np.load(call_folder + 'X_validation.npy')
X_test = np.load(call_folder + 'X_test.npy')
y_train = np.load(call_folder + 'y_binned_train.npy')[:, j]
y_valid = np.load(call_folder + 'y_binned_validation.npy')[:, j]
y_test = np.load(call_folder + 'y_binned_test.npy')[:, j]

img = np.asarray(Image.open(X_train[0]), dtype=np.uint8)
print(img.shape)
x_row, y_col,_ = img.shape
del img

print('Training Set:   ', X_train.shape, y_train.shape)
print('Validation Set: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

with open(os.path.join(store_folder, 'Dataset_size.csv'), 'w+') as fp:
    fp.write("Set,X_data,y_data\n")
    line = "Training_Set:" + "," + str(X_train.shape) + "," + str(y_train.shape) + "\n"
    fp.write(line)
    line = "Validation_Set:" + "," + str(X_valid.shape) + "," + str(y_valid.shape) + "\n"
    fp.write(line)
    line = "Test_Set:" + "," + str(X_test.shape) + "," + str(y_test.shape) + "\n"
    fp.write(line)

##############################################################################
# GRAPH TRAINING

## create a graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    CNN(classes, x_row, y_col, learning_rate)

##############################################################################
# TRAINING
print()
print('Training... ')
with tf.Session(graph=g, config=config) as sess:
    [avg_loss_plot, val_accuracy_plot, test_accuracy_plot] = train(path=store_folder, sess=sess,
                                                                   epochs=epochs, channel=k,
                                                                   training_set=(X_train, y_train),
                                                                   validation_set=(X_valid, y_valid),
                                                                   test_set=(X_test, y_test),
                                                                   batch_size=batch_size,
                                                                   initialize=True)

    np.save(os.path.join(store_folder, channel + '_' + parameter + '_avg_loss_plot.npy'), avg_loss_plot)
    np.save(os.path.join(store_folder, channel + '_' + parameter + '_val_accuracy_plot.npy'), val_accuracy_plot)
    np.save(os.path.join(store_folder, channel + '_' + parameter + '_test_accuracy_plot.npy'), test_accuracy_plot)

del g
##############################################################################
# GRAPH PREDICTION

## create a graph
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    CNN(classes, x_row, y_col, learning_rate)

    ## Saver
    saver = tf.train.Saver()
##############################################################################
# PREDICTION
print()
print('Prediction... ')
with tf.Session(graph=g2, config=config) as sess:
    epoch = np.argmax(val_accuracy_plot) + 1
    print(epoch)
    load(saver=saver, sess=sess, epoch=epoch, path=store_folder)

    # LABELS
    y_pred = np.full((len(X_test)), 0)
    X = np.full((1, x_row, y_col, 1), 0.)

    for i in range(len(X_test)):
        X[0, :, :, :] = np.array(Image.open(str(X_test[i])))[:, :, k:(k+1)]
        y_pred[i] = predict(sess, X, return_proba=False)
    test_acc = 100*np.sum((y_pred == y_test)/len(y_test))
    print('Test Acc: %7.3f%%' % test_acc)
    with open(os.path.join(store_folder, channel + '_' + parameter + '_AccuracyTest.txt'), 'w') as fp:
        fp.write('%.3f%%' % (test_acc))

    # PROBABILITIES
    np.set_printoptions(precision=3, suppress=True)

    y_pred_proba = np.full((len(X_test), classes), 0.)
    X = np.full((1, x_row, y_col, 1), 0.)

    for i in range(len(X_test)):
        X[0, :, :, :] = np.array(Image.open(str(X_test[i])))[:, :, k:(k+1)]
        y_pred_proba[i] = predict(sess, X, return_proba=True)
    print(y_pred_proba)
    np.save(os.path.join(store_folder, channel + '_' + parameter + '_pred_proba.npy'), y_pred_proba)
