import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image
import time
from utils_preprocessing import standardize

def save(saver, sess, epoch, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))


def batch_generator(X_train, y_train, batch, i, row, col, channel, loops):
    
    if i == (loops-1):
        batch_size = len(X_train) - (i * batch)
    else:
        batch_size = batch

    X_send = np.full((batch_size, row, col, 1), 0., dtype = np.float32)
    
    for k in range(0, batch_size):
        img = np.array(Image.open(str(X_train[k + i * batch])))[:, :, channel:(channel+1)]
        img = standardize(img)
        X_send[k, :, :, :] = img
    y_send = y_train[i * batch:(i + 1) * batch]

    (X_send, y_send) = shuffle(X_send, y_send)

    return(X_send, y_send)

def train(sess, epochs, channel, training_set, validation_set, test_set, 
          batch_size, initialize=True, dropout=0.5):

    X_data_train = np.array(training_set[0])
    y_data_train = np.array(training_set[1])
    training_loss = []

    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    avg_loss_plot = []
    val_accuracy_plot = []
    test_accuracy_plot = []
    row, col, _ = np.array(Image.open(str(X_data_train[0]))).shape
    
    loops = int(np.floor(len(X_data_train) / batch_size)) + 1
    print('Loops: ', loops)
    for epoch in range(1, epochs+1):
        avg_loss = 0.0   
        start_time = time.time()
        for i in range(0, loops):
            batch_x, batch_y = batch_generator(X_data_train, y_data_train, batch=batch_size, i=i, row=row, col=col, channel=channel, loops=loops)
            feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'fc_keep_prob:0': dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss

        avg_loss_plot.append(np.mean(avg_loss))
        print('\nEpoch %02d Training Avg. Loss: %7.4f' % (epoch, avg_loss), end=' ')
        del batch_x, batch_y

        if validation_set is not None:
            X_data = np.array(validation_set[0])
            y_data = np.array(validation_set[1])
            y_pred = np.full((len(X_data)), 0)
            x_row, y_col, _ = np.array(Image.open(str(X_data[0]))).shape
            X = np.full((1, x_row, y_col, 1), 0.)

            for i in range(len(X_data)):
                X[0, :, :, :] = np.array(Image.open(str(X_data[i])))[:, :, channel:(channel+1)]
                X = standardize(X)
                y_pred[i] = predict(sess, X, return_proba=False)
            valid_acc = 100*np.sum((y_pred == y_data)/len(y_data))
            val_accuracy_plot.append(valid_acc)
            print(' Validation Acc: %7.3f%%' % valid_acc, end=' ')
        else:
            print()

        if test_set is not None:
            X_data = np.array(test_set[0])
            y_data = np.array(test_set[1])
            y_pred = np.full((len(X_data)), 0)
            x_row, y_col, _ = np.array(Image.open(str(X_data[0]))).shape
            X = np.full((1, x_row, y_col, 1), 0.)

            for i in range(len(X_data)):
                X[0, :, :, :] = np.array(Image.open(str(X_data[i])))[:, :, channel:(channel+1)]
                X = standardize(X)
                y_pred[i] = predict(sess, X, return_proba=False)
            test_acc = 100*np.sum((y_pred == y_data)/len(y_data))
            test_accuracy_plot.append(test_acc)
            print(' Test Acc: %7.3f%%' % test_acc)
        else:
            print()
            
        end_time = time.time()
        print("Total time taken this loop [s]: ", end_time - start_time)
        if epoch == 2:
            print('Termination time will be:  ', time.ctime(start_time + (end_time - start_time)*epochs))

    return avg_loss_plot, val_accuracy_plot, test_accuracy_plot

def predict(sess, X, return_proba=False):
    feed = {'tf_x:0': X, 'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
