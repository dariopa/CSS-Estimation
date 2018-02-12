import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image

##############################################################################
# GENERATE BATCH

def batch_generator(X, y, batch_size):

    row, col, _ = np.array(Image.open(str(X[0]))).shape

    for i in range(0, int(np.floor(len(X) / batch_size)) + 1):
        if i == (int(np.floor(len(X) / batch_size))):
            batch_size = len(X) - (i * batch_size)

        X_send = np.full((batch_size, row, col, 1), 0, dtype = np.uint8)
        
        for k in range(0, batch_size):
            img = np.array(Image.open(str(X[k + i * batch_size])))
            X_send[k, :, :, :] = img[:,:,0:1]                         # NUMERISCHER WERT - ÄNDERN!
        y_send = y[i * batch_size:(i + 1) * batch_size]

        (X_send, y_send) = shuffle(X_send, y_send)

        yield(X_send, y_send)


##############################################################################
# GENERATE BATCH
def batch_generator_V2(X, y, batch_size):
       
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

##############################################################################
# UTILITIES
def save(saver, sess, epoch, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))


def train(sess, epochs, training_set, validation_set, test_set,
          batch_size, initialize=True, dropout=0.5):

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    avg_loss_plot = []
    val_accuracy_plot = []
    test_accuracy_plot = []
    for epoch in range(1, epochs+1):
        avg_loss = []
        batch_gen = batch_generator(X_data, y_data, batch_size=batch_size)
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'fc_keep_prob:0': dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss.append(loss)

        avg_loss_plot.append(np.mean(avg_loss))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, np.mean(avg_loss)), end=' ')

        if validation_set is not None:
            X_data = np.array(validation_set[0])
            y_data = np.array(validation_set[1])
            y_pred = np.full((len(X_data)),0)
            x_row, y_col, _ = np.array(Image.open(str(X_data[0]))).shape
            X = np.full((1, x_row, y_col, 1), 0)

            for i in range(len(X_data)):
                X[0, :, :, :] = np.array(Image.open(str(X_data[i])))[:,:,0:1] # NUMERISCHER WERT - ÄNDERN!
                y_pred[i] = predict(sess, X, return_proba=False)
            valid_acc = 100*np.sum((y_pred == y_data)/len(y_data))
            val_accuracy_plot.append(valid_acc)
            print(' Validation Acc: %7.3f%%' % valid_acc, end=' ')
        else:
            print()

        if test_set is not None:
            X_data = np.array(test_set[0])
            y_data = np.array(test_set[1])
            x_row, y_col, _ = np.array(Image.open(str(X_data[0]))).shape
            X = np.full((1, x_row, y_col, 1), 0)

            for i in range(len(X_data)):
                X[0, :, :, :] = np.array(Image.open(str(X_data[i])))[:,:,0:1] # NUMERISCHER WERT - ÄNDERN!
                y_pred[i] = predict(sess, X, return_proba=False)
            test_acc = 100*np.sum((y_pred == y_data)/len(y_data))
            test_accuracy_plot.append(test_acc)
            print(' Test Acc: %7.3f%%' % test_acc)
        else:
            print()


    return avg_loss_plot, val_accuracy_plot, test_accuracy_plot

def predict(sess, X, return_proba=False):
    feed = {'tf_x:0': X, 'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
