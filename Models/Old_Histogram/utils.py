import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from sklearn.utils import shuffle

##############################################################################
# GENERATE BATCH
def batch_generator(X, y, batch_size):
    (X, y) = shuffle(X, y)
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
        batch_gen = batch_generator(X_data, y_data, batch_size=batch_size)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'fc_keep_prob:0': dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss

        avg_loss_plot.append(avg_loss)
        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss), end=' ')

        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0], 'tf_y:0': validation_set[1], 'fc_keep_prob:0':1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            val_accuracy_plot.append(valid_acc)
            print(' Validation Acc: %7.3f' % valid_acc, end=' ')
        else:
            print()

        if test_set is not None:
            feed = {'tf_x:0': test_set[0], 'tf_y:0': test_set[1], 'fc_keep_prob:0':1.0}
            test_acc = sess.run('accuracy:0', feed_dict=feed)
            test_accuracy_plot.append(test_acc)
            print(' Test Acc: %7.3f' % test_acc)
        else:
            print()

    return avg_loss_plot, val_accuracy_plot, test_accuracy_plot

def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test, 'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
