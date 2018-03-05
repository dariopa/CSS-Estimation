import os
import struct
import numpy as np
import scipy.misc
import fnmatch
from PIL import Image

store_folder = '/home/dario/Documents/SemThes_Local/Data_MNIST'

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                                % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist('./', kind='train')
print('Rows: %d,  Columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./', kind='t10k')
print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))


X_data = np.vstack((X_train, X_test))
X_data = np.reshape(X_data, (len(X_data), 28, 28))

y_data = np.concatenate((y_train, y_test))

print('Dataset:   ', X_data.shape, y_data.shape)

batch_counter = 1
for i in range(0,len(X_data)):

    scipy.misc.toimage(X_data[i, :, :], cmin=0, cmax=1).save(os.path.join(store_folder + '/Images/', str(batch_counter) + '.jpg'))
    batch_counter = batch_counter + 1

# STORE PATH OF IMAGES
print()
print('Saving datapath...')

nr_images = len(fnmatch.filter(os.listdir(store_folder + '/Images/'), '*.jpg'))
data = []
for i in range(0, nr_images):
    data.append(os.path.join(store_folder + '/Images/', str(i+1) + '.jpg'))
# print(data)
np.save(store_folder + '/datapath.npy', data)
print('Done!')

np.save(os.path.join(store_folder, 'y_data.npy'), y_data)

##########################################################################

# LOAD DATA
X_data = np.load(store_folder + '/datapath.npy')
y_data = np.load(store_folder + '/y_data.npy')

X_train = X_data[0:50000]
X_validation = X_data[50000:60000]
X_test = X_data[60000:]

y_train = y_data[0:50000]
y_validation = y_data[50000:60000]
y_test = y_data[60000:]

img = np.asarray(Image.open(X_train[0]), dtype=np.uint8)
print(img.shape)
x_row, y_col = img.shape
del img

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_validation.shape, y_validation.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

##############################################################################
# SAVE DATA IN BINARY FORMAT
print()
print('Saving Data...')
np.save(os.path.join(store_folder, 'X_train.npy'), X_train)
np.save(os.path.join(store_folder, 'X_validation.npy'), X_validation)
np.save(os.path.join(store_folder, 'X_test.npy'), X_test)
np.save(os.path.join(store_folder, 'y_binned_train.npy'), y_train)
np.save(os.path.join(store_folder, 'y_binned_validation.npy'), y_validation)
np.save(os.path.join(store_folder, 'y_binned_test.npy'), y_test)

print('Done!')