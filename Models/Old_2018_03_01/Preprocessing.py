import os
import fnmatch
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import numpy as np

##############################################################################
# Folder Path
# call_folder = '/home/dario/Documents/SemThes_Local/Data_32_32'
# call_folder = '/home/dario/Documents/SemThes_Local/Data_224_224'

# call_folder = '/scratch_net/biwidl102/dariopa/Data_32_32'
call_folder = '/scratch_net/biwidl102/dariopa/Data_224_224'

# How much data to use?
use_data = 1

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 7./10
Val_split = 1./10
Test_split = 2./10

##############################################################################
# LOAD DATA
print()
print('Loading Data...')
X_addr = np.load(os.path.join(call_folder, 'datapath.npy'))
y = np.load(os.path.join(call_folder, 'CSS.npy'))
y_binned = np.load(os.path.join(call_folder, 'CSS_binned.npy'))
print('Done!')

##############################################################################
# SHUFFLE DATA
print()
print('Shuffling Data...')
for i in range(1,5):
    (X_addr, y, y_binned) = shuffle(X_addr, y, y_binned)
print('Done!')

##############################################################################
# SPLIT DATA
print()
print('Splitting Data...')
nr_images = int(use_data * len(fnmatch.filter(os.listdir(call_folder + '/Images/'), '*.jpg')))

X_train = X_addr[0:int(Train_split * nr_images)]
y_train = y[0:int(Train_split * nr_images), :, :]
y_binned_train = y_binned[0:int(Train_split * nr_images), :, :]

X_validation = X_addr[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images)]
y_validation = y[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :, :]
y_binned_validation = y_binned[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :, :]

X_test = X_addr[int((Train_split + Val_split) * nr_images):nr_images]
y_test = y[int((Train_split + Val_split) * nr_images):nr_images, :, :]
y_binned_test = y_binned[int((Train_split + Val_split) * nr_images):nr_images, :, :]

print('Shape of X_train: ' + str(X_train.shape))
print('Shape of y_train: ' + str(y_train.shape))
print('Shape of y_binned_train: ' + str(y_binned_train.shape))
print('Shape of X_validation: ' + str(X_validation.shape))
print('Shape of y_validation: ' + str(y_validation.shape))
print('Shape of y_binned_validation: ' + str(y_binned_validation.shape))
print('Shape of X_test: ' + str(X_test.shape))
print('Shape of y_test: ' + str(y_test.shape))
print('Shape of y_binned_test: ' + str(y_binned_test.shape) + '\n')
print('Done!')

##############################################################################
# SAVE DATA IN BINARY FORMAT
print()
print('Saving Data...')
np.save(os.path.join(call_folder, 'X_train.npy'), X_train)
np.save(os.path.join(call_folder, 'X_validation.npy'), X_validation)
np.save(os.path.join(call_folder, 'X_test.npy'), X_test)
np.save(os.path.join(call_folder, 'y_train.npy'), y_train)
np.save(os.path.join(call_folder, 'y_validation.npy'), y_validation)
np.save(os.path.join(call_folder, 'y_test.npy'), y_test)
np.save(os.path.join(call_folder, 'y_binned_train.npy'), y_binned_train)
np.save(os.path.join(call_folder, 'y_binned_validation.npy'), y_binned_validation)
np.save(os.path.join(call_folder, 'y_binned_test.npy'), y_binned_test)
print('Done!')
