import os
import fnmatch
# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
# import cv2

##############################################################################
# Folder Path
call_folder = '/scratch_net/biwidl102/dariopa/Data_224_224/'
# call_folder = '/media/dario/Semesterarbeit/Data_28_28/'
store_folder = '/Data_224_224/'
# store_folder = '/Data_28_28/'

# How many data to use?
use_data = 0.1/10

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 8./10
Val_split = 1./10
Test_split = 1./10

##############################################################################

# Number of Images:
nr_images = int(use_data * len(fnmatch.filter(os.listdir(call_folder), '*.jpeg')))

# Pixels in x- and y-direction:
arr = np.array(Image.open(call_folder + '/Image1.jpeg'))
x_row, y_col, _ = arr.shape

##############################################################################
# IMPORT AND PROCESS X
print()
print('Processing X Data...')
X = np.full((nr_images,x_row,y_col,3),0, dtype = np.uint8)
# X = []
for i in range(1, nr_images+1):
    img = Image.open(call_folder + '/Image' + str(i) + '.jpeg')
    img = np.array(img, dtype = np.uint8)
    X[i-1,:,:,:] = img
    # img = cv2.imread(call_folder + '/Image' + str(i) + '.jpeg')
    # X.append(img, dtype=np.uint8)

print('Done!')

##############################################################################
# IMPORT AND PROCESS Y
print()
print('Processing y Data...')
y = np.load(call_folder + '/CSS.npy')[0:nr_images,:,:]
y_binned = np.zeros([len(y), 3, 3])


for i in range(0,len(y)):
    # binning r_alpha values
    class_r_alpha, bins_r_alpha = np.histogram(y[i, 0, 0], bins=10, range=[0.5, 0.6])
    y_binned[i, 0, 0] = np.argmax(class_r_alpha)
    # binning g_alpha values
    class_g_alpha, bins_g_alpha = np.histogram(y[i, 0, 1], bins=10, range=[0.5, 0.6])
    y_binned[i, 0, 1] = np.argmax(class_g_alpha)
    # binning b_alpha values
    class_b_alpha, bins_b_alpha = np.histogram(y[i, 0, 2], bins=10, range=[0.5, 0.6])
    y_binned[i, 0, 2] = np.argmax(class_b_alpha)

    # binning r_mean values
    class_r_mean, bins_r_mean = np.histogram(y[i, 1, 0], bins=10, range=[600, 615])
    y_binned[i, 1, 0] = np.argmax(class_r_mean)
    # binning g_mean values
    class_g_mean, bins_g_mean = np.histogram(y[i, 1, 1], bins=10, range=[525, 540])
    y_binned[i, 1, 1] = np.argmax(class_g_mean)
    # binning b_mean values
    class_b_mean, bins_b_mean = np.histogram(y[i, 1, 2], bins=10, range=[460, 475])
    y_binned[i, 1, 2] = np.argmax(class_b_mean)

    # binning r_sigma values:
    class_r_sigma, bins_r_sigma = np.histogram(y[i, 2, 0], bins=10, range=[28, 35])
    y_binned[i, 2, 0] = np.argmax(class_r_sigma)
    # binning g_sigma values
    class_g_sigma, bins_g_sigma = np.histogram(y[i, 2, 1], bins=10, range=[28, 35])
    y_binned[i, 2, 1] = np.argmax(class_g_sigma)
    # binning b_sigma values
    class_b_sigma, bins_b_sigma = np.histogram(y[i, 2, 2], bins=10, range=[28, 35])
    y_binned[i, 2, 2] = np.argmax(class_b_sigma)


np.savetxt(os.path.join(store_folder, 'Bins.csv'), (bins_r_alpha, bins_r_mean, bins_r_sigma, bins_g_alpha, bins_g_mean, bins_g_sigma, bins_b_alpha, bins_b_mean, bins_b_sigma), delimiter=',')

print('Done!')

##############################################################################
# SHUFFLE DATA
print()
print('Shuffling Data...')
for i in range(1,5):
    (X, y, y_binned) = shuffle(X, y, y_binned)
print('Done!')
##############################################################################
# SPLIT DATA
print()
print('Splitting Data...')
X_train = X[0:int(Train_split * nr_images), :, :, :]
y_train = y[0:int(Train_split * nr_images), :, :]
y_binned_train = y_binned[0:int(Train_split * nr_images), :, :]
X_validation = X[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :, :, :]
y_validation = y[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :, :]
y_binned_validation = y_binned[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :, :]
X_test = X[int((Train_split + Val_split) * nr_images):nr_images, :, :, :]
y_test = y[int((Train_split + Val_split) * nr_images):nr_images, :, :]
y_binned_test = y_binned[int((Train_split + Val_split) * nr_images):nr_images, :, :]

print()
print('Image Shape: {}'.format(X_train[0].shape))
print()
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
np.save(os.path.join(store_folder, 'X_train.npy'), X_train)
np.save(os.path.join(store_folder, 'X_validation.npy'), X_validation)
np.save(os.path.join(store_folder, 'X_test.npy'), X_test)
np.save(os.path.join(store_folder, 'y_train.npy'), y_train)
np.save(os.path.join(store_folder, 'y_validation.npy'), y_validation)
np.save(os.path.join(store_folder, 'y_test.npy'), y_test)
np.save(os.path.join(store_folder, 'y_binned_train.npy'), y_binned_train)
np.save(os.path.join(store_folder, 'y_binned_validation.npy'), y_binned_validation)
np.save(os.path.join(store_folder, 'y_binned_test.npy'), y_binned_test)
print('Done!')

##############################################################################
# TRIALS

# print(y_test[:,1,0])
# print(y_binned_test[:,1,0])

# arr1 = X_train[59]
# arr2 = X_validation[0]
# arr = arr1 - arr2
# imshow(arr1)
# plt.show()
# imshow(arr2)
# plt.show()
