import os
import numpy as np
import scipy.io as sio
import scipy.misc
import fnmatch
import random
import math
import cv2
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def Generate(call_folder, store_folder, X_shape, Y_shape, X_shape_output, Y_shape_output, resize, alpha, r_mean, g_mean, b_mean, sigma, classes):
    _, nr_param = alpha.shape
    nr_hyp_images = len(fnmatch.filter(os.listdir(call_folder), '*.mat'))
    batch_counter = 1


    X_data = np.full((nr_param * nr_hyp_images, 255, 3), 0)
    CSS_calc = np.full((3, 31), 0, dtype = np.float16)
    CSS = np.full((nr_hyp_images * nr_param, 2), 0, dtype = np.float16)

    for i in range(0, nr_hyp_images):
        print('Evaluating Image ' + str(i+1))
        mat_contents = sio.loadmat(os.path.join(call_folder, 'RAD_' + str(i+1) + '.mat'))
        rad =mat_contents['rad_new']

        x_row, y_col, spect = rad.shape
        n_features = x_row * y_col

        rad_reshaped = np.reshape(rad, (1, n_features, spect))[0]
        rad_reshaped = np.swapaxes(rad_reshaped, 0, 1)

        # Compute RGB image from Power Spectrum and CSS
        for counter in range(0, nr_param):
            q = 0
            for i in range(401, 711, 10):
                CSS_calc[0,q] = alpha[0, counter] * math.exp(-(i-r_mean) ** 2 / (2 * sigma ** 2))
                CSS_calc[1,q] = alpha[0, counter] * math.exp(-(i-g_mean) ** 2 / (2 * sigma ** 2))
                CSS_calc[2,q] = alpha[0, counter] * math.exp(-(i-b_mean) ** 2 / (2 * sigma ** 2))
                q = q + 1

            # I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
            I = np.matmul(CSS_calc, rad_reshaped) / 4095
            I[I > 1] = 1
            red_distr, red_bins = np.histogram(I[0, :], bins=255, range=[0, 1])
            green_distr, green_bins = np.histogram(I[1, :], bins=255, range=[0, 1])
            blue_distr, blue_bins = np.histogram(I[2, :], bins=255, range=[0, 1])
            
            X_data[batch_counter -1, :, 0] = red_distr
            X_data[batch_counter -1, :, 1] = green_distr
            X_data[batch_counter -1, :, 2] = blue_distr

            # Fill CSS Array
            CSS[batch_counter - 1, :] = [alpha[0, counter], sigma]
            batch_counter = batch_counter + 1

    batch_counter = batch_counter - 1 # just to have right amount of images
    print('batch_counter:   ', batch_counter)
    CSS = CSS[0:batch_counter, :]
    np.save(os.path.join(store_folder, 'CSS.npy'), CSS)
    print('CSS parameters:  ', len(CSS))
    np.save(os.path.join(store_folder, 'X_data.npy'), X_data)
    print('X_data parameters:  ', len(X_data))
    print(X_data)

    # PROCESS Y
    print()
    print('Binning y Data...')

    nr_images = len(CSS)

    y_binned = np.zeros([nr_images, 2])

    for i in range(0,nr_images):
        # binning r_alpha values
        class_alpha, bins_alpha = np.histogram(CSS[i, 0], bins=classes, range=[0.5, 0.6])
        y_binned[i, 0] = np.argmax(class_alpha)


    print('Done!')

    # SAVE DATA IN BINARY FORMAT
    print()
    print('Saving y binned Data...')
    np.save(os.path.join(store_folder, 'CSS_binned.npy'), y_binned)

    print('Done!')

    ##############################################################################
def Preprocess(store_folder, use_data, Train_split, Val_split, Test_split):

    # LOAD DATA
    print()
    print('Loading Data...')
    X_data = np.load(os.path.join(store_folder, 'X_data.npy'))
    y = np.load(os.path.join(store_folder, 'CSS.npy'))
    y_binned = np.load(os.path.join(store_folder, 'CSS_binned.npy'))

    print('Done!')

    # SHUFFLE DATA
    print()
    print('Shuffling Data...')
    for i in range(1,5):
        (X_data, y, y_binned) = shuffle(X_data, y, y_binned)
    print('Done!')

    # SPLIT DATA
    print()
    print('Splitting Data...')
    nr_images = int(use_data * len(X_data))

    X_train = X_data[0:int(Train_split * nr_images)]
    y_train = y[0:int(Train_split * nr_images), :]
    y_binned_train = y_binned[0:int(Train_split * nr_images), :]

    X_validation = X_data[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images)]
    y_validation = y[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :]
    y_binned_validation = y_binned[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :]

    X_test = X_data[int((Train_split + Val_split) * nr_images):nr_images]
    y_test = y[int((Train_split + Val_split) * nr_images):nr_images, :]
    y_binned_test = y_binned[int((Train_split + Val_split) * nr_images):nr_images, :]

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

#################################################################################################################################################
# Generate images?
generate = False
# Preprocess data?
preprocess = True
#################################

call_folder = '/home/dario/Documents/SemThes_Local/Images_RAD/'
# call_folder = '/scratch_net/biwidl102/dariopa/Images_RAD/'

# store_folder = '/home/dario/Documents/SemThes_Local/Data_32_32'
store_folder = '/home/dario/Documents/SemThes_Local/Data_150_150'

# store_folder = '/scratch_net/biwidl102/dariopa/Data_32_32'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_150_150'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224_5_classes'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_150_150_5_classes'

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 150
Y_shape = 150

# Downscaling of images
X_shape_output = 150
Y_shape_output = 150

# Want to resize image?
resize = False

alpha = np.array([[0.501, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.599]])

r_mean = 615
g_mean = 530
b_mean = 465
sigma = 28

# classes
classes = 10


# FOR PREPROCESSING ----------------------
# How much data to use?
use_data = 1

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 8./10
Val_split = 1./10
Test_split = 1./10

if generate == True:
    print('Generating images')
    Generate(call_folder, store_folder, X_shape, Y_shape, X_shape_output, Y_shape_output, resize, alpha, r_mean, g_mean, b_mean, sigma, classes)
if preprocess == True:
    print('\nPreprocess Data')
    Preprocess(store_folder, use_data, Train_split, Val_split, Test_split)


