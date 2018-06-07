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

def Generate(call_folder, store_folder, X_shape, Y_shape, alpha, r_mean, g_mean, b_mean, sigma, classes):
    _, nr_param = alpha.shape
    nr_hyp_images = len(fnmatch.filter(os.listdir(call_folder), '*.mat'))
    batch_counter = 1

    CSS_calc = np.full((3, 31), 0.)
    CSS = np.full((int(np.floor(1392/X_shape)) * int(np.floor(1300/Y_shape)) * nr_hyp_images * nr_param, 2), 0.)

    for i in range(0, nr_hyp_images):
        print('Evaluating Image ' + str(i+1))
        mat_contents = sio.loadmat(os.path.join(call_folder, 'RAD_' + str(i+1) + '.mat'))
        rad =mat_contents['rad_new']

        x_row, y_col, spect = rad.shape
        X_window = int(np.floor(x_row / X_shape))
        Y_window = int(np.floor(y_col / Y_shape))

        rad = rad[0:X_window * X_shape, 0:Y_window * Y_shape, :]
        x_row, y_col, spect = rad.shape
        n_features = x_row * y_col

        rad_reshaped = np.reshape(rad, (1, n_features, spect))[0]
        rad_reshaped = np.swapaxes(rad_reshaped, 0, 1)

        # Compute RGB image from Power Spectrum and CSS
        for counter in range(0, nr_param):
            q = 0
            for i in range(401, 711, 10):
                CSS_calc[0,q] = alpha * math.exp(-(i-r_mean) ** 2 / (2 * sigma[0, counter] ** 2))
                CSS_calc[1,q] = alpha * math.exp(-(i-g_mean) ** 2 / (2 * sigma[0, counter] ** 2))
                CSS_calc[2,q] = alpha * math.exp(-(i-b_mean) ** 2 / (2 * sigma[0, counter] ** 2))
                q = q + 1

            # I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
            I = np.matmul(CSS_calc, rad_reshaped) / 4095

            I_image = np.reshape(I, (3, x_row, y_col))
            I_image = np.swapaxes(I_image, 0, 1)
            I_image = np.swapaxes(I_image, 1, 2)
            I_image[I_image > 1] = 1

            # Now store batches of Image!
            for i in range(0,X_window):
                for j in range (0,Y_window):
                    I_image_batch = I_image[(0 + i * X_shape):(i * X_shape + X_shape), (0 + j * Y_shape):(j * Y_shape + Y_shape), :]
                    scipy.misc.toimage(I_image_batch, cmin=0., cmax=1.).save(os.path.join(store_folder + '/Images/', str(batch_counter) + '.jpg'))

                    # Fill CSS Array
                    CSS[batch_counter - 1, :] = [alpha, sigma[0, counter]]
                    batch_counter = batch_counter + 1

    batch_counter = batch_counter - 1 # just to have right amount of images
    print('batch_counter:   ', batch_counter)
    CSS = CSS[0:batch_counter, :]
    np.save(os.path.join(store_folder, 'CSS.npy'), CSS)
    print('CSS parameters:  ', len(CSS))

    # IMPORT AND PROCESS Y
    print()
    print('Binning y Data...')
    nr_images = len(fnmatch.filter(os.listdir(store_folder + '/Images/'), '*.jpg'))

    y_binned = np.zeros([len(CSS), 2])

    for i in range(0,len(CSS)):
        # binning r_alpha values
        class_sigma, bins_alpha = np.histogram(CSS[i, 0], bins=classes, range=[28, 35])
        y_binned[i, 0] = np.argmax(class_sigma)

    print('Done!')

    # SAVE DATA IN BINARY FORMAT
    print()
    print('Saving y binned Data...')
    np.save(os.path.join(store_folder, 'CSS_binned.npy'), y_binned)

    print('Done!')

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

##############################################################################
def Split(store_folder, use_data, Train_split, Val_split, Test_split):

    # LOAD DATA
    print()
    print('Loading Data...')
    X_addr = np.load(os.path.join(store_folder, 'datapath.npy'))
    y = np.load(os.path.join(store_folder, 'CSS.npy'))
    y_binned = np.load(os.path.join(store_folder, 'CSS_binned.npy'))
    print('Done!')

    # SHUFFLE DATA
    print()
    print('Shuffling Data...')
    (X_addr, y, y_binned) = shuffle(X_addr, y, y_binned)
    print('Done!')

    # SPLIT DATA
    print()
    print('Splitting Data...')
    nr_images = int(use_data * len(fnmatch.filter(os.listdir(store_folder + '/Images/'), '*.jpg')))

    X_train = X_addr[0:int(Train_split * nr_images)]
    y_train = y[0:int(Train_split * nr_images), :]
    y_binned_train = y_binned[0:int(Train_split * nr_images), :]

    X_validation = X_addr[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images)]
    y_validation = y[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :]
    y_binned_validation = y_binned[int(Train_split * nr_images):int((Train_split + Val_split) * nr_images), :]

    X_test = X_addr[int((Train_split + Val_split) * nr_images):nr_images]
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