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
                CSS_calc[0,q] = alpha[0, counter] * math.exp(-(i-r_mean) ** 2 / (2 * sigma[0, counter] ** 2))
                CSS_calc[1,q] = alpha[0, counter] * math.exp(-(i-g_mean) ** 2 / (2 * sigma[0, counter] ** 2))
                CSS_calc[2,q] = alpha[0, counter] * math.exp(-(i-b_mean) ** 2 / (2 * sigma[0, counter] ** 2))
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
                    CSS[batch_counter - 1, :] = [alpha[0, counter], sigma[0, counter]]
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

    y_binned = np.zeros([len(CSS), 1])

    bla = 0
    for i in range(0,len(CSS)):
        # categorize alpha and sigma values: 
        if CSS[i, 0] == 0.5 and CSS[i,1] == 28.:
            y_binned[i] = 0
        elif CSS[i, 0] == 0.533 and CSS[i,1] == 30.:
            y_binned[i] = 1
        elif CSS[i, 0] == 0.566 and CSS[i,1] == 32.:
            y_binned[i] = 2
        elif CSS[i, 0] == 0.6 and CSS[i,1] == 34.:
            y_binned[i] = 3
        elif CSS[i, 0] == 0.5 and CSS[i,1] == 34.:
            y_binned[i] = 4
        elif CSS[i, 0] == 0.533 and CSS[i,1] == 28.:
            y_binned[i] = 5
        elif CSS[i, 0] == 0.566 and CSS[i,1] == 30.:
            y_binned[i] = 6
        elif CSS[i, 0] == 0.6 and CSS[i,1] == 32.:
            y_binned[i] = 7
        elif CSS[i, 0] == 0.5 and CSS[i,1] == 32.:
            y_binned[i] = 8
        elif CSS[i, 0] == 0.533 and CSS[i,1] == 34.:
            y_binned[i] = 9
        elif CSS[i, 0] == 0.566 and CSS[i,1] == 28.:
            y_binned[i] = 10
        elif CSS[i, 0] == 0.6 and CSS[i,1] == 30.:
            y_binned[i] = 11
        elif CSS[i, 0] == 0.5 and CSS[i,1] == 30.:
            y_binned[i] = 12
        elif CSS[i, 0] == 0.533 and CSS[i,1] == 32.:
            y_binned[i] = 13
        elif CSS[i, 0] == 0.566 and CSS[i,1] == 34.:
            y_binned[i] = 14
        elif CSS[i, 0] == 0.6 and CSS[i,1] == 28.:
            y_binned[i] = 15
        else:
            print('No matching class!')
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

# Generate Data?
generate = True
# Split Data?
split = True

call_folder = '../Images_RAD/'
store_folder = '../Data_Ambiguity'
if not os.path.isdir(store_folder):
    os.makedirs(store_folder)
if not os.path.isdir(os.path.join(store_folder, 'Images')):
    os.makedirs(os.path.join(store_folder, 'Images'))

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 1000
Y_shape = 1000

alpha = np.array([[0.500, 0.533, 0.566, 0.600, 0.500, 0.533, 0.566, 0.600, 0.500, 0.533, 0.566, 0.600, 0.500, 0.533, 0.566, 0.600]])
r_mean = 615
g_mean = 530
b_mean = 465
sigma = np.array([[28., 30., 32., 34., 34., 28., 30., 32., 32., 34., 28., 30., 30., 32., 34., 28.]])

# classes
classes = len(alpha)

# How much data to use?
use_data = 1.

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 8./10
Val_split = 1./10
Test_split = 1./10

if generate == True:
    print('Generating images')
    Generate(call_folder, store_folder, X_shape, Y_shape, alpha, r_mean, g_mean, b_mean, sigma, classes)

