import os
import numpy as np
import scipy.io as sio
import fnmatch
import random
import math
from PIL import Image


##############################################################################
call_folder = '/scratch_net/biwidl102/dariopa/Image_RAD/'
store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224'

X_shape = 224
Y_shape = 224


# FIRST COMPUTE PARAMETERS
Red_boundaries = np.array([0.5, 600, 28; 0.6, 615, 35])
Green_boundaries = np.array([0.5, 525, 28; 0.6, 540, 35])
Blue_boundaries = np.array([0.5, 460, 28; 0.6, 475, 35])

# Calculate parameters for 3 channels:
nr_images = 200; # for each image in dataset, it creates "nr_images"
##############################################################################

# red channel
r_alpha = np.zeros(nr_images)
r_mean = np.zeros(nr_images)
r_sigma = np.zeros(nr_images)
# green channel
g_alpha = np.zeros(nr_images)
g_mean = np.zeros(nr_images)
g_sigma = np.zeros(nr_images)
# blue channel
b_alpha = np.zeros(nr_images)
b_mean = np.zeros(nr_images)
b_sigma = np.zeros(nr_images)

for counter in range(0,nr_images+1)
    # r = a + (b-a).*rand(N,1)
    # Red channel
    r_alpha[0,counter] = random.randrange(Red_boundaries[0,0], Red_boundaries[1,0], 0.05)
    r_mean[0,counter] = random.randrange(Red_boundaries[0,1], Red_boundaries[1,1], 1)
    r_sigma[0,counter] = random.randrange(Red_boundaries[0,2], Red_boundaries[1,2], 0.05)

    # Green channel
    g_alpha[0,counter] = random.randrange(Green_boundaries[0,0], Green_boundaries[1,0], 0.05)
    g_mean[0,counter] = random.randrange(Green_boundaries[0,1], Green_boundaries[1,1], 1)
    g_sigma[0,counter] = random.randrange(Green_boundaries[0,2], Green_boundaries[1,2], 0.05)

    # Blue channel
    g_alpha[0,counter] = random.randrange(Green_boundaries[0,0], Green_boundaries[1,0], 0.05)
    g_mean[0,counter] = random.randrange(Green_boundaries[0,1], Green_boundaries[1,1], 1)
    g_sigma[0,counter] = random.randrange(Green_boundaries[0,2], Green_boundaries[1,2], 0.05)

##############################################################################
nr_hyp_images = int(use_data * len(fnmatch.filter(os.listdir(call_folder), '*.mat')))
batch_counter = 1
for i in range(1, nr_images):
    mat_contents = sio.loadmat(os.path.join(call_folder, str(i) + '.mat'))
    rad = mat_contents['rad_new']

    X_window = np.floor(np.shape(rad[0]) / X_shape)
    Y_window = np.floor(np.shape(rad[1]) / Y_shape)
    rad = rad[0:X_window * X_shape, 0:Y_window * Y_shape, :]

    row, col, spect = rad.shape
    n_features = row*col
    rad_reshaped = np.reshape(rad, (1, n_features, spect)
    rad_reshaped = np.swapaxes(rad_reshaped,0,2)
    print(rad_reshaped.shape)

    # Calculate RGB image from Power Spectrum and CSS
    
    for counter in range(1, nr_hyp_images):
        q = 1
        for i in range(401, 710, 10):
            CSS_calc[0,q] = r_alpha[0, counter] * math.exp(-(i-r_mean[0, counter]) ** 2 / (2 * r_sigma[0, counter] ** 2))
            CSS_calc[1,q] = g_alpha[0, counter] * math.exp(-(i-g_mean[0, counter]) ** 2 / (2 * g_sigma[0, counter] ** 2))
            CSS_calc[2,q] = b_alpha[0, counter] * math.exp(-(i-b_mean[0, counter]) ** 2 / (2 * b_sigma[0, counter] ** 2))
            q = q + 1

        # I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
        I = CSS_calc * rad_reshaped / 4095
        I_image = np.reshape(I, (3, row, column))
        I_image = np.swapaxes(I_image, 0, 1)
        I_image = np.swapaxes(I_image, 1, 2) 
        
        # Now store batches of Image!
        for i in range(0,X_window-1):
            for j in range (0,Y_window-1):
                I_image_batch = I_image[(0 + i * X_shape):(i * X_shape + X_shape), (0) + j * Y_shape):(j * Y_shape + Y_shape), :]
                im = Image.fromarray(I_image_batch)
                im.save(os.join.path(store_folder, str(batch_counter) + '.png'))

                # Fill CSS Array
                CSS[batch_counter - 1, :, 0) = [r_alpha(1,counter), r_mean(1,counter), r_sigma(1,counter)]
                CSS(batch_counter - 1, :, 1) = [g_alpha(1,counter), g_mean(1,counter), g_sigma(1,counter)]
                CSS(batch_counter - 1, :, 2) = [b_alpha(1,counter), b_mean(1,counter), b_sigma(1,counter)]
                batch_counter = batch_counter + 1



print('Done!')

