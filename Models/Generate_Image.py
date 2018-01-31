import os
import numpy as np
import scipy.io as sio
import scipy.misc
import fnmatch
import random
import math

##############################################################################
call_folder = '/scratch_net/biwidl102/dariopa/Images_RAD/'
# call_folder = '/media/dario/Semesterarbeit/Images_RAD/'
store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224/'
# store_folder = '/media/dario/Semesterarbeit/Data_224_224_new/'

X_shape = 224
Y_shape = 224


# FIRST COMPUTE PARAMETERS
Red_boundaries = np.array([[0.5, 600, 28], [0.6, 615, 35]])
Green_boundaries = np.array([[0.5, 525, 28], [0.6, 540, 35]])
Blue_boundaries = np.array([[0.5, 460, 28], [0.6, 475, 35]])

# Calculate parameters for 3 channels:
nr_images = 150; # for each image in dataset, it creates "nr_images"
##############################################################################
# CREATE CSS PARAMETERS

# red channel
r_alpha = np.arange(nr_images, dtype=np.float16)
r_mean = np.arange(nr_images, dtype=np.float16)
r_sigma = np.arange(nr_images, dtype=np.float16)
# green channel
g_alpha = np.arange(nr_images, dtype=np.float16)
g_mean = np.arange(nr_images, dtype=np.float16)
g_sigma = np.arange(nr_images, dtype=np.float16)
# blue channel
b_alpha = np.arange(nr_images, dtype=np.float16)
b_mean = np.arange(nr_images, dtype=np.float16)
b_sigma = np.arange(nr_images, dtype=np.float16)


for counter in range(0, nr_images):
    # Red channel
    r_alpha[counter] = round(random.uniform(Red_boundaries[0, 0], Red_boundaries[1, 0]), 3)
    r_mean[counter] = round(random.uniform(Red_boundaries[0, 1], Red_boundaries[1, 1]), 1)
    r_sigma[counter] = round(random.uniform(Red_boundaries[0, 2], Red_boundaries[1, 2]), 1)

    # # Green channel
    g_alpha[counter] = round(random.uniform(Green_boundaries[0, 0], Green_boundaries[1, 0]), 3)
    g_mean[counter] = round(random.uniform(Green_boundaries[0, 1], Green_boundaries[1, 1]), 1)
    g_sigma[counter] = round(random.uniform(Green_boundaries[0, 2], Green_boundaries[1, 2]), 1)

    # # Blue channel
    b_alpha[counter] = round(random.uniform(Blue_boundaries[0, 0], Blue_boundaries[1, 0]), 3)
    b_mean[counter] = round(random.uniform(Blue_boundaries[0, 1], Blue_boundaries[1, 1]), 1)
    b_sigma[counter] = round(random.uniform(Blue_boundaries[0, 2], Blue_boundaries[1, 2]), 1)

np.savetxt(os.path.join(store_folder, 'Parameters.csv'), (r_alpha, r_mean, r_sigma, g_alpha, g_mean, g_sigma, b_alpha, b_mean, b_sigma), delimiter=',')

##############################################################################
# GENERATE RGB IMAGES
nr_hyp_images = len(fnmatch.filter(os.listdir(call_folder), '*.mat'))
batch_counter = 1

CSS_calc = np.full((3, 31), 0, dtype = np.float16)
CSS = np.full((int(np.floor(1392/X_shape)) * int(np.floor(1300/Y_shape)) * nr_hyp_images * nr_images, 3, 3), 0, dtype = np.float16)
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

    # Calculate RGB image from Power Spectrum and CSS
    
    for counter in range(0, nr_images):
        q = 0
        for i in range(401, 711, 10):
            CSS_calc[0,q] = r_alpha[counter] * math.exp(-(i-r_mean[counter]) ** 2 / (2 * r_sigma[counter] ** 2))
            CSS_calc[1,q] = g_alpha[counter] * math.exp(-(i-g_mean[counter]) ** 2 / (2 * g_sigma[counter] ** 2))
            CSS_calc[2,q] = b_alpha[counter] * math.exp(-(i-b_mean[counter]) ** 2 / (2 * b_sigma[counter] ** 2))
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
                scipy.misc.toimage(I_image_batch, cmin=0, cmax=1).save(os.path.join(store_folder, str(batch_counter) + '.jpg'))

                # Fill CSS Array
                CSS[batch_counter - 1, :, 0] = [r_alpha[counter], r_mean[counter], r_sigma[counter]]
                CSS[batch_counter - 1, :, 1] = [g_alpha[counter], g_mean[counter], g_sigma[counter]]
                CSS[batch_counter - 1, :, 2] = [b_alpha[counter], b_mean[counter], b_sigma[counter]]
                batch_counter = batch_counter + 1

batch_counter = batch_counter - 1
CSS = CSS[0:batch_counter, :, :]
np.save(os.path.join(store_folder, 'CSS.npy'), CSS)

print('Done!')
