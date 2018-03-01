import os
import numpy as np
import scipy.io as sio
import scipy.misc
import fnmatch
import random
import math
import cv2

##############################################################################
call_folder = '/home/dario/Documents/SemThes_Local/Images_RAD/'
# call_folder = '/scratch_net/biwidl102/dariopa/Images_RAD/'

# store_folder = '/home/dario/Documents/SemThes_Local/Data_32_32'
store_folder = '/home/dario/Documents/SemThes_Local/Data_150_150'

# store_folder = '/scratch_net/biwidl102/dariopa/Data_32_32'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224'

# Batch size of images
X_shape = 224
Y_shape = 224

# Downscaling of images
X_shape_output = 150
Y_shape_output = 150

# Want to resize image?
resize = True


# FIRST COMPUTE PARAMETERS
Red_boundaries = np.array([[0.5, 600, 28], [0.6, 615, 35]])
Green_boundaries = np.array([[0.5, 525, 28], [0.6, 540, 35]])
Blue_boundaries = np.array([[0.5, 460, 28], [0.6, 475, 35]])

##############################################################################
# CREATE CSS PARAMETERS

alpha = np.array([[0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]])
r_mean = 615
g_mean = 530
b_mean = 465
sigma = np.array([[28., 28.5, 29., 29.5, 30., 30.5, 31., 31.5, 32., 32.5, 33.]])
_, nr_param = alpha.shape

##############################################################################
# GENERATE RGB IMAGES
nr_hyp_images = len(fnmatch.filter(os.listdir(call_folder), '*.mat'))
batch_counter = 1
nr_hyp_images = 50 # HARD PARAMETER - DELETE AFTERWARDS!

CSS_calc = np.full((3, 31), 0, dtype = np.float16)
CSS = np.full((int(np.floor(1392/X_shape)) * int(np.floor(1300/Y_shape)) * nr_hyp_images * nr_param, 2), 0, dtype = np.float16)

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
                if resize == True:
                    I_image_batch = cv2.resize(I_image_batch, (X_shape_output, Y_shape_output))
                scipy.misc.toimage(I_image_batch, cmin=0, cmax=1).save(os.path.join(store_folder + '/Images/', str(batch_counter) + '.jpg'))

                # Fill CSS Array
                CSS[batch_counter - 1, :] = [alpha[0, counter], sigma[0, counter]]
                batch_counter = batch_counter + 1

batch_counter = batch_counter - 1 # just to have right amount of images
print(batch_counter)
CSS = CSS[0:batch_counter, :]
np.save(os.path.join(store_folder, 'CSS.npy'), CSS)
print(len(CSS))
##############################################################################
# IMPORT AND PROCESS Y
print()
print('Binning y Data...')
nr_images = len(fnmatch.filter(os.listdir(store_folder + '/Images/'), '*.jpg'))

y_binned = np.zeros([len(CSS), 2])

for i in range(0,len(CSS)):
    # binning r_alpha values
    class_alpha, bins_alpha = np.histogram(CSS[i, 0], bins=10, range=[0.5, 0.6])
    y_binned[i, 0] = np.argmax(class_alpha)

    # binning r_sigma values:
    class_sigma, bins_sigma = np.histogram(CSS[i, 1], bins=10, range=[28, 33])
    y_binned[i, 1] = np.argmax(class_sigma)


np.savetxt(os.path.join(store_folder, 'Bins.csv'), (bins_alpha, bins_sigma), delimiter=',')

print('Done!')

##############################################################################
# SAVE DATA IN BINARY FORMAT
print()
print('Saving y binned Data...')
np.save(os.path.join(store_folder, 'CSS_binned.npy'), y_binned)

print(CSS)
print(y_binned)
print()

print('Done!')

##############################################################################
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