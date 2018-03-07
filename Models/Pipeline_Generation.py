import numpy as np
from Gen_Pre import Generate, Preprocess

#################################
# Generate images?
generate = True
# Preprocess data?
preprocess = True
#################################

# call_folder = '/home/dario/Documents/SemThes_Local/Images_RAD/'
call_folder = '/scratch_net/biwidl102/dariopa/Images_RAD/'

# store_folder = '/home/dario/Documents/SemThes_Local/Data_32_32'
# store_folder = '/home/dario/Documents/SemThes_Local/Data_150_150'

# store_folder = '/scratch_net/biwidl102/dariopa/Data_32_32'
# store_folder = '/scratch_net/biwidl102/dariopa/Data_150_150'
store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224'

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 224
Y_shape = 224

# Downscaling of images
X_shape_output = 150
Y_shape_output = 150

# Want to resize image?
resize = False

alpha = np.array([[0.501, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.599]])
# alpha = np.array([[0.51, 0.55, 0.599]])
r_mean = 615
g_mean = 530
b_mean = 465
sigma = np.array([[28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28.]])

# FOR PREPROCESSING ----------------------
# How much data to use?
use_data = 1

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 7./10
Val_split = 1./10
Test_split = 2./10


if generate == True:
    print('Generating images')
    Generate(call_folder, store_folder, X_shape, Y_shape, X_shape_output, Y_shape_output, resize, alpha, r_mean, g_mean, b_mean, sigma)
if preprocess == True:
    print('\nPreprocess Data')
    Preprocess(store_folder, use_data, Train_split, Val_split, Test_split)