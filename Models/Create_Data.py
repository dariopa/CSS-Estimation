import numpy as np
from utils_synthetic_data import Generate, Split

# Generate images?
generate = True
# Preprocess data?
split = True

# call_folder = '/home/dario/Documents/SemThes_Local/Images_RAD/'
call_folder = '/scratch_net/biwidl102/dariopa/Images_RAD/'

# store_folder = '/home/dario/Documents/SemThes_Local/Data_150_150'

# store_folder = '/scratch_net/biwidl102/dariopa/Data_150_150'
store_folder = '/scratch_net/biwidl102/dariopa/Data_224_224_big'

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 224
Y_shape = 224

# alpha = np.array([[0.501, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.599]])
alpha = np.array([[0.501, 0.505, 0.51, 0.515, 0.52, 05.25, 0.53, 05.35, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.599]])
r_mean = 615
g_mean = 530
b_mean = 465
sigma = 28

# classes
classes = 10

# FOR PREPROCESSING ----------------------
# How much data to use?
use_data = 1.

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 8./10
Val_split = 1./10
Test_split = 1./10

if generate == True:
    print('Generating images')
    Generate(call_folder, store_folder, X_shape, Y_shape, alpha, r_mean, g_mean, b_mean, sigma, classes)
if split == True:
    print('\nPreprocess Data')
    Split(store_folder, use_data, Train_split, Val_split, Test_split)
