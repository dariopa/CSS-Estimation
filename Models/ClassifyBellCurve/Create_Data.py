import os
import numpy as np
from utils_synthetic_data import Generate, Split

# Generate Data?
generate = True
# Split Data?
split = True

call_folder = '../../Images_RAD/'
store_folder = '../../Data_TEST'
if not os.path.isdir(store_folder):
    os.makedirs(store_folder)
if not os.path.isdir(os.path.join(store_folder, 'Images'))
    os.makedirs(os.path.join(store_folder, 'Images'))

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 150
Y_shape = 150

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
if split == True:
    print('\nSplit Data')
    Split(store_folder, use_data, Train_split, Val_split, Test_split)
