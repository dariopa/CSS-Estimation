import os
import numpy as np
from utils_synthetic_data import Generate, Split


call_folder = '../../Images_RAD/'
store_folder = '../../dariopa/Data_150_150_5_classes_sigma'
if not os.path.isdir(store_folder):
    os.makedirs(store_folder)
if not os.path.isdir(os.path.join(store_folder, 'Images')):
    os.makedirs(os.path.join(store_folder, 'Images'))

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 150
Y_shape = 150

alpha = 0.55
r_mean = 615
g_mean = 530
b_mean = 465
sigma = np.array([[28.01, 28.7, 29.4, 30.1, 30.8, 31.5, 32.2, 32.9, 33.6, 34.3, 34.599]])

# classes
classes = 5

# FOR PREPROCESSING ----------------------
# How much data to use?
use_data = 1.

# Divison factor for Training, Validation and Test data [0,1]:
Train_split = 8./10
Val_split = 1./10
Test_split = 1./10

print('Generating images')
Generate(call_folder, store_folder, X_shape, Y_shape, alpha, r_mean, g_mean, b_mean, sigma, classes)
print('\nPreprocess Data')
Split(store_folder, use_data, Train_split, Val_split, Test_split)
