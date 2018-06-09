import os
import numpy as np
from utils_synthetic_data_alpha import Generate, Split


call_folder = '../../Images_RAD/'
store_folder = '../../Data_150_150_5_classes_alpha'
if not os.path.isdir(store_folder):
    os.makedirs(store_folder)
if not os.path.isdir(os.path.join(store_folder, 'Images')):
    os.makedirs(os.path.join(store_folder, 'Images'))

# FOR IMAGE GENERATION ----------------------
# Batch size of images
X_shape = 150
Y_shape = 150

alpha = np.array([[0.501, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.599]])
# alpha = np.array([[0.501, 0.505, 0.51, 0.515, 0.52, 05.25, 0.53, 05.35, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.599]])
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

print('Generating images')
Generate(call_folder, store_folder, X_shape, Y_shape, alpha, r_mean, g_mean, b_mean, sigma, classes)
print('\nPreprocess Data')
Split(store_folder, use_data, Train_split, Val_split, Test_split)
