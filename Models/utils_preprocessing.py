import numpy as np

def standardize(X_data):
    X_new = X_data
    X_data = (X_data[0, :, :, 0] - np.mean(X_data[0, :, :, 0])) / np.std(X_data[0, :, :, 0])
    X_new[0:1, :, :, 0:1] = X_data
    return X_new