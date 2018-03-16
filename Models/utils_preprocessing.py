import numpy as np

def standardize(X_data):
    X_new[:, :, 0:1] = (X_data[:, :, 0:1] - np.mean(X_data[:, :, 0:1])) / np.std(X_data[:, :, 0:1])
    return X_new