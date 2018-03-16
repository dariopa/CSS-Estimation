import numpy as np

def standardize(X_data, channel):
    X_new = X_data
    for i in range(channel):
        X_new[:, :, i] =(X_data[:, :, i] - np.mean(X_data[:, :, i])) / np.std(X_data[:, :, i])
    return X_new