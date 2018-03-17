import numpy as np

def standardize(X_data):
    X_data = (X_data - np.mean(X_data)) / np.std(X_data)
    return X_data