import numpy as np
import cv2 as cv
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


def standardize(X_data):
    X_data = (X_data - np.mean(X_data)) / np.std(X_data)
    return X_data
    
def norm(X_data):
    for i in range(3):
        X_data[:, :, i] = Normalizer().fit_transform(X_data[:, :, i])
    return X_data

def scale(X_data):
    for i in range(3):
        X_data[:, :, i] = MinMaxScaler().fit_transform(X_data[:, :, i])
    return X_data
