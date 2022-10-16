import numpy as np


def load_data(path_dataset):
    """load data."""
    y = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1], converters={1: lambda x: 0 if b"b" in x else 1})
    x = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=range(2,32))
    return x, y


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
