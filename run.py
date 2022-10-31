import numpy as np
from implementations import *
from logistic_regression import *
from helpers import *

def imputer(data):
    data[data == -999] = np.nan
    col_mean = np.nanmedian(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])
    return data

def build_poly(poly, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    """
    for i in range(poly.shape[1]):
        for j in range(2, degree + 1):
            poly = np.concatenate((poly, np.array(poly[:, i]**j).reshape(poly.shape[0], 1)), axis=1)
    return poly

def replace(data, value, new_value):
    data[data == value] = new_value
    return data

def one_hot_missing(x):
    features_with_missing = np.count_nonzero(x == -999, axis = 0)
    duplicates = np.unique(features_with_missing)
    col1 = []
    cols_1 = []
    for i in range(len(duplicates)):
        col1.append(len(np.argwhere(features_with_missing == np.amax(features_with_missing)).flatten().tolist()))
        cols_1.append([1 if t==-999 else 0 for t in x[:, i]])
        features_with_missing = np.delete(features_with_missing, col1, axis=0)
        
    return col1, cols_1


def preprocess(x, y, degree):
    # 1
    col_missing, representation = one_hot_missing(x)
    # 2 
    x = imputer(x)
    col = []
    for i in np.unique(x[:, -8]):
        col.append([1 if t==i else 0 for t in x[:, -8]])
    x = np.delete(x, 22, axis = 1)
    # 3
    x_log_transform = np.sign(x)*np.log(np.abs(x)+1) 
    x = np.hstack([x, x_log_transform])
    # 4
    x = build_poly(x, degree)
    # 5
    x = standardize(x)[0]
    # 6
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    for i in range(len(col_missing)):
        x = np.concatenate((x, np.array(representation[i]).reshape(len(representation[i]), 1)), axis=1)
    x = np.concatenate((x, np.array(col).reshape(len(col[-1]), 4)), axis = 1)
    # 7
    y = replace(y, -1, 0)
    return x, y

def run():
    y_train, x_train, _ = load_csv_data("data/train.csv")
    y_test, x_test, id = load_csv_data("data/test.csv")
    x_train, y_train = preprocess(x_train, y_train, 2)
    x_test, y_test = preprocess(x_test, y_test, 2)

    model_lg = Logistic_Regression(0.35)
    model_lg.train(x_train, y_train)
    model_lg.test(x_test, y_test)
    predictions = replace(model_lg.test_predictions, 0, -1)
    create_csv_submission(id, predictions, "sample-submission_final.csv")