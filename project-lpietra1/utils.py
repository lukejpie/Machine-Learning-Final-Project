"""
Utils:
Author: Luke Pietrantonio
Date: 4/28/19
Description: Utility Functions
"""
from dataset import Dataset
from sklearn.model_selection import train_test_split


def split(X,y):
    """
    Description: splits all of the data into testing and training
    Params: Lists of X and y dataframes
    Returns: Lists of split X and y dataframes, for training and testing
    """
    num_datasets = len(X)

    X_train = [0] * num_datasets
    X_test = [0] * num_datasets
    y_train = [0] * num_datasets
    y_test = [0] * num_datasets

    for i in range(num_datasets):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.2)

    return X_train, X_test, y_train, y_test
