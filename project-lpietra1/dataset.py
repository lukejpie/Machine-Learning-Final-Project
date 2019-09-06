"""
Dataset Class:
Author: Luke Pietrantonio
Date: 4/18/19
Description: Dataset class to handle the training and testing data
"""

import pandas as pd

class Dataset:


    def __init__(self):
        self.X = None
        self.y = None
        self.size = None

    def get_x(self):
        return self.X

    def get_y(self):
        return self.y

    def get_size(self):
        return self.size

    def load(self, filename):
        df = pd.read_csv(filename)
        self.y = df['cnt']
        self.X = df.drop(columns = ['instant','cnt','casual','registered','dteday','yr'])
        #labels,uniques = pd.factorize(df['dteday'])
        #self.X['dteday'] = labels
        self.size = df.shape[0]
