"""
Run Random Forrest:
Author: Luke Pietrantonio
Created: 4/28/19
Description: Run all of the regression models on the sample data
"""

import matplotlib.pyplot as plt
import math as m
import numpy as np

from utils import *
from dataset import Dataset
from sklearn import utils
from sklearn.model_selection import cross_val_predict, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def rfr_model(X_train, y_train, X_test, y_test):

    rfr = RandomForestRegressor(max_depth=7, n_estimators=100,random_state=False, verbose=False)
    rfr.fit(X_train,y_train)

    predictions = rfr.predict(X_test)

    score = (rfr.score(X_test,y_test))
    #Prints out Feature Importance
    featureimportances = np.array(list(map(lambda x:round (x,11),rfr.feature_importances_)))
    return score,featureimportances

def all_models(X_train,y_train,X_test,y_test):
    """
    Runs all models regression
    Returns List of scores from each model and an averaged feature importance
    """
    scores = []

    #Keeps track of feature importance across all models
    fi = np.zeros(X_train[0].shape[1])


    for i in range(len(X_train)):
        score,fiT = rfr_model(X_train[i], y_train[i], X_test[i], y_test[i])
        scores.append(score)
        fi  = np.add(fi,fiT)

    print("Scores for season,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed:")
    fi = fi/len(X_train)
    print(fi)
    return scores, fi

def plotFI(featureimportances):
    objects = ('season','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed')

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, featureimportances, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Importance')
    plt.title('Feature Importance Over All Models')

    plt.show()

def main():
    divisions = 100
    ds = Dataset()
    ds.load("Bike-Sharing-Dataset/hour.csv")
    size = ds.get_size()

    X = []
    y =[]
    percentages = []
    #Full X and y from dataset
    all_X = ds.get_x()
    all_y = ds.get_y()


    #Shuffle data and split into divisons
    for i in range(1,divisions+1):
        percentage = (1/divisions * i)
        percentages.append(percentage)

        all_X,all_y = utils.shuffle(all_X,all_y)
        X.append(all_X[:int(size*percentage)])
        y.append(all_y[:int(size*percentage)])

    X_train, X_test, y_train, y_test = split(X,y)


    scores,featureimportances = all_models(X_train,y_train,X_test,y_test)
    print("scores")
    print(scores)

    plt.scatter(percentages,scores)
    plt.ylabel('Score')
    plt.xlabel('Percentage of Original Dataset')
    plt.title('Percentage of Original Dataset vs Score')
    plt.show()

    plotFI(featureimportances)


if __name__ == '__main__':
    main()
