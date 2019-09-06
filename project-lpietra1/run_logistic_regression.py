"""
Run Logistic Regression:
Author: Luke Pietrantonio
Created: 5/7/19
Description: Transforming the linear regression problem into a logistic problem by
having "high" and low days, as opposed to linear regression
"""
import math as m
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from dataset import Dataset
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

def create_regressions(num_regr):
    """
    Description: Creates all of instances of the regressions that we need
    Params: Number of regressions
    Returns: List of instantiated regressions
    """
    regrs = []
    for i in range(num_regr):
        regrs.append(LogisticRegression(solver='lbfgs',max_iter=500))
    return regrs

def train_regressions(regrs,X_train,y_train):
    """
    Description: Trains all of the regressions with each of the corresponding datasets
    Params: regrs-List of instantiated regressions, X_train-List of X_train datasets,
    y_train-List of y_train datasets
    Returns: Fitted regressions
    """
    i = 0
    for regr in regrs:
        regr.fit(X_train[i],y_train[i])
        i += 1
        #print('reg done')
    return regrs

def predict(regrs,X_test):
    """
    Description: Performs predictions for each regression on its appropriate test dataset
    Params: regrs-List of trained regressions, X_test-List of X_test datasets
    Returns: y_preds-List of list of predictions of y values from X_test
    """
    y_preds = []
    i = 0
    for regr in regrs:
        y_preds.append(regr.predict(X_test[i]))
        i += 1
        #print('pred done')

    return y_preds

def score(regrs,X_test,y_test):
    """
    Description: Returns the Scores of each regression
    Params: X_test, y_test
    Returns: List of scores
    """
    scores = []

    i=0
    for regr in regrs:
        scores.append(regr.score(X_test[i],y_test[i]))
        i+=1

    return scores

def linToLog(y):
    """
    Description: Reclassifies linear output as high or low
    Params: y-All of the y values from the dataset
    Returns: Classified y values
    """
    mid = y.median()

    y[y<=mid] = 0
    y[y> mid] = 1


    return y

def featureimportances(X_train,regrs):
    """
    Description: Averages all of the feature importances across all of the models
    Params: regrs-List of all of the models
    Returns: List of averaged feature importances
    """
    numFeatures = X_train[0].shape[1]
    fi = np.zeros(numFeatures)
    for i in range(len(regrs)):
        fiT = regrs[i].coef_.flatten()
        fiT = X_train[i].std(0)*fiT

        fi  = np.add(fi,fiT)
    fi = fi/len(regrs)
    return fi

def plotFI(featureimportances):
    objects = ('season','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed')

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, featureimportances, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Importance')
    plt.title('Feature Importance Over All Models')

    plt.show()

def main():

    divisions = 500
    ds = Dataset()
    ds.load("Bike-Sharing-Dataset/hour.csv")
    size = ds.get_size()

    X = []
    y =[]
    percentages = []
    #Full X and y from dataset
    all_X = ds.get_x()
    all_y = ds.get_y()

    print(all_y.median())
    #Transforms linear classifications into classifications
    all_y = linToLog(all_y)
    #Shuffle data and split into divisons
    for i in range(1,divisions+1):
        percentage = (1/divisions * i)
        percentages.append(percentage)

        all_X,all_y = utils.shuffle(all_X,all_y)
        X.append(all_X[:int(size*percentage)])
        y.append(all_y[:int(size*percentage)])

    #Splits training and testing data
    X_train, X_test, y_train, y_test = split(X,y)

    #Create List of Regressions
    regrs = create_regressions(divisions)
    #Train all regressions
    regrs = train_regressions(regrs,X_train,y_train)

    #Perform Predictions
    #y_pred = predict(regrs,X_test)

    # Calculate Scores from the Models
    scores = score(regrs,X_test,y_test)

    print('Scores:')
    print(scores)

    print('mean')
    print(y_test[0].mean(axis=0))

    # print(regrs[0].coef_)


    plt.scatter(percentages,scores)
    plt.ylabel('Score')
    plt.xlabel('Percentage of Original Dataset')
    plt.title('Percentage of Original Dataset vs Score')
    plt.show()

    fi = featureimportances(X_train,regrs)

    plotFI(fi)

if __name__ == '__main__':
    main()
