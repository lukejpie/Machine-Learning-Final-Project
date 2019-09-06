"""
Run regression:
Author: Luke Pietrantonio
Created: 4/18/19
Description: Run all of the regression models on the sample data
"""
import math as m
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from dataset import Dataset
from sklearn import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def create_regressions(num_regr):
    """
    Description: Creates all of instances of the regressions that we need
    Params: Number of regressions
    Returns: List of instantiated regressions
    """
    regrs = []
    for i in range(num_regr):
        regrs.append(LinearRegression())
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

def error(y_preds,y_tests):
    """
    Description: Computes and returns the RMSE of each regression
    Params: y_preds-List of lists of y predictions, y_tests-List of lists of y test values
    Returns: List of errors
    """
    errors = []

    for i in range(len(y_preds)):
        errors.append(m.sqrt(mean_squared_error(y_tests[i],y_preds[i]))/y_tests[i].mean())

    return errors

def score(X_tests,y_tests,regrs):
    """
    Description: Computes and returns the score of each regression
    Params: X_test-List of lists of X test values, y_tests-List of lists of y test values
    Returns: List of scores
    """
    scores = []

    for i in range(len(X_tests)):
        scores.append(regrs[i].score(X_tests[i],y_tests[i]))

    return scores

def poly(X_train,X_test,deg):
    X_train_poly=[]
    X_test_poly=[]
    poly = PolynomialFeatures(degree=deg)

    for i in range(len(X_train)):
        X_train_poly.append(poly.fit_transform(X_train[i]))
        X_test_poly.append(poly.fit_transform(X_test[i]))

    return X_train_poly,X_test_poly

def featureimportances(X_train,regrs):
    """
    Description: Averages all of the feature importances across all of the models
    Params: regrs-List of all of the models
    Returns: List of averaged feature importances
    """
    numFeatures = X_train[0].shape[1]
    fi = np.zeros(numFeatures)
    for i in range(len(regrs)):
        fiT = regrs[i].coef_
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


    #Shuffle data and split into divisons
    for i in range(1,divisions+1):
        percentage = (1/divisions * i)
        percentages.append(percentage)

        all_X,all_y = utils.shuffle(all_X,all_y)
        X.append(all_X[:int(size*percentage)])
        y.append(all_y[:int(size*percentage)])

    #Splits training and testing data
    X_train, X_test, y_train, y_test = split(X,y)

    print(X_train[0].shape)

    #Transforms the data into polynomial features
    #X_train,X_test = poly(X_train,X_test,1)

    print(X_train[0].shape)
    #Create List of Regressions
    regrs = create_regressions(divisions)
    #Train all regressions
    regrs = train_regressions(regrs,X_train,y_train)

    #Perform Predictions
    #y_pred = predict(regrs,X_test)

    # Calculate Errors from the Models
    #errors = error(y_pred,y_test)

    # Calcualte Scores from all of the Models
    scores = score(X_test,y_test,regrs)

    # print('Errors:')
    # print(errors)

    print('Scores:')
    print(scores)

    #Averages Feature importances across all models
    fi = featureimportances(X_train,regrs)
    print('Feature importances:')
    print(fi)

    plt.scatter(percentages,scores)
    plt.ylabel('Score')
    plt.xlabel('Percentage of Original Dataset')
    plt.title('Percentage of Original Dataset vs Score')
    plt.show()

    plotFI(fi)


if __name__ == '__main__':
    main()
