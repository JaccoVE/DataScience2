# Load libraries
import time
import os
import pandas as pd
import numpy as np
import json
import openpyxl

from scipy.stats import uniform
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def scaleFeatures(features):

    # https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm
    # Standardizing the features using Standard Scaler
    features = StandardScaler().fit_transform(features)

    # Standardizing the features using MinMax Scaler
    #features =  MinMaxScaler().fit_transform(features)

    # Standardizing the features using Normalizer Scaler
    #features = Normalizer().fit_transform(features)

    return features

def distibution(labels):

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T

def combinations(hyperparameters):

    count = 1

    for parameter in hyperparameters:
        count = count * len(hyperparameters[parameter])

    return count

# Function for a random hyperparameter gird search
def randomGridSearch(estimator, hyperparameters, n_iter, cv, scoring, trainFeatures, trainLabels):

    print("\nPerforming randomGridSearch...")

    # Randomzed grid search of the hyperparameters
    clf = RandomizedSearchCV(estimator = estimator,
                            param_distributions = hyperparameters,
                            n_iter = n_iter,
                            cv = cv,
                            scoring = scoring,
                            verbose = 1,
                            n_jobs = -1,
                            error_score=np.nan,
                            iid = False)

    # Train the numerous models
    clf.fit(trainFeatures, trainLabels)

    # Identify optimal hyperparameter values
    bestHyperparameters = clf.best_params_

    print("The best performing hyperparameters values are:")
    print(bestHyperparameters)

    return bestHyperparameters

# Function for training
def train(estimator, bestHyperparameters, trainFeatures, trainLabels):

    print("\nTraining with best hyperparameter...")

    # Setup the model with the best hyperparameter
    clf = estimator.set_params( **bestHyperparameters)

    # Train the model
    clf.fit(trainFeatures, trainLabels)

    print("done\n")

    return clf

# Function for analyzing the performance/score of the model
def score(clf, features, labels):

    print("\nEvaluating the score of the model...")

    predictions = clf.predict(features)

    report = metrics.classification_report(labels, predictions)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictions), 3)
    print ("Overall Accuracy:", accuracy)

    return report, accuracy

# Function to write results to excel
def saveResults(bestHyperparameters, trainReport, trainAccuracy, testReport,
            testAccuracy, clf, fileNameModel, fileNameResults):

    # save the model to disk
    joblib.dump(clf, fileNameModel)

    print('\nModel saved as:')
    print(fileNameModel)

    properties_model = [str(bestHyperparameters),
                        trainAccuracy,
                        testAccuracy]

    book = openpyxl.load_workbook(fileNameResults)
    sheet = book.active
    sheet.append(properties_model)
    book.save(fileNameResults)
    time.sleep(0.1)

    print('\nResults saved as:')
    print(fileNameResults)

    return "Files successfully saved"

# ------------------------------------------------------------------------------

# Save locations
fileNameModel = "../results/logisticRegression.sav"
fileNameResults = "../results/logisticRegression.xlsx"

# Load the train and test data
trainData = np.loadtxt("../data/train_data.txt")
testData = np.loadtxt("../data/test_data.txt")

# Split the features and labels
trainFeatures = scaleFeatures(trainData[:,1:45])
trainLabels = trainData[:,0].astype(int)

testFeatures = scaleFeatures(testData[:,1:45])
testLabels = testData[:,0].astype(int)

# Check the data distibution
distTrain = distibution(trainLabels)
distTest = distibution(testLabels)

# Print data distibution
print("\nDistibution of the training data:")
print(distTrain)

print("\nDistibution of the test data:")
print(distTest)

# Estimator to use
estimator = LogisticRegression()

# Hyperparameter combinations to test
hyperparameters = { 'penalty': ['l2'],
                    'dual' : [False],
                    'C' : [0.001, 0.01, 0.05, 0.07, 0.09, 0.1, 1, 10, 100],
                    'fit_intercept' : [True, False],
                    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter' : [1000000000]}

print("\nPossible hyperparameter combinations:")
print(str(combinations(hyperparameters)))

# Algorithm Settings
n_iter = combinations(hyperparameters)
cv = 5
scoring = "roc_auc"

# Random Grid search
bestHyperparameters  = randomGridSearch(estimator,
                                        hyperparameters,
                                        n_iter,
                                        cv,
                                        scoring,
                                        trainFeatures,
                                        trainLabels)

# Train classifier using optimal hyperparameter values
clf = train(estimator, bestHyperparameters, trainFeatures, trainLabels)

# Check the score on the model on the training and test set
print("\nScore on training set:")
trainReport, trainAccuracy = score(clf, trainFeatures, trainLabels)

print("\nScore on test set:")
testReport, testAccuracy = score(clf, testFeatures, testLabels)

# Save results
saveResults(bestHyperparameters,
            trainReport,
            trainAccuracy,
            testReport,
            testAccuracy,
            clf,
            fileNameModel,
            fileNameResults)
