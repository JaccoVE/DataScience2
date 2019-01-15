# Load libraries
import time
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
from sklearn.ensemble import RandomForestClassifier

def scalingData(trainData):

    # https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm
    # Standardizing the features using Standard Scaler
    features = StandardScaler().fit_transform(trainData[:,1:45])
    labels = trainData[:,0]
    trainDataStandard = np.c_[labels, trainData]

    # Standardizing the features using MinMax Scaler
    features =  MinMaxScaler().fit_transform(trainData[:,1:45])
    labels = trainData[:,0]
    trainDataMinMax = np.c_[labels, trainData]

    # Standardizing the features using Normalizer Scaler
    features = Normalizer().fit_transform(trainData[:,1:45])
    labels = trainData[:,0]
    trainDataNorm = np.c_[labels, trainData]

    return trainDataStandard, trainDataMinMax, trainDataNorm

def distibution(labels):

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T

# Function for a random hyperparameter gird search
def randomGridSearch(estimator, hyperparameters, n_iter, cv, scoring, trainFeatures, trainLabels):

    print("Performing randomGridSearch...")

    # Randomzed grid search of the hyperparameters
    clf = RandomizedSearchCV(estimator = estimator,
                            param_distributions = hyperparameters,
                            n_iter = n_iter,
                            cv = cv,
                            scoring = scoring,
                            verbose = 1,
                            n_jobs= -1)

    # Train the numerous models
    clf.fit(trainFeatures, trainLabels)

    # Identify optimal hyperparameter values
    best_n_estim      = clf.best_params_['n_estimators']
    best_max_features = clf.best_params_['max_features']

    print("The best performing n_estimators value is: {:5d}".format(best_n_estim))
    print("The best performing max_features value is: {:5d}".format(best_max_features))

    return best_n_estim, best_max_features

# Function for training
def train(estimator, best_n_estim, best_max_features, trainFeatures, trainLabels):

    print("Training with best hyperparameter...")

    # Setup the model with the best hyperparameter
    clf = estimator.set_params( n_estimators=best_n_estim,
                                max_features=best_max_features)

    # Train the model
    clf.fit(trainFeatures, trainLabels)

    print("done\n")

    return clf

# Function for analyzing the performance/score of the model
def score(clf, features, labels):

    print("Evaluating the score of the model...")

    predictions = clf.predict(features)

    report = metrics.classification_report(features, predictions)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictions), 3)
    print ("Overall Accuracy:", accuracy)

    return report, accuracy

# Function to write results to excel
def saveResults(best_n_estim, best_max_features, trainReport, trainAccuracy, testReport,
            testAccuracy, clf, fileNameModel, fileNameResults):

    # save the model to disk
    joblib.dump(clf, fileNameModel)

    print('Model saved as:')
    print(fileNameModel)

    properties_model = [best_n_estim,
                        best_max_features,
                        trainAccuracy,
                        testAccuracy]

    book = openpyxl.load_workbook(fileNameResults)
    sheet = book.active
    sheet.append(properties_model)
    book.save(fileNameResults)
    time.sleep(0.1)

    print('Results saved as:')
    print(fileNameResults)

    return "Files successfully saved"

# ------------------------------------------------------------------------------

# Save locations
fileNameModel = '../results/randomForest.sav'
fileNameResults = '../results/randomForest.xlsx'

# Load the train and test data
trainData = np.loadtxt("../data/train_data.txt")
testData = np.loadtxt("../data/test_data.txt")

# Split the features and labels
trainFeatures = trainData[:,1:45]
trainLabels = trainData[:,0].astype(int)

testFeatures = testData[:,1:45]
testLabels = testData[:,0].astype(int)

# Check the data distibution
distTrain = distibution(trainLabels)
distTest = distibution(testLabels)

# Print data distibution
print("Distibution of the training data:")
print(distTrain)

print("Distibution of the test data:")
print(distTest)

# Estimator to use
estimator = RandomForestClassifier()

# Random Grid Search Settings
n_estimators = np.random.uniform(70, 80, 5).astype(int)
max_features = np.random.normal(6, 3, 5).astype(int)

# Check max_features>0 & max_features<=total number of features
max_features[max_features <= 0] = 1
max_features[max_features > trainFeatures.shape[1]] = trainFeatures.shape[1]

hyperparameters = {'n_estimators': list(n_estimators),
                   'max_features': list(max_features)}

print (hyperparameters)

# Algorithm Settings
n_iter = 1000
cv = 5
scoring = "roc_auc"

# Random Grid search
best_n_estim, best_max_features  = randomGridSearch(estimator,
                                                    hyperparameters,
                                                    n_iter,
                                                    cv,
                                                    scoring,
                                                    trainFeatures,
                                                    trainLabels)

# Train classifier using optimal hyperparameter values
clf = train(estimator, best_n_estim, best_max_features, trainFeatures, trainLabels)

# Check the score on the model on the training and test set
print("Score on training set:")
trainReport, trainAccuracy = score(clf, trainFeatures, trainLabels)

print("Score on test set:")
testReport, testAccuracy = score(clf, testFeatures, testLabels)

# Save results
saveResults(best_n_estim,
            best_max_features,
            trainReport,
            trainAccuracy,
            testReport,
            testAccuracy,
            clf,
            fileNameModel,
            fileNameResults)
