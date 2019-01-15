# Load libraries
import time
import pandas as pd
import numpy as np
import json
import openpyxl

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
from sklearn.svm import SVC

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
def randomGridSearch(estimator, C, kernel, degree, coef0, n_iter, cv, scoring, trainFeatures, trainLabels):

    print("Performing randomGridSearch...")

    # Estimator that is used
    clf = estimator

    # Random grid
    random_grid = {'C': C,
                   'kernel': kernel,
                   'degree': degree,
                   'coef0': coef0}

    # Randomzed grid search of the hyperparameters
    rf_random = RandomizedSearchCV( estimator = clf,
                                    param_distributions = random_grid,
                                    scoring = scoring,
                                    n_iter = n_iter,
                                    verbose = 1,
                                    cv = cv,
                                    n_jobs= -1)

    # Train the numerous models
    rf_random.fit(trainFeatures, trainLabels)

    # Store the best hyperparameters
    best_hp = rf_random.best_params_

    print("done\n")

    return best_hp

# Function for training
def train(best_hp, trainFeatures, trainLabels):

    print("Training with best hyperparameter...")

    # Setup the model with the best hyperparameter
    clf = SVC(C=best_hp['C'],kernel=best_hp['kernel'],degree=best_hp['degree'],coef0=best_hp['coef0'])

    # Train the model
    clf.fit(trainFeatures, trainLabels)

    print("done\n")

    return clf

# Function for analyzing the performance/score of the model
def score(clf, features, labels):

    print("Evaluating the score of the model...")

    score = clf.score(features, labels)

    print("done\n")

    return score

# Function to write results to excel
def saveResults(best_hp, scoreTrain, scoreTest, clf, fileNameModel, fileNameResults):

    # save the model to disk
    joblib.dump(clf, fileNameModel)

    print('Model saved as:')
    print(fileNameModel)

    properties_model = [best_hp['C'],
                        best_hp['kernel'],
                        best_hp['degree'],
                        best_hp['coef0'],
                        scoreTrain,
                        scoreTest]

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
fileNameModel = '../results/SVM.sav'
fileNameResults = '../results/SVM.xlsx'

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
estimator = SVC(gamma='auto')

# Random Grid Search Settings
C = [0.001, 0.01, 0.1, 1, 10]
kernel = ['linear', 'poly', 'rbf']
degree = [2,4,8,10]
coef0 = [-8,-4,-2,2,4,8]
n_iter = 1000
cv = 5
scoring = "roc_auc"

# Random Grid search
best_hp = randomGridSearch( estimator,
                            C,
                            kernel,
                            degree,
                            coef0,
                            n_iter,
                            cv,
                            scoring,
                            trainFeatures,
                            trainLabels)

# Train the model
clf = train(best_hp, trainFeatures, trainLabels)

# Check the score on the model on the training and test set
scoreTrain = score(clf, trainFeatures, trainLabels)
scoreTest = score(clf, testFeatures, testLabels)

# Print the score of the model
print("Score on training set: " + str(scoreTrain))
print("Score on test set: " + str(scoreTest))

# Save results
saveResults(best_hp, scoreTrain, scoreTest, clf, fileNameModel, fileNameResults)
