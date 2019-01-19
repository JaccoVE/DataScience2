# Load libraries
import pandas as pd
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def distibution(labels):

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T

def numberOfCombinations(hyperparameters):

    count = 1

    for parameter in hyperparameters:
        count = count * len(hyperparameters[parameter])

    return count

# Function for a random hyperparameter gird search
def randomGridSearch(estimator, hyperparameters, n_iter, cv, scoring, trainFeatures, trainLabels):

    print("\nPerforming randomGridSearch...")

    # Randomzed grid search of the hyperparameters
    #clf = RandomizedSearchCV(estimator = estimator,
    #                        param_distributions = hyperparameters,
    #                        n_iter = n_iter,
    #                        cv = cv,
    #                        scoring = scoring,
    #                        verbose = 1,
    #                        n_jobs= 23)

    clf = GridSearchCV(estimator = estimator,
                            param_grid = hyperparameters,
                            cv = cv,
                            scoring = scoring,
                            verbose = 1,
                            n_jobs= 23)

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

    print('\nResults saved as:')
    print(fileNameResults)

    return "Files successfully saved"

# ------------------------------------------------------------------------------

# Save locations
fileNameModel = "../results/12hr_logisticRegression.sav"
fileNameResults = "../results/12hr_logisticRegression.xlsx"

# Load the train and test data
trainData = np.loadtxt("../data/12hr_train_data.txt")
testData = np.loadtxt("../data/12hr_test_data.txt")

# Split the features and labels
trainFeatures = trainData[:,1:45]
trainLabels = trainData[:,0].astype(int)

testFeatures = testData[:,1:45]
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
                    'C' : [1.0, 10.0, 100.0],
                    'fit_intercept' : [True, False],
                    'solver' : ['newton-cg', 'lbfgs', 'sag'],
                    'max_iter' : [100000000]}

# Algorithm Settings
n_iter = 10000
cv = 5
scoring = "roc_auc"

print("\nHyperparameter combinations:")
print("Testing " + str(n_iter) + " of " + str(numberOfCombinations(hyperparameters)) + " combinations")

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
