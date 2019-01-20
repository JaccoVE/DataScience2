# Load libraries
import pandas as pd
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

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

# ------------------------------------------------------------------------------

# Save locations
fileNameModel = "../results/24hr_MLP.sav"
fileNameResults = "../results/24hr_MLP.xlsx"

# Load the train and test data
trainData = np.loadtxt("../data/24hr_train_data.txt")
testData = np.loadtxt("../data/24hr_test_data.txt")

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
estimator = MLPClassifier()

# Hyperparameter combinations to test
#hyperparameters = { 'hidden_layer_sizes': np.arange(5, 15, 1),
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : 10.0 ** -np.arange(6, 8),
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.01, 0.001],
#                    'power_t' : [0.5],
#                    'max_iter' : [3000],
#                    'random_state' : [4, 5, 6, 7, 8],
#                    'momentum' : [0.9, 0.95],
#                    'beta_1' : [0.9, 0.95]}

#hyperparameters = { 'hidden_layer_sizes': [10],
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : 10.0 ** -np.arange(3, 8),
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.01, 0.001, 0.0001, 0.00001],
#                    'power_t' : [0.5],
#                    'max_iter' : [3000],
#                    'random_state' : [4, 5, 6, 7, 8],
#                    'momentum' : np.arange(0.9, 0.99, 0.01),
#                    'beta_1' : np.arange(0.9, 0.99, 0.01)}

#hyperparameters = { 'hidden_layer_sizes': [10],
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : [10.0 ** -7],
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.0001],
#                    'power_t' : [0.5],
#                    'max_iter' : [3000],
#                    'random_state' : [5, 6],
#                    'momentum' : [0.97],
#                    'beta_1' : [0.9]}

#hyperparameters = { 'hidden_layer_sizes': [10],
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : [10.0 ** -7],
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.0001],
#                    'power_t' : [0.5],
#                    'max_iter' : [10000],
#                    'random_state' : [5, 6],
#                    'tol' : [10.0 ** -4, 10.0 ** -5, 10.0 ** -6],
#                    'momentum' : [0.97],
#                    'beta_1' : [0.9]}

#hyperparameters = { 'hidden_layer_sizes': [10],
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : [10.0 ** -7],
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.0001],
#                    'power_t' : [0.5],
#                    'max_iter' : [10000],
#                    'random_state' : [5, 6],
#                    'tol' : [10.0 ** -4, 10.0 ** -5, 10.0 ** -6],
#                    'momentum' : [0.97],
#                    'beta_1' : [0.9]}

#hyperparameters = { 'hidden_layer_sizes': [10],
#                    'activation' : ['relu'],
#                    'solver' : ['sgd'],
#                    'alpha' : [10.0 ** -7],
#                    'learning_rate' : ['constant'],
#                    'learning_rate_init': [0.0001],
#                    'power_t' : [0.5],
#                    'max_iter' : [1000000],
#                    'random_state' : [5, 6],
#                    'tol' : [10.0 ** -4, 10.0 ** -5, 10.0 ** -6],
#                    'momentum' : [0.97],
#                    'beta_1' : [0.9]}

hyperparameters = { 'hidden_layer_sizes': [10],
                    'activation' : ['relu'],
                    'solver' : ['lbfgs'],
                    'alpha' : 10.0 ** -np.arange(4, 8),
                    'learning_rate' : ['constant'],
                    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
                    'power_t' : [0.5],
                    'max_iter' : np.arange(500, 5000, 200),
                    'random_state' : np.arange(4, 8, 1),
                    'tol' : [10.0 ** -4, 10.0 ** -5, 10.0 ** -6],
                    'momentum' : [0.9],
                    'beta_1' : [0.9]}

# Algorithm Settings
n_iter = numberOfCombinations(hyperparameters) #1000
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
