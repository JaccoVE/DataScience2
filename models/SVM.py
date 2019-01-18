# Load libraries
import pandas as pd
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.svm import SVC

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
def saveResults(i, bestHyperparameters, trainReport, trainAccuracy, testReport,
            testAccuracy, clf, fileNameModel, fileNameResults):

    # save the model to disk
    joblib.dump(clf, str(fileNameModel) + str(i) + ".sav")

    print('\nModel saved as:')
    print(str(fileNameModel) + str(i) + ".sav")

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
fileNameModel = "../results/SVM"
fileNameResults = "../results/SVM.xlsx"

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
print("\nDistibution of the training data:")
print(distTrain)

print("\nDistibution of the test data:")
print(distTest)

# Estimator to use
estimator = SVC()

# Algorithm Settings
#n_iter = 42
cv = 5
scoring = "roc_auc"

# Combinations to test
combinations =  [1.83, 1.84]#[1.80, 1.82, 1.84, 1.86]
count = 0

for value in combinations:

    count = count + 1

    print("-------------------------------------------------------------------")
    print("\nIteration: " + str(count) + " of " + str(len(combinations)))

    # Hyperparameter combinations to test
    hyperparameters = { 'C': [value],
                        'kernel' : ['linear'],
                        'gamma' : ['auto', 'scale'],
                        'shrinking': [True]}

    print("\nHyperparameter combinations:")
    print("Testing " + str(numberOfCombinations(hyperparameters)) + " of " + str(numberOfCombinations(hyperparameters)) + " combinations")

    # Random Grid search
    bestHyperparameters  = randomGridSearch(estimator,
                                            hyperparameters,
                                            numberOfCombinations(hyperparameters),
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
    saveResults(count,
                bestHyperparameters,
                trainReport,
                trainAccuracy,
                testReport,
                testAccuracy,
                clf,
                fileNameModel,
                fileNameResults)
