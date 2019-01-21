# Load libraries
import pandas as pd
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import statistics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def distibution(labels):

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T

def numberOfCombinations(hyperparameters):

    count = 1

    for parameter in hyperparameters:
        count = count * len(hyperparameters[parameter])

    return count

# Function for training
def train(estimators, trainFeatures, trainLabels):

    print("\nTraining estimators with best hyperparameter...")

    trainedEstimators = []

    # Train the models
    for estimator in estimators:
        trainedEstimator = estimator.fit(trainFeatures, trainLabels)
        trainedEstimators.append(trainedEstimator)

    print("done\n")

    return trainedEstimators

# Function for analyzing the performance/score of the model
def score(estimators, features, labels):

    reportEstimators = []
    accuracyEstimators = []
    roc_aucEstimators = []
    count = 0

    # Determine the performance of each estimator
    for estimator in estimators:
        count = count + 1
        print("\nEstimator " + str(count) + ":")
        predictions = estimator.predict(features)

        report = metrics.classification_report(labels, predictions)
        reportEstimators.append(report)
        print (report)

        accuracy = round(metrics.accuracy_score(labels, predictions), 3)
        accuracyEstimators.append(accuracy)
        print ("Accuracy:", accuracy)

        roc_auc = round(metrics.roc_auc_score(labels, predictions), 3)
        roc_aucEstimators.append(roc_auc)
        print ("ROC_AUC:", roc_auc)

    return reportEstimators, accuracyEstimators, roc_aucEstimators

# Method: Voting
def scoreEnsembledVoting(estimators, features, labels):

    predictions = np.zeros((features.shape[0], len(estimators)))

    column = 0

    # Make a prediction for each data point with each model
    for estimator in estimators:
        prediction = estimator.predict(features)
        predictions[:,column] = prediction

        column = column + 1

    predictionEnsembled = []

    for row in range(0, (features.shape[0])):
        nrClassOne = np.count_nonzero(predictions[row,:])
        if nrClassOne >= 3:
            predictionEnsembled.append(1)
        else:
            predictionEnsembled.append(0)

    report = metrics.classification_report(labels, predictionEnsembled)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictionEnsembled), 3)
    print ("Accuracy:", accuracy)

    roc_auc = round(metrics.roc_auc_score(labels, predictionEnsembled), 3)
    print ("ROC_AUC:", roc_auc)

    return report, accuracy, roc_auc

# Method: Averaging
def scoreEnsembledAveraging(estimators, features, labels):

    probabilities = np.zeros((features.shape[0], len(estimators)))

    column = 0

    # Make a prediction for each data point with each model
    for estimator in estimators:
        probability = estimator.predict_proba(features)[:,0]
        probabilities[:,column] = probability

        column = column + 1

    predictionEnsembled = []

    for row in range(0, (features.shape[0])):
        probabilityAvg = statistics.mean(probabilities[row,:])
        if probabilityAvg >= 0.6:
            predictionEnsembled.append(0)
        else:
            predictionEnsembled.append(1)

    report = metrics.classification_report(labels, predictionEnsembled)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictionEnsembled), 3)
    print ("Accuracy:", accuracy)

    roc_auc = round(metrics.roc_auc_score(labels, predictionEnsembled), 3)
    print ("ROC_AUC:", roc_auc)

    return report, accuracy, roc_auc

# Function to write results to excel
def saveResults(trainReport, trainAccuracy, trainROC_AUC,
                testReport, testAccuracy, testROC_AUC,
                estimators, fileNameResults):

    for i in range(0, len(estimators)):
        columns = [str(estimators[i]),
                str(trainReport[i]),
                str(testReport[i]),
                trainAccuracy[i],
                testAccuracy[i],
                trainROC_AUC[i],
                testROC_AUC[i]]

        book = openpyxl.load_workbook(fileNameResults)
        sheet = book.active
        sheet.append(columns)
        book.save(fileNameResults)


    print('\nResults saved as:')
    print(fileNameResults)

# ------------------------------------------------------------------------------

# Save locations
fileNameResults = "../results/12hr_ensembled.xlsx"

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

# Estimators to use
estimator_KNN = KNeighborsClassifier()
estimator_logisticRegression = LogisticRegression()
estimator_MLP = MLPClassifier()
estimator_randomForest = RandomForestClassifier()
estimator_SVM = SVC(probability = True)

# Algorithm Settings
cv = 5
scoring = "roc_auc"

# KNN ---------------------------------------------------------------------------------------------------
print("\nKNN:")
hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 10, 'p': 2}
estimator_KNN = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN = cross_val_score(estimator_KNN, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN.mean(), cross_score_KNN.std() * 2))

# logisticRegression -------------------------------------------------------------------------------------
print("\nlogisticRegression:")
hyperparameters_logisticRegression = {'C': 10.0, 'dual': False, 'fit_intercept': True, 'max_iter': 100000000, 'penalty': 'l2', 'solver': 'newton-cg'}
estimator_logisticRegression = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression = cross_val_score(estimator_logisticRegression, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression.mean(), cross_score_logisticRegression.std() * 2))

# MLP ---------------------------------------------------------------------------------------------------
print("\nMLP:")
hyperparameters_MLP = {'activation': 'relu', 'alpha':  1e-7, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 3000, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'sgd'}
estimator_MLP = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP = cross_val_score(estimator_MLP, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP.mean(), cross_score_MLP.std() * 2))

# randomForest --------------------------------------------------------------------------------------------
print("\nrandomForest:")
hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 2000}
estimator_randomForest = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest = cross_val_score(estimator_randomForest, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest.mean(), cross_score_randomForest.std() * 2))

# SVM ---------------------------------------------------------------------------------------------------
print("\nSVM:")
hyperparameters_SVM = {'C': 0.1, 'coef0': 7.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True}
estimator_SVM = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM = cross_val_score(estimator_SVM, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM.mean(), cross_score_SVM.std() * 2))

# Individual Models ---------------------------------------------------------------------------------------------------
# Train the estimators using optimal hyperparameter values
estimators = [estimator_KNN, estimator_logisticRegression, estimator_MLP, estimator_randomForest, estimator_SVM]
estimators = train(estimators, trainFeatures, trainLabels)

# Check the score of each individual model on the training and test set
print("\n\nIndividual models:")
print("\nScore on training set:")
trainReport, trainAccuracy, trainROC_AUC = score(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReport, testAccuracy, testROC_AUC = score(estimators, testFeatures, testLabels)

# EnsembledVoting ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Voting:")
print("\nScore on training set:")
trainReportEnsembled, trainAccuracyEnsembled, trainROC_AUCEnsembled = scoreEnsembledVoting(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReportEnsembled, testAccuracyEnsembled, testROC_AUCEnsembled = scoreEnsembledVoting(estimators, testFeatures, testLabels)

# Add the results of the ensembled models to the individual results
trainReport.append(trainReportEnsembled)
trainAccuracy.append(trainAccuracyEnsembled)
trainROC_AUC.append(trainROC_AUCEnsembled)
testReport.append(testReportEnsembled)
testAccuracy.append(testAccuracyEnsembled)
testROC_AUC.append(testROC_AUCEnsembled)

# EnsembledAveraging ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Averaging:")
print("\nScore on training set:")
trainReportEnsembled, trainAccuracyEnsembled, trainROC_AUCEnsembled = scoreEnsembledAveraging(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReportEnsembled, testAccuracyEnsembled, testROC_AUCEnsembled = scoreEnsembledAveraging(estimators, testFeatures, testLabels)

# Add the results of the ensembled models to the individual results
trainReport.append(trainReportEnsembled)
trainAccuracy.append(trainAccuracyEnsembled)
trainROC_AUC.append(trainROC_AUCEnsembled)
testReport.append(testReportEnsembled)
testAccuracy.append(testAccuracyEnsembled)
testROC_AUC.append(testROC_AUCEnsembled)
estimators.append("ensembledVoting")
estimators.append("ensembledAveraging")

# Save the results of the ensembled models
saveResults(trainReport, trainAccuracy, trainROC_AUC,
            testReport, testAccuracy, testROC_AUC,
            estimators, fileNameResults)
