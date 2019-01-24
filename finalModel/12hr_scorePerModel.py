# Load libraries
import pandas as pd
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score

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

# Function for analyzing the performance/score of the model
def score(clf, features, labels):

    print("\nEvaluating the score of the model...")

    predictions = clf.predict(features)

    report = metrics.classification_report(labels, predictions)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictions), 3)
    print ("Overall Accuracy:", accuracy)

    return report, accuracy

# ------------------------------------------------------------------------------

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
estimator_SVM = SVC()

# Algorithm Settings
cv = 5
scoring = "roc_auc"

# KNN ---------------------------------------------------------------------------------------------------
print("\nKNN:")

hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 10, 'p': 2}
clf = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN1.mean(), cross_score_KNN1.std() * 2))

hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 11, 'p': 1, 'weights': 'distance'}
clf = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN2.mean(), cross_score_KNN2.std() * 2))

# logisticRegression -------------------------------------------------------------------------------------
print("\nlogisticRegression:")

hyperparameters_logisticRegression = {'C': 10.0, 'dual': False, 'fit_intercept': True, 'max_iter': 100000000, 'penalty': 'l2', 'solver': 'newton-cg'}
clf = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression.mean(), cross_score_logisticRegression.std() * 2))

# MLP ---------------------------------------------------------------------------------------------------
print("\nMLP:")

hyperparameters_MLP = {'activation': 'relu', 'alpha': 1e-07, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 3000, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'sgd'}
clf = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP1.mean(), cross_score_MLP1.std() * 2))

hyperparameters_MLP = {'activation': 'relu', 'alpha': 1e-06, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 3000, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'lbfgs'}
clf = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP2.mean(), cross_score_MLP2.std() * 2))

hyperparameters_MLP = {'activation': 'relu', 'alpha': 1e-06, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.1, 'max_iter': 500, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'lbfgs'}
clf = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP3 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP3.mean(), cross_score_MLP3.std() * 2))

# randomForest --------------------------------------------------------------------------------------------
print("\nrandomForest:")

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 2000}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest.mean(), cross_score_randomForest.std() * 2))

# SVM ---------------------------------------------------------------------------------------------------
print("\nSVM:")

hyperparameters_SVM = {'C': 0.1, 'coef0': 7.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True}
clf = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM.mean(), cross_score_SVM.std() * 2))
