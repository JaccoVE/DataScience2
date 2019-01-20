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

hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}
clf = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN.mean(), cross_score_KNN.std() * 2))

# logisticRegression -------------------------------------------------------------------------------------
print("\nlogisticRegression:")

hyperparameters_logisticRegression = {'C': 1.0, 'dual': False, 'fit_intercept': True, 'max_iter': 100000000, 'penalty': 'l2', 'solver': 'newton-cg'}
clf = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression1.mean(), cross_score_logisticRegression1.std() * 2))

hyperparameters_logisticRegression = {'C': 0.1, 'dual': False, 'fit_intercept': True, 'max_iter': 100000000, 'penalty': 'l2', 'solver': 'newton-cg'}
clf = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression2.mean(), cross_score_logisticRegression2.std() * 2))

hyperparameters_logisticRegression = {'C': 0.1, 'dual': False, 'fit_intercept': True, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'newton-cg'}
clf = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression3 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression3.mean(), cross_score_logisticRegression3.std() * 2))

# MLP ---------------------------------------------------------------------------------------------------
print("\nMLP:")

hyperparameters_MLP = {'activation': 'relu', 'alpha': 1e-05, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.1, 'max_iter': 500, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'lbfgs', 'tol': 1e-05}
clf = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP1.mean(), cross_score_MLP1.std() * 2))

hyperparameters_MLP = {'activation': 'relu', 'alpha': 1e-07, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.0001, 'max_iter': 10000, 'momentum': 0.97, 'power_t': 0.5, 'random_state': 5, 'solver': 'sgd', 'tol': 1e-05}
clf = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP2.mean(), cross_score_MLP2.std() * 2))

# randomForest --------------------------------------------------------------------------------------------
print("\nrandomForest:")

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 46}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest1.mean(), cross_score_randomForest1.std() * 2))

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 19, 'max_features': 'sqrt', 'max_leaf_nodes': 12, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 55}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest2.mean(), cross_score_randomForest2.std() * 2))

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 12, 'max_features': 'sqrt', 'max_leaf_nodes': 9, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 60}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest3 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest3.mean(), cross_score_randomForest3.std() * 2))

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 15, 'max_features': 'sqrt', 'max_leaf_nodes': 11, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest4 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest4.mean(), cross_score_randomForest4.std() * 2))

hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 18, 'max_features': 'sqrt', 'max_leaf_nodes': 9, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 500}
clf = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest5 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest5.mean(), cross_score_randomForest5.std() * 2))

# SVM ---------------------------------------------------------------------------------------------------
print("\nSVM:")

hyperparameters_SVM = {'C': 0.001, 'coef0': -10, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True}
clf = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM1 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM1.mean(), cross_score_SVM1.std() * 2))

hyperparameters_SVM = {'C': 0.01, 'coef0': -15, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True}
clf = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM2 = cross_val_score(clf, trainFeatures, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM2.mean(), cross_score_SVM2.std() * 2))
