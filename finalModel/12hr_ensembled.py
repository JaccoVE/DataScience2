# Load libraries
import numpy as np
import openpyxl
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import statistics
import math
import matplotlib.pyplot as plt

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
    count = 0

    # Train the models
    for estimator in estimators:
        trainedEstimator = estimator.fit(trainFeatures[count], trainLabels)
        trainedEstimators.append(trainedEstimator)
        count = count + 1

    print("done\n")

    return trainedEstimators

# Function for analyzing the performance/score of the model
def score(estimators, features, labels):

    reportEstimators = []
    accuracyEstimators = []
    roc_aucEstimators = []
    rocCurveEstimators_fpr = []
    rocCurveEstimators_tpr = []
    rocCurveEstimators_auc = []
    count = 0

    # Determine the performance of each estimator
    for estimator in estimators:
        print("\nEstimator " + str(count) + ":")
        predictions = estimator.predict(features[count])

        report = metrics.classification_report(labels, predictions)
        reportEstimators.append(report)
        print (report)

        accuracy = round(metrics.accuracy_score(labels, predictions), 3)
        accuracyEstimators.append(accuracy)
        print ("Accuracy:", accuracy)

        roc_auc = round(metrics.roc_auc_score(labels, predictions), 3)
        roc_aucEstimators.append(roc_auc)
        print ("ROC_AUC:", roc_auc)

        # ROC curve
        probability = estimator.predict_proba(features[count])[:,1]
        fpr, tpr, threshold = metrics.roc_curve(labels, probability, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        rocCurveEstimators_fpr.append(fpr)
        rocCurveEstimators_tpr.append(tpr)
        rocCurveEstimators_auc.append(auc)

        count = count + 1

    return reportEstimators, accuracyEstimators, roc_aucEstimators, rocCurveEstimators_fpr, rocCurveEstimators_tpr, rocCurveEstimators_auc

# Method: Voting
def scoreEnsembledVoting(estimators, features, labels):

    predictions = np.zeros((features[0].shape[0], len(estimators)))

    column = 0

    # Make a prediction for each data point with each model
    for estimator in estimators:
        prediction = estimator.predict(features[column])
        predictions[:,column] = prediction

        column = column + 1

    predictionEnsembled = []

    for row in range(0, (features[0].shape[0])):
        nrClassOne = np.count_nonzero(predictions[row,:])
        if nrClassOne >= math.ceil(len(estimators)/2.0):
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
def scoreEnsembledAveraging(estimators, features, labels, threshold):

    probabilities = np.zeros((features[0].shape[0], len(estimators)))

    column = 0

    # Make a prediction for each data point with each model
    for estimator in estimators:
        probability = estimator.predict_proba(features[column])[:,0]
        probabilities[:,column] = probability

        column = column + 1

    predictionEnsembled = []

    for row in range(0, (features[0].shape[0])):
        probabilityAvg = statistics.mean(probabilities[row,:])
        if probabilityAvg >= threshold:
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

def plotROC(testFPR, testTPR, testAUC, estimatorNames):
    for i in range(0,len(testFPR)):
        plt.figure()
        plt.plot(testFPR[i], testTPR[i], label='ROC curve (area = %0.2f)' % testAUC[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve ' + str(estimatorNames[i]))
        plt.legend(loc="lower right")
        plt.show()

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

# Select the best features for each individual model
trainFeatures_KNN = trainFeatures#[:,[3,5,16,18,20,21,22,24,33,42, 0,1,2,7,8,12,13,14,15,17,23,25,26,29,30,34,35,36]]
testFeatures_KNN = testFeatures#[:,[3,5,16,18,20,21,22,24,33,42, 0,1,2,7,8,12,13,14,15,17,23,25,26,29,30,34,35,36]]

trainFeatures_logisticRegression = trainFeatures#trainFeatures[:,[6,8,12,18,24,30,32,34,37,39,41,43]]
testFeatures_logisticRegression = testFeatures#testFeatures[:,[6,8,12,18,24,30,32,34,37,39,41,43]]
#trainFeatures_logisticRegression = np.delete(trainFeatures_logisticRegression, 42, axis=1)
#testFeatures_logisticRegression = np.delete(testFeatures_logisticRegression, 42, axis=1)

trainFeatures_MLP = trainFeatures#[:,[6,10,15,33,34,35,37,38,39, 13,17,40,43]]
testFeatures_MLP = testFeatures#[:,[6,10,15,33,34,35,37,38,39, 13,17,40,43]]

trainFeatures_randomForest = trainFeatures
testFeatures_randomForest = testFeatures

trainFeatures_SVM = trainFeatures#[:,[0,1,3,5,6,7,11,12,18,19,20,22,24,28,34,36,37, 2,4,8,13,14,15,16,17,21,23,25,26,27,29,30,31,32,33,35,38,40]]
testFeatures_SVM = testFeatures#[:,[0,1,3,5,6,7,11,12,18,19,20,22,24,28,34,36,37, 2,4,8,13,14,15,16,17,21,23,25,26,27,29,30,31,32,33,35,38,40]]

# KNN ---------------------------------------------------------------------------------------------------
print("\nKNN:")
hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 10, 'p': 2}
estimator_KNN = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN = cross_val_score(estimator_KNN, trainFeatures_KNN, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN.mean(), cross_score_KNN.std() * 2))

# logisticRegression -------------------------------------------------------------------------------------
print("\nlogisticRegression:")
hyperparameters_logisticRegression = {'C': 10.0, 'dual': False, 'fit_intercept': True, 'max_iter': 100000000, 'penalty': 'l2', 'solver': 'newton-cg'}
estimator_logisticRegression = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression = cross_val_score(estimator_logisticRegression, trainFeatures_logisticRegression, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression.mean(), cross_score_logisticRegression.std() * 2))

# MLP ---------------------------------------------------------------------------------------------------
print("\nMLP:")
hyperparameters_MLP = {'activation': 'relu', 'alpha':  1e-7, 'beta_1': 0.9, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 3000, 'momentum': 0.9, 'power_t': 0.5, 'random_state': 6, 'solver': 'sgd'}
estimator_MLP = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP = cross_val_score(estimator_MLP, trainFeatures_MLP, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP.mean(), cross_score_MLP.std() * 2))

# randomForest --------------------------------------------------------------------------------------------
print("\nrandomForest:")
hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 2000}
estimator_randomForest = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest = cross_val_score(estimator_randomForest, trainFeatures_randomForest, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest.mean(), cross_score_randomForest.std() * 2))

# SVM ---------------------------------------------------------------------------------------------------
print("\nSVM:")
hyperparameters_SVM = {'C': 0.1, 'coef0': 7.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True}
estimator_SVM = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM = cross_val_score(estimator_SVM, trainFeatures_SVM, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM.mean(), cross_score_SVM.std() * 2))

# Individual Models ---------------------------------------------------------------------------------------------------
# Train the estimators using optimal hyperparameter values
estimators = [  estimator_KNN,
                estimator_logisticRegression,
                estimator_MLP,
                estimator_randomForest,
                estimator_SVM]
estimatorNames = [  'KNN',
                    'Logistic Regression',
                    'MLP',
                    'Random Forest',
                    'SVM']
trainFeatures = [   trainFeatures_KNN,
                    trainFeatures_logisticRegression,
                    trainFeatures_MLP,
                    trainFeatures_randomForest,
                    trainFeatures_SVM]
testFeatures = [testFeatures_KNN,
                testFeatures_logisticRegression,
                testFeatures_MLP,
                testFeatures_randomForest,
                testFeatures_SVM]

estimators = train(estimators, trainFeatures, trainLabels)

# Check the score of each individual model on the training and test set
print("\n\nIndividual models:")
print("\nScore on training set:")
trainReport, trainAccuracy, trainROC_AUC, trainFPR, trainTPR, trainThreshold = score(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReport, testAccuracy, testROC_AUC, testFPR, testTPR, testAUC = score(estimators, testFeatures, testLabels)

# Plot the ROC curve of each estimator
plotROC(testFPR, testTPR, testAUC, estimatorNames)

# Save the results of the ensembled models
saveResults(trainReport, trainAccuracy, trainROC_AUC,
            testReport, testAccuracy, testROC_AUC,
            estimators, fileNameResults)

# EnsembledVoting ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Voting:")
print("\nScore on training set:")
trainReportEnsembled, trainAccuracyEnsembled, trainROC_AUCEnsembled = scoreEnsembledVoting(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReportEnsembled, testAccuracyEnsembled, testROC_AUCEnsembled = scoreEnsembledVoting(estimators, testFeatures, testLabels)

# Save the results of the ensembled models
saveResults([trainReportEnsembled], [trainAccuracyEnsembled], [trainROC_AUCEnsembled],
            [testReportEnsembled], [testAccuracyEnsembled], [testROC_AUCEnsembled],
            ["ensembledVoting"], fileNameResults)

# EnsembledAveraging ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Averaging:")

thresholds = [0.5, 0.55, 0.6, 0,65, 0.7, 0.75, 0.8]

for threshold in thresholds:

    print("\nScore on training set:")
    trainReportEnsembled, trainAccuracyEnsembled, trainROC_AUCEnsembled = scoreEnsembledAveraging(estimators, trainFeatures, trainLabels, threshold)

    print("\nScore on test set:")
    testReportEnsembled, testAccuracyEnsembled, testROC_AUCEnsembled = scoreEnsembledAveraging(estimators, testFeatures, testLabels, threshold)

    # Save the results of the ensembled models
    saveResults([trainReportEnsembled], [trainAccuracyEnsembled], [trainROC_AUCEnsembled],
                [testReportEnsembled], [testAccuracyEnsembled], [testROC_AUCEnsembled],
                ["ensembledAveraging " + str(threshold)], fileNameResults)
