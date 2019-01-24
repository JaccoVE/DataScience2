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
def score(estimators, estimatorNames, features, labels):

    reportEstimators = []
    accuracyEstimators = []
    precisionEstimators = []
    recallEstimators = []
    aucEstimators = []
    count = 0

    # Determine the performance of each estimator
    for estimator in estimators:
        print("\nEstimator " + str(count) + ":")
        predictions = estimator.predict(features[count])

        # Make and print report
        report = metrics.classification_report(labels, predictions)
        reportEstimators.append(report)
        print (report)

        # Calculate and report accuracy
        accuracy = round(metrics.accuracy_score(labels, predictions), 3)
        accuracyEstimators.append(accuracy)
        print ("Accuracy:", accuracy)

        # Determine precision and recall and print both
        conf_matrix = metrics.confusion_matrix(labels, predictions)

        TP = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        TN = conf_matrix[1][1]

        precision = round(TP / (TP + FP), 3)
        recall = round(TP / (TP + FN), 3)

        precisionEstimators.append(precision)
        recallEstimators.append(recall)
        print ("Precision:", precision)
        print ("Recall:", recall)

        # Make ROC curve
        probability = estimator.predict_proba(features[count])[:,1]
        fpr, tpr, threshold = metrics.roc_curve(labels, probability, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        aucEstimators.append(auc)
        print ("AUC:", auc)

        plotROC(fpr, tpr, auc, estimatorNames[count])

        count = count + 1

    return reportEstimators, accuracyEstimators, precisionEstimators, recallEstimators, aucEstimators

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

    conf_matrix = metrics.confusion_matrix(labels, predictionEnsembled)

    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]

    precision = round(TP / (TP + FP), 3)
    recall = round(TP / (TP + FN), 3)
    auc = "NaN"

    print ("Precision:", precision)
    print ("Recall:", recall)

    return report, accuracy, precision, recall, auc


# Method: Averaging
def scoreEnsembledAveraging(estimators, features, labels, threshold = 0.5):

    probabilities = np.zeros((features[0].shape[0], len(estimators)))

    column = 0

    # Make a prediction for each data point with each model
    for estimator in estimators:
        probability = estimator.predict_proba(features[column])[:,0]
        probabilities[:,column] = probability

        column = column + 1

    predictionEnsembled = []
    probabilityAvgEnsembled = []

    for row in range(0, (features[0].shape[0])):
        probabilityAvg = statistics.mean(probabilities[row,:])
        probabilityAvgEnsembled.append(probabilityAvg)
        if probabilityAvg >= threshold:
            predictionEnsembled.append(0)
        else:
            predictionEnsembled.append(1)

    report = metrics.classification_report(labels, predictionEnsembled)
    print (report)

    accuracy = round(metrics.accuracy_score(labels, predictionEnsembled), 3)
    print ("Accuracy:", accuracy)

    conf_matrix = metrics.confusion_matrix(labels, predictionEnsembled)

    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]

    precision = round(TP / (TP + FP), 3)
    recall = round(TP / (TP + FN), 3)

    print ("Precision:", precision)
    print ("Recall:", recall)

    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(labels, probabilityAvgEnsembled, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print ("ROC_AUC:", auc)

    # Plot the ROC curve of each estimator
    plotROC(fpr, tpr, auc, "Averaging")

    return report, accuracy, precision, recall, auc

def plotROC(testFPR, testTPR, testAUC, estimatorName):
    plt.figure()
    plt.plot(testFPR, testTPR, label='ROC curve (area = %0.2f)' % testAUC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + str(estimatorName))
    plt.legend(loc="lower right")
    plt.show()

# Function to write results to excel
def saveResults(trainReport, trainAccuracy, trainPrecision, trainRecall, trainAUC,
                testReport, testAccuracy, testPrecision, testRecall, testAUC,
                estimators, fileNameResults):

    for i in range(0, len(estimators)):
        columns = [str(estimators[i]),
                trainAUC[i],
                testAUC[i],
                trainAccuracy[i],
                testAccuracy[i],
                trainPrecision[i],
                testPrecision[i],
                trainRecall[i],
                testRecall[i],
                str(trainReport[i]),
                str(testReport[i])]

        book = openpyxl.load_workbook(fileNameResults)
        sheet = book.active
        sheet.append(columns)
        book.save(fileNameResults)


    print('\nResults saved as:')
    print(fileNameResults)

# ------------------------------------------------------------------------------

# Save locations
fileNameResults = "../results/24hr_ensembled2.xlsx"

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
estimator_SVM = SVC(probability = True)

# Algorithm Settings
cv = 5
scoring = "roc_auc"

# Select the best features for each individual model
trainFeatures_KNN = trainFeatures#[:,[3,5,16,18,20,21,22,24,33,42, 0,1,2,7,8,12,13,14,15,17,23,25,26,29,30,34,35,36]]
testFeatures_KNN = testFeatures#[:,[3,5,16,18,20,21,22,24,33,42, 0,1,2,7,8,12,13,14,15,17,23,25,26,29,30,34,35,36]]

trainFeatures_logisticRegression = trainFeatures#[:,[6,8,12,18,24,30,32,34,37,39,41,43]]
testFeatures_logisticRegression = testFeatures#[:,[6,8,12,18,24,30,32,34,37,39,41,43]]
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
hyperparameters_KNN = {'algorithm': 'auto', 'leaf_size': 1, 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}
estimator_KNN = estimator_KNN.set_params( **hyperparameters_KNN)
cross_score_KNN = cross_val_score(estimator_KNN, trainFeatures_KNN, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_KNN.mean(), cross_score_KNN.std() * 2))

# logisticRegression -------------------------------------------------------------------------------------
print("\nlogisticRegression:")
hyperparameters_logisticRegression = {'C': 0.1, 'dual': False, 'fit_intercept': True, 'max_iter': 10000, 'penalty': 'l2', 'solver': 'newton-cg'}
estimator_logisticRegression = estimator_logisticRegression.set_params( **hyperparameters_logisticRegression)
cross_score_logisticRegression = cross_val_score(estimator_logisticRegression, trainFeatures_logisticRegression, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_logisticRegression.mean(), cross_score_logisticRegression.std() * 2))

# MLP ---------------------------------------------------------------------------------------------------
print("\nMLP:")
hyperparameters_MLP = {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': 10, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'lbfgs'}
estimator_MLP = estimator_MLP.set_params( **hyperparameters_MLP)
cross_score_MLP = cross_val_score(estimator_MLP, trainFeatures_MLP, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_MLP.mean(), cross_score_MLP.std() * 2))

# randomForest --------------------------------------------------------------------------------------------
print("\nrandomForest:")
hyperparameters_randomForest = {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 2000}
estimator_randomForest = estimator_randomForest.set_params( **hyperparameters_randomForest)
cross_score_randomForest = cross_val_score(estimator_randomForest, trainFeatures_randomForest, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_randomForest.mean(), cross_score_randomForest.std() * 2))

# SVM ---------------------------------------------------------------------------------------------------
print("\nSVM:")
hyperparameters_SVM = {'C': 0.01, 'coef0': -15, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': True}
estimator_SVM = estimator_SVM.set_params( **hyperparameters_SVM)
cross_score_SVM = cross_val_score(estimator_SVM, trainFeatures_SVM, trainLabels, cv=cv, n_jobs = -1, scoring = scoring)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (cross_score_SVM.mean(), cross_score_SVM.std() * 2))

# Individual Models ---------------------------------------------------------------------------------------------------
# Train the estimators using optimal hyperparameter values
estimators = [  #estimator_KNN,
                estimator_logisticRegression,
                #estimator_MLP,
                estimator_randomForest,
                estimator_SVM]
estimatorNames = [  #'KNN',
                    'Logistic Regression',
                    #'MLP',
                    'Random Forest',
                    'SVM']
trainFeatures = [   #trainFeatures_KNN,
                    trainFeatures_logisticRegression,
                    #trainFeatures_MLP,
                    trainFeatures_randomForest,
                    trainFeatures_SVM]
testFeatures = [#testFeatures_KNN,
                testFeatures_logisticRegression,
                #testFeatures_MLP,
                testFeatures_randomForest,
                testFeatures_SVM]

estimators = train(estimators, trainFeatures, trainLabels)

# Check the score of each individual model on the training and test set
print("\n\nIndividual models:")
print("\nScore on training set:")
trainReport, trainAccuracy, trainPrecision, trainRecall, trainAUC = score(estimators, estimatorNames, trainFeatures, trainLabels)

print("\nScore on test set:")
testReport, testAccuracy, testPrecision, testRecall, testAUC = score(estimators, estimatorNames, testFeatures, testLabels)

# Save the results of the ensembled models
saveResults(trainReport, trainAccuracy, testPrecision, testRecall, trainAUC,
            testReport, testAccuracy, testPrecision, testRecall, testAUC,
            estimators, fileNameResults)

# EnsembledVoting ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Voting:")
print("\nScore on training set:")
trainReportEnsembled, trainAccuracyEnsembled, trainPrecisionEnsembled, trainRecallEnsembled, trainAUCEnsembled = scoreEnsembledVoting(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReportEnsembled, testAccuracyEnsembled, testPrecisionEnsembled, testRecallEnsembled, testAUCEnsembled = scoreEnsembledVoting(estimators, testFeatures, testLabels)

# Save the results of the ensembled models
saveResults([trainReportEnsembled], [trainAccuracyEnsembled], [trainPrecisionEnsembled], [trainRecallEnsembled], [trainAUCEnsembled],
            [testReportEnsembled], [testAccuracyEnsembled], [testPrecisionEnsembled], [testRecallEnsembled], [testAUCEnsembled],
            ["ensembledVoting"], fileNameResults)

# EnsembledAveraging ---------------------------------------------------------------------------------------------------
# Check the score of the ensembled models using voting on the training and test set
print("\n\nEnsembled Averaging:")
print("\nScore on training set:")
trainReportEnsembled, trainAccuracyEnsembled, trainPrecisionEnsembled, trainRecallEnsembled, trainAUCEnsembled = scoreEnsembledAveraging(estimators, trainFeatures, trainLabels)

print("\nScore on test set:")
testReportEnsembled, testAccuracyEnsembled, testPrecisionEnsembled, testRecallEnsembled, testAUCEnsembled = scoreEnsembledAveraging(estimators, testFeatures, testLabels)

# Save the results of the ensembled models
saveResults([trainReportEnsembled], [trainAccuracyEnsembled], [trainPrecisionEnsembled], [trainRecallEnsembled], [trainAUCEnsembled],
            [testReportEnsembled], [testAccuracyEnsembled], [testPrecisionEnsembled], [testRecallEnsembled], [testAUCEnsembled],
            ["ensembledAveraging"], fileNameResults)
