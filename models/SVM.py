# Load libraries
import time
import pandas as pd
import numpy as np
import json
import openpyxl

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
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

def loadData():

    # (1) Load training data from file
    trainData = np.loadtxt("../data/train_data.txt")                           # Load the train data
    testData = np.loadtxt("../data/test_data.txt")                             # Load the test data
    trainDataM, valDataM = train_test_split(trainData, test_size=0.3)         # Split train set into 80% for training and 20% for validation

    return trainDataM, valDataM, testData, trainDataM

# Function for a random hyperparameter gird search
def randomGridSearch(C,kernel,degree,coef0,n_iter, cv, scoring, trainDataM):

    random_grid = {'C': C,
                   'kernel': kernel,
                   'degree': degree,
                   'coef0': coef0}

    clf = SVC()                                                                          # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,                      # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(trainDataM[:,1:45], trainDataM[:,0].astype(int))                                          # Train the numerous models
    best_hp = rf_random.best_params_                                                                        # Store the best hyperparameters

    return best_hp

# Function for training and performance assesment
def performance(best_hp, trainDataM, valDataM):

    clf = SVC(C=best_hp['C'],kernel=best_hp['kernel'],degree=best_hp['degree'],coef0=best_hp['coef0'])
    # The mean score and the 95% confidence interval
    scoresTrain = cross_val_score(clf, trainDataM[:,1:45], trainDataM[:,0], cv=5,  scoring='roc_auc')
    scoresVal = cross_val_score(clf, valDataM[:,1:45], valDataM[:,0], cv=5, scoring='roc_auc')
    print("Accuracy Train: %0.2f (+/- %0.2f)" % (scoresTrain.mean(), scoresTrain.std() * 2))
    print("Accuracy Val: %0.2f (+/- %0.2f)" % (scoresVal.mean(), scoresVal.std() * 2))
    print(scoresTrain)
    print(scoresVal)

    clf.fit(trainDataM[:,1:45], trainDataM[:,0].astype(int))                              # Train the model
    val_pred = clf.predict(valDataM[:,1:45])                                              # Prediction of classes on the validation data
    tra_pred = clf.predict(trainDataM[:,1:45])                                              # Prediction of classes on the training data
    acc_val = metrics.accuracy_score(valDataM[:,0].astype(int), val_pred)                 # Accuracy of validation data
    acc_train = metrics.accuracy_score(trainDataM[:,0].astype(int), tra_pred)               # Accuracy of training data
    print("Acc val:", acc_val)
    print("Acc train:", acc_train)

    return clf, scoresTrain, scoresVal

# Function to write results to excel
def saveResults(best_hp, scoresTrain, scoresVal, clf, fileName, saveName):

        # save the model to disk
    joblib.dump(clf, fileName)

    print('Results saved as:')

    saveLocation = '../results/'                                                                # Location for the save

    print(saveLocation + saveName)

    properties_model = [scoresTrain.mean(),
                        scoresTrain.std()*2,
                        scoresVal.mean(),
                        scoresVal.std() * 2,
                        best_hp['C'],
                        best_hp['kernel'],
                        best_hp['degree'],
                        best_hp['coef0']]                   # Which values are saved
    book = openpyxl.load_workbook(saveLocation + saveName)
    sheet = book.active
    sheet.append(properties_model)
    book.save(saveLocation + saveName)
    time.sleep(0.1)

    return "File successfully saved"

# (1) Load the data
[trainDataM, valDataM, testData, trainData] = loadData()
[trainDataStandard, trainDataMinMax, trainDataNorm] = scalingData(trainData)
trainDataM, valDataM = train_test_split(trainDataM, test_size=0.3)
# (2) Initialize grid
C=[0.001, 0.01, 0.1, 1, 10]
kernel=['linear', 'poly', 'rbf']
degree=[2,4,8,10]
coef0=[-8,-4,-2,2,4,8]
# (3) Algorithm settings
n_iter, cv, scoring = 1000, 3,  "roc_auc"
# (4) Save locations
fileName = '../results/finalized_model_svm.sav'
saveName = 'svm_rgs.xlsx'
# (5) Random Grid search
best_hp = randomGridSearch(C,kernel,degree,coef0,n_iter,cv,scoring,trainDataM)
# (6) Performance assesment
[clf, scoresTrain, scoresVal] = performance(best_hp, trainDataM, valDataM)
# (7) Save results
saveResults(best_hp,scoresTrain,scoresVal,clf,fileName,saveName)
