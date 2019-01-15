import time
import pandas as pd
import numpy as np
import json
import openpyxl

from sklearn.neighbors import KNeighborsClassifier
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

def scalingData(trainData):

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
def randomGridSearch(n_neighbors,weights,leaf_size,p,n_iter,cv,scoring,trainDataM):

    random_grid = {'n_neighbors': n_neighbors,
                   'weights': weights,
                   'leaf_size': leaf_size,
                   'p': p}

    clf = KNeighborsClassifier()                                                                          # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,                      # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(trainDataM[:,1:45], trainDataM[:,0].astype(int))                                          # Train the numerous models
    best_hp = rf_random.best_params_                                                                        # Store the best hyperparameters

    return best_hp

# Function for training and performance assesment
def performance(best_hp, trainDataM, valDataM):

    clf = KNeighborsClassifier(n_neighbors=best_hp['n_neighbors'],weights=best_hp['weights'],leaf_size=best_hp['leaf_size'],p=best_hp['p'])
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
                        best_hp['n_neighbors'],
                        best_hp['weights'],
                        best_hp['leaf_size'],
                        best_hp['p']]                   # Which values are saved
    book = openpyxl.load_workbook(saveLocation + saveName)
    sheet = book.active
    sheet.append(properties_model)
    book.save(saveLocation + saveName)
    time.sleep(0.1)

    return "File successfully saved"

# (1) Load data
[trainDataM, valDataM, testData, trainData] = loadData()
# () Scale data
[trainDataStandard, trainDataMinMax, trainDataNorm] = scalingData(trainData)
trainDataM, valDataM = train_test_split(trainDataStandard, test_size=0.3)
# (2) Initialize grid
n_neighbors=[2,3,4,5,6,7,8]
weights=['uniform','distance']
leaf_size=[10,20,30,40,50]
p=[1,2,3,4,5]
# (3) Algorithm settings
n_iter, cv, scoring = 10, 3,  "roc_auc"
# (4) Save locations
fileName = '../results/finalized_model_knn.sav'
saveName = 'knn_rgs.xlsx'
# (5) Random Grid search
best_hp = randomGridSearch(n_neighbors,weights,leaf_size,p,n_iter,cv,scoring,trainDataM)
# (6) Performance assesment
[clf, scoresTrain, scoresVal] = performance(best_hp, trainDataM, valDataM)
# (7) Save results
saveResults(best_hp,scoresTrain,scoresVal,clf,fileName,saveName)
