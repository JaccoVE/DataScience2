# Load libraries
import time
import pandas as pd
import numpy as np
import json
import openpyxl

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

def splitData():

    # (Only required 1 time) Load dataset
    #df = pd.read_excel('../data/featuresNEW_12hrs.xls').values            # Read the values from the Excel file
    #train_data, test_data = train_test_split(df, test_size=0.2)           # Split data set in train and test data set
    #np.savetxt("train_data.txt", train_data)                              # Save data for training
    #np.savetxt("test_data.txt", test_data)                                # Save data for testing

    return print("Data has been split and saved, one time action.")

def loadData():

    # (1) Load training data from file
    trainData = np.loadtxt("../data/train_data.txt")                           # Load the train data
    testData = np.loadtxt("../data/test_data.txt")                             # Load the test data
    trainDataM, valDataM = train_test_split(trainData, test_size=0.3)         # Split train set into 80% for training and 20% for validation

    return trainDataM, valDataM, testData, trainData

def loadModel(filename_model):

    # load the model from disk
    #loaded_model = joblib.load(filename_model)
    #result = loaded_model.score(X_test, Y_test)
    #print(result)

    return print("Model has been loaded.")

# Function for PCA and split of the data
def pcaSplit(trainData, expl_var):

    features = StandardScaler().fit_transform(trainData[:,1:45])                    # Standardizing the features
    labels = trainData[:,0]                                                         # Split labels from data frame

    pca = PCA(expl_var)                                                      # Principal component analysis of explained variance
    PCAdf_tra = pca.fit_transform(features)                                  # Transform features train
    expl_var_ratio = pca.explained_variance_ratio_                           # Shows the percentage of variance explained by principal component
    PCAdf = np.c_[labels, PCAdf_tra]                                         # Add labels to data frame

    trainData, testData = train_test_split(PCAdf, test_size=0.3)             # Split data set in train and test data set
    trainDataM, valDataM = train_test_split(trainData, test_size=0.3)        # Split train set into 80% for training and 20% for validation

    return trainData, testData, trainDataM, valDataM

# Function for a random hyperparameter gird search
def randomGridSearch(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring, trainDataM):

    random_grid = {'n_estimators': n_estimators,                                                        # Dictionary for the hyperparameters
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion' : criterion}

    clf = RandomForestClassifier()                                                                          # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,                      # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(trainDataM[:,1:45], trainDataM[:,0].astype(int))                                          # Train the numerous models
    best_hp = rf_random.best_params_                                                                        # Store the best hyperparameters

    return best_hp

# Function for a hyperparameter grid search
def gridSearch(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, cv, scoring, trainDataM):

    grid = {'n_estimators': n_estimators,                                                        # Dictionary for the hyperparameters
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'criterion' : criterion}

    clf = RandomForestClassifier()                                                                      # Estimator that is used
    rf_random = GridSearchCV(estimator = clf, param_grid = grid,                  # Randomzed grid search of the hyperparameters
                                   scoring = scoring, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(trainDataM[:,1:45], trainDataM[:,0].astype(int))
    best_hp = rf_random.best_params_

    return best_hp

# Function for training and performance assesment
def performance(best_hp, trainDataM, valDataM):

    clf = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_features=best_hp['max_features'],        # Initialize the model
                                 max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'],
                                 min_samples_leaf=best_hp['min_samples_leaf'], bootstrap=best_hp['bootstrap'], criterion=best_hp['criterion'])
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
                        best_hp['n_estimators'],
                        best_hp['min_samples_split'],
                        best_hp['min_samples_leaf'],
                        best_hp['max_features'],
                        best_hp['max_depth'],
                        str(best_hp['criterion']),
                        str(best_hp['bootstrap'])]                   # Which values are saved
    book = openpyxl.load_workbook(saveLocation + saveName)
    sheet = book.active
    sheet.append(properties_model)
    book.save(saveLocation + saveName)
    time.sleep(0.1)

    return "File successfully saved"

# filename = '../results/finalized_model.sav' for random grid search
# saveName = 'randomforests.xlsx' for random grid search
# filename = '../results/finalized_model_gs.sav' for grid search
# saveName = 'randomforests_gs.xlsx' for grid search

def runGridSearch():
    # (1) Load the data
    trainDataM, valDataM, testData, trainData = loadData()
    # (2) Initialize grid
    bootstrap = [False]; criterion = ['entropy']; max_features = ['auto']
    n_estimators = [2000]; max_depth = [5]; min_samples_split = [10]
    min_samples_leaf = [2]
    # (3) Algorithm settings
    cv, scoring = 3,  "roc_auc"
    # (4) Location and Name
    fileName = '../results/finalized_model_gs.sav'
    saveName = 'randomforests_gs.xlsx'
    # (5) Grid search
    best_hp = gridSearch(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, cv, scoring, trainDataM)
    # (6) Performance assesment
    [clf, scoresTrain, scoresVal] = performance(best_hp, trainDataM, valDataM)
    # (7) Save results
    saveResults(best_hp, scoresTrain, scoresVal, clf, fileName, saveName)

    return print("End Grid Search!")

runGridSearch()

def runRandomGridSearch():
    # (1) Load the data
    trainDataM, valDataM, testData, trainData = loadData()
    # (2) Initialize grid
    bootstrap = [True, False]
    criterion = ['gini', 'entropy']
    max_features = ['auto', 'sqrt']
    n_estimators =[int(x) for x in np.linspace(start = 200, stop = 2500, num = 20)]
    max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]; max_depth.append(None)
    min_samples_split = [2, 5, 10, 20, 40]
    min_samples_leaf = [1, 2, 4, 8, 16, 32]
    # (3) Algorithm settings
    n_iter, cv, scoring = 1000, 3,  "roc_auc"
    # (4) Save locations
    fileName = '../results/finalized_model_2.sav'
    saveName = 'randomforests.xlsx'
    # (5) Random Grid search
    best_hp = randomGridSearch(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring, trainDataM)
    # (6) Performance assesment
    [clf, scoresTrain, scoresVal] = performance(best_hp, trainDataM, valDataM)
    # (7) Save results
    saveResults(scoresTrain, scoresVal, clf, fileName, saveName)

    return print("End Random Grid Search")
