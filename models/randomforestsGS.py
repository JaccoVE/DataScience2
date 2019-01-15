# Load libraries
import time
import pandas as pd
import numpy as np
import json
import openpyxl

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# Function for a random hyperparameter grid search
def HG_search(n_estimators, criterion, max_features, max_depth, bootstrap, cv, scoring, tra_mdl_data):

    random_grid = {'n_estimators': n_estimators,                                                        # Dictionary for the hyperparameters
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion' : criterion}

    clf = RandomForestClassifier()                                                                      # Estimator that is used
    rf_random = GridSearchCV(estimator = clf, param_grid = random_grid,                  # Randomzed grid search of the hyperparameters
                                   scoring = scoring, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(tra_mdl_data[:,1:45], tra_mdl_data[:,0].astype(int))
    best_hp = rf_random.best_params_
    print(best_hp)
    return best_hp

# Function for training and performance assesment
def train_perf(best_hp, tra_mdl_data, val_mdl_data):
    clf = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_features=best_hp['max_features'],        # Initialize the model
                                 max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'],
                                 min_samples_leaf=best_hp['min_samples_leaf'], bootstrap=best_hp['bootstrap'])

    clf.fit(tra_mdl_data[:,1:45], tra_mdl_data[:,0].astype(int))                              # Train the model
    val_pred = clf.predict(val_mdl_data[:,1:45])                                              # Prediction of classes on the validation data
    tra_pred = clf.predict(tra_mdl_data[:,1:45])                                              # Prediction of classes on the training data
    mat_val = confusion_matrix(val_mdl_data[:,0].astype(int), val_pred).ravel()               # Confusion matrix of validation data
    mat_tra = confusion_matrix(tra_mdl_data[:,0].astype(int), tra_pred).ravel()               # Confusion matrix of training data
    acc_val = metrics.accuracy_score(val_mdl_data[:,0].astype(int), val_pred)                 # Accuracy of validation data
    acc_train = metrics.accuracy_score(tra_mdl_data[:,0].astype(int), tra_pred)               # Accuracy of training data

    return mat_val, mat_tra, acc_val, acc_train, clf

# Function to write results to excel
def save_results(mat_val, mat_tra, acc_val, acc_train, clf):

    # save the model to disk
    filename = '../results/finalized_model_gs.sav'
    joblib.dump(clf, filename)

    print('Results saved as:')

    saveLocation = '../results/'                                                                # Location for the save
    saveName = 'randomforests_gs.xlsx'  # Name of the save

    print(saveLocation + saveName)

    properties_model = ["[" + str(mat_val[0]) + " " + str(mat_val[1]) + "; " + str(mat_val[2]) + " " + str(mat_val[3]) + "]",
                        "[" + str(mat_tra[0]) + " " + str(mat_tra[1]) + "; " + str(mat_tra[2]) + " " + str(mat_tra[3]) + "]",
                        acc_val,
                        acc_train,
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

# (1) Load training data from file
train_data = np.loadtxt("../data/train_data.txt")                           # Load the train data
test_data = np.loadtxt("../data/test_data.txt")                             # Load the test data
tra_mdl_data, val_mdl_data = train_test_split(train_data, test_size=0.2)    # Split train set into 80% for training and 20% for validation

# (2) Initialize the grid for RandomizedSearchCV
bootstrap = [False]
criterion = ['entropy']
max_features = ['auto']
n_estimators = [321,250,300,350,400]
max_depth = [15,20,25]
min_samples_split = [30,40,50]
min_samples_leaf = [7,8,9]
bootstrap = [False]

# (3) Algorithm settings
cv, scoring = 3,  "roc_auc"

# Algorithm
best_hp = HG_search(n_estimators, criterion, max_features, max_depth,                                 # Random Grid Search of the hyperparameters
                     bootstrap, cv, scoring, tra_mdl_data)
[mat_val, mat_tra, acc_val, acc_train, clf] = train_perf(best_hp, tra_mdl_data, val_mdl_data)         # Train the model and asses the performance
save_results(mat_val, mat_tra, acc_val, acc_train, clf)                                               # Save the results in excel
