# Load libraries

import time
import pandas as pd
import numpy as np
import json
import openpyxl

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Function to apply PCA
def PCA_function(df, expl_var, i):

    features = StandardScaler().fit_transform(df[:,1:45])                            # Standardizing the features
    labels = df[:,0]

    pca = PCA(expl_var)                                                      # Principal component analysis of explained variance

    PCAdf_tra = pca.fit_transform(features)                                  # Transform features train
    print(np.abs(pca.components_[i]).argsort()[:pca.explained_variance_.shape[0]][::-1])
    print(pca.explained_variance_)
    PCAdf = np.c_[labels, PCAdf_tra]                                         # Add labels to data frame

    train_data, test_data = train_test_split(PCAdf, test_size=0.2)           # Split data set in train and test data set
    tra_mdl_data, val_mdl_data = train_test_split(train_data, test_size=0.2) # Split train set into 80% for training and 20% for validation

    return train_data, test_data, tra_mdl_data, val_mdl_data

# Function for a random hyperparameter grid search
def RHG_search(n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring, PCAdf_tra, tra_mdl_data):

    random_grid = {'n_estimators': n_estimators,                                                        # Dictionary for the hyperparameters
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = RandomForestClassifier()                                                                      # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,                  # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(tra_mdl_data[:,1:45], tra_mdl_data[:,0].astype(int))
    best_hp = rf_random.best_params_
    return best_hp                                                                                      # The best hyperparameters from the RHG search

# Function for training and performance assesment
def train_perf(best_hp, tra_mdl_data, val_mdl_data):
    clf = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_features=best_hp['max_features'],        # Initializa the model
                                 max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'],
                                 min_samples_leaf=best_hp['min_samples_leaf'], bootstrap=best_hp['bootstrap'])

    clf.fit(tra_mdl_data[:,1:45], tra_mdl_data[:,0].astype(int))                                      # Train the model
    val_pred = clf.predict(val_mdl_data[:,1:45])                                              # Prediction of classes on the validation data
    tra_pred = clf.predict(tra_mdl_data[:,1:45])                                              # Prediction of classes on the training data
    mat_val = confusion_matrix(val_mdl_data[:,0].astype(int), val_pred).ravel()               # Confusion matrix of validation data
    mat_tra = confusion_matrix(tra_mdl_data[:,0].astype(int), tra_pred).ravel()               # Confusion matrix of training data
    acc_val = metrics.accuracy_score(val_mdl_data[:,0].astype(int), val_pred)                 # Accuracy of validation data
    acc_train = metrics.accuracy_score(tra_mdl_data[:,0].astype(int), tra_pred)               # Accuracy of training data

    return mat_val, mat_tra, acc_val, acc_train

# Function to write results to excel
def save_results(i, mat_val, mat_tra, acc_val, acc_train):

    print('Results saved as:')

    saveLocation = '../results/'                                                                # Location for the save
    saveName = 'randomforests.xlsx'  # Name of the save

    print(saveLocation + saveName)

    properties_model = [i,
                        "[" + str(mat_val[0]) + " " + str(mat_val[1]) + "; " + str(mat_val[2]) + " " + str(mat_val[3]) + "]",
                        "[" + str(mat_tra[0]) + " " + str(mat_tra[1]) + "; " + str(mat_tra[2]) + " " + str(mat_tra[3]) + "]",
                        acc_val,
                        acc_train]    # Which values are saved
    book = openpyxl.load_workbook(saveLocation + saveName)
    sheet = book.active
    sheet.append(properties_model)
    book.save(saveLocation + saveName)
    time.sleep(0.1)

    return "File successfully saved"

# (1) Load dataset
df = pd.read_excel('../data/featuresNEW_12hrs.xls').values


# (4) Initialize the grid for RandomizedSearchCV
bootstrap = [True, False]
max_features = ['auto', 'sqrt']
n_estimators =[int(x) for x in np.linspace(start = 200, stop = 5000, num = 20)]
max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]; max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 40]
min_samples_leaf = [1, 2, 4, 8, 16, 32]
bootstrap = [True, False]

# Algorithm settings
n_iter = 1
cv = 3
scoring = "roc_auc"
expl_var = [0.5,0.6,0.7,0.8,0.9]

# Algorithm
for i in range(0,5,1):

    print("Start PCA interval:", expl_var[i])

    [train_data, test_data, tra_mdl_data, val_mdl_data] = PCA_function(df, expl_var[i], i)

    best_hp = RHG_search(n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring, tra_mdl_data, val_mdl_data)

    [mat_val, mat_tra, acc_val, acc_train] = train_perf(best_hp, tra_mdl_data, val_mdl_data)

    # Save to file
    save_results(expl_var[i], mat_val, mat_tra, acc_val, acc_train)
