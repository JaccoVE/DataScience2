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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# Function to apply PCA
def PCA_function(df, expl_var):

    features = StandardScaler().fit_transform(df[:,1:45])                    # Standardizing the features
    labels = df[:,0]                                                         # Split labels from data frame
    pca = PCA(expl_var)                                                      # Principal component analysis of explained variance
    PCAdf_tra = pca.fit_transform(features)                                  # Transform features train
    expl_var_ratio = pca.explained_variance_ratio_                           # Shows the percentage of variance explained by principal component
    PCAdf = np.c_[labels, PCAdf_tra]                                         # Add labels to data frame

    train_data, test_data = train_test_split(PCAdf, test_size=0.2)           # Split data set in train and test data set
    tra_mdl_data, val_mdl_data = train_test_split(train_data, test_size=0.2) # Split train set into 80% for training and 20% for validation

    return train_data, test_data, tra_mdl_data, val_mdl_data

# Function for a random hyperparameter grid search
def RHG_search(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring, tra_mdl_data):

    random_grid = {'n_estimators': n_estimators,                                                        # Dictionary for the hyperparameters
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion' : criterion}

    clf = RandomForestClassifier()                                                                      # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,                  # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, verbose = 1, cv = cv, n_jobs= -1)
    rf_random.fit(tra_mdl_data[:,1:45], tra_mdl_data[:,0].astype(int))
    best_hp = rf_random.best_params_
    print(best_hp)
    return best_hp                                                                                      # The best hyperparameters from the RHG search

# Function for training and performance assesment
def train_perf(best_hp, tra_mdl_data, val_mdl_data):
    clf = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_features=best_hp['max_features'],        # Initializa the model
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
    filename = '../results/finalized_model.sav'
    joblib.dump(clf, filename)

    print('Results saved as:')

    saveLocation = '../results/'                                                                # Location for the save
    saveName = 'randomforests.xlsx'  # Name of the save

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

# (Only required 1 time) Load dataset
#df = pd.read_excel('../data/featuresNEW_12hrs.xls').values            # Read the values from the Excel file
#train_data, test_data = train_test_split(df, test_size=0.2)           # Split data set in train and test data set
#np.savetxt("train_data.txt", train_data)                              # Save data for training
#np.savetxt("test_data.txt", test_data)                                # Save data for testing

# (1) Load training data from file
train_data = np.loadtxt("../data/train_data.txt")                           # Load the train data
test_data = np.loadtxt("../data/test_data.txt")                             # Load the test data
tra_mdl_data, val_mdl_data = train_test_split(train_data, test_size=0.2)    # Split train set into 80% for training and 20% for validation

# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)

# (2) Initialize the grid for RandomizedSearchCV
bootstrap = [True, False]
criterion = ['gini', 'entropy']
max_features = ['auto', 'sqrt']
n_estimators =[int(x) for x in np.linspace(start = 200, stop = 2500, num = 20)]
max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]; max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 40]
min_samples_leaf = [1, 2, 4, 8, 16, 32]
bootstrap = [True, False]

# (3) Algorithm settings
n_iter, cv, scoring = 100, 3,  "roc_auc"

# (4) Algorithm for finding the hyperparameters to initialize the model

# (4) Run Algorithm
best_hp = RHG_search(n_estimators, criterion, max_features, max_depth, min_samples_leaf, min_samples_split, # Random Grid Search of the hyperparameters
                     bootstrap, n_iter, cv, scoring, tra_mdl_data)
[mat_val, mat_tra, acc_val, acc_train, clf] = train_perf(best_hp, tra_mdl_data, val_mdl_data)         # Train the model and asses the performance
save_results(mat_val, mat_tra, acc_val, acc_train, clf)                                               # Save the results in excel
