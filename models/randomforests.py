# Load libraries
import time
import pandas as pd
import numpy as np
import json
import openpyxl

from sklearn.preprocessing import StandardScaler
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Function to apply PCA
def PCA_function(tra_mdl_data, val_mdl_data, expl_var):

    features_tra = StandardScaler().fit_transform(tra_mdl_data[:,1:45]) # Standardizing the features
    features_val = StandardScaler().fit_transform(val_mdl_data[:,1:45]) # Standardizing the features

    pca = PCA(expl_var)                                                 # Principal component analysis of explained variance

    PCAdf_tra = pca.fit_transform(features_tra)                         # Transform features train
    PCAdf_val = pca.fit_transform(features_val)                         # Transform features validation

    return PCAdf_tra, PCAdf_val

# Function for a random hyperparameter grid search
def RHG_search(n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, n_iter, cv, scoring):

    random_grid = {'n_estimators': n_estimators,                                                 # Dictionary for the hyperparameters
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = RandomForestClassifier()                                                               # Estimator that is used
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,           # Randomzed grid search of the hyperparameters
                                   scoring = scoring, n_iter = n_iter, cv = cv, verbose=1, n_jobs= -1)
    best_hp = rf_random.best_params_
    return best_hp                                                                               # The best hyperparameters from the RHG search

# Function for training and performance assesment
def train_perd(best_hp, PCAdf_tra, PCAdf_val):
    clf = RandomForestClassifier(n_estimators=best_hp['n_estimators'], max_features=best_hp['max_features'],        # Initializa the model
                                 max_depth=best_hp['max_depth'], min_samples_split=best_hp['min_samples_split'],
                                 min_samples_leaf=best_hp['min_samples_leaf'], bootstrap=best_hp['bootstrap'])

    clf.fit(PCAdf_tra, tra_mdl_data[0,:])                                         # Train the model
    val_pred = clf.predict(PCAdf_val)                                             # Prediction of classes on the validation data
    tra_pred = clf.predict(PCAdf_tra)                                             # Prediction of classes on the training data
    mat_val = confusion_matrix(val_mdl_data[:,0], val_pred).ravel()               # Confusion matrix of validation data
    mat_tra = confusion_matrix(tra_mdl_data[:,0], tra_pred).ravel()               # Confusion matrix of training data
    acc_val = metrics.accuracy_score(val_mdl_data[:,0], val_pred)                 # Accuracy of validation data
    acc_train = metrics.accuracy_score(tra_mdl_data[:,0], tra_pred)               # Accuracy of training data

    return mat_val, mat_tra, acc_val, acc_train

# Function to write results to excel
def save_results(i, iterations, crossVal, mat_val, mat_tra, acc_val, acc_train):
    print('Results saved as:')
    saveLocation = '../results/'
    saveName = 'randomforests-' + 'iter_' + str(iterations) + '-cv_' + str(crossVal) + '.xlsx'
    print(saveLocation + saveName)

    properties_model = [i, mat_val, mat_tra, acc_val, acc_train]
    book = openpyxl.load_workbook(saveLocation + saveName)
    sheet = book.active
    sheet.append(properties_model)
    book.save(saveLocation + saveName)
    time.sleep(0.1)

    return "File successfully saved"

# (1) Load dataset
df = pd.read_excel('../data/featuresNEW_12hrs.xls').values

# (2) Split data set in train and test data set
train_data, test_data = train_test_split(df, test_size=0.2)

# (3) Split train set into 80% for training and 20% for validation
tra_mdl_data, val_mdl_data = train_test_split(train_data, test_size=0.2)

# (4) Initialize the grid for RandomizedSearchCV
bootstrap = [True, False]
max_features = ['auto', 'sqrt']
n_estimators =[int(x) for x in np.linspace(start = 200, stop = 5000, num = 20)]
max_depth = [int(x) for x in np.linspace(3, 110, num = 11)]; max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 40]
min_samples_leaf = [1, 2, 4, 8, 16, 32]
bootstrap = [True, False]

# Algorithm settings
iterations = 1000
crossVal = 3
scoring = "roc_auc"

# Algorithm
for i
    PCA_function(tra_mdl_data, val_mdl_data, expl_var)

    RHG_search(n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split, bootstrap, iterations, crossVal, scoring)

    train_perd(best_hp, PCAdf_tra, PCAdf_val)

    # Save to file
    save_results(i, iterations, crossVal, 1, 2, 3, 4)
