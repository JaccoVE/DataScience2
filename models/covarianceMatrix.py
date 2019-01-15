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

df = pd.read_excel('../data/featuresNEW_12hrs_sortrows.xls').values

features = StandardScaler().fit_transform(df[:,1:45])                            # Standardizing the features
pca = PCA(expl_var)                                                      # Principal component analysis of explained variance
PCAdf_tra = pca.fit_transform(features)                                  # Transform features train
print(pca.get_covariance())
