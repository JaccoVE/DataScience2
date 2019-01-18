# Load libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def scaleFeatures(features):

    # https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm
    # Standardizing the features using Standard Scaler
    scaler = StandardScaler()

    # Standardizing the features using MinMax Scaler
    #scaler =  MinMaxScaler().fit_transform(features)

    # Standardizing the features using Normalizer Scaler
    #scaler = Normalizer().fit_transform(features)

    return scaler.fit_transform(features)

def distibution(labels):

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T

# ------------------------------------------------------------------------------

# Open data from Excel
df = pd.read_excel('../data/featuresNEW_12hrs_sortrows.xls').values

# Print the first five rows of the dataset
print("Original dataset:")
print(df[0:5,:])

# Split the features and labels
features = df[:,1:45]
labels = df[:,0]

# Scale the features
features = scaleFeatures(features)

# Print shape of the features and labels
print("Feature shape:")
print(features.shape)
print("Labels shape")
print(labels.shape)

# Combine the features and labels
dfScaled = np.append(labels.reshape(len(labels),1), features, axis=1)

# Print shape of the scaled dataset
print("Scaled dataset shape:")
print(dfScaled.shape)

# Print the first five rows of the dataset
print("Scaled dataset:")
print(dfScaled[0:5,:])

# Split data set in train and test data set
trainData, testData = train_test_split(dfScaled, test_size=0.2)

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

# Save the train and test data
#np.savetxt("../data/train_data.txt", trainData)
#np.savetxt("../data/test_data.txt", testData)
