import pickle
from sklearn import neighbors
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the file
with open('dataset_small.pkl', 'rb') as file:
	data = pickle.load(file, encoding='latin1')
x, y = data

# Randomize the data indecies
perm = np.random.permutation(y.size)
print(perm)

# Precision: Uses 70% of the data set
PRC = 0.7
split_point = int(np.ceil(y.shape[0]*PRC))
print(split_point)

## Training subsets

# takes a list of all items from index 0 to 2898
# get all the records between these indecies and return all the columns associated with them
x_train = x[perm[:split_point].ravel(), :]

# Only one column, therfore ":" (all the columns) does not need to be passed as arg
y_train = y[perm[:split_point].ravel()]

## Testing subsets
# Uses the last 30% of the permutation set

x_test = x[perm[split_point:].ravel(), :]
y_test = y[perm[split_point:].ravel()]


# shape is the columns and rows of the data set
print('Training Shape: ' + str(x_train.shape) + ' Training Targets Shape: ' + str(y_train.shape))
print('Training Shape: ' + str(x_test.shape) + ' Training Targets Shape: ' + str(y_test.shape))
print(int(np.sqrt(y_train.shape[0]) / 53) + 3)

# For the K value, start by using the square root of the data set size.
knn = neighbors.KNeighborsClassifier(int(np.sqrt(y_train.shape[0]) / 53) + 3)
# Callculating ecludian distance between nodes (neigbors)
knn.fit(x_train, y_train)
yhat = knn.predict(x_train)
print('Classification Accuracy: ', metrics.accuracy_score(yhat, y_train))
print('confusion matrix \n' + str(metrics.confusion_matrix(y_train, yhat)))


## Automation of splitting data
PRC = 0.8
acc = np.zeros((10,))

for i in range(10):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=PRC)
	knn = neighbors.KNeighborsClassifier(n_neighbors=5)
	knn.fit(x_train, y_train)
	yhat = knn.predict(x_test)
	acc[i] = metrics.accuracy_score(yhat, y_test)

print(acc)
acc.shape = (1,10)
print('Mean expected error: ' + str(np.mean(acc[0])))


## How can you improve splitting the data?
## SVM, Descision Tree....
