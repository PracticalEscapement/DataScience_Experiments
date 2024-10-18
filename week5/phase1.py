import pickle
from sklearn import neighbors
from sklearn import datasets
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For the K value, start by using the square root of the data set size.
knn = neighbors.KNeighborsClassifier(n_neighbors=11)

with open('dataset_small.pkl', 'rb') as file:
	data = pickle.load(file, encoding='latin1')
x, y = data
x_shape = None
y_shape = None

if hasattr(x, 'shape'):
	x_shape = x.shape
if hasattr(y, 'shape'):
	y_shape = y.shape

print(x_shape)
print(y_shape)

knn.fit(x, y)
yhat = knn.predict(x)
print('Prediction value: ' + str(yhat[-1]), ', real target: ' + str(y[-1]))
knn.score(x, y)
print(knn.score(x, y))

plt.pie(np.c_[np.sum(np.where(y==1,1,0)), np.sum(np.where(y==-1,1,0))][0], 
		labels=['Not fully funed', 'Full Amount'], colors=['g','r'],
		shadow=False,
		autopct='%.3F'
		)

plt.gcf().set_size_inches((6, 6))
plt.savefig('pie.png', dpi=300, bbox_inches='tight')

TP = np.sum(np.logical_and(yhat == -1, y == -1))
TN = np.sum(np.logical_and(yhat == 1, y == 1))
FP = np.sum(np.logical_and(yhat == -1, y == 1))
FN = np.sum(np.logical_and(yhat == 1, y == -1))

print('TP: ' + str(TP), 'FP: ' + str(FP))
print('FN: ' + str(FN), 'TN: ' + str(TN))

print(metrics.confusion_matrix(yhat, y))