import pickle
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the file
with open('dataset_small.pkl', 'rb') as file:
	data = pickle.load(file, encoding='latin1')
x, y = data

PRC = 0.1
#10 expermients 4 algorithms to test
acc_r = np.zeros((10,4))
for i in range(10):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=PRC)
	nn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
	nn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
	# Support vector machine
	svc = svm.SVC()
	# Descision tree
	dt = tree.DecisionTreeClassifier() 

	## Training models
	nn1.fit(x_train, y_train)
	nn3.fit(x_train, y_train)
	svc.fit(x_train, y_train)
	dt.fit(x_train, y_train)

	yhat_nn1 = nn1.predict(x_test)
	yhat_nn3 = nn3.predict(x_test)
	yhat_svc = svc.predict(x_test)
	yhat_dt = dt.predict(x_test)

	## Measuring accuracy of each test
	acc_r[i][0] = metrics.accuracy_score(yhat_nn1, y_test)
	acc_r[i][1] = metrics.accuracy_score(yhat_nn3, y_test)
	acc_r[i][2] = metrics.accuracy_score(yhat_svc, y_test)
	acc_r[i][3] = metrics.accuracy_score(yhat_dt, y_test)
#end for

# show the accuracy matrix
print(acc_r)

plt.boxplot(acc_r)
for i in range(4):
	xderiv = (i+1) * np.ones(acc_r[:,1].shape) + (np.random.rand(10,)-0.5)*0.1
	plt.plot(xderiv, acc_r[:, i], 'ro', alpha=0.3)
ax = plt.gca()
ax.set_xticklabels(['1-NN', '3-NN', 'SVM', 'Decisson Tree'])
plt.ylabel('Accuracy')
plt.show()
plt.savefig('error_ms_accuracy.png', dpi=300, bbox_inches='tight')
