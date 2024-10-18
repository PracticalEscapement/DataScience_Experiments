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

MAXN = 700
fig = plt.figure()
fig.set_size_inches(5, 5)
plt.plot(1.25*np.random.rand(MAXN, 1), 1.25*np.random.rand(MAXN), 'r.', alpha=0.3)
plt.plot(8+1.5*np.random.rand(MAXN, 1), 5+1.5*np.random.rand(MAXN), 'y.', alpha=0.3)
plt.plot(5+1.5*np.random.rand(MAXN, 1), 1.25*np.random.rand(MAXN), 'g.', alpha=0.3)
plt.show()