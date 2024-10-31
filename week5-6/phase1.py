# Simple Linear Regeression Model

import seaborn as sns # Dataset
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1, rc={"line.linewidth": 2, 'font.family': [u'times']})
import matplotlib.pyplot as plt
import numpy as np

#plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=14)
plt.rc('figure', figsize=(12,5))

x = [5, 3, 2, 12, 9, 25, 16, 23]
y = [65, 30, 80, 74, 55, 38, 61, 35]

#plt.scatter(x, y, color='red')
#plt.show()

X1 = np.random.randn(300, 2)
A = np.array([[0.6, 0.4], [0.4, 0.6]])
X2 = np.dot(X1, A)
plt.plot(X2[:,0], X2[:,1], "o", alpha=0.3)
#plt.show()
model1 = [0+1*x for x in np.arange(-2, 3)]
model2 = [0.3+0.9*x for x in np.arange(-2, 3)]
model3 = [0-0.1*x for x in np.arange(-2, 3)]
plt.plot(X2[:,0], X2[:,1], 'o', alpha=0.3)
plt.plot(np.arange(-2,3), model1, 'r')
plt.plot(np.arange(-2,3), model2, 'g')
plt.plot(np.arange(-2,3), model3, 'b')
plt.show()