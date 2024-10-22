import numpy as np
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

# Depth of decsion tree
C = 3
MAXN = 1000

# Comparison matrix
yhat_test = np.zeros((10, 299, 2))      # 10 diff matrices n*m matrices (299*2)
yhat_train = np.zeros((10, 299, 2))

for i in range(10):
    # Data is scewed to show better representation
    X = np.concatenate([1.25 * np.random.randn(MAXN, 2), 5+1.5*np.random.randn(MAXN, 2)])
    X = np.concatenate([X, [8,5] + 1.5 * np.random.randn(MAXN, 2)])
    y = np.concatenate([np.ones((MAXN, 1)), -np.ones((MAXN, 1))])
    y = np.concatenate([y, np.ones((MAXN, 1))])
    # Permutation matrix
    perm = np.random.permutation(y.size)
    X = X[perm, :]
    y = y[perm]   # Only one column
    
    X_test = np.concatenate([1.25*np.random.randn(MAXN, 2), 5+1.5*np.random.randn(MAXN, 2)])
    X_test = np.concatenate([X_test, [8, 5] + 1.5 * np.random.randn(MAXN, 2)])
    y_test = np.concatenate([np.ones((MAXN, 1)), -np.ones((MAXN, 1))])
    y_test = np.concatenate([y_test, np.ones((MAXN, 1))])
    j=0
    for N in range(10, 3000, 10):
        Xr = X[:N, :]
        yr = y[:N]
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C)
        clf.fit(Xr, yr.ravel())
        yhat_test[i, j, 0] = 1.0 - metrics.accuracy_score(clf.predict(X_test), y_test.ravel())
        yhat_train[i, j, 0] = 1.0 - metrics.accuracy_score(clf.predict(Xr), yr.ravel())
        j=j+1

p1, = plt.plot(np.mean(yhat_test[:,:,0].T, axis=1), 'pink')
p2, = plt.plot(np.mean(yhat_train[:,:,0].T, axis=1), 'c')
fig = plt.gcf()
fig.set_size_inches(12, 5)
plt.xlabel('Number of samples x10')
plt.ylabel('Error rate')
plt.legend([p1,p2], [f"Test C = {C}", f"Train C = {C}"])
plt.show()
plt.savefig("Learning_curve_1.png", dpi=300, bbox_inches='tight')

