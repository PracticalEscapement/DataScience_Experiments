import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree

# Max confidence
MAXC = 20
# interations
MAXN = 3000
NTEST = 4000
# the number of matrices (multi dim matrix)
ITERS=10


yhat_test = np.zeros((ITERS, MAXC, 2))
yhat_train = np.zeros((ITERS, MAXC, 2))
for i in range(ITERS):
    # Data is scewed to show better representation
    X = np.concatenate([1.25 * np.random.randn(MAXN, 2), 5+1.5*np.random.randn(MAXN, 2)])
    X = np.concatenate([X, [8,5] + 1.5 * np.random.randn(MAXN, 2)])
    y = np.concatenate([np.ones((MAXN, 1)), -np.ones((MAXN, 1))])
    y = np.concatenate([y, np.ones((MAXN, 1))])
    # Permutation matrix
    perm = np.random.permutation(y.size)
    X = X[perm, :]
    y = y[perm]

    X_test = np.concatenate([1.25*np.random.randn(MAXN, 2), 5+1.5*np.random.randn(MAXN, 2)])
    X_test = np.concatenate([X_test, [8, 5] + 1.5 * np.random.randn(MAXN, 2)])
    y_test = np.concatenate([np.ones((MAXN, 1)), -np.ones((MAXN, 1))])
    y_test = np.concatenate([y_test, np.ones((MAXN, 1))])

    j=0
    for C in range(1, MAXC+1):
        # all nodes must have at least 1 leaf,
        # max depth of tree is 5 levels to prevent overfeeding
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=5)
        clf.fit(X, y.ravel())
        yhat_test[i, j, 0] = 1. - metrics.accuracy_score(clf.predict(X_test), y_test.ravel())
        yhat_train[i, j, 0] = 1. - metrics.accuracy_score(clf.predict(X), y.ravel())
        j=j+1

p1, = plt.plot(np.mean(yhat_test[:,:,0].T, axis=1), 'r')
p2, = plt.plot(np.mean(yhat_train[:,:,0].T, axis=1), 'y')
fig = plt.gcf()
fig.set_size_inches(7, 8)
plt.xlabel('Complexity')
plt.ylabel('Error Rate')
plt.legend([p1, p2], ['Testing Error', 'Training Error'])
plt.show()


