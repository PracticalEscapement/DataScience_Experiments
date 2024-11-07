import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ice = pd.read_csv('SeaIce.txt', delim_whitespace=True)
# Tuple of (rows, columns)
print('Shape:', ice.shape)
# Get first 5 rows
print(ice.head())
# Not statistical mode, only selects columns with numerical data
print(ice.mode(numeric_only=True).mean())

x = ice.year
y = ice.extent

# plt.scatter(x, y, color='red')
# plt.xlabel('Year')
# plt.ylabel('Extent')

#plt.show()
# Observe the outliers in the data set
print("Different values in data fields", np.unique(ice.data_type.values))
print(ice[(ice.data_type != "Goddard") & (ice.data_type != "NRTSI-G")])

# Exclude the outliers
ice2 = ice[ice.data_type != "-9999"]
x = ice2.year
y = ice2.extent

# plt.scatter(x, y, color='blue')
# plt.xlabel('Year')
# plt.ylabel('Extent')
# plt.show()

sns.lmplot(x="mo", y="extent", data=ice2, aspect=2)
plt.savefig("Phase-3-figure1.png", dpi=300, bbox_inches='tight')

grouped = ice2.groupby('mo')
month_means = grouped.extent.mean()
month_variance = grouped.extent.var()
print('Mean: ', month_means)
print("Variance: ", month_variance)

for i in range(12):
    mask = ice2['mo'] == i+1
    ice2.loc[mask, 'extent'] = 100*(ice2.loc[mask, 'extent'] - month_means[i+1]) / month_means.mean()
sns.lmplot(x="mo", y="extent", data=ice2, aspect=2)
plt.savefig('Phase-3-figure2.png')

print('mean: ', ice2.extent.mean())
print('var: ', ice2.extent.var())

sns.lmplot(x="year", y="extent", data=ice2, aspect=2)
plt.savefig("phase-3-figure.png")

# Data for january
jan = ice2[ice2.mo == 1]
sns.lmplot(x="year", y="extent", data=jan, height=6, aspect=2)
plt.savefig("phase-3-figure4.png")

# Data for Augest
aug = ice2[ice2.mo == 8]
sns.lmplot(x="year", y="extent", data=aug, height=6, aspect=2)
plt.savefig("phase-3-figure5.png")

print(scipy.stats.pearsonr(ice2.year.values, ice2.extent.values))


# Linear Regression model training

est = LinearRegression(fit_intercept=True)
x = ice2[['year']]
y = ice2[['extent']]
est.fit(x, y)

print("Coefficient: ", est.coef_)
print("Intercept: ", est.intercept_)

y_hat = est.predict(x)
plt.plot(x, y, 'o', alpha=0.5)
plt.plot(x, y_hat, 'r', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Extent")
plt.show()
print("MSE: ", metrics.mean_squared_error(y_hat, y))
print("R^2: ", metrics.r2_score(y, y_hat))
print("Var: ", y.var())

x_jan = jan[['year']]
y_jan = jan[['extent']]
jan_model = LinearRegression()
jan_model.fit(x_jan, y_jan)
y_jan_hat = jan_model.predict(x_jan)

plt.figure()
plt.plot(x_jan, y_jan, 'o', alpha=0.5)
plt.plot(x_jan, y_jan_hat, 'g', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Extent")
plt.show()


