import matplotlib.pylab as plt
from matplotlib import cm
import math
import pandas as pd
import numpy as np
import random

#print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')

#if you need to make the usetex true the following package need to be installed
# on your local machine  using the following command
#sudo apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng

plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', size=12) 

data = pd.read_csv("ACCIDENTS_GU_BCN_20131.csv", encoding='latin-1')
print(data.columns)
#Create a new column which is the date
data['Date'] = '2013-'+data['Month of the year'].apply(lambda x : str(x)) + '-' +  data['Day of month'].apply(lambda x : str(x))
data['Date'] = pd.to_datetime(data['Date'])
accidents = data.groupby(['Date']).size()

def meanBootstrap(X,numberb):
    import numpy as np
    x = [0]*numberb
    for i in range(numberb):
        sample = [X[_] for _ in np.random.randint(len(X), size=len(X))]
        x[i] = np.mean(sample)
    return x

m = meanBootstrap(accidents, 10000)

data = pd.read_csv("ACCIDENTS_GU_BCN_20131.csv", encoding='latin-1')
print(data.columns)
#Create a new column which is the date
data['Date'] = '2013-'+data['Month of the year'].apply(lambda x : str(x)) + '-' +  data['Day of month'].apply(lambda x : str(x))
data['Date'] = pd.to_datetime(data['Date'])
accidents = data.groupby(['Date']).size()

#Confidence intervals.
m = accidents.mean()
se = accidents.std()/math.sqrt(len(accidents))
ci = [m - se*1.96, m + se*1.96]
print("Confidence interval:", ci)


m = meanBootstrap(accidents, 10000)
sample_mean = np.mean(m)
sample_se =  np.std(m)

print("Mean estimate:", sample_mean)
print("SE of the estimate:", sample_se)

ci = [np.percentile(m,2.5), np.percentile(m,97.5)]
print("Confidence interval:", ci)


#What is the real meaning of CI?
df = accidents   

n = 100                                               # number of observations
N_test = 100                                          # number of samples with n observations
means = np.array([0.0] * N_test)                      # samples' mean
s = np.array([0.0] * N_test)                          # samples' std
ci = np.array([[0.0,0.0]] * N_test)
tm = df.mean()                                        # "true" mean

for i in range(N_test):                               # sample generation and CI computation
    rows = np.random.choice(df.index.values, n)
    sampled_df = df.loc[rows]
    means[i] = sampled_df.mean()
    s[i] = sampled_df.std()
    ci[i] = means[i] + np.array([-s[i] *1.96/np.sqrt(n), s[i]*1.96/np.sqrt(n)])    

out1 = ci[:,0] > tm                                   # CI that do not contain the "true" mean
out2 = ci[:,1] < tm

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ind = np.arange(1, N_test+1)
ax.axhline(y = tm, 
           xmin = 0, 
           xmax = N_test+1, 
           color = [0, 0, 0])
ci = np.transpose(ci)
ax.plot([ind,ind], 
        ci, 
        color = '0.75', 
        marker = '_', 
        ms = 0, 
        linewidth = 3)
ax.plot([ind[out1],ind[out1]], 
        ci[:, out1], 
        color = [1, 0, 0, 0.8], 
        marker = '_', 
        ms = 0, 
        linewidth = 3)
ax.plot([ind[out2],ind[out2]], 
        ci[:, out2], 
        color = [1, 0, 0, 0.8], 
        marker = '_',
        ms = 0, 
        linewidth = 3)
ax.plot(ind, 
        means, 
        color = [0, .8, .2, .8], 
        marker = '.',
        ms = 10, 
        linestyle = '')
ax.set_ylabel("Confidence interval for the samples' mean estimate",
              fontsize = 12)
ax.set_xlabel('Samples (with %d observations). '  %n, 
              fontsize = 12)
plt.savefig("confidence.png",
            dpi = 300, 
            bbox_inches = 'tight')
plt.show()
