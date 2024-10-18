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

#Variability in estimates.
df = accidents.to_frame()
m = []

for i in range(10):
    df['for_testing'] = False
    # get a 25% sample 
    sampled_ids = np.random.choice(df.index,
                                   size=np.int64(np.ceil(df.index.size * 0.25)),
                                   replace=False)
    df.loc[sampled_ids, 'for_testing'] = True
    accidents_sample = df[df['for_testing'] == True]
    m.append(accidents_sample[0].mean())
    print('Sample '+str(i)+': Mean', '%.2f' % accidents_sample[0].mean())


fig, ax = plt.subplots(1, 1, figsize=(12, 2))
x = range(10)
ax.step(x,m, where='mid')
ax.set_ylabel('Mean')
ax.set_xlabel('Sample')


#Sampling distribution of point estimates
plt.autumn()

# population
df = accidents.to_frame()    
N_test = 10000              
elements = 200             

# mean array of samples
means = [0] * N_test             

# sample generation
for i in range(N_test):          
    rows = np.random.choice(df.index.values, elements)
    sampled_df = df.loc[rows]
    means[i] = sampled_df.mean()
    
fig, ax = plt.subplots(1, 1, figsize=(12,3))

plt.hist(np.array(means),bins=50)
plt.ylabel('Frequency')
plt.xlabel('Sample mean value')
ax.axvline(x = np.array(means).mean(), 
           ymin = 0, 
           ymax = 700, 
           color = [1, 0, 0])
plt.savefig("empiricalmean.png",dpi=300, bbox_inches='tight')
plt.show()
plt.set_cmap(cmap=cm.Pastel2)

print("Sample mean:", np.array(means).mean())

#Standard error of the mean
rows = np.random.choice(df.index.values, 200)
sampled_df = df.loc[rows]
est_sigma_mean = sampled_df.std()/math.sqrt(200)

print('Direct estimation of SE from one sample of 200 elements:', \
       est_sigma_mean[0])
print('Estimation of the SE by simulating 10000 samples of 200 elements:',  \
       np.array(means).std())


#Bootstrapping the standard error of the mean.
def meanBootstrap(X,numberb):
    import numpy as np
    x = [0]*numberb
    for i in range(numberb):
        sample = [X[_] for _ in np.random.randint(len(X), size=len(X))]
        x[i] = np.mean(sample)
    return x

#Bootstrapping the standard error of the mean.

m = meanBootstrap(accidents, 10000)
print("Mean estimate:", np.mean(m))

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plt.ylabel('Frequency')
plt.xlabel('Sample mean value')
plt.hist(m, 
         bins = 50, 
         density = True)
ax.axvline(x = np.mean(m), 
           ymin = 0.0, 
           ymax = 1.0, 
           color = [1, 0, 0])


def medBootstrap(X,numberb):
    import numpy as np
    x = [0]*numberb
    for i in range(numberb):
        sample = [X[_] for _ in np.random.randint(len(X), size=len(X))]
        x[i] = np.median(sample)
    return x

med = medBootstrap(accidents, 10000)
print("Median estimate:", np.mean(med) )
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plt.hist(med, bins=5, density=True)
plt.ylabel('Frequency')
plt.xlabel('Sample median value')
ax.axvline(x = np.array(med).mean(), 
           ymin = 0, 
           ymax = 1.0, 
           color = [1, 0, 0])
