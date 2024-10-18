import matplotlib.pylab as plt
from matplotlib import cm
import math
import pandas as pd
import numpy as np
import random

#Hypothesis testing
#Testing hypotheses using confidence intervals.

data = pd.read_csv("ACCIDENTS_GU_BCN_20101.csv", encoding='latin-1')
#Create a new column which is the date
data['Date'] = data['Day of month'].apply(lambda x : str(x)) + '-' +  \
               data['Month of the year'].apply(lambda x : str(x))
data2 = data['Date']
counts2010 =data['Date'].value_counts()
print('2010: Mean', counts2010.mean())

data = pd.read_csv("ACCIDENTS_GU_BCN_20131.csv", encoding='latin-1')
#Create a new column which is the date
data['Date'] = data['Day of month'].apply(lambda x : str(x)) + '-' +  \
               data['Month of the year'].apply(lambda x : str(x))
data2 = data['Date']
counts2013 = data['Date'].value_counts()
print('2013: Mean', counts2013.mean())

n = len(counts2013)
mean = counts2013.mean()
s = counts2013.std()
ci = [mean - s*1.96/np.sqrt(n),  mean + s*1.96/np.sqrt(n)] 
print('2010 accident rate estimate:', counts2010.mean())
print('2013 accident rate estimate:', counts2013.mean())
print('CI for 2013:',ci)


#Testing hypotheses using P-values.
m = len(counts2010)
n = len(counts2013)
p = (counts2013.mean() - counts2010.mean())
print('m:',m, 'n:', n)
print('mean difference: ', p)


x = counts2010
y = counts2013
pool = np.concatenate([x,y])
np.random.shuffle(pool)

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plt.hist(pool, 
         bins = 25, 
         density = True)
plt.ylabel('Frequency')
plt.xlabel('Number of accidents')
plt.title("Pooled distribution")


N = 10000 # number of samples
diff = np.arange(N)
for i in np.arange(N):
    p1 = [random.choice(pool) for _ in np.arange(n)]
    p2 = [random.choice(pool) for _ in np.arange(n)]
    diff[i] = (np.mean(p1)-np.mean(p2))

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plt.hist(diff, bins = 50, density = True)
plt.ylabel('Frequency')
plt.xlabel('Difference in the mean')

# counting how many differences are larger than the observed one
diff2 = np.array(diff)
w1 = np.where(diff2 > p)[0]      
len(w1)

print('p-value (Simulation)=', len(w1)/float(N), '(', len(w1)/float(N)*100 ,'%)', 'Difference =', p)
if len(w1)/float(N)<0.05:
    print('The effect is likely')
else:
    print('The effect is not likely')