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
data['Date'] = '2013-' + data['Month of the year'].apply(lambda x : str(x)) + '-' +  data['Day of month'].apply(lambda x : str(x))
data['Date'] = pd.to_datetime(data['Date'])
accidents = data.groupby(['Date']).size()
print("Mean:", accidents.mean())

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.ylabel('Number of accidents')
plt.xlabel('Day')
plt.plot(range(0, 365), np.array(accidents), 'b-+', lw=0.7, alpha=0.7)
plt.plot(range(0, 365), [accidents.mean()]*365, 'r-', lw=0.7, alpha=0.9)
plt.show()

#exit()
#We can plot the distribution of our variable of interest: the daily number of accidents.

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plt.ylabel('Frequency')
plt.xlabel('Number of accidents')
plt.hist(np.array(accidents), bins=20)
ax.axvline(x=accidents.mean(), ymin=0, ymax=40, color=[1, 0, 0])
plt.savefig("bootmean.png",dpi=300, bbox_inches='tight')
plt.show()

print("Mean:", accidents.mean(), "; STD:", accidents.std())
