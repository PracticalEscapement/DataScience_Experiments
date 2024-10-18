import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

edu = pd.read_csv('educ_figdp_1_Data.csv', 
na_values = ':',
usecols = ["TIME","GEO","Value"])

#print("\n\n\n")
#print(edu.head())  # returns the first 5 rows in the file
#
#print("\n\n\n")
#print(edu.tail())   # returns the last 5 rows in the file
#
#print("\n\n\n")
#print(edu.describe())
#
#print("\n\n\n")
#print(edu['Value'])
#
#print("\n\n\n")
#print(edu[10:14])   #returns results from range 10 inclusive to 14 exclusive.
#
##edu.xi[90:94 , ['TIME ','GEO']] ##depricated
#
#print("\n\n\n")
#print(edu.iloc[90:94, [edu.columns.get_loc('TIME'), edu.columns.get_loc('GEO')]])
#
#print("\n\n\n")
#print(edu[edu['Value'] > 6.5].tail())
#print("\n\n\n")
#print(edu[edu["Value"].isnull()])    #filter values with null value.
#
#print("\n\n\n")
#print(edu.max(axis = 0))

#print("Pandas max function", edu['Value'].max())
#print("Python max function", max(edu['Value']))
#
#s = edu['Value']/100
#print(s.head())

#sqr = edu['Value'].apply(np.sqrt)   # use numpy to find square root.
#print(sqr)
#exp = edu['Value'].apply(lambda d: d*d)  # use numpy to find exponent
#print(exp)

#edu['ValueNorm'] = edu['Value'] / edu['Value'].max()    #observe column added 'ValueNorm'
#print(edu['ValueNorm'])
#print(edu.head())
#
#edu.drop('ValueNorm', axis = 1, inplace = True) # removal of 'ValueNorm' column
#print(edu)
