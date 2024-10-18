import pandas as pd
import matplotlib as matlib

file = open('./adult.data', 'r')


def chr_int(a):
	if a.isdigit():
		return int(a)
	else: 
		return 0

data = []

for line in file:
	data1 = line.split(',')
	if len(data1) == 15:
		data.append([
			chr_int(data1[0]),
			data1[1], 
			chr_int(data1[2]), 
			data1[3], 
			chr_int(data1[4]), 
			data1[5], 
			chr_int(data1[6]), 
			data1[7], 
			data1[8], 
			data1[9], 
			chr_int(data1[10]), 
			chr_int(data1[11]), 
			chr_int(data1[12]), 
			data1[13], 
			data1[14].replace('\n', '')
		])
print(data[1:2])

df = pd.DataFrame(data)
print(df.shape)
counts = df.groupby(df[13]).size()
print(counts)

ml = df.groupby(df[9]).size()
print(ml)

df1 = df[df[14] == ' >50K']
print("The number of people with income less than 50K: ", (int(len(df1)))/float(len(df))*100)

print("The average age is: ", df1[0].mean())
print("The average age variance: ", df[0].var())
print("The std is: ", df1[0].std())
print("The median is: ", df1[0].median())
print("The min is: ", df1[0].min())
print("The max is: ", df1[0].max())

age_col = df1[0]
age_col.hist(histtype='stepfilled', bins = 20)
matlib.pyplot(age_col)