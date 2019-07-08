import pandas as pd
import numpy as np

data = pd.read_csv('results.csv')

'''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

sk_data = data.loc[:, ['Cd','Cl']].sample(frac=0.2)
print(sk_data)

relevant = data.loc[:, ['Cd', 'Cl']]

fit = scaler.fit(relevant)

print(fit.transform(relevant))
print(fit.inverse_transform(relevant))
'''

def rescale(col, oldMin, oldMax, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (((col - oldMin) * (newMax - newMin)) / (oldMax - oldMin))
	#print(x)
	return x

relevant = data.loc[:, ['Cd']]
cd_scaled = rescale(relevant['Cd'], relevant['Cd'].min(), relevant['Cd'].max(), -1, 1)
#cl_scaled = rescale(relevant['Cl'], -1, 1)
val = data.loc[:, ['Cd', 'Cl']].sample(frac=0.2)

