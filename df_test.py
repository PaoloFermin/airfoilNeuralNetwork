from __future__ import division
import pandas as pd
import numpy as np

data = pd.read_csv('results.csv')

import sys
print(sys.version)



print("original data")
print(data)



'''
nn_data = pd.read_csv('results_nn.csv')
nn_data.rename(columns={'Unnamed: 0':'case number'}, inplace=True)
print("neural network output:")
print(nn_data)

temp = data.iloc[nn_data['case number'],:]
original_test_data = temp.loc[:, ('Cd', 'Cl')]
print("original test data:")
print(original_test_data)

original_test_data = original_test_data.reset_index(drop=False)

comparison = pd.concat([original_test_data, nn_data.loc[:, ['real_Cd', 'real_Cl']]], axis=1)
print("comparison:")
print(comparison)
'''

def rescale(col, oldMin, oldMax, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (((col - oldMin) * (newMax - newMin)) / (oldMax - oldMin))
	#print(x)
	return x

data['Cd_scaled'] = rescale(data.loc[:,'Cd'], data.loc[:, 'Cd'].min(), data.loc[:, 'Cd'].max(), -1, 1)
val_data = data.loc[:, ['Cd', 'Cd_scaled']].sample(frac=.2)
print("validation data: ")
sk_data = val_data.copy()

#val_data['Cd_scaled'] = rescale(val_data.loc[:, 'Cd'], -1, 1)
val_data['Cd_unscaled'] = rescale(val_data.loc[:, 'Cd_scaled'], -1, 1, data['Cd'].min(), data['Cd'].max())
val_data['Cd_error'] = abs((val_data['Cd_unscaled'] - val_data['Cd']) / (val_data['Cd'])) * 100 

'''
val_data['Cl_scaled'] = rescale(val_data.loc[:, 'Cl'], -1, 1)
val_data['Cl_unscaled'] = rescale(val_data.loc[:, 'Cl_scaled'], val_data['Cl'].min(), val_data['Cl'].max())
val_data['Cl_error'] = abs((val_data['Cl_unscaled'] - val_data['Cl']) / (val_data['Cl'])) * 100 
'''

val_data = val_data[['Cd', 'Cd_scaled', 'Cd_unscaled', 'Cd_error']]
print(val_data)

'''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
sk_data['Cd_scaled'] = scaler.fit_transform(sk_data)
sk_data['Cd_unscaled'] = scaler.inverse_transform(sk_data)
print(sk_data)
'''

'''
scaled_data = rescale(data[target], -1, 1)
#results = pd.concat([original_test_data, scaled_data], axis=1)
results = pd.DataFrame(scaled_data)
#results.rename(columns={target:'scaled_data'}, inplace=True)
results['unscaled_data'] = rescale(scaled_data, data[target].min(), data[target].max())
'''
#print("results:")
#print(results)

