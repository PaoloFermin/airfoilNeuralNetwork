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

def rescale(col, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	#print(x)
	return x


validation_data = data.loc[:, ['Cd', 'Cl']].sample(frac=.2)
print("validation data: ")
sk_data = validation_data.copy()

validation_data['Cd_scaled'] = rescale(validation_data.loc[:, 'Cd'], -1, 1)
validation_data['Cd_unscaled'] = rescale(validation_data.loc[:, 'Cd_scaled'], validation_data['Cd'].min(), validation_data['Cd'].max())
validation_data['Cd_error'] = abs((validation_data['Cd_unscaled'] - validation_data['Cd']) / (validation_data['Cd'])) * 100 

validation_data['Cl_scaled'] = rescale(validation_data.loc[:, 'Cl'], -1, 1)
validation_data['Cl_unscaled'] = rescale(validation_data.loc[:, 'Cl_scaled'], validation_data['Cl'].min(), validation_data['Cl'].max())
validation_data['Cl_error'] = abs((validation_data['Cl_unscaled'] - validation_data['Cl']) / (validation_data['Cl'])) * 100 

validation_data = validation_data[['Cd', 'Cd_scaled', 'Cd_unscaled', 'Cd_error', 'Cl', 'Cl_unscaled', 'Cl_error']]
print(validation_data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
sk_data['Cd_scaled'] = scaler.fit_transform(sk_data)
sk_data['Cd_unscaled'] = scaler.inverse_transform(sk_data)
print(sk_data)


'''
scaled_data = rescale(data[target], -1, 1)
#results = pd.concat([original_test_data, scaled_data], axis=1)
results = pd.DataFrame(scaled_data)
#results.rename(columns={target:'scaled_data'}, inplace=True)
results['unscaled_data'] = rescale(scaled_data, data[target].min(), data[target].max())
'''
#print("results:")
#print(results)

