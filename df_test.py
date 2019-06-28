import pandas as pd
import numpy as np

data = pd.read_csv('results.csv')
print(data)

def rescale(col, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	#print(x)
	return x

target = 'Cl'

scaled_data = rescale(data[target], -1, 1)
results = pd.DataFrame(scaled_data)
#results.rename(columns={target:'scaled_data'})
results['unscaled_data'] = rescale(scaled_data, data[target].min(), data[target].max())

print(results)

nn_data = pd.read_csv('results_nn.csv')
print(nn_data)

