import pandas as pd
import numpy as np

data = pd.read_csv('results.csv')

def rescale(col, newMin, newMax):
	print(col)	
	print(col.min())
	print(type(col))	
	a = col - col.min()	
	b = col.max() - col.min()	
	c = newMax - newMin	
	d = a*c / b
#	return newMin + d
	return ((col - col.min()) * (newMax - newMin) / (col.max() - col.min())) + newMin

np_rescale = np.frompyfunc(rescale, 3, 1)

#data.apply(np_rescale, args=(data['U'],-1, 1))

#scaled_data = data['U'].apply(rescale, kwargs=(value=data['U'], newMin=0,newMax=1)) 

scaled_data = rescale(data['U'], -1, 1)


print(scaled_data)

