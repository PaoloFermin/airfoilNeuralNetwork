from keras.layers import Input, Dense, Conv2D
from keras.models import Model, Sequential
from scipy import stats

import tensorflow as tf
import pandas as pd
import numpy as np
import os

#remove pesky deprecation warnings which I should probably care about but meh
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#import tensorflow_model_optimization as tfmot

#########################################

#knobs
inputs = ['U', 'angle']
outputs = ['Cd', 'Cl']
cases = [1, 2, 4, 8, 16, 32]

#add callbacks
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping_monitor = EarlyStopping(patience=150) #stop training when it won't improve anymore

#create log
i = 0
while os.path.exists('./logs_nn/%s' % i):
	i += 1
logdir = './logs_nn/%s' % i
tb = TensorBoard(log_dir=logdir)
print('\n Run number %d \n' % i)

#PREPROCESS DATA
#load in data and format it for training
data = pd.read_csv('results.csv')
#print(data)

#rescale data
#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(-1, 1))


def rescale(col, newMin, newMax, reverse=False, original=None):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	#if reverse:
		#x = ((col * (original.max() - original.min()) - newMin) / (newMax - newMin)) + original.min()
	#else:
	x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	#print(x)
	
	#x = scaler.fit_transform(col.values.astype(float)) 
	return x

df_normalized = data.copy() #separate copy from original
df_normalized['U'] = rescale(data['U'], -1, 1)
df_normalized['angle'] = rescale(data['angle'], -1, 1)
df_normalized['Ux'] = rescale(data['Ux'], -1, 1)
df_normalized['Uy'] = rescale(data['Uy'], -1, 1)
df_normalized['Cl'] = rescale(data['Cl'], -1, 1)
df_normalized['Cd'] = rescale(data['Cd'], -1, 1)

print("original data:\n")
print(data)
print("normalized data:\n")
print(df_normalized)

train_x = df_normalized[inputs]
train_y = data[outputs]	#the target column

#split between training and testing data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)

#get number of columns in training data
n_cols = train_x.shape[1]

#CREATE AND BUILD MODEL
#create model
model = Sequential()
model.add(Dense(8, kernel_initializer='normal', activation='relu', input_shape=(n_cols,))) #add input layer and first hidden layer
#model.add(Dense(24, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(8, kernel_initializer='normal', activation='relu')) 
model.add(Dense(len(outputs)))#output layer

#add optimizer
from keras import optimizers
adam = optimizers.Adam(lr=0.01)

#compile model
model.compile(optimizer=adam, loss='mean_squared_error')
print(model.summary())

#train model
fit = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10000, callbacks=[tb, early_stopping_monitor], verbose=False)


print("x_test = ")
print(x_test)
print("y_test = ")
print(y_test)
print(type(y_test))


#GET RESULTS
predictions = model.predict(x_test) #returns numpy array of predictions
print(predictions)

#get test x values and predicted y values into one dataframe and display
results = x_test
analyzed_results = pd.DataFrame(cases)

for i in range(len(outputs)):
	real = 'real_%s'%outputs[i]
	pred = 'pred_%s'%outputs[i]
	err = 'error_%s (%%)'%outputs[i]
	
	#get original y values
	temp = data.iloc[y_test.index.values,:]
	results[real] = temp.loc[:, outputs[i]]

	#use this if output scaling
	#results[pred] = rescale(predictions[:, i], data[outputs[i]].min(), data[outputs[i]].max())
	
	#use this if no output scaling
	results[pred] = predictions[:, i]

	#results[real] = rescale(y_test[outputs[i]], -1, 1, reverse=True, original=data[outputs[i]])
	#results[pred] = rescale(predictions[:, i], -1, 1, reverse=True, original=data[outputs[i]])	

	results[err] = abs((results[pred] - results[real]) / (results[real])) * 100 
	trimmed_results = results[(np.abs(stats.zscore(results)) < 3).all(axis=1)]
	print("Average error of %s is %.2f" % (outputs[i], results[err].mean()))
	numOutliers = results.shape[0] - trimmed_results.shape[0]
	print("Removed %d outliers: Average error of %s is %.2f" % (numOutliers, outputs[i], trimmed_results[err].mean()))

print(results)
print(trimmed_results)
#write output to csv file
results.to_csv('./results_nn.csv', index=True) 

