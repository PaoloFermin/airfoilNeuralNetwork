from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.losses import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error, mean_absolute_percentage_error
from keras import optimizers
from scipy import stats

import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import mean
import talos as ta
from talos import Evaluate, Predict
import os

#remove pesky deprecation warnings which I should probably care about but meh
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#import tensorflow_model_optimization as tfmot

#########################################


#knobs
inputs = ['U', 'angle']
outputs = ['Cd', 'Cl']

#create hyperparameter dictionary
p = {
	'first_neuron': [4, 8, 16, 32],
	'second_neuron': [8, 16, 32],
	'losses': ['mean_absolute_error'],
	'epochs': [5000, 10000, 15000],
 	'lr': [0.001, 0.01]
}
	
#add callbacks
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=200) 

#create log
i = 0
while os.path.exists('./logs_nn/%s' % i):
	i += 1
logdir = './logs_nn/%s' % i
tb = TensorBoard(log_dir=logdir)
print('\n Run number %d \n' % i)

#PREPROCESS DATA
data = pd.read_csv('results.csv')
#print(data)



def rescale(col, newMin, newMax, reverse=False, original=None):
	x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	return x

df_normalized = data.copy() #separate copy from original
df_normalized['U'] = rescale(data['U'], -1, 1)
df_normalized['angle'] = rescale(data['angle'], -1, 1)
df_normalized['Ux'] = rescale(data['Ux'], -1, 1)
df_normalized['Uy'] = rescale(data['Uy'], -1, 1)
df_normalized['Cd'] = data['Cd']
df_normalized['Cl'] = data['Cl']

#df_normalized['Cl'] = rescale(data['Cl'], -1, 1)
#df_normalized['Cd'] = rescale(data['Cd'], -1, 1)

train_x = df_normalized[inputs].values
train_y = df_normalized[outputs].values	#the target column

#split between training and testing data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)

#convert dataframe objects into numpy arrays
#x_train = x_train.values
#y_train = y_train.values
#print(type(x_train))


#get number of columns in training data
n_cols = train_x.shape[1]
print("number of inputs columns: " + str(n_cols))

#CREATE AND BUILD MODEL

def create_model(x_train, y_train, x_test, y_test, params):
	global model
	
	#add optimizer
	adam = optimizers.Adam(lr=params['lr'])	

	model = Sequential()
	model.add(Dense(params['first_neuron'], kernel_initializer='normal', activation='relu', input_shape=(n_cols,))) 
	model.add(Dense(params['second_neuron'], kernel_initializer='normal', activation='relu')) 
	model.add(Dense(len(outputs)))#output layer

	#compile model
	model.compile(optimizer=adam, loss=params['losses'])
	print(model.summary())

	#train model
	fit = model.fit(x_train, y_train, validation_split=0.2, epochs=params['epochs'], verbose=False)

	return fit, model


t = ta.Scan(
	x=x_train, 
	y=y_train, 
	params=p, 
	model=create_model,
	dataset_name='talos_airfoil',
	experiment_no='1',
	grid_downsample=0.25
)
	
print(t.details)

#report results
r = ta.Reporting(t)

print(r.data)
print(r.best_params(metric='val_loss', ascending=True))

print("sorted results: ")
results = r.table(metric='val_loss', ascending=True)
print(results)
print(type(results))
#r.plot_line()

p = Predict(t)

print(t.best_model(metric='val_loss'))

#p.predict(x_test)

#evaluate predictions
e = Evaluate(t)

evaluation = e.evaluate(x_test, y_test, metric='val_loss', asc=True, mode='regression', folds=10)
print("mean error: " + str(mean(evaluation)))


'''
for case in cases:
	
	tf.keras.backend.clear_session()
	#graph is a platform for creating a tf model
	#reset the graph for each case
	graph = tf.Graph()


	with tf.Session(graph=graph):
		
		#create model
		create_model(case)
		
		#train model
		train_model()

		#GET RESULTS
		predictions = model.predict(x_test) #returns numpy array of predictions
		#print(predictions)

		#get test x values and predicted y values into one dataframe and display
		results = x_test


		for i in range(len(outputs)):
			real = 'real_%s'%outputs[i]
			pred = 'pred_%s'%outputs[i]
			err = 'error_%s (%%)'%outputs[i]
			
			#get original y values
			temp = data.iloc[y_test.index.values,:]
			results[real] = temp.loc[:, outputs[i]]

			#use this if output scaling
			results[pred] = rescale(predictions[:, i], data[outputs[i]].min(), data[outputs[i]].max())
			
			#use this if no output scaling
			#results[pred] = predictions[:, i]

			#results[real] = rescale(y_test[outputs[i]], -1, 1, reverse=True, original=data[outputs[i]])
			#results[pred] = rescale(predictions[:, i], -1, 1, reverse=True, original=data[outputs[i]])	

			results[err] = abs((results[pred] - results[real]) / (results[real])) * 100 
			
			#remove outliers
			trimmed_results = results[(np.abs(stats.zscore(results)) < 3).all(axis=1)]
			avg_error = trimmed_results[err].mean()
			print("Average error of %s is %.2f" % (outputs[i], avg_error))
			np.append(case_errors, avg_error)
			numOutliers = results.shape[0] - trimmed_results.shape[0]
			print("Removed %d outliers: Average error of %s is %.2f" % (numOutliers, outputs[i], trimmed_results[err].mean()))

		#print(results)
		print(trimmed_results)
		#write output to csv file
		results.to_csv('./results_nn.csv', index=True) 
		


	analyzed_results['error'] = case_errors
	print(analyzed_results)

'''
