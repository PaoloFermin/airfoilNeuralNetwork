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
from talos import Evaluate, Predict, Deploy, live
import os

#remove pesky deprecation warnings which I should probably care about but meh
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#import tensorflow_model_optimization as tfmot

#########################################


#knobs
inputs = ['U', 'angle']
outputs = ['Cl', 'Cd']

#create hyperparameter dictionary
p = {
	'first_neuron': [8, 16, 24, 32],
	'second_neuron': [8, 16, 24, 32],
	#'third_neuron': [8, 16, 24, 32],
	'losses': ['mean_squared_error'],
	'epochs': [15000],
 	'lr': [0.0001]
}
	
#add callbacks
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=200) 

#create log
#PREPROCESS DATA
data = pd.read_csv('results.csv')
#print(data)



def rescale(col, newMin, newMax, reverse=False, oldMin=-1, oldMax=1):
	if reverse:
		x = newMin + (((col - oldMin) * (newMax - newMin)) / (oldMax - oldMin))
	else:		
		x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	return x

df_normalized = data.copy() #separate copy from original
df_normalized['U'] = rescale(data['U'], -1, 1)
df_normalized['angle'] = rescale(data['angle'], -1, 1)
df_normalized['Ux'] = rescale(data['Ux'], -1, 1)
df_normalized['Uy'] = rescale(data['Uy'], -1, 1)
df_normalized['Cd'] = rescale(data['Cd'], -1, 1)
df_normalized['Cl'] = rescale(data['Cl'], -1, 1)

#df_normalized['Cl'] = rescale(data['Cl'], -1, 1)
#df_normalized['Cd'] = rescale(data['Cd'], -1, 1)

train_x = df_normalized[inputs]
train_y = df_normalized[outputs]	#the target column

#split between training and testing data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)

#convert dataframe objects into numpy arrays
#x_train = x_train.values
#y_train = y_train.values
#print(type(x_train))
print(y_test)
print(type(y_test))

#get number of columns in training data
n_cols = train_x.shape[1]
print("number of input columns: " + str(n_cols))

#CREATE AND BUILD MODEL

def create_model(x_train, y_train, x_test, y_test, params):
	global model
	
	#add optimizer
	adam = optimizers.Adam(lr=params['lr'])	

	model = Sequential()
	model.add(Dense(params['first_neuron'], kernel_initializer='normal', activation='relu', input_shape=(n_cols,))) 
	model.add(Dense(params['second_neuron'], kernel_initializer='normal', activation='relu')) 
	#model.add(Dense(params['third_neuron'], kernel_initializer='normal', activation='relu'))
	model.add(Dense(len(outputs)))#output layer

	#compile model
	model.compile(optimizer=adam, loss=params['losses'])
	print(model.summary())
	
	

	i = 0
	while os.path.exists('./logs_nn/%s' % i):
		i += 1
	logdir = './logs_nn/%s' % i
	tb = TensorBoard(log_dir=logdir)
	print('\n Run number %d \n' % i)

	#train model
	fit = model.fit(x_train, y_train, 
		validation_split=0.2, 
		epochs=params['epochs'], 
		verbose=False,
		callbacks=[early_stopping_monitor, tb]
	)
	
	return fit, model


i = 0
while os.path.exists('./talos_airfoil_%s.csv' % i):
	i += 1
t = ta.Scan(
	x=x_train.values, 
	y=y_train.values, 
	params=p, 
	model=create_model,
	dataset_name='talos_airfoil',
	experiment_no=str(i),
	reduction_method='correlation', 
	reduction_interval=25, 
	reduction_metric='val_loss',
	reduce_loss=True
)
	
print(t.details)

#report results
r = ta.Reporting(t)

print(r.data)	#returns a dataframe
print(r.best_params(metric='val_loss', ascending=True))

print("sorted results: ")
results = r.table(metric='val_loss', ascending=True)
print(results)	#returns a dataframe
#r.plot_line()

def best_model_by_loss(scan, metric, loss):

	isolated_df = scan.data.loc[scan.data['losses']==loss,:]
	best = isolated_df.sort_values(metric, ascending=True).iloc[0].name
	
	return best


#p = Predict(t)

print(t.best_model(metric='val_loss'))

#p.predict(x_test)

#evaluate models based on test dataset
e = Evaluate(t)

for loss in p['losses']:
	evaluation = e.evaluate(x_test.values, y_test.values, model_id=best_model_by_loss(t, 'val_loss', loss), metric='val_loss', asc=True, mode='regression', folds=10)
	print("predictions for %s is " % loss)
	print(evaluation)
	print("mean error for %s is %.6f" % (loss, mean(evaluation)))


i = 0
while os.path.exists('./optimized_airfoil_nn_%s.zip' % i):
	i += 1
deploy_dir = './optimized_airfoil_nn_%s' % i
Deploy(t, deploy_dir, metric='val_loss', asc=True)

validation_data = x_test

#get original y values
temp = data.iloc[y_test.index.values,:]
validation_data[['Cd', 'Cl']] = temp.loc[:, ['Cd', 'Cl']]
validation_data.to_csv('./optimization_validation_data.csv')

