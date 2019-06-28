from keras.layers import Input, Dense, Conv2D
from keras.models import Model, Sequential

#import tensorflow_model_optimization as tfmot

import pandas as pd

#load in data and format it for training
data = pd.read_csv('results.csv')
#print(data)

#normalize data

def rescale(col, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	#print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (col - col.min()) * (newMax - newMin) / (col.max() - col.min())
	#print(x)
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

inputs = ['U', 'angle']
outputs = ['Cd']

train_x = df_normalized[inputs]
train_y = df_normalized[outputs]	#the target column


#get number of columns in training data
n_cols = train_x.shape[1]

#create model
model = Sequential()
model.add(Dense(24, kernel_initializer='normal', activation='relu', input_shape=(n_cols,))) #add input layer and first hidden layer
#model.add(Dense(24, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu')) 
model.add(Dense(len(outputs)))#output layer

#add optimizer
from keras import optimizers
adam = optimizers.Adam(lr=0.01)

#compile model
model.compile(optimizer=adam, loss='logcosh')

print(model.summary())

#split between training and testing data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping_monitor = EarlyStopping(patience=150) #stop training when it won't improve anymore

import os

i = 0
while os.path.exists('./logs_nn/%s' % i):
	i += 1
logdir = './logs_nn/%s' % i
tb = TensorBoard(log_dir=logdir)


#train model
fit = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10000, callbacks=[tb, early_stopping_monitor], verbose=False)

print("x_test = ")
print(x_test)
print("y_test = ")
print(y_test)
print(type(y_test))

predictions = model.predict(x_test) #returns numpy array of predictions
print(predictions)


#get test x values and predicted y values into one dataframe and display
results = x_test

for i in range(len(outputs)):
	real = 'real_%s'%outputs[i]
	pred = 'pred_%s'%outputs[i]
	results[real] = rescale(y_test[outputs[i]], data[outputs[i]].min(), data[outputs[i]].max())
	results[pred] = rescale(predictions[:, i], data[outputs[i]].min(), data[outputs[i]].max())
	
	results['error_%s (%%)'%outputs[i]] = ((results[pred] - results[real]) / (results[real])) * 100 



#output['unscaled_Ux'] = scaler.inverse_transform(data_scaled)
#output['unscaled_Uy'] = scaler.inverse_transform({1, output.loc[:,'Uy']})


print(results)

#write output to csv file
results.to_csv('./results_nn.csv', index=False) 

