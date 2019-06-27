from keras.layers import Input, Dense, Conv2D
from keras.models import Model, Sequential

#import tensorflow_model_optimization as tfmot

import pandas as pd

#load in data and format it for training
data = pd.read_csv('results.csv')
#print(data)

#normalize data
from sklearn import preprocessing

'''
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
data_scaled = scaler.fit_transform(data.loc[:,['Ux', 'Uy']].values.astype(float))
df_normalized = pd.DataFrame(data_scaled)
df_normalized['Cd'] = data.loc[:,'Cd']
df_normalized['Cl'] = data.loc[:,'Cl']
'''

def rescale(col, newMin, newMax):
	#minmaxscaler: x = newmin + (x - xmin)(newmax -newmin) / (xmax -xmin)
	print("scaling %s to min %d and max %d" % (col.head, newMin, newMax))	
	x = newMin + (col - col.min()) * (newMax - newMin) / (col.max() - col.min())
	print(x)
	return x

df_normalized = data.copy() #separate copy from original
df_normalized['U'] = rescale(data['U'], -1, 1)
df_normalized['angle'] = rescale(data['angle'], -1, 1)
df_normalized['Ux'] = rescale(data['Ux'], -1, 1)
df_normalized['Uy'] = rescale(data['Uy'], -1, 1)
#df_normalized = pos_normalized.join(neg_normalized) 
#df_normalized.columns = ['Ux','Uy','Cd','Cl']

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
model.compile(optimizer=adam, loss='mean_squared_error')

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

'''
import tensorflow_model_optimization as tfmot

#set up model pruning
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
	initial_sparsity=0.0, final_sparsity=0.5, 
	begin_step=2000, end_step=4000)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
'''

#train model
fit = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10000, callbacks=[tb, early_stopping_monitor])

print("x_test = ")
print(x_test)
print(type(x_test))
print("y_test = ")
print(y_test)

predictions = model.predict(x_test) #returns numpy array of predictions
print(predictions)

#get test x values and predicted y values into one dataframe and display
results = x_test

for i in range(len(outputs)):
	pred = 'pred_%s'%outputs[i]
	real = 'real_%s'%outputs[i]
	results[pred] = predictions[:, i]
	results[real] = y_test
	#results['unscaled_input_%s'%inputs[i]] = rescale(x_test[inputs[i]], data[inputs[i]].min(), data[inputs[i]].max()) 
	results['error_%s (%%)'%outputs[i]] = ((results[pred] - results[real]) / (results[real])) * 100 
#output.rename(columns={0:'case number'}, inplace=True)



#output['unscaled_Ux'] = scaler.inverse_transform(data_scaled)
#output['unscaled_Uy'] = scaler.inverse_transform({1, output.loc[:,'Uy']})


print(results)

#write output to csv file
results.to_csv('./results_nn.csv', index=False) 
	


'''
real_outputs = output[['pred_Cd','pred_Cl']].apply(reverse_normalized, axis=1)
print("real output: ")
print(real_outputs)
'''
