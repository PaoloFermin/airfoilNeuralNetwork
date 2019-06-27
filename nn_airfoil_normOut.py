from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model, Sequential

#import tensorflow_model_optimization as tfmot

import pandas as pd

#load in data and format it for training
data = pd.read_csv('results.csv')
print(data)

#normalize data
from sklearn import preprocessing

input_features = data.loc[:,['Ux', 'Uy']]
output_features  = data.loc[:,['Cd','Cl']]

print(input_features)
print(output_features)

input_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
input_scaled = input_scaler.fit_transform(input_features.values.astype(float))
output_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
output_scaled = output_scaler.fit_transform(output_features.values.astype(float))

#output_scaled = output_features * 100

print(output_scaled)

input_scaled_df = pd.DataFrame(input_scaled)
output_scaled_df = pd.DataFrame(output_scaled)

df_normalized = pd.concat([input_scaled_df,output_scaled_df], axis=1)

#df_normalized = pos_normalized.join(neg_normalized) 
df_normalized.columns = ['Ux','Uy','Cd','Cl']

print("normalized data:\n")
print(df_normalized)

train_x = df_normalized[['Ux','Uy']]
train_y = df_normalized[['Cl','Cd']]	#the target column


#get number of columns in training data
n_cols = train_x.shape[1]

#create model
model = Sequential()
model.add(Dense(24, kernel_initializer='normal', activation='relu', input_shape=(n_cols,))) #add input layer and first hidden layer
#model.add(BatchNormalization())
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.3))

model.add(Dense(8, kernel_initializer='normal', activation='relu')) 
#model.add(BatchNormalization())
model.add(Dense(2))#output layer

#add optimizer
from keras import optimizers
adam = optimizers.Adam(lr=0.001)

#compile model
model.compile(optimizer=adam, loss='logcosh')

print(model.summary())

#split between training and testing data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=100) #stop training when it won't improve anymore

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
fit = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10000, callbacks=[tb, early_stopping_monitor],verbose=False)

print("x_test = ")
print(x_test)
print(type(x_test))
print("y_test = ")
print(y_test)

test_outputs = model.predict(x_test) #returns numpy array of predictions
#test_outputs /= 100

print(test_outputs)

#get test x values and predicted y values into one dataframe and display
output = x_test
pred_Cd = test_outputs[:, 0]
pred_Cl = test_outputs[:, 1]
 
output['pred_Cd'] = pred_Cd
output['pred_Cl'] = pred_Cl

print(output)


def reverse_normalized(value, df, column):
	#normalized_value * (max(x) - min(x)) + min(x)
		
	colMin = df[column].min()
	colMax = df[column].max()
	print("min = " + str(colMin))
	print("max = " + str(colMax))
	
	return value * (colMax - colMin) + colMin	

real_Ux = reverse_normalized(x_test['Ux'], data, 'Ux')
print("real Ux = " + str(real_Ux)) 

real_Uy = reverse_normalized(x_test['Uy'], data, 'Uy')
print("real Uy = " + str(real_Uy)) 

real_Cd = reverse_normalized(pred_Cd, data, 'Cd')
print("real Cd = " + str(real_Cd))	

real_Cl = reverse_normalized(pred_Cl, data, 'Cl')
print("real Cl = " + str(real_Cl))

output['real_Ux'] = real_Ux
output['real_Uy'] = real_Uy
output['real_Cd'] = real_Cd
output['real_Cl'] = real_Cl

#output.rename(columns={0:'case number'}, inplace=True)


#print(output)

#write output to csv file
output.to_csv('./results_nn.csv') 
	
#plot regression values during training
import matplotlib.pyplot as plt
print(fit.history.keys())



'''
real_outputs = output[['pred_Cd','pred_Cl']].apply(reverse_normalized, axis=1)
print("real output: ")
print(real_outputs)
'''
