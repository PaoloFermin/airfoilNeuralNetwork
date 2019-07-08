from talos import Evaluate, Restore
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import os
from matplotlib import pyplot as plt

plt.switch_backend('TkAgg')
print(plt.get_backend())


validation_df = pd.read_csv('optimization_validation_data.csv')
validation_df.columns = ['case number', 'U', 'angle', 'Cd', 'Cl']

validation_df.sort_values(by=['angle'], inplace=True)

print(validation_df)

x_val = validation_df[['U','angle']]
y_val = validation_df[['Cd', 'Cl']]

i = 0
while os.path.exists('optimized_airfoil_nn_%s.zip' % i):
	i += 1
i -= 1
net = Restore('optimized_airfoil_nn_%s.zip'%i)

pred = net.model.predict(x_val)
print("Cd predictions: ")
print(pred[:,0])
print("Cl predictions: ")
print(pred[:, 1])
validation_df['pred_Cd'] = pred[:, 1]
validation_df['pred_Cl'] = pred[:, 0]


validation_df['error_Cd'] = abs((validation_df['pred_Cd'] - validation_df['Cd']) / validation_df['Cd']) * 100

validation_df['error_Cl'] = abs((validation_df['pred_Cl'] - validation_df['Cl']) / validation_df['Cl']) * 100

validation_df.plot(x='angle', y=['Cl','pred_Cl'])
#plt.show()
#validation_df.plot(kind='scatter', x='angle', y='pred_Cl')

validation_df.plot(x='angle', y=['Cd', 'pred_Cd'])
print(pred)

print(validation_df)
plt.show()
