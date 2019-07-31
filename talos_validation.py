from talos import Evaluate, Restore
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('TkAgg')
print(plt.get_backend())

def rescale(col, newMin, newMax, reverse=False, oldMin=-1, oldMax=1):
	if reverse:
		x = newMin + (((col - oldMin) * (newMax - newMin)) / (oldMax - oldMin))
	else:		
		x = newMin + (((col - col.min()) * (newMax - newMin)) / (col.max() - col.min()))
	return x

orig_df = pd.read_csv('results.csv')
validation_df = pd.read_csv('optimization_validation_data.csv')
validation_df.columns = ['case number', 'U', 'angle', 'Cd', 'Cl']

validation_df.sort_values(by=['angle'], inplace=True)

print(validation_df)

x_val = validation_df[['U','angle']]
y_val = validation_df[['Cd', 'Cl']]

i = 0
while os.path.exists('./optimized_networks/optimized_airfoil_nn_%s.zip' % i):
	i += 1
i -= 1
net = Restore('./optimized_networks/optimized_airfoil_nn_%s.zip'%i)

pred = net.model.predict(x_val)
print("Cd predictions: ")
print(pred[:,0])
print("Cl predictions: ")
print(pred[:, 1])
validation_df['pred_Cd'] = rescale(pred[:, 1], orig_df['Cd'].min(), orig_df['Cd'].max(), reverse=True)
validation_df['pred_Cl'] = rescale(pred[:, 0], orig_df['Cl'].min(), orig_df['Cl'].max(), reverse=True)


validation_df['error_Cd'] = abs((validation_df['pred_Cd'] - validation_df['Cd']) / validation_df['Cd']) * 100

validation_df['error_Cl'] = abs((validation_df['pred_Cl'] - validation_df['Cl']) / validation_df['Cl']) * 100

validation_df.plot(x='angle', y=['Cl','pred_Cl'])
#plt.show()
#validation_df.plot(kind='scatter', x='angle', y='pred_Cl')

validation_df.plot(x='angle', y=['Cd', 'pred_Cd'])
print(pred)

print(validation_df)
#plt.show()

#threedee = plt.figure().gca(projection='3d')
#threedee.plot_trisurf(validation_df['angle'], validation_df['U'], validation_df['pred_Cd'])
#threedee.plot_trisurf(validation_df['angle'], validation_df['U'], validation_df['Cd'])
plt.show()

