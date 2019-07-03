from talos import Evaluate, Restore
import pandas as pd
import matplotlib.pyplot as plt


validation_df = pd.read_csv('optimization_validation_data.csv')
validation_df.columns = ['case number', 'U', 'angle', 'Cd', 'Cl']

print(validation_df)

x_val = validation_df[['U','angle']]
y_val = validation_df[['Cd', 'Cl']]

net = Restore('optimized_airfoil_nn_2.zip')

pred = net.model.predict(x_val)
print("Cd predictions: ")
print(pred[:,0])
print("Cl predictions: ")
print(pred[:, 1])
validation_df['pred_Cd'] = pred[:, 0]
validation_df['pred_Cl'] = pred[:, 1]


validation_df['error_Cd'] = abs((validation_df['pred_Cd'] - validation_df['Cd']) / validation_df['Cd']) * 100

validation_df['error_Cl'] = abs((validation_df['pred_Cl'] - validation_df['Cl']) / validation_df['Cl']) * 100

validation_df.plot(kind='scatter', x='angle', y='Cl')
plt.show()
validation_df.plot(kind='scatter', x='angle', y='pred_Cl')

print(pred)

print(validation_df)
plt.show()
