from talos import Evaluate, Restore
import pandas as pd

validation_df = pd.read_csv('validation_results.csv')
validation_df.columns = ['Ux', 'Uy', 'U', 'angle', 'Cd', 'Cl']

print(validation_df)

x_val = validation_df[['U','angle']]
y_val = validation_df[['Cd', 'Cl']]

net = Restore('optimized_airfoil_nn.zip')

pred = net.model.predict(x_val)

print(pred)
