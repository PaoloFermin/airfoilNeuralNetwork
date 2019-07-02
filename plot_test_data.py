import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('validation_results.csv')
df.plot(kind='scatter', x='angle', y='Cl')
df.plot(kind='scatter', x='angle', y='Cd')
plt.show()

