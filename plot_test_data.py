import pandas as pd
import matplotlib.pyplot as plt

print(plt.get_backend())

df = pd.read_csv('validation_results.csv')
df.plot(kind='scatter', x='angle', y='Cl')
df.plot(kind='scatter', x='angle', y='Cd')
plt.show()

