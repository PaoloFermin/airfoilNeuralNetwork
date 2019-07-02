import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results.csv')
df.plot(kind='scatter', x='U', y='Cl')
plt.show()

