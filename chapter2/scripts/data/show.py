"""
Obtiene los datos de entrada clasificados y los muestra en una gráfica.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../../data/iris.data', header=None)

# extraer la longitud de sépalo y pétalo
x = df.iloc[0:100, [0, 2]].values

# representar los datos
plt.scatter(x[:50, 0], x[:50, 1],
            color='red',
            marker='o',
            label='setosa')

plt.scatter(x[50:100, 0], x[50:100, 1],
            color='blue',
            marker='x',
            label='versicolor')

plt.xlabel('Longitud del sépalo [cm]')
plt.ylabel('Longitud del pétalo [cm]')
plt.legend(loc='upper left')
plt.show()
