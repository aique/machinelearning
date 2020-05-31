"""
Coste del algoritmo Adaline antes de la normalización de datos de entrada.

Se puede comprobar que el rendimiento para un rango de aprendizaje de 0.01
no sólo es peor que el del positrón, sino que el número de errores crece
en cada época.

Es necesario un rango de aprendizaje de 0.0001 para obtener valores aceptables.
"""

import matplotlib.pyplot as plt
from algorithms.adaline import Adaline
from data.data import Data
from plot.plot import Plot

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

x, y = Data.read_data()

ada1 = Adaline(n_iter=10, eta=0.01).fit(x, y)
ada2 = Adaline(n_iter=10, eta=0.0001).fit(x, y)

Plot.plot_adaline_log_err(ax[0], ada1)
Plot.plot_adaline_log_err(ax[1], ada2)

plt.show()
