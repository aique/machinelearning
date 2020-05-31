"""
Coste del algoritmo Adaline después de la normalización de datos de entrada.

La convergencia es mucho mayor que en el ejemplo previo a la normalización,
a pesar de que el rango de aprendizaje es 0.01.
"""

import matplotlib.pyplot as plt
from algorithms.adaline import Adaline
from data.data import Data
from plot.plot import Plot

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

x, y = Data.read_data()
x = Plot.normalization(x)

ada = Adaline(n_iter=15, eta=0.01).fit(x, y)

Plot.plot_prediction(plt=ax[0], x=x, y=y, classifier=ada, normalized=True)
Plot.plot_adaline_log_err(ax[1], ada)

plt.show()
