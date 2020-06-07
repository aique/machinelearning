"""
Coste del algoritmo Adaline con gradiente estocástico
"""

import matplotlib.pyplot as plt
from chapter2.algorithms.adaline_sgd import AdalineSGD
from chapter2.data.data import Data
from chapter2.plot.plot import Plot

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

x, y = Data.read_data()
Plot.normalization(x)

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(x, y)

Plot.plot_prediction(plt=ax[0], x=x, y=y, classifier=ada, normalized=True)
Plot.plot_adaline_avg_cost(ax[1], ada)

plt.show()
