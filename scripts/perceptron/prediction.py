"""
Muestra el modelo de predicción deducido por el perceptrón
"""
import matplotlib.pyplot as plt
from perceptron import Perceptron
from plot import Plot
from data.data import Data

x, y = Data().read_data()
ppn = Perceptron(eta=0.1, n_iter=10).fit(x, y)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

Plot.plot_prediction(ax, x, y, classifier=ppn)
plt.show()
