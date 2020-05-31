"""
Muestra los errores del perceptrón durante su entrenamiento.
"""
import matplotlib.pyplot as plt
from perceptron import Perceptron
from data.data import Data

x, y = Data().read_data()
ppn = Perceptron(0.1, 10)
ppn.fit(x, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de actualizaciones')
plt.show()
