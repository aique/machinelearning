from sklearn.svm import SVC
from chapter3.plot.plot import Plot
import numpy as np

# genera un conjunto de datos con límite de decisión no lineal
np.random.seed(1)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0] > 0,
                       x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# representa un modelo de decisión kernelizado que intenta ajustarse al problema no lineal
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(x_xor, y_xor)
plt = Plot.plot_decision_regions(x_xor, y_xor, classifier=svm)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='upper left')
plt.show()
