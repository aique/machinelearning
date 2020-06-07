import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from chapter3.data.data import Data
from chapter3.plot.plot import Plot

# compone un conjunto de datos de entrenamiento y otro de pruebas
x_train, x_test, y_train, y_test = Data.iris_data_sets()

# escalado de características para optimizar el rendimiento (normalización de datos de entrada visto anteriormente)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# entrenamiento del modelo seleccionado
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(x_train_std, y_train)

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))
plt = Plot.plot_decision_regions(x=x_combined_std,
                                 y=y_combined,
                                 classifier=ppn,
                                 test_idx=range(105, 150))

plt.xlabel('petal length [standarized]')
plt.ylabel('petal width [standarized]')
plt.legend(loc='upper left')
plt.show()
