import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from chapter3.data.data import Data
from chapter3.plot.plot import Plot

# compone un conjunto de datos de entrenamiento y otro de pruebas
x_train, x_test, y_train, y_test = Data.iris_data_sets()

# escalado de características para optimizar el rendimiento (normalización de datos de entrada visto anteriormente)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

# C - inverso del parámetro de regularización
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(x_train_std, y_train)
plt = Plot.plot_decision_regions(x_combined_std,
                                 y_combined,
                                 classifier=lr,
                                 test_idx=range(105, 150))

plt.xlabel('petal length [standarized]')
plt.ylabel('petal width [standarized]')
plt.legend(loc='upper left')
plt.show()
