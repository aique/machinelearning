from chapter3.algorithms.logistic_regression import LogisticRegression
from chapter3.data.data import Data
from chapter3.plot.plot import Plot

# compone un conjunto de datos de entrenamiento y otro de pruebas
x_train, x_test, y_train, y_test = Data.iris_data_sets()

x_train_01_subset = x_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lr = LogisticRegression(eta=0.5, n_iter=1000, random_state=1)
lr.fit(x_train_01_subset, y_train_01_subset)
plt = Plot.plot_decision_regions(x=x_train_01_subset,
                                 y=y_train_01_subset,
                                 classifier=lr)

plt.xlabel('petal length [standarized]')
plt.ylabel('petal width [standarized]')
plt.legend(loc='upper left')
plt.show()
