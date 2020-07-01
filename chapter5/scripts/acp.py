from chapter3.plot.plot import Plot
from chapter5.data.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

x_train_std, x_test_std, y_train, y_test = Data().wine_data_sets()

"""
se toman los 2 componentes principales para la transformación de
características, no obstante lo ideal es determinar el número de componentes
mediante una compensación entre el coste computacional y el rendimiento predictivo del algoritmo.
"""
pca = PCA(n_components=2)

lr = LogisticRegression()
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

lr.fit(x_train_pca, y_train)

plt = Plot().plot_decision_regions(x_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
