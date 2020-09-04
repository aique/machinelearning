import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from chapter3.plot.plot import Plot
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

x, y = make_moons(n_samples=100, random_state=123)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
x_kpca = kpca.fit_transform(x)

"""
se aplica la técnica acp y se obtiene una clasificación correcta. 
"""
pca = PCA(n_components=2)

lr = LogisticRegression()
x_pca = pca.fit_transform(x_kpca)

lr.fit(x_pca, y)

plt = Plot().plot_decision_regions(x_pca, y, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
