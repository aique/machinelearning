import numpy as np
from matplotlib.colors import ListedColormap
from algorithms.adaline import Adaline


class Plot(object):

    @staticmethod
    def plot_prediction(plt, x, y, classifier, resolution=0.02, normalized=False):
        # definir un generador de marcadores y un mapa de colores
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # representar la superficie de decisión
        x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        x1_arange = np.arange(x1_min, x1_max, resolution)
        x2_arange = np.arange(x2_min, x2_max, resolution)
        xx1, xx2 = np.meshgrid(x1_arange, x2_arange)
        z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
        plt.set_xlim(xx1.min(), xx1.max())
        plt.set_ylim(xx2.min(), xx2.max())

        # representar muestras de clase
        for idx, c1 in enumerate(np.unique(y)):
            plt.scatter(x=x[y == c1, 0],
                        y=x[y == c1, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=c1,
                        edgecolor='black')

        plt.set_xlabel('Logitud del sépalo [cm]' + (' normalizado' if normalized else ''))
        plt.set_ylabel('Logitud del pétalo [cm]' + (' normalizado' if normalized else ''))
        plt.set_title('Predicción')
        plt.legend(loc='upper left')

        return plt

    @staticmethod
    def plot_adaline_log_err(subplot, ada: Adaline):
        subplot.plot(range(1, len(ada.cost_) + 1),
                     np.log10(ada.cost_),
                     marker='o')
        subplot.set_xlabel('Épocas')
        subplot.set_ylabel('log(suma errores cuadráticos)')
        subplot.set_title('Rango de aprendizaje ' + str(ada.eta))

    @staticmethod
    def plot_adaline_avg_cost(subplot, ada: Adaline):
        subplot.plot(range(1, len(ada.cost_) + 1),
                     ada.cost_,
                     marker='o')
        subplot.set_xlabel('Épocas')
        subplot.set_ylabel('Media de coste')
        subplot.set_title('Rango de aprendizaje ' + str(ada.eta))

    @staticmethod
    def normalization(x):
        x_std = np.copy(x)
        x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
        x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

        return x_std
