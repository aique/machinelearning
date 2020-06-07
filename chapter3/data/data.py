from sklearn import datasets
from sklearn.model_selection import train_test_split


class Data(object):

    @staticmethod
    def iris_data_sets():
        # obtiene el conjunto de datos
        iris = datasets.load_iris()
        # longitud y anchura del pétalo (columnas 2 y 3)
        x = iris.data[:, [2, 3]]
        # etiquetas (tipos de flor)
        y = iris.target

        # compone un conjunto de datos de entrenamiento y otro de pruebas
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

        return x_train, x_test, y_train, y_test
