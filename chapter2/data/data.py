import os
import numpy as np
import pandas as pd


class Data(object):

    @staticmethod
    def read_data():
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../data/iris.data', header=None)

        # obtener las etiquetas verdaderas de clase
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)

        # extraer la longitud de sépalo y pétalo
        x = df.iloc[0:100, [0, 2]].values

        return x, y
