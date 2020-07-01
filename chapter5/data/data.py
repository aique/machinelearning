import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Data(object):

    @staticmethod
    def wine_data_sets():
        # obtiene el conjunto de datos
        df_wine = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../data/wine.data', header=None)
        x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
        # estandarización de las características
        sc = StandardScaler()
        x_train_std = sc.fit_transform(x_train)
        x_test_std = sc.transform(x_test)

        return x_train_std, x_test_std, y_train, y_test
