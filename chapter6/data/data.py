import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data(object):

    @staticmethod
    def read_data():
        le = LabelEncoder()

        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../data/wdbc.data', header=None)

        x = df.loc[:, 2:].values
        y = df.loc[:, 1].values
        y = le.fit_transform(y)
        le.transform(['M', 'B'])

        return x, y

    @staticmethod
    def wdbc_data_sets():
        x, y = Data.read_data()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=1
        )

        return x_train, x_test, y_train, y_test
