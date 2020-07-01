"""
Obtiene los datos de entrada clasificados e imprime una reducida muestra.
"""
import os
import pandas as pd

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../../data/iris.data', header=None)
print(df.tail())
