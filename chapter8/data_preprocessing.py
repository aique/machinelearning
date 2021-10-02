import pyprind
import pandas as pd
import os
import numpy as np

# Las críticas se preprocesarán para ser movidos a un fichero csv
# con su contenido y categoría correspondiente (positiva o negativa)

basepath = '/home/aique/Projects/personal/machinelearning/data/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

for i in ('test', 'train'):
    for j in ('pos', 'neg'):
        path = os.path.join(basepath, i, j)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[j]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))  # barajamos el data frame
df.to_csv('movie_data.csv', index=False, encoding='utf-8')  # volcamos los resultados a fichero

# verificamos el resultado mostrando los 3 primeros registros
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))
