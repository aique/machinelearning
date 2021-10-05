import pyprind
import pandas as pd
import os
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

basepath = '/home/aique/Projects/personal/machinelearning/data/aclImdb'
stop = stopwords.words('english')


# Las críticas se preprocesarán para ser movidos a un fichero csv
# con su contenido y categoría correspondiente (positiva o negativa)

def create_data_frame():
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()

    for i in ('test', 'train'):
        for j in ('pos', 'neg'):
            path = os.path.join(basepath, i, j)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = preprocessor(infile.read())
                df = df.append([[txt, labels[j]]], ignore_index=True)
                pbar.update()

    df.columns = ['review', 'sentiment']

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))  # barajamos el data frame
    df.to_csv('movie_data.csv', index=False, encoding='utf-8')  # volcamos los resultados a fichero
    df = pd.read_csv('movie_data.csv', encoding='utf-8')

    return df


# Esta función realiza las siguientes tareas: elimina el código HTML, obtiene todos los emoticonos,
# elimina los caracteres no alfabéticos, convierte el texto a minúsculas e inserta al final los emoticonos encontrados.

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)  #
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)  #
    text = (re.sub('[\W]+', ' ', text.lower())) + ' '.join(emoticons).replace('-', '')

    return text


def tokenizer(text):
    return text.split()


# Obtiene un array de las palabras más relevantes del texto, eliminando sus declinaciones mediante
# el algoritmo de Porter, además de aquellas que se encuentran en el diccionario de palabras irrelevantes.

def tokenizer_porter(text):
    porter = PorterStemmer()

    return [porter.stem(word) for word in text.split() if word not in stop]


df = create_data_frame()

x_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, penalty='l1', solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(x_train, y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(x_test, y_test))
