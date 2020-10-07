import numpy as np
from chapter6.data.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# carga de datos de cáncer de mama
x_train, x_test, y_train, y_test = Data().wdbc_data_sets()

pipe_lr = make_pipeline(
    StandardScaler(),  # estandarización de características
    PCA(n_components=2),  # análisis de componentes principales
    LogisticRegression(random_state=1)  # estimación mediante regresión logística
)

scores = cross_val_score(estimator=pipe_lr,
                         X=x_train,
                         y=y_train,
                         cv=10,  # número de iteraciones en el proceso de validación cruzada
                         n_jobs=1)  # número de CPUs (tareas en paralelo) que se utilizarán

print('Cross value accuracy scores: %s' % scores)
print('Cross value accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
