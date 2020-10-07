from chapter6.data.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# carga de datos de cáncer de mama
x_train, x_test, y_train, y_test = Data().wdbc_data_sets()

pipe_lr = make_pipeline(
    StandardScaler(),  # estandarización de características
    PCA(n_components=2),  # análisis de componentes principales
    LogisticRegression(random_state=1)  # estimación mediante regresión logística
)

pipe_lr.fit(x_train, y_train)

print('Simple logistic regression Accuracy: %.3f' % pipe_lr.score(x_test, y_test))
