from chapter3.plot.plot import Plot
from chapter5.data.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_train_std, x_test_std, y_train, y_test = Data().wine_data_sets()

lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train_std, y_train)

lr = LogisticRegression()
lr.fit(x_train_lda, y_train)

plt = Plot().plot_decision_regions(x_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
