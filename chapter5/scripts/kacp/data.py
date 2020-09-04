import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=100, random_state=123)

plt.scatter(x[y==0, 0], x[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(x[y==1, 0], x[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
