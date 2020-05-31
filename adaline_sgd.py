import numpy as np


class AdalineSGD(object):
    """
    NEURONA LINEAL ADAPTATIVA (DESCENSO DE GRADIENTE ESTOCÁSTICO)

    En lugar de actualizar los pesos por lotes, lo hará de forma incremental
    para cada muestra de entrenamiento.

    Esto permite converger más rápidamente, reducir el coste computacional y
    optimizar el modelo para el aprendizaje online, en el que el modelo se
    entrena según van llegando los datos.

    Los datos pueden ir acumulándose poco a poco y no es necesario almacenar
    aquellos que ya se han utilizado para el entrenamiento.

    Parámetros
    ----------
    eta: float
    n_iter: int
    shuffle: bool (default: True)
    random_state: int

    Atributos
    ---------
    w_: 1d-array
    cost_: list
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        # rango de aprendizaje
        self.eta = eta
        # épocas o iteraciones por el conjunto de entrenamiento
        self.n_iter = n_iter
        # previene la formación de ciclos
        self.shuffle = shuffle
        # semilla para la generación aleatoria de pesos
        self.random_state = random_state

    def fit(self, x, y):
        """
        Inicializa los pesos en self._w iterando las épocas establecidas
        sobre el conjunto de entrenamiento recibido como primer parámetro,
        teniendo en cuenta las etiquetas de clase verdadera recibidas como
        segundo.
        """
        # tamaño de los datos de entrada
        x_len = x.shape[1]
        self.initialize_weights(x_len)
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self.do_shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def initialize_weights(self, m):
        # generador de números aleatorios
        self.rgen = np.random.RandomState(self.random_state)
        # array de pesos inicializados por números aleatorios siguiendo una distribución normal
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def do_shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]

    def update_weights(self, xi, target):
        """
        Aplica la regla de aprendizaje y actualiza los pesos.
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2

        return cost

    def net_input(self, x):
        """
        Cálculo de la entrada de red (self.w_^t)*x.
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def partial_fit(self, x, y):
        """
        Aplica la regla de aprendizaje sin actualizar los pesos.
        """
        if not self.w_initialized:
            self.initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(x, y)

        return self

    def activation(self, x):
        return x

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)
