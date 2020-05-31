import numpy as np


class Adaline(object):
    """
    NEURONA LINEAL ADAPTATIVA

    Parámetros
    ----------
    eta: float
    n_iter: int
    random_state: int

    Atributos
    ---------
    w_: 1d-array
    cost_: list
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # rango de aprendizaje
        self.eta = eta
        # épocas o iteraciones por el conjunto de entrenamiento
        self.n_iter = n_iter
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
        # tamaño del array de pesos en base a los datos de entrada mas el sesgo
        w_size = x_len + 1
        # generador de números aleatorios
        rgen = np.random.RandomState(self.random_state)
        # array de pesos inicializados por números aleatorios siguiendo una distribución normal
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=w_size)

        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = y - output
            # actualiza todos los pesos salvo el sesgo multiplicando la matriz de características por el vector de error
            self.w_[1:] += self.eta * x.T.dot(errors)
            # actualiza el sesgo
            self.w_[0] += self.eta * errors.sum()
            # función de coste
            cost = 0.5 * (errors**2).sum()
            # variación de la función de coste en cada época para comprobar la convergencia
            self.cost_.append(cost)

        return self

    def net_input(self, x):
        """
        Cálculo de la entrada de red (self.w_^t)*x.
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return x

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)
