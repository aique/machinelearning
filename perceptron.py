import numpy as np


class Perceptron(object):
    """
    PERCEPTRON

    Parámetros
    ----------
    eta: float
    n_iter: int
    random_state: int

    Atributos
    ---------
    w_: 1d-array
    errors_: list
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

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                """
                xi: dtos de entrada sobre las que se genera la predicción de etiqueta
                target: etiqueta de clase verdadera
                """
                predict = self.predict(xi)
                update = self.eta * (target - predict)
                # actualiza todos los pesos salvo el sesgo teniendo en cuenta sus valores previos
                self.w_[1:] += update * xi
                # Actualiza el sesgo
                self.w_[0] += update
                # sumatorio de errores encontrados con fines estadsíticos
                errors += int(update != 0.0)
            # número de errores encontrados en cada época para comprobar la convergencia
            self.errors_.append(errors)

        return self

    def net_input(self, x):
        """
        Cálculo de la entrada de red (self.w_^t)*x.
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """
        Realiza una predicción de la etiqueta de clase.

        Si net_input(x) es mayor que 0 devolverá 1,
        en caso contrario devolverá -1.
        """
        return np.where(self.net_input(x) >= 0.0, 1, -1)
