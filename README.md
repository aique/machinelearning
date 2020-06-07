# Aprendizaje computacional

En este repositorio se recogen ejemplos prácticos de aprendizaje computacional realizados
siguiendo la lectura del libro `Python Machine Learning` de Sebastian Raschka y Vahid Mirjalili.

## Tipos de aprendizaje automático

### Aprendizaje supervisado

- **Objetivo** - Aprender un modelo a partir de datos de entrenamiento etiquetados, que permite hacer predicciones de
futuro.

- **Ejemplos** - Filtro de correo no deseado. Este es un ejemplo de **tarea de clasificación** con asignación
de etiquetas. Otro ejemplo podría ser de este tipo de aprendizaje la **regresión**, donde el valor predicho puede ser un
valor continuo.

### Aprendizaje reforzado

- **Objetivo** - Desarrollar un agente que mejore su rendimiento basado en interacciones con el entorno. En este caso,
durante el periodo de aprendizaje los resultados no se contrastan con la etiqueta o el valor correcto de los datos de
entrada, sino con una **función de recompensa**.

- **Ejemplos** - Motor de ajedrez, donde el agente elige entre una serie de movimientos y la recompensa se puede definir
como la victoria o la derrota al final del juego.

### Aprendizaje no supervisado

- **Objetivo** - Explorar la estructura de un conjunto de datos de entrada sin etiquetar para extraer información
significativa de ellos.

- **Ejemplos** - Descubrir grupos de clientes basados en sus intereses con el fin de desarrollar estrategias de
marketing.

## Algoritmos simples de aprendizaje automático

### Algunos algoritmos básicos

#### Perceptrón

Algoritmo capaz de optimizar un conjunto de **coeficientes de peso** en base a los cuales realizar predicciones.

Durante la fase de entrenamiento se utilizan un conjunto de datos de entrada etiquetados. En una serie de iteracioneso o 
**épocas** realizadas sobre ellos, llamadas **entrenamiento**, el algoritmo es capaz de optimizar los pesos.

Esta optimización se realiza obteniendo el producto de los pesos con los valores de entrada, el cual se contrasta 
con una función escalón en base a la etiqueta de los valores de entrada.

La **convergencia** sólo está garantizada si las clases son linealmente separables y el rango de aprendizaje es 
suficientemente pequeño.

#### Neuronas lineales adaptativas (Adaline)

La diferencia con respecto al perceptrón reside en que los pesos se actualizan en base a una **función de activación 
lineal** (o **función objetivo** o **función de coste**), en lugar de hacerlo en base a una función escalón unitario. 
Minificar el coste de esta función lineal permitirá desarrollar algoritmos más avanzados.

Algunas de las estrategias para disminuir este coste son:

- **Optimizar el rango de aprendizaje** - Experimentación con distintos rangos de aprendizaje en contraste con la
convergencia y el número de errores obtenido.

- **Descenso de gradiente** - Algoritmo de aproximación mediante sucesivas iteraciones.

- **Descenso de gradiente estocástico** - Similar al descenso de gradiente, sin embargo la actualización de pesos se
realiza de forma incremental para cada muestra de entrenamiento.

- **Escalado de características** - Tratamiento de los datos de entrada que favorece la convergencia.

#### Regresión logística

A pesar de su nombre, se trata de un modelo para clasificación, similar a los anteriores modelos.

Es muy similar al modelo Adaline, sin embargo en lugar de utilizar una función de activación lineal para la optimización
de pesos, se utiliza una función sigmoide, que se interpreta como la probabilidad de que una muestra pertenezca a una
etiqueta determinada.

Una de las ventajas de este modelo es que no sólo es capaz de **predecir la etiqueta** de la muestra, **también la
probabilidad** de que ésto ocurra. Por este motivo es un módelo popular para predicciones metereológicas o cálculo de
probabilidades de que un paciente padezca una enfermedad concreta en función a determinados síntomas.

### Sobreajuste y regularización

El sobreajuste es un problema común en el que un algoritmo funciona bien durante el entrenamiento pero no generaliza 
correctamente con los datos de prueba. En este caso se dice que el modelo tiene una **alta varianza**, causada 
generalmente por un modelo demasiado complejo.

De forma parecida, un modelo puede no ser lo suficientemente complejo para capturar correctamente el patrón en los datos
de entrenamiento, provocando una situación de **subajuste o underfitting**.

Estos problemas se dan en modelos de decisión no lineales, y una estrategia de compensación es la denominada
**regularización**.