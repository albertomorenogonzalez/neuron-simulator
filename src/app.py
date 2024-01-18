import streamlit as st
import numpy as np

st.set_page_config(page_title="Simulador de Neurona")

st.image("media/neurons.jpg")

st.title('Simulador de Neurona')

import numpy as np

# Definición del perceptrón como clase
class Neuron:
    """
    Esta clase representa un perceptrón (neurona) en una red neuronal.

    Atributos:
    - weights (list): Lista de pesos para la neurona.
    - bias (float): Término de sesgo para la neurona.
    - func (str): Función de activación a utilizar ("sigmoid", "relu" o "tanh").
    """

    def __init__(self, weights, bias, func):
        """
        Inicializa una neurona con pesos, sesgo y función de activación.

        Parámetros:
        - weights (list): Lista de pesos para la neurona.
        - bias (float): Término de sesgo para la neurona.
        - func (str): Función de activación a utilizar ("sigmoid", "relu" o "tanh").
        """
        self.weights = weights
        self.bias = bias
        self.func = func

    def __str__(self) -> str:
        """
        Devuelve una representación en cadena de la neurona.

        Devuelve:
        - str: Representación en cadena de la neurona.
        """
        return f"Perceptrón con pesos: {self.weights}, sesgo = {self.bias} y utilizando la función {self.func}"


    def sigmoid(self, x):
        """
        Calcula la función de activación sigmoidal.

        Parámetros:
        - x (float): Valor de entrada.

        Devuelve:
        - float: Resultado de la función sigmoidal.
        """
        return 1 / (1 + np.e ** -x)


    def relu(self, x):
        """
        Calcula la función de activación de la unidad lineal rectificada (ReLU).

        Parámetros:
        - x (float): Valor de entrada.

        Devuelve:
        - float: Resultado de la función ReLU.
        """
        if x < 0:
            return 0
        else:
            return x


    def tanh(self, x):
        """
        Calcula la función de activación tangente hiperbólica (tanh).

        Parámetros:
        - x (float): Valor de entrada.

        Devuelve:
        - float: Resultado de la función tanh.
        """
        return np.tanh(x)


    def predict(self, input_data):
        """
        Realiza una predicción basada en los datos de entrada.

        Parámetros:
        - input_data (list): Lista de valores de entrada.

        Devuelve:
        - float: Resultado de la predicción.
        """
        y = sum(np.multiply(input_data, self.weights)) + self.bias

        if self.func == "sigmoid":
            return self.sigmoid(y)
        elif self.func == "relu":
            return self.relu(y)
        elif self.func == "tanh":
            return self.tanh(y)


    def changeBias(self, bias):
        """
        Cambia el término de sesgo de la neurona.

        Parámetros:
        - bias (float): Nuevo valor de sesgo.
        """
        self.bias = bias


entry_num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

st.subtitle("Pesos")

for i in entry_num:
    st.number_input("w"+i, key="w"+i)

st.subtitle("Entradas")

for i in entry_num:
    st.number_input("x"+i, key="x"+i)

colBias, colActivation = st.columns(2)

with colBias:
    st.subtitle("Sesgo")
    st.number_input("Introduce el valor del sesgo")

with colActivation:
    st.subtitle("Función de Activación")
    st.select("")