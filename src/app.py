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

st.subheader("Pesos")
weights = []

weight_columns = st.columns(entry_num)


for i in range(0, entry_num):

    with weight_columns[i]:
        st.markdown(f"Peso w<sub>{i}</sub>", unsafe_allow_html=True)
        weights.append(st.number_input(f"w{i}", key=f"w{i}", label_visibility="collapsed"))

st.write(f"w = {weights}")


st.subheader("Entradas")
entries = []

entry_columns = st.columns(entry_num)

for i in range(0, entry_num):

    with entry_columns[i]:
        st.markdown(f"Entrada x<sub>{i}</sub>", unsafe_allow_html=True)
        entries.append(st.number_input(f"x{i}", key=f"x{i}", label_visibility="collapsed"))

st.write(f"x = {entries}")


colBias, colActivation = st.columns(2)


with colBias:
    st.subheader("Sesgo")
    st.number_input("Introduce el valor del sesgo")

with colActivation:
    st.subheader("Función de Activación")
    st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tanh')
    )


if st.button("Calcular la salida"):
        st.write("La salida de la neurona es ")



st.divider()

st.write("Autor: Alberto Moreno González - ")
st.write("Máster FP en Inteligencia Artificial y Big Data del Centro Integrado en CPIFP Alan Turing (Málaga)")