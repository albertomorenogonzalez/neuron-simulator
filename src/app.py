# Library imports
import streamlit as st
import numpy as np

# Neuron class import
# The class is located in another file in this directory
# The name of the file is neuron.py
from neuron import Neuron

# We set the name of the page in the top of the tap
st.set_page_config(page_title="Simulador de Neurona", layout="wide")
# st.set_page_config(layout="wide")

# We set the style of the page, first in a variable
# and then applied with an st.markdown.

# style = f"""
# <style>
#    .appview-container .main .block-container{{
#        max-width: 90%
#    }}
# </style>
# """

 # st.markdown(style, unsafe_allow_html=True)

# We set the image and title
st.image("media/neurons.jpg", use_column_width="always")

st.title('Simulador de Neurona')

# We set the slider to show which number of entries and weights we want
entry_num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

# Setting the weights depending of the number we set before
st.subheader("Pesos")
weights = []

weight_columns = st.columns(entry_num)

for i in range(0, entry_num):

    with weight_columns[i]:
        st.markdown(f"Peso w<sub>{i}</sub>", unsafe_allow_html=True)
        weights.append(st.number_input(f"w{i}", key=f"w{i}", label_visibility="collapsed"))

st.write(f"w = {weights}")


# Setting the entries depending of the number we set before
st.subheader("Entradas")
entries = []

entry_columns = st.columns(entry_num)

for i in range(0, entry_num):

    with entry_columns[i]:
        st.markdown(f"Entrada x<sub>{i}</sub>", unsafe_allow_html=True)
        entries.append(st.number_input(f"x{i}", key=f"x{i}", label_visibility="collapsed"))

st.write(f"x = {entries}")


# We set the bias and activation function input
colBias, colActivation = st.columns(2)

with colBias:
    st.subheader("Sesgo")
    bias = st.number_input("Introduce el valor del sesgo")

with colActivation:
    st.subheader("Función de Activación")
    function = st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tanh')
    )


# Neuron instance and operations
n = Neuron(weights, bias, function)
output = n.predict(entries)


# We set the button to run and show the results
if st.button("Calcular la salida"):
        st.write(f"La salida de la neurona es {output}")



st.divider()

st.write("Autor: Alberto Moreno González - ")
st.write("Máster FP en Inteligencia Artificial y Big Data del Centro Integrado en CPIFP Alan Turing (Málaga)")