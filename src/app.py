import streamlit as st
import numpy as np
from neuron import Neuron

st.set_page_config(page_title="Simulador de Neurona")

st.image("media/neurons.jpg", use_column_width="always")

st.title('Simulador de Neurona')

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
    bias = st.number_input("Introduce el valor del sesgo")

with colActivation:
    st.subheader("Función de Activación")
    function = st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tanh')
    )


# Instancia de la neurona y cálculos
n = Neuron(weights, bias, function)
output = n.predict(entries)


if st.button("Calcular la salida"):
        st.write(f"La salida de la neurona es {output}")



st.divider()

st.write("Autor: Alberto Moreno González - ")
st.write("Máster FP en Inteligencia Artificial y Big Data del Centro Integrado en CPIFP Alan Turing (Málaga)")