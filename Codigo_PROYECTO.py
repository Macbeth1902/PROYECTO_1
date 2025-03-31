### PROYECTO 1 ###

## Importamos librerías necesarias para el proyecto
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import streamlit as st
from scipy.stats import kurtosis, skew

## Configuración de la página
st.set_page_config(page_title="📊 Análisis Financiero", layout="wide")

st.title("📈 Análisis de datos de Google")
st.write("🔍 En este proyecto, analizaremos el comportamiento de las acciones de Google desde el **01 de Enero de 2010** hasta la fecha actual.")

## Función para obtener datos del precio de cierre
@st.cache_data
def obtener_datos(stocks):
    inicio = "2010-01-01"
    fin = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download(stocks, start=inicio, end=fin)[["Close"]]
    df = df.reset_index()  ## Convertimos la fecha en columna
    return df

## Función para calcular los rendimientos diarios logarítmicos
@st.cache_data
def rendimientos_logaritmicos(df):
    df_rendimientos = df.copy()
    df_rendimientos["Returns"] = np.log(df_rendimientos["Close"] / df_rendimientos["Close"].shift(1))
    df_rendimientos = df_rendimientos.dropna().reset_index(drop=True)  # Eliminamos valores NaN
    return df_rendimientos

## Carga de datos con spinner
with st.spinner('⏳ Cargando datos...'):
    df = obtener_datos(['GOOGL'])
    df_rendimientos = rendimientos_logaritmicos(df)

## Calculamos las métricas estadísticas
media = df_rendimientos["Returns"].mean()
curtosis_valor = kurtosis(df_rendimientos["Returns"], fisher=True)
sesgo_valor = skew(df_rendimientos["Returns"])

st.subheader("📊 Estadísticas de los rendimientos de GOOGLE")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="📏 Media", value=f"{media:.5f}")

with col2:
    st.metric(label="📈 Curtosis", value=f"{curtosis_valor:.2f}")

with col3:
    st.metric(label="📉 Sesgo", value=f"{sesgo_valor:.2f}")

## Elaboramos una gráfica con los rendimientos diarios:
st.subheader("📈 Evolución de los rendimientos de Google")
fg, ax = plt.subplots(figsize=(13, 5))
ax.plot(df_rendimientos["Date"], df_rendimientos["Returns"], color="blue")
ax.axhline(y=0, color='r', linestyle='--')
ax.legend()
ax.set_title("Evolución de los rendimientos de Google")
ax.set_xlabel("Fecha")
ax.set_ylabel("Rendimiento Diario")
st.pyplot(fg)

## Elaboramos un histograma de los rendimientos diarios del activo
st.subheader("📊 Histograma de los rendimientos de Google")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_rendimientos["Returns"], bins=50, color="blue", edgecolor="black")
ax.axvline(x=media, color='r', linestyle='--', label=f"Media: {media:.5f}")
ax.legend()
ax.set_title("Histograma de los rendimientos de Google")
ax.set_xlabel("Rendimiento Diario")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)