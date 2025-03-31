### PROYECTO 1 ###

## Importamos librerías necesarias para el proyecto
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import datetime
import streamlit as st
from scipy.stats import kurtosis, skew

## Definimos la configuración de la página
st.set_page_config(page_title="📊 Análisis Financiero", layout="wide")

st.title("📈 Análisis de datos de Google")
st.write("🔍 En este proyecto, analizaremos el comportamiento de las acciones de Google desde el **01 de Enero de 2010** hasta la fecha actual.")

## Definimos la función para obtener datos del precio de cierre
@st.cache_data
def obtener_datos(stocks):
    inicio = "2010-01-01"
    fin = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download(stocks, start=inicio, end=fin)[["Close"]]
    df = df.reset_index()  ##Convertimos la fecha en columna (más tarde se requerirá para visualizar los datos)
    return df

## Definimos la función para calcular los rendimientos logarítmicos
@st.cache_data
def calcular_rendimientos(df):
    df_rendimientos = df.copy()
    df_rendimientos["Returns"] = df_rendimientos["Close"].pct_change()
    df_rendimientos = df_rendimientos.dropna().reset_index()  ##Reseteamos el índice 
    return df_rendimientos

## Definimos una carga de datos con spinner, para una mejor visualización
with st.spinner('⏳ Cargando datos...'):
    df = obtener_datos(['GOOGL'])
    df_rendimientos = calcular_rendimientos(df)

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
