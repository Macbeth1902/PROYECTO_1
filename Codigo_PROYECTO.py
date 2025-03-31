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
from scipy.stats import norm, t

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
    df.rename(columns={"Close": "Precio_Cierre"}, inplace=True)
    return df

## Función para calcular los rendimientos diarios logarítmicos
@st.cache_data
def rendimientos_logaritmicos(df):
    df_rendimientos = df.copy()
    df_rendimientos["Returns"] = np.log(df_rendimientos["Precio_Cierre"] / df_rendimientos["Precio_Cierre"].shift(1))
    df_rendimientos = df_rendimientos.dropna().reset_index(drop=True)  # Eliminamos valores NaN
    return df_rendimientos

## Hacemos la carga de datos, con spinner para el tiempo de espera
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
    st.metric(label="📈 Curtosis", value=f"{curtosis_valor:.5f}")

with col3:
    st.metric(label="📉 Sesgo", value=f"{sesgo_valor:.5f}")

## Elaboramos una gráfica con los rendimientos diarios:
st.subheader("📈 Evolución de los rendimientos de Google")
fg, ax = plt.subplots(figsize=(10, 5))
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

####################################
## Calculamos el VaR y ES de acuerdo a diversos métodos.

## Paramétrico
@st.cache_data
def var_es_parametrico(rendimientos, alpha, dist='normal'):
    if dist == 'normal':
        var = norm.ppf(1 - alpha, loc=rendimientos.mean(), scale=rendimientos.std())
        es = -rendimientos[rendimientos <= var].mean()
    elif dist == 't-student':
        df = len(rendimientos) - 1  # Grados de libertad
        var = t.ppf(1 - alpha, df, loc=rendimientos.mean(), scale=rendimientos.std())
        es = -rendimientos[rendimientos <= var].mean()
    else:
        raise ValueError("La distribución debe ser 'normal' o 't-student'")
    return var, es

## Método histórico
@st.cache_data
def var_es_historico(rendimientos, alpha):
    var = rendimientos.quantile(1 - alpha)
    es = -rendimientos[rendimientos <= var].mean()
    return var, es

## Montecarlo
@st.cache_data
def var_es_montecarlo(rendimientos, alpha, n_sim=10000):
    media = rendimientos.mean()
    std = rendimientos.std()
    sim_rendimientos = np.random.normal(media, std, n_sim)  # Asumiendo distribución normal (para simular solamente)
    var = np.percentile(sim_rendimientos, 100 * (1 - alpha))
    es = -sim_rendimientos[sim_rendimientos <= var].mean()
    return var, es

## Definimos nuestro vector de alphas:
alphas = [0.95, 0.975, 0.99]
resultados = []

## Utilizamos 'df_logrendimientos' (el DataFrame de rendimientos logarítmicos)
## Ya que 'Returns' es el nombre de la columna con los rendimientos logaritmicos:
rendimientos = df_rendimientos['Returns']

### Hacemos que corra sobre el vector de alphas
for alpha in alphas:
    var_normal, es_normal = var_es_parametrico(rendimientos, alpha, dist='normal')
    var_t, es_t = var_es_parametrico(rendimientos, alpha, dist='t-student')
    var_hist, es_hist = var_es_historico(rendimientos, alpha)
    var_mc, es_mc = var_es_montecarlo(rendimientos, alpha)

    resultados.append([alpha, var_normal, es_normal, var_t, es_t, var_hist, es_hist, var_mc, es_mc])

## Elaboramos un DataFrame del VaR Obtenido y el ES:

columnas = ['Alpha', 'VaR Normal', 'ES Normal', 'VaR t-Student', 'ES t-Student',
           'VaR Histórico', 'ES Histórico', 'VaR Monte Carlo', 'ES Monte Carlo']
df_resultados = pd.DataFrame(resultados, columns=columnas)

## Imprimimos los resultados
print(df_resultados)

## Mostramos los resultados en la página
st.subheader("📊 Resultados del VaR y ES")
bonito_df = df_resultados.style.format("{:.5f}").applymap(lambda x: "color: red;" if x < 0 else "")
st.dataframe(bonito_df, use_container_width=True)