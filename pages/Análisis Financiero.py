import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import streamlit as st
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.stats import norm, t
from utils import obtener_datos, rendimientos_logaritmicos

## Configuración de la página
st.set_page_config(page_title="📊 Análisis Financiero", layout="wide")

st.title("📈 Análisis de datos de Google")
st.markdown("""
    <div style="font-size: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333333; line-height: 1.6; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
        <p><strong>📊 Comportamiento histórico de la acción de Google (GOOGL)</strong></p>
        <p>En esta sección se presentará el comportamiento histórico de la acción de Google (GOOGL), con el objetivo de evaluar sus rendimientos y riesgos asociados. 
        Para ello, se utilizará como fuente de datos la plataforma <strong style="color: #0073e6;">Yahoo Finance</strong>, de la cual se extraerá la información correspondiente 
        al precio de cierre diario de la acción desde el <strong>1 de enero de 2010</strong> hasta la fecha actual.</p>
    </div>
""", unsafe_allow_html=True)


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

st.markdown("""
Para el cálculo de los rendimientos diarios, se optó por calcular los **Rendimientos Logarítmicos**. 
            La elección se debe a que, a comparación de los rendimientos lineales, los rendimientos logarítmicos suelen estar más cerca de 
            adaptarse a una distribución normal. Lo anterior facilita aplicar modelos estadísticos que requieren dicha distribución, como el 
            cálculo del VaR paramétrico. 
""", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_rendimientos["Date"], df_rendimientos["Returns"], color="#7F7FFF", width=14, edgecolor='none')
ax.axhline(y=0, color='#ea314e', linestyle='--', linewidth=1.5)
ax.set_title("Evolución de los rendimientos de Google")
ax.set_xlabel("Fecha")
ax.set_ylabel("Rendimiento Diario")
st.pyplot(fig)



## Elaboramos un histograma de los rendimientos diarios del activo
st.subheader("📊 Histograma de los rendimientos de Google")
fig, ax = plt.subplots(figsize=(10, 5))

# Usamos seaborn para hacer el histograma con kde (estimación de densidad)
sns.histplot(df_rendimientos["Returns"], bins=50, kde=True, color="blue", edgecolor="black", ax=ax)

ax.axvline(x=media, color='#f84848', linestyle='--', label=f"Media: {media:.5f}")
ax.legend()
ax.set_title("Histograma de los rendimientos de Google")
ax.set_xlabel("Rendimiento Diario")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

st.markdown("""
    <div style="font-size: 20px; font-family: Arial, sans-serif; color: #333333; line-height: 1.6; background-color: #f0f0f0; padding: 20px; border-radius: 8px;">
        <p><strong>Notemos que el activo presenta una gran volatilidad:</strong> existen periodos donde los rendimientos son estables, y otros donde hay grandes fluctuaciones. Por ejemplo,
        entre 2020 y 2022 se aprecia un comportamiento extremo, lo que probablemente esté asociado con el impacto de eventos globales como la pandemia del COVID-19.</p>
    </div>
""", unsafe_allow_html=True)
