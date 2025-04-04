"""
En este código se almacenan las funciones que se utilizan en el proyecto constantemente. 
Esto, con la finalidad de evitar repeticiones en cada página del código.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

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
def rendimientos_logaritmicos(df):
    df_rendimientos = df.copy()
    df_rendimientos["Returns"] = np.log(df_rendimientos["Precio_Cierre"] / df_rendimientos["Precio_Cierre"].shift(1))
    df_rendimientos = df_rendimientos.dropna().reset_index(drop=True)  ## Eliminamos valores NaN
    return df_rendimientos