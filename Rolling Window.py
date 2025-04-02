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
st.set_page_config(page_title="🔄 Rolling Window", layout="wide")

st.title("🔄 Rolling Window de Google")
st.write("🔍 ....")


