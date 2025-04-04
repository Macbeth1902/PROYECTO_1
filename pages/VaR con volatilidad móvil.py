### INCISO F ###

## Importamos librerías necesarias para el proyecto
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy.stats import norm
from utils import obtener_datos, rendimientos_logaritmicos

# Configuración de la página
st.set_page_config(page_title="📉 Cálculo de VaR con VM y DN", layout="wide")

st.title("💻 Cálculo de VaR con Volatilidad Móvil y Distribución Normal para GOOGLE")
st.write("🔍 Esta sección muestra el cálculo del VaR basado en volatilidad móvil y distribución normal.")

## Carga de datos con spinner
ticker = "GOOGL"
with st.spinner('⏳ Cargando datos...'):
    df = obtener_datos(ticker)
    df_rendimientos = rendimientos_logaritmicos(df)

# Parámetros
tamaño_ventana = 252
alphas = [0.05, 0.01]
q_alphas = [norm.ppf(alpha) for alpha in alphas]
VaR_moviles = {alpha: [] for alpha in alphas}

# Cálculo del VaR con rolling window
for i in range(tamaño_ventana, len(df_rendimientos)):
    ventana = df_rendimientos["Returns"].iloc[i - tamaño_ventana:i]  # Tomamos rendimientos, no precios
    sigma_t = ventana.std()
    for alpha, q_alpha in zip(alphas, q_alphas):
        VaR = q_alpha * sigma_t
        VaR_moviles[alpha].append(VaR)

# DataFrame con los VaR
VaR_moviles_df = pd.DataFrame(VaR_moviles, index=df_rendimientos.index[tamaño_ventana:])

# Mostrar datos en Streamlit
st.write("📊 Datos de retornos logarítmicos:", df_rendimientos.head())
st.write("📉 VaR calculado:", VaR_moviles_df.head())

# Gráfica de resultados
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_rendimientos["Date"][tamaño_ventana:], df_rendimientos["Returns"][tamaño_ventana:], label='Retornos Logarítmicos', color = '#7F7FFF')
ax.plot(df_rendimientos["Date"][tamaño_ventana:], VaR_moviles_df[0.05], label='VaR 95%', color='firebrick', linestyle='-')
ax.plot(df_rendimientos["Date"][tamaño_ventana:], VaR_moviles_df[0.01], label='VaR 99%', color='gold', linestyle='-')
ax.set_title('VaR con Volatilidad Móvil y Distribución Normal para GOOGL')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor')
ax.legend()
ax.grid(True)
st.pyplot(fig)

## Cálculo de violaciones
## Cálculo de violaciones
violaciones_data = []

for alpha in alphas:
    violaciones = (df_rendimientos["Returns"][tamaño_ventana:] < VaR_moviles_df[alpha]).sum()
    porcentaje_violaciones = (violaciones / len(VaR_moviles_df)) * 100
    violaciones_data.append([f"{(1-alpha):.0%}", f"{porcentaje_violaciones:.4f}%", violaciones])

df_violaciones = pd.DataFrame(violaciones_data, columns=["Nivel de Confianza", "Porcentaje de Violaciones", "Número de Violaciones"])

st.subheader("📌 **Tabla de Violaciones del VaR**")
st.table(df_violaciones)

st.markdown("""
## 📝 **Conclusiones**

### **Resultados obtenidos**
El gráfico muestra que el VaR con volatilidad móvil se ajusta dinámicamente a las variaciones en la volatilidad de los retornos de GOOGL. El porcentaje de violaciones para el VaR al 95% es de 4.63%, ligeramente inferior al nivel de significancia esperado del 5%. En cambio, el VaR al 99% presenta un porcentaje de violaciones de 2.09%, que es superior al nivel esperado del 1%.

### **Perfiles de inversionista**

- **VaR 95%**: Un inversionista con un nivel moderado de aversión al riesgo podría utilizar el VaR al 95% como medida de la pérdida máxima potencial en un día con un 95% de confianza. El modelo parece ser ligeramente conservador, lo que sugiere que la pérdida real podría ser menor a la predicha por el VaR.
  
- **VaR 99%**: Un inversionista con una mayor aversión al riesgo, que busca protección ante pérdidas extremas, podría optar por el VaR al 99%. Sin embargo, el modelo podría subestimar el riesgo real en escenarios extremos, lo que sugiere que la pérdida real podría ser mayor que la predicha por el VaR.

### **Algunas interpretaciones de inversión**

- **Compra**: Si el precio actual de GOOGL está significativamente por encima del VaR calculado, podría ser una señal de que el activo está sobrevalorado, lo que aumenta el riesgo de una corrección a la baja. En este caso, un inversionista podría considerar posponer una compra o incluso vender si ya posee el activo.
  
- **Venta**: Si el precio de GOOGL cae por debajo del VaR calculado, especialmente para el VaR al 99%, podría indicar que el mercado está sobre reaccionando y el activo está infravalorado. En este caso, un inversionista con un horizonte de inversión a largo plazo y mayor tolerancia al riesgo podría considerar comprar más acciones de GOOGL.

⚠️ **Nota importante**: Ambos escenarios están sujetos a incertidumbre. Las decisiones de inversión deben considerar factores económicos, estructurales y otros eventos difíciles de predecir. El VaR es solo una herramienta más en la toma de decisiones, no una garantía de rendimiento.
""")
