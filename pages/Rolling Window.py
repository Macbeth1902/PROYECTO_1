import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy.stats import norm, t
from utils import obtener_datos, rendimientos_logaritmicos

##Configuraos la página en Streamlit
st.set_page_config(page_title="📊 Análisis de Riesgo en los Rendimientos", layout="wide")

## Introducción a la página: 
st.markdown("""
## 📈 Sobre las Métricas de Riesgo
En esta sección, se aplicaron dos métricas de riesgo sobre los rendimientos diarios de la acción de Google: 
- **Value at Risk (VaR)** 
- **Expected Shortfall (ES)**

Asimismo, se emplearon dos enfoques distintos para el cálculo de dichas métricas:
1. **Paramétrico:** Se asume una distribución normal de los rendimientos logarítmicos.
2. **Histórico:** Se basa en los datos observados de los rendimientos logarítmicos.

Cada métrica se calculó bajo niveles de confianza del **95% y 99%**, utilizando una ventana de 252 días. Lo anterior facilitó contar con una mejor visualización 
            de la evolución del riesgo de la acción, a lo largo del periodo estudiado.
""")

## Definimos el tamaño de la ventana
tamaño_ventana = 252

## Hacemos la carga de datos, con spinner para el tiempo de espera
with st.spinner('⏳ Cargando datos...'):
    df = obtener_datos(['GOOGL'])
    df_logrendimientos = rendimientos_logaritmicos(df)

## Para evitar errores en el código, convertimos la columna de fechas a una columna por separado:
df_logrendimientos['Date'] = df_logrendimientos.index

## Obtenemos la media y la desviación estándar de los rendimientos logarítmicos
rolling_mean = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).mean()
rolling_std = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).std()

## Calculamos el VaR histórico y paramétrico al 95% y 99%

## VaR Paramétrico al 95%
VaR_95_Para = norm.ppf(1-0.95, rolling_mean, rolling_std)
VaR_Para_df_95 = pd.DataFrame({'Date': df_logrendimientos.index, '95% VaR Paramétrico': VaR_95_Para.squeeze()})
VaR_Para_df_95.set_index('Date', inplace=True)
VaR_Para_df_95 = VaR_Para_df_95.dropna()

## VaR Paramétrico al 99%
VaR_99_Para = norm.ppf(1-0.99, rolling_mean, rolling_std)
VaR_Para_df_99 = pd.DataFrame({'Date': df_logrendimientos.index, '99% VaR Paramétrico': VaR_99_Para.squeeze()})
VaR_Para_df_99.set_index('Date', inplace=True)
VaR_Para_df_99 = VaR_Para_df_99.dropna()

## VaR Histórico al 95%
VaR_95_hist = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).quantile(0.05)
vaR_hist_df_95 = pd.DataFrame({'Date': df_logrendimientos.index, '95% VaR Histórico': VaR_95_hist.squeeze()})
vaR_hist_df_95.set_index('Date', inplace=True)
vaR_hist_df_95 = vaR_hist_df_95.dropna()

## VaR Histórico al 99%
VaR_99_hist = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).quantile(0.01)
vaR_hist_df_99 = pd.DataFrame({'Date': df_logrendimientos.index, '99% VaR Histórico': VaR_99_hist.squeeze()})
vaR_hist_df_99.set_index('Date', inplace=True)
vaR_hist_df_99 = vaR_hist_df_99.dropna()

## Calculamos el ES histórico y paramétrico al 95% y 99%
## ES Histórico al 95%
ES_95_hist_df = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).apply(lambda x: x[x <= x.quantile(0.05)].mean())
ES_95_hist_df = ES_95_hist_df.dropna()
ES_95_hist_df = ES_95_hist_df.rename( "ES histórico al 95%")

## ES Histórico al 99%
ES_99_hist_df = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).apply(lambda x: x[x <= x.quantile(0.01)].mean())
ES_99_hist_df = ES_99_hist_df.dropna()
ES_99_hist_df = ES_99_hist_df.rename( "ES histórico al 99%")

## ES Paramétrico al 95%
ES_95_Para_df = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).apply(lambda x: x[x <= norm.ppf(1-0.95, np.mean(x), np.std(x))].mean())
ES_95_Para_df = ES_95_Para_df.dropna()
ES_95_Para_df = ES_95_Para_df.rename( "ES paramétrico al 95%")

## ES Paramétrico al 99%
ES_99_Para_df = df_logrendimientos['Returns'].rolling(window=tamaño_ventana).apply(lambda x: x[x <= norm.ppf(1-0.99, np.mean(x), np.std(x))].mean())
ES_99_Para_df = ES_99_Para_df.dropna()
ES_99_Para_df = ES_99_Para_df.rename( "ES paramétrico al 99%")

## Definimos opciones para la selección de métricas en nuestro gráfico:
opciones_metricas = [
    "VaR Histórico 95%",
    "VaR Histórico 99%",
    "VaR Paramétrico 95%",
    "VaR Paramétrico 99%",
    "ES Histórico 95%",
    "ES Histórico 99%",
    "ES Paramétrico 95%",
    "ES Paramétrico 99%",
    "Todas las métricas"
]

## Utilizamos un multiselect para que el usuario pueda elegir las métricas que desea visualizar
seleccion = st.multiselect("Selecciona las métricas a visualizar:", opciones_metricas, default=["Todas las métricas"])

## Si el usuario selecciona "Todas las métricas", mostramos todas
if "Todas las métricas" in seleccion:
    seleccion = opciones_metricas[:-1]

## Creamos la gráfica
st.subheader("📈 Rendimientos logarítmicos diarios con métricas de riesgo")
plt.figure(figsize=(14, 7))

## Graficamos los rendimientos logarítmicos
plt.plot(df_logrendimientos.index, df_logrendimientos['Returns'] * 100, label='Rendimientos Logarítmicos Diarios (%)', color='#7F7FFF', alpha=0.8)

## Graficamos solo las métricas seleccionadas
if "VaR Histórico 95%" in seleccion:
    plt.plot(vaR_hist_df_95.index, vaR_hist_df_95['95% VaR Histórico'] * 100, label='VaR Histórico 95%', color='darkblue')

if "VaR Histórico 99%" in seleccion:
    plt.plot(vaR_hist_df_99.index, vaR_hist_df_99['99% VaR Histórico'] * 100, label='VaR Histórico 99%', color='darkorange')

if "VaR Paramétrico 95%" in seleccion:
    plt.plot(VaR_Para_df_95.index, VaR_Para_df_95['95% VaR Paramétrico'] * 100, label='VaR Paramétrico 95%', color='mediumseagreen')

if "VaR Paramétrico 99%" in seleccion:
    plt.plot(VaR_Para_df_99.index, VaR_Para_df_99['99% VaR Paramétrico'] * 100, label='VaR Paramétrico 99%', color='firebrick')

if "ES Histórico 95%" in seleccion:
    plt.plot(ES_95_hist_df.index, ES_95_hist_df * 100, label='ES Histórico 95%', color='darkviolet')

if "ES Histórico 99%" in seleccion:
    plt.plot(ES_99_hist_df.index, ES_99_hist_df * 100, label='ES Histórico 99%', color='gold')

if "ES Paramétrico 95%" in seleccion:
    plt.plot(ES_95_Para_df.index, ES_95_Para_df * 100, label='ES Paramétrico 95%', color='dodgerblue')

if "ES Paramétrico 99%" in seleccion:
    plt.plot(ES_99_Para_df.index, ES_99_Para_df * 100, label='ES Paramétrico 99%', color='tan')


plt.title('Rendimientos Logarítmicos Diarios con Métricas de Riesgo')
plt.xlabel('Fecha')
plt.ylabel('Valor (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()

## Mostramos la gráfica en Streamlit
st.pyplot(plt)

st.markdown("""
De manera general, se observa que el VaR siempre se encuentra por debajo del ES, lo cual es esperado, ya que el ES promedia las pérdidas que superan el VaR.
""")


## Para mayor claridad, mostramos las métricas de riesgo en un dataframe. Esta continuación es el ejercicio e)
st.subheader("📊Número de Violaciones del VaR o del ES registradas")
st.markdown("""
### 📌 **Evaluación de la Precisión del VaR y ES**
Para evaluar la eficiencia de nuestras estimaciones de **VaR** y **ES**, analizamos el número y porcentaje de violaciones de nuestras medidas de riesgo sobre los rendimientos logarítmicos reales de **Google**.

✅ **Regla general**: Se considera que una buena estimación de riesgo debe generar un porcentaje de violaciones inferior pero cercano al **2.5%**.  
📋 A continuación, se presenta una tabla con los resultados obtenidos:
""", unsafe_allow_html=True)

## Definimos el dataframe con las medidas de riesgo:
df_Medidas_Riesgo = pd.concat([df_logrendimientos['Returns'].iloc[tamaño_ventana-1:], vaR_hist_df_95, vaR_hist_df_99, VaR_Para_df_95, VaR_Para_df_99, ES_95_hist_df, ES_95_Para_df, ES_99_hist_df, ES_99_Para_df], axis=1)

violaciones_VaR_hist_95 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['95% VaR Histórico']).sum()
porcentaje_violaciones_VaR_hist_95 = (violaciones_VaR_hist_95 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el VaR histórico al 95% es de: " , porcentaje_violaciones_VaR_hist_95.round(4), "%" )

violaciones_VaR_hist_99 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['99% VaR Histórico']).sum()
porcentaje_violaciones_VaR_hist_99 = (violaciones_VaR_hist_99 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el VaR histórico al 99% es de: " , porcentaje_violaciones_VaR_hist_99.round(4), "%" )

violaciones_VaR_Para_95 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['95% VaR Paramétrico']).sum()
porcentaje_violaciones_VaR_Para_95 = (violaciones_VaR_Para_95 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el VaR paramétrico al 95% es de: " , porcentaje_violaciones_VaR_Para_95.round(4), "%" )

violaciones_VaR_Para_99 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['99% VaR Paramétrico']).sum()
porcentaje_violaciones_VaR_Para_99 = (violaciones_VaR_Para_99 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el VaR paramétrico al 99% es de: " , porcentaje_violaciones_VaR_Para_99.round(4), "%" )

violaciones_ES_hist_95 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['ES histórico al 95%']).sum()
porcentaje_violaciones_ES_hist_95 = (violaciones_ES_hist_95 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el ES histórico al 95% es de: " , porcentaje_violaciones_ES_hist_95.round(4), "%" )

violaciones_ES_hist_99 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['ES histórico al 99%']).sum()
porcentaje_violaciones_ES_hist_99 = (violaciones_ES_hist_99 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el ES histórico al 99% es de: " , porcentaje_violaciones_ES_hist_99.round(4), "%" )

violaciones_ES_Para_95 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['ES paramétrico al 95%']).sum()
porcentaje_violaciones_ES_Para_95 = (violaciones_ES_Para_95 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el ES paramétrico al 95% es de: " , porcentaje_violaciones_ES_Para_95.round(4), "%" )

violaciones_ES_Para_99 = (df_Medidas_Riesgo['Returns'] < df_Medidas_Riesgo['ES paramétrico al 99%']).sum()
porcentaje_violaciones_ES_Para_99 = (violaciones_ES_Para_99 / len(df_Medidas_Riesgo)) * 100
#print("El porcentaje de violaciones para el ES paramétrico al 99% es de: " , porcentaje_violaciones_ES_Para_99.round(4), "%" )

df_porcentaje_y_numero_de_violaciones = (pd.DataFrame({'VaR Histórico al 95%': [porcentaje_violaciones_VaR_hist_95, violaciones_VaR_hist_95],
                                                      'VaR Histórico al 99%': [porcentaje_violaciones_VaR_hist_99, violaciones_VaR_hist_99],
                                                      'VaR Paramétrico al 95%': [porcentaje_violaciones_VaR_Para_95, violaciones_VaR_Para_95],
                                                      'VaR Paramétrico al 99%': [porcentaje_violaciones_VaR_Para_99, violaciones_VaR_Para_99],
                                                      'ES Histórico al 95%': [porcentaje_violaciones_ES_hist_95, violaciones_ES_hist_95],
                                                      'ES Histórico al 99%': [porcentaje_violaciones_ES_hist_99, violaciones_ES_hist_99],
                                                      'ES Paramétrico al 95%': [porcentaje_violaciones_ES_Para_95, violaciones_ES_Para_99],
                                                      'ES Paramétrico al 99%': [porcentaje_violaciones_ES_Para_99, violaciones_ES_Para_99]}, index = ["Porcentaje de Violaciones", "Número de Violaciones"])).T
df_porcentaje_y_numero_de_violaciones["Porcentaje de Violaciones"] = df_porcentaje_y_numero_de_violaciones["Porcentaje de Violaciones"].astype(float)
df_porcentaje_y_numero_de_violaciones['Número de Violaciones'] = df_porcentaje_y_numero_de_violaciones['Número de Violaciones'].astype(int)

## Mostramos nuestro DataFrame con mejor formato
st.dataframe(df_porcentaje_y_numero_de_violaciones.style.format({
    "Porcentaje de Violaciones": "{:.2f}%",  # Aquí agregamos el símbolo '%'
    "Número de Violaciones": "{:,}"  # Separador de miles
}).set_properties(**{
    'text-align': 'center',
    'font-size': '14px'
}).set_table_styles([{
    'selector': 'thead th',
    'props': [('background-color', '#004c99'), ('color', 'white'), ('font-weight', 'bold')]
}, {
    'selector': 'tbody td',
    'props': [('border', '1px solid #ddd'), ('padding', '8px')]
}]))

st.markdown("""
### 📊 **Conclusiones finales**  

✅ **VaR 95%**: Tanto el **Histórico** como el **Paramétrico** presentan un **alto porcentaje de violaciones**, lo cual indica que se subestima el riesgo.  
✅ **VaR 99%**: Más confiable, pero se sigue sobrepasando el **2.5%**. 

El **Expected Shortfall (ES)** muestra estimaciones más precisas, en comparación con el VaR:  
- El **ES 95%** (**Paramétrico e Histórico**) ofrecen mejores resultados, con valores más cercanos al **2.5%**.  
- El **ES 99%** (**Paramétrico e Histórico**) se tiene una tendencia a sobreestimar el riesgo.

Se recomienda, para este caso particular, el **ES Paramétrico al 95%** como la mejor herramienta para evaluar el riesgo de la acción de Google, pues
            su porcentaje de violaciones es el más cercano al **2.5%**. Si, por otro lado, lo que se busca es una estimación más conservadora, el **ES Paramétrico al 99%** es la mejor opción.
""")
