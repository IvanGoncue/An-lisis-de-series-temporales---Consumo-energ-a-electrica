# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:48:48 2024

@author: GonCue
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\Datos\Banco mundial\Datos_España.xlsx'
# La notación r antes de la cadena de la ruta (r'C:\...') indica que la cadena 
#es una cadena de texto sin procesar, lo que significa que los caracteres de 
#barra invertida \ no se interpretarán como caracteres de escape. 
#Esto es especialmente útil en rutas de archivo en sistemas Windows.

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)
#los años están en la primera columna

# Mostrar el DataFrame
print(df_original)

################################# REPRESENTACION

# Crear un gráfico de línea para la serie 
plt.plot(df_original.index, df_original['Electricity consumption'], label='España')
# Añadir etiquetas y título
plt.xlabel('Año (1990-2022)')
plt.ylabel('Valor Agregado')
plt.title('Consumo de energía eléctrica (TWh)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()  

################################# ARIMA (1,0,1) ENTRENADO 80% DATOS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualizamos los gráficos ACF y PACF
plot_acf(df_original['Electricity consumption'])
plot_pacf(df_original['Electricity consumption'], lags=15)  # Ajusta el número de lags según tu preferencia
plt.show()

# Basándonos en los gráficos, intentaremos un modelo ARIMA(1, 0, 1) como ejemplo
order = (1, 0, 1)

# 6. División de datos
train_size = int(len(df_original) * 0.8)
train, test = df_original['Electricity consumption'][:train_size], df_original['Electricity consumption'][train_size:]

# 7. Ajuste del modelo ARIMA
model = ARIMA(train, order=order)
model_fit = model.fit()

# 8. Evaluación del modelo
predictions = model_fit.forecast(steps=len(test))
mse = ((predictions - test) ** 2).mean()#error cuadrático medio

# 9. Predicciones
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predictions')
plt.title('ARIMA (1,0,1)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

################################# ARIMA (1,0,5) ENTRENADO 80% DATOS

# Basándonos en los gráficos, intentaremos un modelo ARIMA(1, 0, 5) 
order = (1, 0, 5)

# 6. División de datos
train_size = int(len(df_original) * 0.8)
train, test = df_original['Electricity consumption'][:train_size], df_original['Electricity consumption'][train_size:]

# 7. Ajuste del modelo ARIMA
model = ARIMA(train, order=order)
model_fit = model.fit()

# 8. Evaluación del modelo
predictions = model_fit.forecast(steps=len(test))
mse = ((predictions - test) ** 2).mean()#error cuadrático medio

# 9. Predicciones
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predictions')
plt.title('ARIMA (1,0,5)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

