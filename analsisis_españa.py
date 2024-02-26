# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:48:48 2024

@author: GonCue
"""
################################### IMPORTAR LIBRERIAS

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy import stats

#################################### LEER ARCHIVO

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

#################################### ACF Y PACF

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualizamos los gráficos ACF y PACF
plot_acf(df_original['Electricity consumption'])
#Converge a cero pero se hace negativo despues -- posible estacionalidad
plot_pacf(df_original['Electricity consumption'], lags=15)  # Ajusta el número de lags según tu preferencia
plt.show()
#Fuerte componente AR

########################## PRUEBA DICKY FULLER ESTACIONARIEDAD

from statsmodels.tsa.stattools import adfuller

result = adfuller(df_original['Electricity consumption'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
#En la prueba de Dickey-Fuller, la hipótesis nula es que la serie temporal 
#tiene una raíz unitaria, lo que implica que no es estacionaria

#p-valor es 0.243, que es mayor que 0.05. Por lo tanto, no hay evidencia 
#suficiente para rechazar la hipótesis nula. Esto sugiere que la serie 
#podría tener una raíz unitaria y no ser estacionaria.

######################### TOMAR DIFERENCIAS PARA ESTACIONARIZAR

# Crear una figura y ejes con subgráficos
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

# Gráfico 1: Serie Temporal Original
axes[0, 0].plot(df_original.index, df_original['Electricity consumption'], label='Original', color='blue')
axes[0, 0].set_xlabel('Año (1990-2022)')
axes[0, 0].set_ylabel('Valor Agregado', color='blue')
axes[0, 0].set_title('Serie Temporal Original')
axes[0, 0].legend(loc='upper left')
axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# Gráfico 2: ACF para Serie Temporal Original
plot_acf(df_original['Electricity consumption'], ax=axes[0, 1], lags=15)
axes[0, 1].set_title('ACF - Serie Temporal Original')

# Gráfico 3: Primera Diferencia
df_diff1 = df_original['Electricity consumption'].diff().dropna()
axes[1, 0].plot(df_original.index[1:], df_diff1, label='Primera Diferencia', color='orange')
axes[1, 0].set_xlabel('Año (1990-2022)')
axes[1, 0].set_ylabel('Primera Diferencia', color='orange')
axes[1, 0].set_title('Primera Diferencia')
axes[1, 0].legend(loc='upper right')
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

#Cuando calculas la primera diferencia con df_original['Electricity consumption'].diff(), 
#la operación introduce un valor NaN en la primera posición de la serie resultante. 
#Este NaN proviene de la diferencia entre el primer valor y un valor que no existe antes de él.
#Para evitar este NaN en el primer valor de la serie de la primera diferencia, 
#usamos df_original.index[1:] al representar el gráfico.

# Aplicar diferenciación de primer orden
result_diff1 = adfuller(df_diff1)
print('ADF Statistic diferencia de orden 1:', result_diff1[0])
print('p-value:', result_diff1[1])

#p-valor es 0.382, que es mayor que 0.05. Por lo tanto, no hay 
#evidencia suficiente para rechazar la hipótesis nula, sugiriendo 
#que la serie diferenciada podría aún tener una raíz unitaria y 
#no ser estacionaria.

# Gráfico 4: ACF para Primera Diferencia
plot_acf(df_diff1, ax=axes[1, 1], lags=15)
axes[1, 1].set_title('ACF - Primera Diferencia')

# Gráfico 5: Segunda Diferencia
df_diff2 = df_diff1.diff().dropna()
axes[2, 0].plot(df_original.index[2:], df_diff2, label='Segunda Diferencia', color='green')
axes[2, 0].set_xlabel('Año (1990-2022)')
axes[2, 0].set_ylabel('Segunda Diferencia', color='green')
axes[2, 0].set_title('Segunda Diferencia')
axes[2, 0].legend(loc='upper right')
axes[2, 0].grid(True, linestyle='--', alpha=0.7)

# Prueba de Dickey-Fuller para la serie diferenciada de segundo orden
result_diff_2nd = adfuller(df_diff2)
print('ADF Statistic diferencia de orden 2:', result_diff_2nd[0])
print('p-value:', result_diff_2nd[1])

#el p-valor es menor que 0.05, lo cual sugiere que hay evidencia 
#suficiente para rechazar la hipótesis nula de que la serie tiene 
#una raíz unitaria. Por lo tanto, puedes considerar que la serie 
#después de la diferenciación de segundo orden es estacionaria.


# Gráfico 6: ACF para Segunda Diferencia
plot_acf(df_diff2, ax=axes[2, 1], lags=15)
axes[2, 1].set_title('ACF - Segunda Diferencia')

# Ajustar el diseño
plt.tight_layout()

# Mostrar el gráfico
plt.show()



###################################### CALIBRACIÓN###########################################



################################### CALIBRACIÓN MINIMIZANDO MSE
# División de datos
train_size = int(len(df_original) * 0.8)
train, test = df_original['Electricity consumption'][:train_size], df_original['Electricity consumption'][train_size:]

# Ejemplo de ajuste con diferentes parámetros
p_values = range(0, 4)
d_values = range(0, 4)
q_values = range(0, 4)

best_mse = float('inf')
best_params = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test, predictions)

            if mse < best_mse:
                best_mse = mse
                best_params = order

# Ajuste final con los mejores parámetros encontrados
final_model = ARIMA(train, order=best_params)
final_model_fit = final_model.fit()
final_predictions = final_model_fit.forecast(steps=len(test))

# Representar el modelo completo
plt.plot(df_original.index[:train_size], train[:], label='Datos 1990-2015 (train)')
plt.plot(df_original.index[1:train_size], final_model_fit.fittedvalues[1:], label='Modelo entrenado')
plt.plot(df_original.index[train_size:], test, label='Datos 2015-2022')
plt.plot(test.index, final_predictions, label='Predicciones')
plt.title('Calibracion MSE "Minimización de error cuadrático medio" ARIMA (3,0,0)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Verificación de residuos
residuals = final_model_fit.resid

#creamos subplots
plt.figure(figsize=(12, 6))

# Gráfico de series temporales de residuos
plt.subplot(2, 2, 1)
plt.plot(residuals[1:])
plt.title('Gráfico de Residuos a lo largo del tiempo')

# Histograma de los residuos
plt.subplot(2, 2, 2)
sns.histplot(residuals[1:], kde=True)
plt.title('Histograma de Residuos')

# Gráfico de cuantiles-cuantiles (Q-Q)
plt.subplot(2, 2, 3)
stats.probplot(residuals[1:], plot=plt)
plt.title('Gráfico Q-Q de Residuos')

# Gráfico de autocorrelación de residuos

ax = plt.subplot(2, 2, 4)
plot_acf(residuals[1:], lags=15, ax=ax)
plt.title('Gráfico ACF de Residuos')

# Añadir título al conjunto de subplots
plt.suptitle('Calibración por minimización MSE - Análisis de Residuos', fontsize=16)
plt.tight_layout()
plt.show()

# Prueba de normalidad (Jarque-Bera)
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
print(f'Estadística de prueba Jarque-Bera: {jb_stat}')
print(f'p-valor jb: {jb_pvalue}')
print(f'Skewness de los residuos: {skew}')
print(f'Kurtosis de los residuos: {kurtosis}')

# Convertir predictions a Serie de Pandas con el mismo índice que test
predictions_with_index = pd.Series(final_predictions.values, index=test.index)
# Calcular el MSE
mse = ((predictions_with_index - test) ** 2).mean()
print(f'Error cuadrático medio (MSE): {mse}')
mse_libreria = mean_squared_error(test, final_predictions)
print(f'MSE_libreria: {mse_libreria}')
# ("No supported index is available")
#predicciones tienen un índice de tipo entero sin información sobre las fechas.
#ignorar esta advertencia, ya que el objetivo principal es obtener el MSE 
#y la visualización de las predicciones



####################################### CALIBRACION BIC

import numpy as np
from itertools import product

# Función para evaluar un modelo ARIMA con BIC
def evaluate_arima_bic(data, pdq):
    p, d, q = pdq
    model = ARIMA(data, order=(p, d, q))
    try:
        results = model.fit()
        bic = results.bic
    except:
        bic = np.inf
    return bic

# Función para encontrar los mejores parámetros usando BIC
def find_best_arima_params(data, p_range, d_range, q_range):
    # Generar todas las combinaciones posibles de parámetros
    pdq_combinations = list(product(p_range, d_range, q_range))

    # Evaluar cada combinación utilizando BIC
    bics = [evaluate_arima_bic(data, pdq) for pdq in pdq_combinations]

    # Encontrar la combinación con el menor BIC
    best_pdq = pdq_combinations[np.argmin(bics)]
    best_bic = np.min(bics)

    return best_pdq, best_bic

# Rangos de valores para p, d, y q
p_range = range(0, 4)
d_range = range(0, 4)
q_range = range(0, 4)

# Encontrar los mejores parámetros utilizando BIC
best_pdq, best_bic = find_best_arima_params(df_original['Electricity consumption'], p_range, d_range, q_range)

# División de datos
train_size = int(len(df_original) * 0.8)
train, test = df_original['Electricity consumption'][:train_size], df_original['Electricity consumption'][train_size:]

# Ajuste final con los mejores parámetros encontrados
best_model = ARIMA(train, order=best_pdq)
best_model_fit = best_model.fit()
best_predictions = best_model_fit.forecast(steps=len(test))

plt.plot(train.index[:], train[:], label='Datos 1990-2015 (train)')
plt.plot(train.index[2:], best_model_fit.fittedvalues[2:], label='Modelo entrenado')
plt.plot(test.index, test, label='Datos 2015-2022')
plt.plot(test.index, best_predictions, label='Predicciones')
plt.title('Calibracion BIC "Bayesian Information Criterion" ARIMA (0,2,1)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Verificación de residuos
residuals = best_model_fit.resid

#creamos subplots
plt.figure(figsize=(12, 6))

# Gráfico de series temporales de residuos
plt.subplot(2, 2, 1)
plt.plot(residuals[2:])
plt.title('Gráfico de Residuos a lo largo del tiempo')

# Histograma de los residuos
plt.subplot(2, 2, 2)
sns.histplot(residuals[2:], kde=True)
plt.title('Histograma de Residuos')

# Gráfico de cuantiles-cuantiles (Q-Q)
plt.subplot(2, 2, 3)
stats.probplot(residuals[2:], plot=plt)
plt.title('Gráfico Q-Q de Residuos')

# Gráfico de autocorrelación de residuos

ax = plt.subplot(2, 2, 4)
plot_acf(residuals[2:], lags=15, ax=ax)
plt.title('Gráfico ACF de Residuos')

# Añadir título al conjunto de subplots
plt.suptitle('Calibración por minimización BIC - Análisis de Residuos', fontsize=16)
plt.tight_layout()
plt.show()

# Prueba de normalidad (Jarque-Bera)
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
print(f'Estadística de prueba Jarque-Bera: {jb_stat}')
print(f'p-valor jb: {jb_pvalue}')
print(f'Skewness de los residuos: {skew}')
print(f'Kurtosis de los residuos: {kurtosis}')

# Convertir predictions a Serie de Pandas con el mismo índice que test
predictions_with_index = pd.Series(best_predictions.values, index=test.index)
# Calcular el MSE
mse = ((predictions_with_index - test) ** 2).mean()
print(f'Error cuadrático medio (MSE): {mse}')
mse_libreria = mean_squared_error(test, best_predictions)
print(f'MSE_libreria: {mse_libreria}')
# ("No supported index is available")
#predicciones tienen un índice de tipo entero sin información sobre las fechas.
#ignorar esta advertencia, ya que el objetivo principal es obtener el MSE 
#y la visualización de las predicciones



############################ COMPARACIÓN DE LOS DOS MODELOS

# Modelos
orders = [best_params, best_pdq] #model,  final_model y best_model
model_names = ["Calibracion MSE -Minimización de error cuadrático medio- ARIMA (3,0,0)", "Calibracion BIC -Bayesian Information Criterion- ARIMA (0,2,1)"]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

for i, (order, model_name) in enumerate(zip(orders, model_names)):
    # Ajuste del modelo
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    
    # Representación en subgráficos
    axes[i].plot(train.index[:], train[:], label='Datos 1990-2015 (train)')
    axes[i].plot(train.index[2:], model_fit.fittedvalues[2:], label='Modelo entrenado')
    axes[i].plot(test.index, test, label='Datos 2015-2022')
    axes[i].plot(test.index, predictions, label='Predicciones')
    axes[i].set_title(f'{model_name}')
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

############################## GENERAR PREDICCIONES FUERA DE LA MUESTA (2023)

# Asumiendo que tu índice actual es de tipo año
current_index = test.index

# Número adicional de años para predecir más allá de los datos de prueba
num_additional_years = 5

# Crear un nuevo rango de años para las predicciones adicionales
additional_index = current_index.append(pd.Index(range(current_index[-1] + 1, current_index[-1] + 1 + num_additional_years)))

# Predicciones adicionales más allá de los datos de prueba
final_predictions_additional = final_model_fit.forecast(steps=len(test) + num_additional_years)

#
# Visualización de las predicciones
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test', color='red')
#plt.plot(test.index, final_predictions, label='Test predictions', color='green')
plt.plot(additional_index, final_predictions_additional, label='Additional Predictions (hasta 2027)', color='purple')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()




